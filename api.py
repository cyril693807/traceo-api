"""
api.py — API Traceo
===================
Reçoit un point de départ, une forme et une largeur, et renvoie un fichier GPX
qui suit les VRAIES routes (OpenStreetMap) pour dessiner cette forme à vélo.

Si les données de routes ne sont pas joignables, l'API répond par une erreur
claire (503) : elle ne fabrique JAMAIS un faux tracé qui ne suivrait aucune route.

Lancer en local :  uvicorn api:app --reload   puis ouvrir http://127.0.0.1:8000/docs
"""

import datetime
import logging
import math
import time

import gpxpy
import gpxpy.gpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response

from gpx_shape_generator import SHAPES, _haversine_m
from routing import (
    PROFILES, RoadDataUnavailable, RouteError,
    diagnose_overpass, fetch_osm_graph, route_shape_on_roads,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("traceo-api")

app = FastAPI(
    title="Traceo API",
    description="Génère des parcours GPX vélo en forme (cœur, étoile…) qui suivent les vraies routes.",
)

# Les navigateurs ne laissent lire que certains en-têtes de réponse : on expose
# ceux qu'utilise l'interface (qualité du tracé, etc.).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Quality", "X-Mean-Deviation-M",
                    "X-Coverage-Pct", "X-Route-Km", "X-Profile"],
)

PUBLIC_SHAPES = ["heart", "star", "circle", "infinity"]

SHIFT_M = 1000
SEARCH_BUDGET_S = 40
# (décalage est, décalage nord, échelle) : essayés dans l'ordre, on garde le meilleur
CANDIDATES = [(0, 0, 1.0)] + [(dx, dy, 1.0) for dx, dy in
    [(700, 0), (-700, 0), (0, 700), (0, -700), (700, 700), (-700, 700), (700, -700), (-700, -700),
     (1000, 0), (-1000, 0), (0, 1000), (0, -1000)]] + [(0, 0, 1.2), (0, 0, 0.85)]


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/diagnostic")
def diagnostic():
    """Indique quels serveurs OpenStreetMap répondent depuis cet hébergement."""
    return {"serveurs_openstreetmap": diagnose_overpass()}


@app.get("/graph", include_in_schema=False)
def graph_dump(lat: float, lon: float, radius_m: int = Query(3500, ge=500, le=6000), profile: str = "route"):
    """Export compact du réseau routier (pour tester l'algorithme hors ligne)."""
    try:
        G = fetch_osm_graph(lat, lon, radius_m, profile)
    except RoadDataUnavailable as e:
        raise HTTPException(503, str(e))
    ids = {n: i for i, n in enumerate(G.nodes)}
    return {
        "nodes": [[round(G.nodes[n]["y"], 6), round(G.nodes[n]["x"], 6)] for n in G.nodes],
        "edges": [[ids[u], ids[v], round(d["length"], 1), d.get("mult", 1.0)] for u, v, d in G.edges(data=True)],
    }


@app.get("/generate")
def generate(
    lat: float = Query(..., ge=-90, le=90, description="Latitude du centre"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude du centre"),
    shape: str = Query("heart", description=f"Forme : {', '.join(PUBLIC_SHAPES)}"),
    width_m: float = Query(4000, ge=1000, le=10000, description="Largeur de la forme en mètres (1000-10000). Moins de 3000 m donne des formes grossières, sauf en ville très dense."),
    profile: str = Query("route", description="route = routes revêtues, vtt = chemins et pistes inclus"),
):
    if shape not in PUBLIC_SHAPES:
        raise HTTPException(400, f"Forme inconnue '{shape}'. Formes disponibles : {PUBLIC_SHAPES}")
    if profile not in PROFILES:
        raise HTTPException(400, f"Profil inconnu '{profile}'. Profils : {list(PROFILES)}")

    # 1. Forme idéale, en mètres autour du centre
    fn = SHAPES[shape]
    pts = fn(n_points=240, radius_m=width_m / 2) if shape == "circle" else fn(n_points=240, width_m=width_m)

    # 2. Vraies routes autour (zone = forme + marge pour pouvoir décaler la forme)
    radius = int(max(abs(c) for p in pts for c in p) + 500 + SHIFT_M)
    try:
        G = fetch_osm_graph(lat, lon, radius, profile)
    except RoadDataUnavailable as e:
        raise HTTPException(503, str(e))
    except RouteError as e:
        raise HTTPException(422, str(e))

    # 3. Tracer la forme sur ces routes. Le réseau routier n'est pas régulier :
    #    on essaie quelques décalages / tailles et on garde le tracé le plus fidèle.
    t0 = time.monotonic()
    best = None
    for dx, dy, k in CANDIDATES:
        if best is not None and time.monotonic() - t0 > SEARCH_BUDGET_S:
            break
        cand = [(x * k + dx, y * k + dy) for x, y in pts]
        try:
            r, mm = route_shape_on_roads(G, cand, lat, lon)
        except RouteError:
            continue
        except Exception:
            logger.exception("Echec du tracé")
            continue
        score = mm["mean_dev_m"] / max(mm["coverage_pct"], 1.0)
        if best is None or score < best[0]:
            best = (score, r, mm, (dx, dy, k))
        if mm["quality"] == "bon":
            break
    if best is None:
        raise HTTPException(422, "Impossible de tracer cette forme ici. Essaie une forme plus grande ou une zone plus urbaine.")
    _, route, m, chosen = best
    logger.info("candidat retenu dx=%s dy=%s échelle=%s", *chosen)

    # 4. GPX (sans doublons consécutifs, avec altitude et heure fictives :
    #    certains services, dont Garmin Connect, rejettent les traces sans ces champs)
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack(name=f"Traceo - {shape}")
    gpx.tracks.append(track)
    seg = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(seg)
    t = datetime.datetime(2026, 1, 1, 8, 0, 0, tzinfo=datetime.timezone.utc)
    prev = None
    prev_xy = None
    for n in route:
        xy = (G.nodes[n]["y"], G.nodes[n]["x"])
        if xy == prev_xy:
            continue
        if prev_xy is not None:
            d = _haversine_m(prev_xy[0], prev_xy[1], xy[0], xy[1])
            t += datetime.timedelta(seconds=max(1, round(d / 5.5)))  # ~20 km/h, secondes entières
        seg.points.append(gpxpy.gpx.GPXTrackPoint(round(xy[0], 6), round(xy[1], 6), elevation=0, time=t))
        prev_xy = xy

    logger.info("shape=%s profile=%s qualité=%s écart=%.0fm couverture=%.0f%% %.1fkm | détail=%s | réseau utilisé=%d nœuds",
                shape, profile, m["quality"], m["mean_dev_m"], m["coverage_pct"], m["route_km"], m, G.number_of_nodes())
    return Response(
        content=gpx.to_xml(),
        media_type="application/gpx+xml",
        headers={
            "Content-Disposition": f'attachment; filename="traceo_{shape}.gpx"',
            "X-Quality": m["quality"],
            "X-Mean-Deviation-M": str(m["mean_dev_m"]),
            "X-Coverage-Pct": str(m["coverage_pct"]),
            "X-Route-Km": str(m["route_km"]),
            "X-Profile": profile,
        },
    )
