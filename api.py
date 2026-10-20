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
import os
import time

import gpxpy
import gpxpy.gpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response

from gpx_shape_generator import SHAPES, _haversine_m
from routing import (
    PROFILES, RoadDataUnavailable, RouteError,
    diagnose_overpass, fetch_osm_graph, find_best_placements, route_shape_on_roads,
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

SHIFT_M = 1500
SEARCH_BUDGET_S = 45
ROUTE_OVERHEAD = 1.4  # longueur réelle du parcours / longueur de la forme idéale


@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/app")


@app.get("/app", include_in_schema=False)
def web_app():
    """L'interface web de Traceo (une seule page)."""
    return FileResponse(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.html"),
                        media_type="text/html")


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
    lat: float = Query(..., ge=-90, le=90, description="Latitude du point de départ / centre"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude du point de départ / centre"),
    shape: str = Query("heart", description=f"Forme : {', '.join(PUBLIC_SHAPES)}"),
    length_km: float | None = Query(None, ge=10, le=45, description="Longueur visée du parcours en km (10-45). Prioritaire sur width_m."),
    width_m: float = Query(5000, ge=1000, le=10000, description="Largeur de la forme en mètres, si length_km n'est pas donné."),
    profile: str = Query("route", description="route = routes revêtues, vtt = chemins et pistes inclus"),
):
    if shape not in PUBLIC_SHAPES:
        raise HTTPException(400, f"Forme inconnue '{shape}'. Formes disponibles : {PUBLIC_SHAPES}")
    if profile not in PROFILES:
        raise HTTPException(400, f"Profil inconnu '{profile}'. Profils : {list(PROFILES)}")

    fn = SHAPES[shape]

    def make(w):
        return fn(n_points=240, radius_m=w / 2) if shape == "circle" else fn(n_points=240, width_m=w)

    # 1. Taille de la forme. Un parcours sur routes est ~1,4x plus long que la forme idéale.
    if length_km is not None:
        probe = make(4000)
        probe_km = sum(math.dist(probe[i], probe[i + 1]) for i in range(len(probe) - 1)) / 1000.0
        width_m = min(10000.0, max(1500.0, 4000.0 * length_km / (ROUTE_OVERHEAD * probe_km)))
    pts = make(width_m)
    ideal_km = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)) / 1000.0
    target_km = length_km if length_km is not None else ideal_km * ROUTE_OVERHEAD

    # 2. Vraies routes autour (zone = forme + marge pour pouvoir décaler la forme)
    radius = int(max(abs(c) for p in pts for c in p) * 1.2 + SHIFT_M + 300)
    try:
        G = fetch_osm_graph(lat, lon, radius, profile)
    except RoadDataUnavailable as e:
        raise HTTPException(503, str(e))
    except RouteError as e:
        raise HTTPException(422, str(e))

    # 3. Où poser la forme ? On teste de nombreuses positions / tailles / rotations
    #    (évaluation rapide), puis on trace pour de bon les meilleures et on garde la plus fidèle.
    #    Le serveur gratuit est lent : on travaille en deux tours dans un budget de temps.
    t0 = time.monotonic()
    best = None

    def attempt(shape_pts, n_routes):
        """Cherche les meilleures positions de la forme et trace les n_routes premières."""
        nonlocal best
        try:
            placements = find_best_placements(G, shape_pts, lat, lon, top=n_routes, max_shift_m=SHIFT_M)
        except Exception:
            logger.exception("Echec de la recherche de position")
            placements = [{"pts": shape_pts, "dx": 0, "dy": 0, "scale": 1.0, "rot": 0}]
        ratios = []
        for p in placements:
            if best is not None and time.monotonic() - t0 > SEARCH_BUDGET_S:
                break
            try:
                r, mm = route_shape_on_roads(G, p["pts"], lat, lon)
            except RouteError:
                continue
            except Exception:
                logger.exception("Echec du tracé")
                continue
            ideal = mm["shape_km"]
            if ideal > 0:
                ratios.append(mm["route_km"] / ideal)
            km_err = abs(mm["route_km"] - target_km) / target_km
            score = mm["mean_dev_m"] / max(mm["coverage_pct"], 1.0) * (1.0 + 2.0 * km_err)
            if best is None or score < best[0]:
                best = (score, r, mm, p)
        return ratios

    # Tour 1 : taille nominale. On mesure de combien le parcours réel dépasse la forme idéale ici.
    ratios = attempt(pts, 3)
    # Tour 2 : on recale la taille de la forme pour tomber sur la longueur demandée.
    if ratios and time.monotonic() - t0 < SEARCH_BUDGET_S:
        ratio = sorted(ratios)[len(ratios) // 2]
        k = min(1.25, max(0.5, target_km / (ratio * ideal_km)))
        if abs(k - 1.0) > 0.08:
            logger.info("recalage de la taille : x%.2f (parcours réel / forme idéale = %.2f)", k, ratio)
            attempt([(x * k, y * k) for x, y in pts], 4)
    if best is None:
        raise HTTPException(422, "Impossible de tracer cette forme ici. Essaie une zone plus urbaine ou une autre longueur.")
    _, route, m, chosen = best
    logger.info("durée recherche+tracé %.1fs", time.monotonic() - t0)
    logger.info("position retenue dx=%s dy=%s échelle=%s rot=%s cible=%.0fkm", chosen["dx"], chosen["dy"], chosen["scale"], chosen["rot"], target_km)

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
