"""
api.py — API Traceo
===================
Reçoit un point de départ, une forme et une largeur, et renvoie un fichier GPX
qui suit les VRAIES routes (OpenStreetMap) pour dessiner cette forme à vélo.

Si les données de routes ne sont pas joignables, l'API répond par une erreur
claire (503) : elle ne fabrique JAMAIS un faux tracé qui ne suivrait aucune route.

Lancer en local :  uvicorn api:app --reload   puis ouvrir http://127.0.0.1:8000/docs
"""

import logging
import math

import gpxpy
import gpxpy.gpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, Response

from gpx_shape_generator import SHAPES
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

    # 2. Vraies routes autour (zone = forme + marge)
    radius = int(max(abs(c) for p in pts for c in p) + 500)
    try:
        G = fetch_osm_graph(lat, lon, radius, profile)
    except RoadDataUnavailable as e:
        raise HTTPException(503, str(e))
    except RouteError as e:
        raise HTTPException(422, str(e))

    # 3. Tracer la forme sur ces routes
    try:
        route, m = route_shape_on_roads(G, pts, lat, lon)
    except RouteError as e:
        raise HTTPException(422, str(e))
    except Exception:
        logger.exception("Echec du tracé")
        raise HTTPException(500, "Erreur interne pendant le calcul du parcours.")

    # 4. GPX
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack(name=f"Traceo - {shape}")
    gpx.tracks.append(track)
    seg = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(seg)
    for n in route:
        seg.points.append(gpxpy.gpx.GPXTrackPoint(G.nodes[n]["y"], G.nodes[n]["x"]))

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
