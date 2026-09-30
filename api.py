"""
api.py
======
API backend pour Traceo. Reçoit une adresse (ou des coordonnées), une forme
et une largeur, et renvoie DIRECTEMENT le fichier GPX en réponse HTTP —
le navigateur télécharge alors le fichier comme n'importe quel fichier servi
par un serveur (bien plus fiable que le Blob/<a download> côté client qu'on
a vu planter selon les navigateurs/webviews).

Lancer en local :
    pip install fastapi "uvicorn[standard]" osmnx networkx shapely gpxpy numpy python-multipart
    uvicorn api:app --reload

Puis ouvrir http://127.0.0.1:8000/docs pour tester interactivement,
ou appeler directement :
    GET /generate?lat=45.6064&lon=4.8332&shape=heart&width_m=1300&use_real_streets=true

NOTE : dans CE sandbox de conversation, l'API Overpass (donnees OSM reelles)
n'est pas accessible (reseau restreint) — use_real_streets retombera donc
automatiquement sur le reseau synthetique de test avec un avertissement.
Une fois deploye sur un vrai serveur (Render, Railway, Fly.io...), 
use_real_streets=true utilisera le vrai reseau routier.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import io
import logging

from gpx_shape_generator import (
    SHAPES, get_real_network_lite, generate_synthetic_street_network,
    fit_shape_to_network, route_accuracy, export_gpx,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("traceo-api")

app = FastAPI(title="Traceo API", description="Génère des parcours GPX en forme, ajustés au réseau routier.")

# CORS ouvert pour permettre a l'appli web (servie depuis un autre domaine,
# ex. Vercel/Netlify) d'appeler cette API. A restreindre a ton propre domaine
# en production (allow_origins=["https://tondomaine.com"]).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/generate")
def generate(
    lat: float = Query(..., description="Latitude du centre du parcours"),
    lon: float = Query(..., description="Longitude du centre du parcours"),
    shape: str = Query("heart", description=f"Forme parmi : {', '.join(SHAPES.keys())}"),
    width_m: float = Query(1300, ge=200, le=5000, description="Largeur cible en metres (200-5000)"),
    network_type: str = Query("bike", description="Type de reseau OSM : bike, walk, drive"),
    use_real_streets: bool = Query(True, description="Utiliser le vrai reseau OSM (necessite un deploiement avec acces internet)"),
):
    if shape not in SHAPES:
        raise HTTPException(400, f"Forme inconnue '{shape}'. Formes disponibles : {list(SHAPES.keys())}")
    if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        raise HTTPException(400, "Coordonnées invalides.")

    # 1. Recuperer le reseau routier : reel si possible, sinon repli synthetique
    used_real_network = False
    G = None
    if use_real_streets:
        try:
            G = get_real_network_lite(lat, lon, dist_m=max(1500, int(width_m * 1.3)), network_type=network_type)
            used_real_network = True
        except Exception as e:
            logger.warning(f"Reseau OSM reel indisponible ({e}) — repli sur le reseau synthetique.")
    if G is None:
        G = generate_synthetic_street_network(lat, lon, extent_m=max(2000, int(width_m * 1.8)), cell_m=80)

    # 2. Generer la forme et l'ajuster au reseau
    shape_fn = SHAPES[shape]
    shape_pts = shape_fn(n_points=120, width_m=width_m) if shape != "circle" else shape_fn(n_points=120, radius_m=width_m / 2)

    try:
        route_nodes, shape_latlon, _ = fit_shape_to_network(G, shape_pts, lat, lon)
    except Exception as e:
        logger.exception("Echec de l'ajustement au reseau")
        raise HTTPException(500, f"Échec de génération du parcours : {e}")

    if len(route_nodes) < 4:
        raise HTTPException(422, "Réseau routier trop clairsemé ici pour cette forme — essaie une largeur plus petite ou un autre point de départ.")

    mean_err, max_err = route_accuracy(shape_latlon, [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route_nodes])

    # 3. Exporter en GPX (en memoire, pas de fichier temporaire sur disque)
    buf = io.StringIO()
    import gpxpy, gpxpy.gpx
    gpx = gpxpy.gpx.GPX()
    track = gpxpy.gpx.GPXTrack(name=f"Traceo - {shape}")
    gpx.tracks.append(track)
    segment = gpxpy.gpx.GPXTrackSegment()
    track.segments.append(segment)
    for n in route_nodes:
        segment.points.append(gpxpy.gpx.GPXTrackPoint(G.nodes[n]['y'], G.nodes[n]['x']))
    gpx_xml = gpx.to_xml()

    filename = f"traceo_{shape}.gpx"
    return Response(
        content=gpx_xml,
        media_type="application/gpx+xml",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Used-Real-Network": str(used_real_network),
            "X-Mean-Error-M": f"{mean_err:.1f}",
            "X-Max-Error-M": f"{max_err:.1f}",
        },
    )
