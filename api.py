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
import threading
import time

import gpxpy
import gpxpy.gpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response

from gpx_shape_generator import SHAPES, _haversine_m
from routing import (
    PROFILES, RoadDataUnavailable, RouteError,
    check_shapes, connect_start, diagnose_overpass, fetch_osm_graph, find_best_placements, route_shape_on_roads,
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
                    "X-Coverage-Pct", "X-Route-Km", "X-Profile",
                    "X-Shape-Km", "X-Approach-Km"],
)

PUBLIC_SHAPES = ["heart", "star", "circle", "infinity", "triangle", "square", "diamond"]

SHIFT_M = 1500
HEAVY_LOCK = threading.Lock()
MAX_RADIUS_M = 6500
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


def _make(shape, w):
    fn = SHAPES[shape]
    return fn(n_points=240, radius_m=w / 2) if shape == "circle" else fn(n_points=240, width_m=w)


def _length_km(pts):
    return sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)) / 1000.0


def shape_points(shape, length_km=None, width_m=5000.0):
    """Points de la forme (en mètres) dimensionnée pour la longueur demandée."""
    if length_km is not None:
        probe_km = _length_km(_make(shape, 4000))
        width_m = min(10000.0, max(1500.0, 4000.0 * length_km / (ROUTE_OVERHEAD * probe_km)))
    pts = _make(shape, width_m)
    return pts, _length_km(pts)


def area_radius(pts):
    """Rayon de la zone de routes à télécharger : forme + marge pour la décaler."""
    # plafonné : au-delà, le réseau ne tient pas dans la mémoire du serveur gratuit
    return min(MAX_RADIUS_M, int(max(abs(c) for p in pts for c in p) * 1.2 + SHIFT_M + 300))


# Les formes simples supportent mieux un réseau clairsemé que le cœur ou l'étoile,
# dont la pointe et les creux doivent tomber pile sur des routes.
SIMPLE_SHAPES = {"circle", "triangle", "diamond", "square"}


def _check(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    length_km: float = Query(25, ge=10, le=45),
    profile: str = Query("route"),
):
    """Indique, pour ce point de départ, quelles formes ont des chances d'être
    reconnaissables (estimation rapide, sans calcul d'itinéraire)."""
    if profile not in PROFILES:
        raise HTTPException(400, f"Profil inconnu '{profile}'. Profils : {list(PROFILES)}")
    shapes = {s: shape_points(s, length_km)[0] for s in PUBLIC_SHAPES}
    radius = max(area_radius(p) for p in shapes.values())
    try:
        G = fetch_osm_graph(lat, lon, radius, profile)
    except RoadDataUnavailable as e:
        raise HTTPException(503, str(e))
    except RouteError as e:
        raise HTTPException(422, str(e))
    t0 = time.monotonic()
    raw = check_shapes(G, shapes, lat, lon, max_shift_m=SHIFT_M)
    # L'estimation forme par forme est trop bruitée pour trancher seule ; on s'en sert
    # pour juger la densité de la ZONE, puis on applique ce que les essais ont montré :
    # les formes simples passent même quand les routes sont rares, pas le cœur ni l'étoile.
    zone = sum(raw.values()) / len(raw)
    level = "dense" if zone >= 58 else "moyen" if zone >= 44 else "clairseme"
    table = {
        "dense": {"simple": "conseillee", "complexe": "conseillee"},
        "moyen": {"simple": "conseillee", "complexe": "possible"},
        "clairseme": {"simple": "possible", "complexe": "deconseillee"},
    }[level]
    result = {s: {"verdict": table["simple" if s in SIMPLE_SHAPES else "complexe"]} for s in PUBLIC_SHAPES}
    logger.info("check lat=%.4f lon=%.4f %skm %s : zone=%.0f (%s) détail=%s (%.1fs)", lat, lon, length_km,
                profile, zone, level, raw, time.monotonic() - t0)
    advice = {
        "dense": None,
        "moyen": "Routes moyennement denses : les formes simples (cercle, triangle, losange, carré) seront les plus nettes.",
        "clairseme": ("Peu de routes adaptées ici : le cœur, l'étoile et l'infini risquent d'être méconnaissables. "
                      "Préfère une forme simple, le mode VTT, ou un départ plus proche d'une ville."),
    }[level]
    return {"formes": result, "conseil": advice, "zone": level}


def _generate(
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

    # 1. Taille de la forme. Un parcours sur routes est ~1,4x plus long que la forme idéale.
    pts, ideal_km = shape_points(shape, length_km, width_m)
    target_km = length_km if length_km is not None else ideal_km * ROUTE_OVERHEAD

    # 2. Vraies routes autour (zone = forme + marge pour pouvoir décaler la forme)
    radius = area_radius(pts)
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
            km_err = abs(mm["route_km"] - target_km) / target_km
            score = (mm["mean_dev_m"] / max(mm["coverage_pct"], 1.0) * (1.0 + 4.0 * km_err)
                     * (1.0 + mm.get("key_dev_m", 0.0) / 120.0))
            if ideal > 0:
                # rapport longueur réelle / forme idéale, mesuré sur les meilleurs tracés
                ratios.append((mm["mean_dev_m"] / max(mm["coverage_pct"], 1.0), mm["route_km"] / ideal))
            if best is None or score < best[0]:
                best = (score, r, mm, p)
        return ratios

    # Tour 1 : taille nominale. On mesure de combien le parcours réel dépasse la forme idéale ici.
    ratios = attempt(pts, 3)
    # Tour 2 : on recale la taille de la forme pour tomber sur la longueur demandée.
    if ratios and time.monotonic() - t0 < SEARCH_BUDGET_S:
        ratio = min(ratios)[1]
        k = min(1.6, max(0.5, target_km / (ratio * ideal_km)))
        if abs(k - 1.0) > 0.08:
            logger.info("recalage de la taille : x%.2f (parcours réel / forme idéale = %.2f)", k, ratio)
            attempt([(x * k, y * k) for x, y in pts], 4)
    if best is None:
        raise HTTPException(422, "Impossible de tracer cette forme ici. Essaie une zone plus urbaine ou une autre longueur.")
    _, route, m, chosen = best
    # Départ et retour au point choisi par l'utilisateur
    route, approach_m = connect_start(G, route, lat, lon)
    m["approach_km"] = round(approach_m / 1000.0, 2)
    m["total_km"] = round(m["route_km"] + 2 * approach_m / 1000.0, 2)
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
            "X-Route-Km": str(m["total_km"]),
            "X-Shape-Km": str(m["route_km"]),
            "X-Approach-Km": str(m["approach_km"]),
            "X-Profile": profile,
        },
    )


@app.get("/check")
def check(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    length_km: float = Query(25, ge=10, le=45),
    profile: str = Query("route"),
):
    # Un seul calcul lourd à la fois : deux réseaux routiers en mémoire en même
    # temps dépassent les 512 Mo du serveur gratuit (le serveur redémarrait).
    with HEAVY_LOCK:
        return _check(lat=lat, lon=lon, length_km=length_km, profile=profile)


@app.get("/generate")
def generate(
    lat: float = Query(..., ge=-90, le=90, description="Latitude du point de départ / centre"),
    lon: float = Query(..., ge=-180, le=180, description="Longitude du point de départ / centre"),
    shape: str = Query("heart", description=f"Forme : {', '.join(PUBLIC_SHAPES)}"),
    length_km: float | None = Query(None, ge=10, le=45, description="Longueur visée du parcours en km (10-45). Prioritaire sur width_m."),
    width_m: float = Query(5000, ge=1000, le=10000, description="Largeur de la forme en mètres, si length_km n'est pas donné."),
    profile: str = Query("route", description="route = routes revêtues, vtt = chemins et pistes inclus"),
):
    # Un seul calcul lourd à la fois : deux réseaux routiers en mémoire en même
    # temps dépassent les 512 Mo du serveur gratuit (le serveur redémarrait).
    with HEAVY_LOCK:
        return _generate(lat=lat, lon=lon, shape=shape, length_km=length_km, width_m=width_m, profile=profile)
