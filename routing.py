"""
routing.py
==========
Récupération des VRAIES routes (OpenStreetMap) et tracé d'une forme dessus.

Ce module remplace l'ancien calcul "snap + plus court chemin", qui avait deux
défauts :
  1. Il retombait SILENCIEUSEMENT sur un faux réseau de rues en quadrillage
     quand OpenStreetMap ne répondait pas -> GPX qui ne suit aucune route réelle.
  2. Il cherchait le plus court chemin entre points de la forme sans se soucier
     de rester près de la forme -> raccourcis, allers-retours, tracé déformé.

Ici :
  - on interroge plusieurs serveurs OpenStreetMap, avec nouvelles tentatives ;
  - si aucun ne répond, on LÈVE UNE ERREUR (jamais de faux tracé en silence) ;
  - on filtre les voies selon le profil (route / VTT) : pas d'autoroute,
    pas d'escalier, pas de trottoir, pas de voie privée ;
  - chaque tronçon de rue est pénalisé selon son éloignement de la forme, donc
    le trajet "colle" à la forme au lieu de couper au plus court ;
  - on supprime les allers-retours (impasses) et on mesure la fidélité.
"""

import logging
import math
import time

import networkx as nx
import numpy as np
import requests
import shapely
from shapely.geometry import LineString

from gpx_shape_generator import _haversine_m, _resample_polyline

logger = logging.getLogger("traceo-routing")

M_PER_DEG = 111_320.0

OVERPASS_MIRRORS = [
    "https://overpass.openstreetmap.fr/api/interpreter",   # seul serveur joignable depuis Render lors du test du 20/10/2026
    "https://overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]

# Serveurs qui viennent d'échouer : on les essaie en dernier pendant 10 minutes,
# pour ne pas perdre du temps (jusqu'à 35 s chacun) sur un serveur en panne.
_MIRROR_FAILED_AT = {}
_MIRROR_PENALTY_S = 600


def _ordered_mirrors():
    now = time.time()
    def key(item):
        i, url = item
        recently_failed = now - _MIRROR_FAILED_AT.get(url, 0) < _MIRROR_PENALTY_S
        return (recently_failed, i)
    return [url for _, url in sorted(enumerate(OVERPASS_MIRRORS), key=key)]

HEADERS = {
    "User-Agent": "Traceo/1.0 (generateur de parcours GPX en forme; https://traceo-api.onrender.com)",
    "Accept": "application/json",
}

# Multiplicateur de "coût" selon le type de voie (1.0 = neutre, >1 = on évite).
HIGHWAY_COST = {
    "cycleway": 0.8,
    "living_street": 1.0,
    "residential": 1.0,
    "unclassified": 1.0,
    "tertiary": 1.1,
    "tertiary_link": 1.1,
    "service": 1.2,
    "road": 1.2,
    "secondary": 1.5,
    "secondary_link": 1.5,
    "primary": 3.0,
    "primary_link": 3.0,
    "track": 0.9,
    "path": 0.9,
}

PROFILES = {
    "route": {
        "label": "Vélo de route (routes revêtues)",
        "highways": ["cycleway", "living_street", "residential", "unclassified",
                     "tertiary", "tertiary_link", "secondary", "secondary_link",
                     "primary", "primary_link", "service", "road"],
        "paved_only": True,
    },
    "vtt": {
        "label": "VTT / gravel (chemins et pistes inclus)",
        "highways": ["cycleway", "living_street", "residential", "unclassified",
                     "tertiary", "tertiary_link", "secondary", "secondary_link",
                     "primary", "primary_link", "service", "road", "track", "path"],
        "paved_only": False,
    },
}


class RoadDataUnavailable(Exception):
    """Les serveurs OpenStreetMap n'ont pas répondu (ou réponse inutilisable)."""


class RouteError(Exception):
    """Données routières trop pauvres ici pour dessiner cette forme."""


# ----------------------------------------------------------------------------
# 1. Récupération du réseau routier réel
# ----------------------------------------------------------------------------

def build_overpass_query(south, west, north, east, profile="route", timeout=30):
    p = PROFILES[profile]
    highways = "|".join(p["highways"])
    surface = ""
    if p["paved_only"]:
        surface = ('["surface"!~"^(unpaved|dirt|gravel|ground|grass|sand|mud|earth|'
                   'fine_gravel|compacted|pebblestone|woodchips)$"]')
    return (
        f"[out:json][timeout:{timeout}];\n"
        f'way["highway"~"^({highways})$"]'
        f'["bicycle"!~"^(no|dismount)$"]'
        f'["access"!~"^(private|no)$"]'
        f'["service"!~"^(driveway|parking_aisle|drive-through)$"]'
        f"{surface}"
        f"({south:.6f},{west:.6f},{north:.6f},{east:.6f});\n"
        "out body;\n>;\nout skel qt;"
    )


def _graph_from_overpass(data):
    remark = data.get("remark", "")
    if "runtime error" in remark.lower() or "timed out" in remark.lower():
        # Réponse tronquée : un réseau incomplet donnerait un parcours faux.
        raise RoadDataUnavailable("Réponse OpenStreetMap incomplète : " + remark[:120])
    coords = {}
    ways = []
    for el in data.get("elements", []):
        t = el.get("type")
        if t == "node" and "lat" in el and "lon" in el:
            coords[el["id"]] = (el["lat"], el["lon"])
        elif t == "way":
            ways.append(el)

    if not ways:
        remark = data.get("remark", "")
        raise RoadDataUnavailable(
            "Réponse OpenStreetMap sans aucune route" + (f" ({remark})" if remark else "") + "."
        )

    G = nx.Graph()
    for w in ways:
        tags = w.get("tags") or {}
        mult = HIGHWAY_COST.get(tags.get("highway", ""), 1.2)
        nds = w.get("nodes", [])
        for a, b in zip(nds[:-1], nds[1:]):
            if a == b or a not in coords or b not in coords:
                continue
            la, loa = coords[a]
            lb, lob = coords[b]
            if a not in G:
                G.add_node(a, y=la, x=loa)
            if b not in G:
                G.add_node(b, y=lb, x=lob)
            length = _haversine_m(la, loa, lb, lob)
            if G.has_edge(a, b):
                G[a][b]["mult"] = min(G[a][b]["mult"], mult)
            else:
                G.add_edge(a, b, length=length, mult=mult)

    if G.number_of_nodes() < 30:
        raise RouteError("Presque aucune route utilisable autour de ce point pour ce profil.")

    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    biggest = comps[0]
    lats = [G.nodes[n]["y"] for n in G.nodes]
    lons = [G.nodes[n]["x"] for n in G.nodes]
    logger.info(
        "réseau OSM : %d voies, %d nœuds, %d tronçons, %d composantes, plus grande = %d nœuds (%.0f%%), "
        "emprise lat %.4f..%.4f lon %.4f..%.4f, remark=%r",
        len(ways), G.number_of_nodes(), G.number_of_edges(), len(comps), len(biggest),
        100.0 * len(biggest) / G.number_of_nodes(),
        min(lats), max(lats), min(lons), max(lons), remark[:80])
    return G.subgraph(biggest).copy()


_CACHE = {}
_CACHE_MAX = 6
_CACHE_TTL_S = 3600


def fetch_osm_graph(lat, lon, radius_m, profile="route", budget_s=60.0):
    """Télécharge le réseau routier réel autour d'un point.
    Essaie plusieurs serveurs OpenStreetMap, puis lève RoadDataUnavailable
    si aucun ne répond dans le temps imparti (pas de repli silencieux)."""
    if profile not in PROFILES:
        raise ValueError(f"Profil inconnu : {profile}")

    key = (round(lat, 3), round(lon, 3), int(radius_m // 100), profile)
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _CACHE_TTL_S:
        return hit[1]

    dlat = radius_m / M_PER_DEG
    dlon = radius_m / (M_PER_DEG * math.cos(math.radians(lat)))
    query = build_overpass_query(lat - dlat, lon - dlon, lat + dlat, lon + dlon, profile)

    deadline = time.monotonic() + budget_s
    errors = []
    for attempt in range(2):
        for url in _ordered_mirrors():
            remaining = deadline - time.monotonic()
            if remaining < 5:
                break
            host = url.split("/")[2]
            try:
                resp = requests.post(url, data={"data": query}, headers=HEADERS,
                                     timeout=(6.0, min(35.0, remaining)))
                if resp.status_code != 200:
                    errors.append(f"{host}: HTTP {resp.status_code}")
                    _MIRROR_FAILED_AT[url] = time.time()
                    continue
                G = _graph_from_overpass(resp.json())
                if len(_CACHE) >= _CACHE_MAX:
                    _CACHE.pop(next(iter(_CACHE)))
                _CACHE[key] = (time.time(), G)
                return G
            except RouteError:
                raise
            except Exception as e:  # réseau, JSON invalide, serveur surchargé...
                _MIRROR_FAILED_AT[url] = time.time()
                errors.append(f"{host}: {type(e).__name__}")
                logger.warning("Overpass %s : %s %s", host, type(e).__name__, str(e)[:160])
        if time.monotonic() >= deadline - 5:
            break
        time.sleep(2)

    raise RoadDataUnavailable(
        "Impossible de récupérer les routes OpenStreetMap pour l'instant "
        "(serveurs surchargés ou injoignables). Réessaie dans une minute. "
        "Détail : " + "; ".join(errors[-4:])
    )


def diagnose_overpass(lat=45.6064, lon=4.8332):
    """Test rapide de chaque serveur OpenStreetMap, pour voir lequel répond
    depuis l'hébergement (utile pour comprendre une panne). Les serveurs sont
    testés en parallèle, avec le détail de l'erreur."""
    from concurrent.futures import ThreadPoolExecutor

    query = f"[out:json][timeout:10];node(around:300,{lat},{lon});out count;"

    def probe(url):
        t0 = time.monotonic()
        entry = {"serveur": url.split("/")[2]}
        try:
            r = requests.post(url, data={"data": query}, headers=HEADERS, timeout=(8, 20))
            entry["http"] = r.status_code
            entry["ok"] = r.status_code == 200
            if r.status_code != 200:
                entry["reponse"] = r.text[:150].replace("\n", " ")
        except Exception as e:
            entry["http"] = None
            entry["ok"] = False
            entry["erreur"] = type(e).__name__
            entry["detail"] = str(e)[:200]
        entry["ms"] = int((time.monotonic() - t0) * 1000)
        return entry

    with ThreadPoolExecutor(max_workers=len(OVERPASS_MIRRORS)) as pool:
        return list(pool.map(probe, OVERPASS_MIRRORS))


# ----------------------------------------------------------------------------
# 2. Tracer la forme sur les routes
# ----------------------------------------------------------------------------

def _remove_spurs(route):
    """Supprime les allers-retours (A, B, A -> A) et les doublons consécutifs."""
    changed = True
    while changed:
        changed = False
        out = []
        for n in route:
            if len(out) >= 2 and out[-2] == n:
                out.pop()
                changed = True
            elif out and out[-1] == n:
                changed = True
            else:
                out.append(n)
        route = out
    return route


def _samples(line, step=20.0):
    k = max(2, int(line.length // step) + 1)
    return shapely.line_interpolate_point(line, np.linspace(0.0, line.length, k))


def route_shape_on_roads(G, shape_pts_m, lat0, lon0,
                         anchor_spacing_m=90.0, stray_m=25.0, max_snap_m=150.0,
                         prune_saving_m=40.0):
    """Trace la forme sur le réseau G.

    Idée : chaque tronçon a un coût = longueur x type de voie x (1 + (écart/25m)²),
    où "écart" est sa distance à la forme idéale. Le plus court chemin entre deux
    points de la forme reste donc collé à la forme au lieu de couper au plus court.

    Retourne (liste de noeuds, dict de mesures de fidélité).
    """
    kx = M_PER_DEG * math.cos(math.radians(lat0))
    nodes = list(G.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    px = np.array([(G.nodes[n]["x"] - lon0) * kx for n in nodes])
    py = np.array([(G.nodes[n]["y"] - lat0) * M_PER_DEG for n in nodes])

    pts = [tuple(p) for p in shape_pts_m]
    if math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) > 1.0:
        pts.append(pts[0])
    line = LineString(pts)

    # --- coût des tronçons -------------------------------------------------
    edges = list(G.edges(data=True))
    iu = np.array([index[u] for u, _, _ in edges])
    iv = np.array([index[v] for _, v, _ in edges])
    d_u = shapely.distance(shapely.points(np.column_stack([px[iu], py[iu]])), line)
    d_v = shapely.distance(shapely.points(np.column_stack([px[iv], py[iv]])), line)
    mid = np.column_stack([(px[iu] + px[iv]) / 2, (py[iu] + py[iv]) / 2])
    d_m = shapely.distance(shapely.points(mid), line)
    d_edge = (d_u + 2 * d_m + d_v) / 4.0
    for (u, v, data), d in zip(edges, d_edge):
        data["w"] = data["length"] * data.get("mult", 1.0) * (1.0 + (d / stray_m) ** 2)

    # --- points d'ancrage le long de la forme -------------------------------
    n_anchor = max(12, int(round(line.length / anchor_spacing_m)))
    anchors = _resample_polyline(pts, n_anchor + 1)
    sequence, skipped = [], 0
    for ax, ay in anchors:
        d = np.hypot(px - ax, py - ay)
        i = int(np.argmin(d))
        if d[i] > max_snap_m:
            skipped += 1
            continue
        if not sequence or sequence[-1] != nodes[i]:
            sequence.append(nodes[i])
    if len(sequence) < 4:
        raise RouteError("Réseau routier trop clairsemé ici pour cette forme : "
                         "essaie une largeur plus petite ou un autre point de départ.")

    # --- élagage des points d'ancrage qui créent des "piques" -------------------
    # Un point d'ancrage mal placé (sur une rue un peu à côté de la forme) oblige
    # le parcours à faire un détour aller-retour. On retire tout point dont la
    # visite coûte plus cher que d'aller directement au suivant.
    def _cost(a, b, cutoff):
        try:
            return nx.single_source_dijkstra(G, a, b, cutoff=cutoff, weight="w")[0]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return math.inf

    # Chaque point est jugé par rapport à ses voisins d'ORIGINE (pas de réaction en
    # chaîne), et on ne retire jamais deux points consécutifs.
    original = sequence
    pruned = [original[0]]
    dropped_prev = False
    for i in range(1, len(original) - 1):
        prev, cur, nxt = original[i - 1], original[i], original[i + 1]
        via = _cost(prev, cur, 3000.0) + _cost(cur, nxt, 3000.0)
        direct = _cost(prev, nxt, via + 1.0 if math.isfinite(via) else 6000.0)
        saving = via - direct if math.isfinite(via) and math.isfinite(direct) else 0.0
        if saving > prune_saving_m and not dropped_prev:
            dropped_prev = True           # le détour par `cur` ne vaut pas le coup
            continue
        dropped_prev = False
        pruned.append(cur)
    pruned.append(original[-1])
    sequence = pruned

    # --- assemblage du parcours --------------------------------------------
    route = [sequence[0]]
    unreachable = 0
    for a, b in zip(sequence[:-1], sequence[1:]):
        try:
            path = nx.shortest_path(G, a, b, weight="w")
        except nx.NetworkXNoPath:
            unreachable += 1
            continue
        route.extend(path[1:])
    route = _remove_spurs(route)
    if len(route) < 8:
        raise RouteError("Parcours trop court : réseau routier insuffisant pour cette forme ici.")

    # --- mesures de fidélité -------------------------------------------------
    rx = np.array([px[index[n]] for n in route])
    ry = np.array([py[index[n]] for n in route])
    route_line = LineString(np.column_stack([rx, ry]))
    dev = shapely.distance(_samples(route_line), line)       # route -> forme
    cov = shapely.distance(_samples(line), route_line)       # forme -> route
    mean_dev = float(np.mean(dev))
    coverage = float(100.0 * np.mean(cov <= 40.0))

    # La fidélité se juge par rapport à la taille de la forme : 40 m d'écart sur un
    # cœur de 6 km est invisible, sur un cœur de 1,5 km il le déforme.
    xs = [p[0] for p in pts]
    span = max(max(xs) - min(xs), 1.0)
    rel = mean_dev / span
    if rel <= 0.015 and coverage >= 65:
        quality = "bon"
    elif rel <= 0.03 and coverage >= 50:
        quality = "moyen"
    else:
        quality = "faible"

    metrics = {
        "quality": quality,
        "mean_dev_m": round(mean_dev, 1),
        "p90_dev_m": round(float(np.percentile(dev, 90)), 1),
        "max_dev_m": round(float(np.max(dev)), 1),
        "coverage_pct": round(coverage, 1),
        "route_km": round(route_line.length / 1000.0, 2),
        "shape_km": round(line.length / 1000.0, 2),
        "anchors_skipped": skipped,
        "unreachable": unreachable,
        "nodes": len(route),
    }
    return route, metrics


# ----------------------------------------------------------------------------
# 3. Choisir où poser la forme sur la carte
# ----------------------------------------------------------------------------

def find_best_placements(G, shape_pts_m, lat0, lon0, top=3,
                         max_shift_m=1500.0, step_m=500.0,
                         scales=(0.9, 1.0, 1.1), rotations_deg=(-10, 0, 10)):
    """Cherche où poser la forme (décalage, taille, rotation) pour qu'elle tombe
    sur le plus de routes. Évaluation très rapide : on mesure, pour chaque position
    candidate, la distance de la forme aux routes les plus proches (pas de calcul
    d'itinéraire). Retourne les `top` meilleures positions, forme déjà transformée.
    """
    kx = M_PER_DEG * math.cos(math.radians(lat0))
    # routes densifiées : un point tous les ~25 m le long de chaque tronçon
    pts_list = []
    for u, v in G.edges():
        x1 = (G.nodes[u]["x"] - lon0) * kx
        y1 = (G.nodes[u]["y"] - lat0) * M_PER_DEG
        x2 = (G.nodes[v]["x"] - lon0) * kx
        y2 = (G.nodes[v]["y"] - lat0) * M_PER_DEG
        n = max(1, int(math.hypot(x2 - x1, y2 - y1) // 25))
        for i in range(n + 1):
            t = i / n
            pts_list.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
    tree = shapely.STRtree(shapely.points(np.array(pts_list)))

    base = [tuple(p) for p in shape_pts_m]
    line = LineString(base)
    samples = np.array([(p.x, p.y) for p in _samples(line, step=90.0)])

    def evaluate(k, rot, dx, dy):
        c, sn = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        rx = (samples[:, 0] * c - samples[:, 1] * sn) * k
        ry = (samples[:, 0] * sn + samples[:, 1] * c) * k
        q = shapely.points(np.column_stack([rx + dx, ry + dy]))
        _, d = tree.query_nearest(q, return_distance=True, all_matches=False)
        score = float(np.mean(np.minimum(d, 200.0))) + 0.5 * float(np.mean(d > 60.0)) * 100
        # on préfère rester près du point demandé et garder la taille demandée
        return score + 0.004 * math.hypot(dx, dy) + 30.0 * abs(k - 1.0)

    # 1) balayage grossier (rapide), 2) affinage autour des meilleures positions
    t_start = time.monotonic()
    coarse = np.arange(-max_shift_m, max_shift_m + 1, step_m)
    results = []
    for k in scales:
        for rot in rotations_deg:
            for dx in coarse:
                for dy in coarse:
                    if math.hypot(dx, dy) > max_shift_m * 1.2:
                        continue
                    results.append((evaluate(k, rot, dx, dy), float(dx), float(dy), k, rot))
    results.sort(key=lambda r: r[0])
    fine = []
    for _, bx, by, k, rot in results[:6]:
        for dx in (-step_m / 2, 0.0, step_m / 2):
            for dy in (-step_m / 2, 0.0, step_m / 2):
                if time.monotonic() - t_start > 20.0:
                    break
                fine.append((evaluate(k, rot, bx + dx, by + dy), bx + dx, by + dy, k, rot))
    results = sorted(results + fine, key=lambda r: r[0])
    logger.info("recherche de position : %d positions testées en %.1fs", len(results), time.monotonic() - t_start)

    # on garde des positions différentes les unes des autres
    chosen = []
    for r in results:
        if all(math.hypot(r[1] - c[1], r[2] - c[2]) > 400 or abs(r[3] - c[3]) > 0.05 for c in chosen):
            chosen.append(r)
        if len(chosen) >= top:
            break

    out = []
    for score, dx, dy, k, rot in chosen:
        c, s = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        moved = [((x * c - y * s) * k + dx, (x * s + y * c) * k + dy) for x, y in base]
        out.append({"score": score, "dx": dx, "dy": dy, "scale": k, "rot": rot, "pts": moved})
    return out
