"""
gpx_shape_generator.py
=======================

Algorithme central du micro-SaaS : transformer une forme (cœur, cercle, lettre...)
en un parcours GPX qui suit le réseau routier/cyclable réel.

Principe (map matching pour "Strava Art") :
1. On génère une courbe paramétrique pour la forme voulue (ex: cœur), centrée
   sur un point de départ, mise à l'échelle en mètres.
2. On échantillonne N points le long de cette courbe.
3. On "snap" (accroche) chaque point sur le nœud le plus proche du graphe
   routier réel (OpenStreetMap via osmnx).
4. On relie les nœuds accrochés consécutifs par le plus court chemin réel
   (networkx), en respectant le sens des rues.
5. On assemble le tout en une boucle continue et on exporte en .gpx.

NOTE IMPORTANTE SUR CE PROTOTYPE :
Ce sandbox n'a pas accès à l'API Overpass (osmnx.graph_from_point ne peut pas
récupérer les données OSM ici — réseau restreint). Le code est donc écrit pour
fonctionner avec un vrai graphe osmnx EN LOCAL (voir get_real_network ci-dessous),
mais pour tester et valider l'algorithme dans ce sandbox, on utilise
generate_synthetic_street_network() qui simule un quartier réaliste
(grille irrégulière, rues manquantes, comme une vraie ville).

Chez toi, en remplaçant juste l'appel au réseau synthétique par
get_real_network(), tout le reste du pipeline fonctionne à l'identique.
"""

import math
import random
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point as ShapelyPoint

try:
    import gpxpy
    import gpxpy.gpx
    HAS_GPXPY = True
except ImportError:
    HAS_GPXPY = False


# ----------------------------------------------------------------------------
# 1. CONVERSION MÈTRES <-> LAT/LON
# ----------------------------------------------------------------------------

def meters_to_latlon(center_lat, center_lon, dx_m, dy_m):
    """Convertit un décalage en mètres (dx=est, dy=nord) depuis un centre
    en coordonnées lat/lon absolues. Approximation locale (correcte sur
    quelques km, largement suffisant pour un parcours de quelques km)."""
    lat_deg_per_m = 1 / 111_320.0
    lon_deg_per_m = 1 / (111_320.0 * math.cos(math.radians(center_lat)))
    return center_lat + dy_m * lat_deg_per_m, center_lon + dx_m * lon_deg_per_m


# ----------------------------------------------------------------------------
# 2. GÉNÉRATION DE FORMES PARAMÉTRIQUES (en mètres, centrées sur l'origine)
# ----------------------------------------------------------------------------

def shape_heart(n_points=120, width_m=1200):
    """Courbe de cœur classique (équation paramétrique), mise à l'échelle
    pour occuper environ `width_m` mètres de large."""
    t = np.linspace(0, 2 * np.pi, n_points)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    # Normaliser à l'échelle voulue
    scale = width_m / (x.max() - x.min())
    x *= scale
    y *= scale
    return list(zip(x, y))


def shape_circle(n_points=100, radius_m=600):
    t = np.linspace(0, 2 * np.pi, n_points)
    return list(zip(radius_m * np.sin(t), radius_m * np.cos(t)))


def shape_infinity(n_points=140, width_m=1400):
    """Symbole infini — bon exemple de forme non-convexe supplémentaire."""
    t = np.linspace(0, 2 * np.pi, n_points)
    scale = width_m / 2
    x = scale * np.cos(t) / (1 + np.sin(t) ** 2)
    y = scale * np.cos(t) * np.sin(t) / (1 + np.sin(t) ** 2)
    return list(zip(x, y))


def shape_star(n_points=100, width_m=1200, points=5, inner_ratio=0.45):
    """Étoile à N branches. `inner_ratio` contrôle la profondeur des creux
    (0.4-0.5 = étoile classique, plus proche de 1 = presque un cercle)."""
    outer_r = width_m / 2
    inner_r = outer_r * inner_ratio
    verts = []
    for i in range(points * 2):
        r = outer_r if i % 2 == 0 else inner_r
        angle = math.pi / 2 + i * math.pi / points
        verts.append((r * math.cos(angle), r * math.sin(angle)))
    verts.append(verts[0])
    # Ré-échantillonner en n_points régulièrement espacés le long du contour
    return _resample_polyline(verts, n_points)


def _polygon(verts, n_points):
    verts = list(verts) + [verts[0]]
    return _resample_polyline(verts, n_points)


def shape_triangle(n_points=120, width_m=1200):
    """Triangle équilatéral pointe en haut."""
    h = width_m * math.sqrt(3) / 2
    return _polygon([(0, 2 * h / 3), (width_m / 2, -h / 3), (-width_m / 2, -h / 3)], n_points)


def shape_square(n_points=120, width_m=1200):
    """Carré."""
    a = width_m / 2
    return _polygon([(-a, a), (a, a), (a, -a), (-a, -a)], n_points)


def shape_diamond(n_points=120, width_m=1200):
    """Losange (plus haut que large)."""
    a = width_m / 2
    return _polygon([(0, a * 1.35), (a, 0), (0, -a * 1.35), (-a, 0)], n_points)


def shape_text(text, n_points=200, width_m=1500, font_size=200):
    """Convertit un texte (prénom, initiales, mot court) en contour vectoriel
    à parcourir. Utilise le contour de police via matplotlib (TextPath),
    garde le plus grand sous-contour (la boucle extérieure) par lettre et
    les enchaîne. Pour un rendu propre, préfère des mots courts (1-8
    caractères) et des polices bien fermées (éviter le 'i' du 'j' isolé,
    les ponctuations, etc. — chaque lettre doit former une boucle fermée)."""
    from matplotlib.textpath import TextPath
    tp = TextPath((0, 0), text, size=font_size)
    polygons = tp.to_polygons()
    if not polygons:
        raise ValueError("Aucun contour trouvé pour ce texte/cette police.")
    # Garder les polygones significatifs (ignorer les trous internes des
    # lettres type 'a', 'o', 'e' -- on ne garde que les contours extérieurs,
    # concaténés dans l'ordre de lecture gauche->droite)
    polygons = sorted(polygons, key=lambda p: p[:, 0].min())
    merged = []
    for poly in polygons:
        if len(poly) < 6:
            continue  # bruit / trous trop petits
        merged.extend(poly.tolist())
        merged.append(poly[0].tolist())  # revenir au début du sous-contour

    xs = [p[0] for p in merged]
    scale = width_m / (max(xs) - min(xs))
    cx = (max(xs) + min(xs)) / 2
    ys = [p[1] for p in merged]
    cy = (max(ys) + min(ys)) / 2
    scaled = [((x - cx) * scale, (y - cy) * scale) for x, y in merged]
    return _resample_polyline(scaled, n_points)


def _resample_polyline(points, n_points):
    """Ré-échantillonne une polyligne (liste de (x,y)) en n_points points
    régulièrement espacés le long de sa longueur totale — évite d'avoir
    trop de points sur les segments courts et pas assez sur les longs."""
    pts = np.array(points)
    seg_lengths = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cumlen = np.concatenate([[0], np.cumsum(seg_lengths)])
    total = cumlen[-1]
    if total == 0:
        return list(map(tuple, pts))
    targets = np.linspace(0, total, n_points)
    out_x = np.interp(targets, cumlen, pts[:, 0])
    out_y = np.interp(targets, cumlen, pts[:, 1])
    return list(zip(out_x, out_y))


SHAPES = {
    "heart": shape_heart,
    "circle": shape_circle,
    "infinity": shape_infinity,
    "star": shape_star,
    "triangle": shape_triangle,
    "square": shape_square,
    "diamond": shape_diamond,
    "text": shape_text,
}


# ----------------------------------------------------------------------------
# 3. RÉSEAU ROUTIER : réel (osmnx, à utiliser en local) ou synthétique (test)
# ----------------------------------------------------------------------------

def get_real_network(center_lat, center_lon, dist_m=2000, network_type="bike"):
    """À UTILISER EN LOCAL (nécessite un accès internet à l'API Overpass).
    Récupère le vrai réseau routier/cyclable OSM autour d'un point via osmnx.
    Retourne un graphe networkx non orienté avec attributs 'y'/'x' (lat/lon)
    sur les nœuds et 'length' (mètres) sur les arêtes.

    NOTE : osmnx entraîne des dépendances lourdes (geopandas, rasterio...)
    qui peuvent dépasser les limites d'un hébergement gratuit. Pour un
    déploiement en ligne, préférer get_real_network_lite() ci-dessous
    (mêmes données OSM, aucune dépendance lourde).
    """
    import osmnx as ox
    G = ox.graph_from_point((center_lat, center_lon), dist=dist_m,
                             network_type=network_type, simplify=True)
    return ox.convert.to_undirected(G)


_HIGHWAY_FILTERS = {
    "bike": '["highway"]["highway"!~"motorway|motorway_link|proposed|construction|abandoned"]["access"!~"private|no"]',
    "walk": '["highway"]["highway"!~"motorway|motorway_link|proposed|construction|abandoned"]["access"!~"private|no"]',
    "drive": '["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service"]["access"!~"private|no"]',
}


def get_real_network_lite(center_lat, center_lon, dist_m=2000, network_type="bike", timeout=25):
    """Version légère de get_real_network : interroge directement l'API
    Overpass en HTTP (requests) et construit le graphe à la main, sans
    passer par osmnx. Beaucoup plus adapté à un hébergement gratuit
    (Render, Railway...) où les dépendances lourdes d'osmnx posent souvent
    problème (taille de build, mémoire).

    Nécessite seulement `requests` en plus des dépendances déjà utilisées
    par ce module.
    """
    import requests

    lat_delta = dist_m / 111_320.0
    lon_delta = dist_m / (111_320.0 * math.cos(math.radians(center_lat)))
    south, north = center_lat - lat_delta, center_lat + lat_delta
    west, east = center_lon - lon_delta, center_lon + lon_delta

    highway_filter = _HIGHWAY_FILTERS.get(network_type, _HIGHWAY_FILTERS["bike"])
    query = f"""
    [out:json][timeout:{timeout}];
    way{highway_filter}({south},{west},{north},{east});
    (._;>;);
    out body;
    """
    resp = requests.post(
        "https://overpass-api.de/api/interpreter",
        data={"data": query},
        timeout=timeout + 5,
    )
    resp.raise_for_status()
    data = resp.json()

    node_coords = {}
    for el in data.get("elements", []):
        if el["type"] == "node":
            node_coords[el["id"]] = (el["lat"], el["lon"])

    G = nx.Graph()
    for el in data.get("elements", []):
        if el["type"] != "way":
            continue
        nds = el.get("nodes", [])
        for a, b in zip(nds[:-1], nds[1:]):
            if a not in node_coords or b not in node_coords:
                continue
            lat_a, lon_a = node_coords[a]
            lat_b, lon_b = node_coords[b]
            if a not in G:
                G.add_node(a, y=lat_a, x=lon_a)
            if b not in G:
                G.add_node(b, y=lat_b, x=lon_b)
            d = _haversine_m(lat_a, lon_a, lat_b, lon_b)
            G.add_edge(a, b, length=d)

    if G.number_of_nodes() == 0:
        raise RuntimeError("Aucune donnée OpenStreetMap trouvée pour cette zone.")

    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


def generate_synthetic_street_network(center_lat, center_lon, extent_m=2000,
                                       cell_m=90, jitter_ratio=0.25,
                                       removal_rate=0.12, seed=42):
    """Simule un quartier réaliste pour tester l'algorithme hors-ligne :
    grille de rues avec bruit de position (les rues réelles ne sont pas
    parfaitement droites) et suppression aléatoire d'arêtes (tous les
    carrefours n'existent pas dans la vraie vie).
    """
    rng = random.Random(seed)
    n_cells = int(extent_m / cell_m)
    G = nx.Graph()

    # Créer les nœuds avec un léger bruit de position
    for i in range(n_cells + 1):
        for j in range(n_cells + 1):
            node_id = i * (n_cells + 1) + j
            dx = (i - n_cells / 2) * cell_m + rng.uniform(-cell_m, cell_m) * jitter_ratio
            dy = (j - n_cells / 2) * cell_m + rng.uniform(-cell_m, cell_m) * jitter_ratio
            lat, lon = meters_to_latlon(center_lat, center_lon, dx, dy)
            G.add_node(node_id, y=lat, x=lon, _dx=dx, _dy=dy)

    # Relier les nœuds voisins (horizontal + vertical), avec suppressions aléatoires
    def node_id(i, j):
        return i * (n_cells + 1) + j

    for i in range(n_cells + 1):
        for j in range(n_cells + 1):
            for ni, nj in [(i + 1, j), (i, j + 1)]:
                if ni <= n_cells and nj <= n_cells:
                    if rng.random() < removal_rate:
                        continue
                    a, b = node_id(i, j), node_id(ni, nj)
                    d = _haversine_m(G.nodes[a]['y'], G.nodes[a]['x'],
                                      G.nodes[b]['y'], G.nodes[b]['x'])
                    G.add_edge(a, b, length=d)

    # S'assurer que le graphe est connecté (garder la plus grande composante)
    largest_cc = max(nx.connected_components(G), key=len)
    return G.subgraph(largest_cc).copy()


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ----------------------------------------------------------------------------
# 4. CŒUR DE L'ALGO : accrocher la forme au réseau réel et construire la route
# ----------------------------------------------------------------------------

def nearest_node(G, lat, lon):
    """Nœud du graphe le plus proche d'un point donné (recherche brute simple,
    suffisante jusqu'à quelques milliers de nœuds ; à remplacer par un
    KD-tree/BallTree pour de gros graphes réels)."""
    best, best_d = None, float("inf")
    for n, data in G.nodes(data=True):
        d = (data['y'] - lat) ** 2 + (data['x'] - lon) ** 2
        if d < best_d:
            best, best_d = n, d
    return best


def fit_shape_to_network(G, shape_points_m, center_lat, center_lon):
    """Étapes clés de l'algorithme :
    1. Convertit chaque point de la forme (en mètres) en lat/lon absolu.
    2. Accroche chaque point au nœud réel le plus proche du graphe.
    3. Déduplique les nœuds consécutifs identiques (évite les boucles nulles).
    4. Relie chaque paire de nœuds consécutifs par le plus court chemin réel.
    5. Ferme la boucle (retour au point de départ).

    Retourne : (route_nodes, shape_latlon, snapped_latlon)
    """
    shape_latlon = [meters_to_latlon(center_lat, center_lon, dx, dy)
                    for dx, dy in shape_points_m]

    # 2. Snap
    snapped = [nearest_node(G, lat, lon) for lat, lon in shape_latlon]

    # 3. Dédupliquer les répétitions consécutives
    dedup = [snapped[0]]
    for n in snapped[1:]:
        if n != dedup[-1]:
            dedup.append(n)
    if dedup[0] != dedup[-1]:
        dedup.append(dedup[0])  # fermer la boucle

    # 4. Stitching : plus court chemin entre chaque paire consécutive
    full_route = [dedup[0]]
    for a, b in zip(dedup[:-1], dedup[1:]):
        try:
            path = nx.shortest_path(G, a, b, weight="length")
        except nx.NetworkXNoPath:
            continue  # segment isolé : on saute ce point d'ancrage
        full_route.extend(path[1:])

    snapped_latlon = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in dedup]
    return full_route, shape_latlon, snapped_latlon


def route_accuracy(shape_latlon, route_latlon):
    """Mesure grossière de fidélité : distance moyenne (m) entre chaque point
    de la forme idéale et le point le plus proche du parcours réel généré."""
    route_line = LineString([(lon, lat) for lat, lon in route_latlon])
    dists = []
    for lat, lon in shape_latlon:
        p = ShapelyPoint(lon, lat)
        # distance en degrés -> conversion approx en mètres
        d_deg = p.distance(route_line)
        d_m = d_deg * 111_320
        dists.append(d_m)
    return float(np.mean(dists)), float(np.max(dists))


# ----------------------------------------------------------------------------
# 5. EXPORT GPX
# ----------------------------------------------------------------------------

def export_gpx(G, route_nodes, out_path, name="Parcours en forme"):
    coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in route_nodes]
    if HAS_GPXPY:
        gpx = gpxpy.gpx.GPX()
        track = gpxpy.gpx.GPXTrack(name=name)
        gpx.tracks.append(track)
        segment = gpxpy.gpx.GPXTrackSegment()
        track.segments.append(segment)
        for lat, lon in coords:
            segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon))
        with open(out_path, "w") as f:
            f.write(gpx.to_xml())
    else:
        # Fallback minimal sans dépendance
        with open(out_path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<gpx version="1.1" creator="gpx_shape_generator" '
                    'xmlns="http://www.topografix.com/GPX/1/1" '
                    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                    'xsi:schemaLocation="http://www.topografix.com/GPX/1/1 '
                    'http://www.topografix.com/GPX/1/1/gpx.xsd">\n')
            f.write(f'<trk><name>{name}</name><trkseg>\n')
            for lat, lon in coords:
                f.write(f'<trkpt lat="{lat:.6f}" lon="{lon:.6f}"></trkpt>\n')
            f.write('</trkseg></trk></gpx>\n')
    return out_path
