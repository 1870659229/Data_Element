# -*- coding: utf-8 -*-
"""
水上航道智能路径规划系统 - Web API 服务 (轻量版)
直接使用拓扑数据 + networkx 进行路径规划，无需 ML 模型
"""

import sys
import os
import json
import csv
import traceback
from datetime import datetime
from math import radians, cos, sin, asin, sqrt, pi
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from shapely.geometry import Point, LineString, box
from shapely.ops import nearest_points

# WGS-84 → GCJ-02 纠偏（中国大陆地图坐标系转换）
_GCJ_A = 6378245.0
_GCJ_EE = 0.00669342162296594323

def _gcj_transform_lat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * sqrt(abs(x))
    ret += (20.0 * sin(6.0 * x * pi) + 20.0 * sin(2.0 * x * pi)) * 2.0 / 3.0
    ret += (20.0 * sin(y * pi) + 40.0 * sin(y / 3.0 * pi)) * 2.0 / 3.0
    ret += (160.0 * sin(y / 12.0 * pi) + 320.0 * sin(y * pi / 30.0)) * 2.0 / 3.0
    return ret

def _gcj_transform_lon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * sqrt(abs(x))
    ret += (20.0 * sin(6.0 * x * pi) + 20.0 * sin(2.0 * x * pi)) * 2.0 / 3.0
    ret += (20.0 * sin(x * pi) + 40.0 * sin(x / 3.0 * pi)) * 2.0 / 3.0
    ret += (150.0 * sin(x / 12.0 * pi) + 300.0 * sin(x / 30.0 * pi)) * 2.0 / 3.0
    return ret

def wgs84_to_gcj02(lat, lon):
    """WGS-84 → GCJ-02（高德/谷歌中国镜像坐标系）"""
    if _out_of_china(lat, lon):
        return lat, lon
    dlat = _gcj_transform_lat(lon - 105.0, lat - 35.0)
    dlon = _gcj_transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = sin(radlat)
    magic = 1 - _GCJ_EE * magic * magic
    sqrtmagic = sqrt(magic)
    dlat = (dlat * 180.0) / ((_GCJ_A * (1 - _GCJ_EE)) / (magic * sqrtmagic) * pi)
    dlon = (dlon * 180.0) / (_GCJ_A / sqrtmagic * cos(radlat) * pi)
    return lat + dlat, lon + dlon

def _out_of_china(lat, lon):
    return not (72.004 < lon < 137.8347 and 0.8293 < lat < 55.8271)

from ship_navigator import MultiObjectiveNavigator, PhysicalConstraintChecker, ShipCharacteristics, PathType

app = Flask(__name__, static_folder='static', static_url_path='/static')

OUTPUT_DIR = 'output'
WATERWAY_DIR = 'data_osm'
nodes_data = {}
graph_edges = {}
graph_degrees = {}
main_component = set()
edge_waypoints = {}
edge_features = {}
ship_db = {}  # 船舶特征数据库（按船名索引）


def _safe_float(val, fallback=None):
    """安全的浮点数转换：NaN/None/空值 → fallback"""
    import math
    if val is None:
        return fallback
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return fallback
        return v
    except (ValueError, TypeError):
        return fallback


def _normalize_ship_type(raw_type, length_m=None):
    """将 CSV 中各种船型名称规范化为 SHIP_TEMPLATES 兼容名称。

    规则（2026-06-17 改进）：
      - '拖轮' → '拖船'（同义词）
      - '货船' → 按船长细分为 小型/中型/大型 货船
      - 其他类型保持不变（在 SHIP_TEMPLATES 中补充对应模板）
    """
    if not raw_type or not isinstance(raw_type, str):
        return '中型货船'
    t = raw_type.strip()
    if t == '拖轮':
        return '拖船'
    if t == '货船':
        L = _safe_float(length_m, 63)
        if L < 55:
            return '小型货船'
        elif L < 100:
            return '中型货船'
        else:
            return '大型货船'
    return t


# 全局规划器（惰性初始化）
_navigator = None
_constraint_checker = None
_bidirectional_pairs = set()

MAX_EDGE_LENGTH_M = 10000
WATERWAY_PROXIMITY_M = 2000
# 长边穿陆地检测：边长超过此阈值时，进行更严格的水系距离检测
LONG_EDGE_STRICT_THRESHOLD_M = 2000
_waterway_grid = {}
_waterway_geoms = []
_water_polygon_geoms = []
_water_polygon_grid = {}


def haversine_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return c * 6371000


def _rdp_simplify(points, eps_deg):
    """
    Ramer-Douglas-Peucker 简化：递归保留偏离首尾连线 > eps_deg 的点。
    points: [(lon, lat), ...]，坐标按经纬度平面近似（短距 < 10km 误差 < 1%）
    eps_deg: 偏离阈值（度数）。中纬度 1° ≈ 111km → 30m ≈ 0.00027°
    """
    if len(points) < 3:
        return list(points)

    def perp_dist(p, a, b):
        if a[0] == b[0] and a[1] == b[1]:
            return sqrt((p[0] - a[0]) ** 2 + (p[1] - a[1]) ** 2)
        num = abs((b[1] - a[1]) * p[0] - (b[0] - a[0]) * p[1] + b[0] * a[1] - b[1] * a[0])
        den = sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
        return num / den

    def rec(start, end):
        dmax, idx = 0.0, start
        for i in range(start + 1, end):
            d = perp_dist(points[i], points[start], points[end])
            if d > dmax:
                dmax, idx = d, i
        if dmax > eps_deg:
            keep[idx] = True
            rec(start, idx)
            rec(idx, end)

    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    rec(0, len(points) - 1)
    return [p for p, k in zip(points, keep) if k]


def load_data():
    global nodes_data, graph_edges

    nodes_path = os.path.join(OUTPUT_DIR, 'topology_nodes.csv')
    with open(nodes_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nid = int(row['node_id'])
            nodes_data[nid] = {
                'lat': float(row['lat']),
                'lon': float(row['lon']),
                'type': row.get('type', 'unknown'),
                'frequency': int(row.get('frequency', 0)),
                'ship_count': int(row.get('ship_count', 0))
            }

    edges_path = os.path.join(OUTPUT_DIR, 'topology_edges.csv')
    global _bidirectional_pairs
    _bidirectional_pairs = set()
    with open(edges_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            u, v = int(row['from_node']), int(row['to_node'])
            graph_edges[(u, v)] = {
                'weight': float(row.get('weight', 1))
            }
            if row.get('is_bidirectional', '').lower() == 'true':
                _bidirectional_pairs.add((u, v))

    # 修复（2026-06-07）：先加载 waypoints，再调 _filter_edges，
    # 因为水域面检查（_is_edge_near_waterway）需要按 waypoints 物理路径采样，
    # 而不是按 u-v 直线（直线可能穿岛，但实际轨迹是绕行的）
    waypoints_path = os.path.join(OUTPUT_DIR, 'edge_waypoints.csv')
    if os.path.exists(waypoints_path):
        with open(waypoints_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (int(row['from_node']), int(row['to_node']))
                if key not in edge_waypoints:
                    edge_waypoints[key] = []
                edge_waypoints[key].append({
                    'lat': float(row['lat']),
                    'lon': float(row['lon']),
                    'sequence': int(row['sequence'])
                })
        for key in edge_waypoints:
            edge_waypoints[key].sort(key=lambda w: w['sequence'])
        print(f"边航点加载完成: {len(edge_waypoints)} 条边, {sum(len(v) for v in edge_waypoints.values())} 个航点")
    else:
        print("未找到 edge_waypoints.csv，路由将使用直线连接")

    _filter_edges()

    edge_features_path = os.path.join(OUTPUT_DIR, 'edge_features_dynamic_weights.csv')
    if os.path.exists(edge_features_path):
        with open(edge_features_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                u, v = int(row['from_node']), int(row['to_node'])
                key = (u, v)
                if key not in edge_features:
                    edge_features[key] = {
                        'avg_distance': float(row.get('avg_distance', 100)) / 1000,
                        'avg_travel_time': float(row.get('avg_travel_time', 30)) / 60,
                        'segment_count': int(row.get('segment_count', 0)),
                        'waterway_type': row.get('waterway_type', 'open'),
                        'avg_actual_speed': float(row.get('avg_actual_speed', 5)),
                    }

    _compute_graph_topology()

    print(f"数据加载完成: {len(nodes_data)} 个节点, {len(graph_edges)} 条边, {len(edge_features)} 条边特征")

    # 加载船舶特征数据库（2026-06-17 改进：NaN 安全 + 船型规范化）
    ship_db_path = os.path.join(OUTPUT_DIR, 'ship_characteristics_db.csv')
    if os.path.exists(ship_db_path):
        _ship_df = pd.read_csv(ship_db_path)
        for _, row in _ship_df.iterrows():
            raw_length = _safe_float(row.get('length'))
            raw_type = row.get('ship_type', '货船')
            normalized_type = _normalize_ship_type(raw_type, raw_length)
            tpl = SHIP_TEMPLATES.get(normalized_type, SHIP_TEMPLATES['中型货船'])

            ship_db[row['ship_name']] = {
                'ship_name': row['ship_name'],
                'ship_type': normalized_type,
                'length': _safe_float(raw_length, tpl['length']),
                'width': _safe_float(row.get('width'), tpl['width']),
                'draft': _safe_float(row.get('draft'), tpl['draft']),
                'height': _safe_float(row.get('height'), tpl['height']),
                'tonnage': _safe_float(row.get('tonnage'), tpl['tonnage']),
                'max_speed': _safe_float(row.get('max_speed'), tpl['max_speed']),
                'mmsi': row.get('mmsi', ''),
                'data_source': row.get('data_source', ''),
            }
        print(f"船舶特征数据库加载完成: {len(ship_db)} 艘")


def _init_navigator():
    """惰性初始化多目标导航器"""
    global _navigator, _constraint_checker
    if _navigator is not None:
        return

    import networkx as nx
    import pandas as pd

    # 构建有向图
    G = nx.DiGraph()
    for nid, attrs in nodes_data.items():
        G.add_node(nid, **attrs)
    for (u, v), attrs in graph_edges.items():
        G.add_edge(u, v, weight=attrs.get('weight', 1))
    # 为双向边补充反向边
    for (u, v) in _bidirectional_pairs:
        if not G.has_edge(v, u):
            G.add_edge(v, u, weight=graph_edges.get((u, v), {}).get('weight', 1))

    # 为导航器加载边特征（原始单位：米/秒，与 app.py 的 km/分钟格式不同）
    nav_edge_features = {}
    ef_path = os.path.join(OUTPUT_DIR, 'edge_features_dynamic_weights.csv')
    if os.path.exists(ef_path):
        import csv as _csv
        with open(ef_path, 'r', encoding='utf-8-sig') as f:
            reader = _csv.DictReader(f)
            for row in reader:
                u, v = int(row['from_node']), int(row['to_node'])
                key = (u, v)
                if key not in nav_edge_features:
                    nav_edge_features[key] = {
                        'avg_distance': float(row.get('avg_distance', 100)),
                        'avg_travel_time': float(row.get('avg_travel_time', 30)),
                        'segment_count': int(row.get('segment_count', 0)),
                        'waterway_type': row.get('waterway_type', 'open'),
                        'avg_actual_speed': float(row.get('avg_actual_speed', 5)),
                    }
    # 为双向边镜像 nav_edge_features
    for (u, v) in _bidirectional_pairs:
        if (u, v) in nav_edge_features and (v, u) not in nav_edge_features:
            nav_edge_features[(v, u)] = dict(nav_edge_features[(u, v)])

    # 修复（2026-06-07）：拓扑数据中部分边只有单向（如 (405,40) 存在但 (40,405) 不存在），
    # 但 graph_edges 已包含 (u,v)，仅反向缺失时，补全反向边。
    # 关键限制：只在 graph_edges 已有该边（任一方向）时才补全，避免把
    # 因穿陆地/拓扑原因被过滤的边通过 nav_edge_features 重新加回图。
    _fixed_dir = 0
    for (u, v) in list(graph_edges.keys()):
        if G.has_edge(u, v) and not G.has_edge(v, u):
            G.add_edge(v, u, weight=graph_edges[(u, v)].get('weight', 1))
            _fixed_dir += 1
        elif G.has_edge(v, u) and not G.has_edge(u, v):
            G.add_edge(u, v, weight=graph_edges[(u, v)].get('weight', 1))
            _fixed_dir += 1
    if _fixed_dir:
        print(f"  - 补全有向图方向: {_fixed_dir} 条（基于 graph_edges）")

    _constraint_checker = PhysicalConstraintChecker(nav_edge_features, nodes_data, G)
    _navigator = MultiObjectiveNavigator(G, nav_edge_features, _constraint_checker)
    print(f"多目标导航器已初始化: {len(nav_edge_features)} 条边特征")


def _load_waterways():
    global _waterway_geoms, _waterway_grid, _water_polygon_geoms, _water_polygon_grid

    waterway_path = os.path.join(WATERWAY_DIR, 'waterways.geojson')
    if not os.path.exists(waterway_path):
        print("ERROR: 未找到水系数据 (data_osm/waterways.geojson)，水系过滤已禁用！所有边将默认通过验证")
        return False

    with open(waterway_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    features = data.get('features', [])
    if not features:
        print("水系数据为空")
        return False

    from shapely.geometry import shape as shapely_shape

    geoms = []
    for feat in features:
        geom = feat.get('geometry')
        if geom:
            try:
                g = shapely_shape(geom)
                if g.is_valid and not g.is_empty:
                    geoms.append(g)
            except Exception:
                pass

    _waterway_geoms = geoms
    print(f"水系数据加载完成: {len(geoms)} 条")

    _build_waterway_grid()

    # 加载水域面数据（用于检测长边是否穿过陆地）
    polygon_path = os.path.join(WATERWAY_DIR, 'water_polygons.geojson')
    if os.path.exists(polygon_path):
        with open(polygon_path, 'r', encoding='utf-8') as f:
            pdata = json.load(f)
        pfeatures = pdata.get('features', [])
        poly_geoms = []
        for feat in pfeatures:
            geom = feat.get('geometry')
            if geom:
                try:
                    g = shapely_shape(geom)
                    if g.is_valid and not g.is_empty:
                        if g.geom_type == 'Polygon':
                            poly_geoms.append(g)
                        elif g.geom_type == 'MultiPolygon':
                            poly_geoms.extend(g.geoms)
                except Exception:
                    pass
        _water_polygon_geoms = poly_geoms
        print(f"水域面数据加载完成: {len(poly_geoms)} 个面")
        _build_water_polygon_grid()
    else:
        print("未找到水域面数据 (data_osm/water_polygons.geojson)，长边穿陆地检测不可用")

    return True


def _build_waterway_grid():
    global _waterway_grid
    _waterway_grid = {}

    grid_size = 0.02

    for idx, geom in enumerate(_waterway_geoms):
        try:
            minx, miny, maxx, maxy = geom.bounds
            col_start = int(minx / grid_size)
            col_end = int(maxx / grid_size) + 1
            row_start = int(miny / grid_size)
            row_end = int(maxy / grid_size) + 1

            for col in range(col_start, col_end):
                for row in range(row_start, row_end):
                    key = (col, row)
                    if key not in _waterway_grid:
                        _waterway_grid[key] = []
                    _waterway_grid[key].append(idx)
        except Exception:
            continue

    print(f"  空间网格索引已建立: {len(_waterway_grid)} 个网格")


def _build_water_polygon_grid():
    global _water_polygon_grid
    _water_polygon_grid = {}

    grid_size = 0.02

    for idx, geom in enumerate(_water_polygon_geoms):
        try:
            minx, miny, maxx, maxy = geom.bounds
            col_start = int(minx / grid_size)
            col_end = int(maxx / grid_size) + 1
            row_start = int(miny / grid_size)
            row_end = int(maxy / grid_size) + 1

            for col in range(col_start, col_end):
                for row in range(row_start, row_end):
                    key = (col, row)
                    if key not in _water_polygon_grid:
                        _water_polygon_grid[key] = []
                    _water_polygon_grid[key].append(idx)
        except Exception:
            continue

    print(f"  水域面空间网格索引已建立: {len(_water_polygon_grid)} 个网格")


def _is_point_in_water(lat, lon):
    """检查点是否在水域面内（使用空间网格加速）"""
    if not _water_polygon_geoms:
        return True  # 无水域面数据时默认通过

    grid_size = 0.02
    col = int(lon / grid_size)
    row = int(lat / grid_size)
    nearby_indices = _water_polygon_grid.get((col, row), [])

    if not nearby_indices:
        for dc in range(-1, 2):
            for dr in range(-1, 2):
                if dc == 0 and dr == 0:
                    continue
                nearby_indices.extend(_water_polygon_grid.get((col + dc, row + dr), []))
        nearby_indices = list(set(nearby_indices))

    if not nearby_indices:
        return False

    point = Point(lon, lat)
    for idx in nearby_indices:
        if _water_polygon_geoms[idx].contains(point):
            return True
    return False


def _is_edge_near_waterway(u, v):
    """检查边 u->v 是否全程靠近水系（多点采样，防止直线段穿过陆地/岛屿）"""
    n1 = nodes_data.get(u)
    n2 = nodes_data.get(v)
    if not n1 or not n2:
        return True

    if not _waterway_geoms:
        return True

    # 修复（2026-06-15）：加载 waypoint 物理路径（只加载一次，两个检查共用）
    # 沿 waypoint 全段采样，而不是沿 u-v 直线 —— 真实轨迹是绕着岛屿的折线，
    # 直线穿岛不代表这条边不合法；桥接边两端各自在独立水系边，直线恰好落两水系之间。
    edge_phys_path = None
    if edge_waypoints and ((u, v) in edge_waypoints or (v, u) in edge_waypoints):
        pts = edge_waypoints.get((u, v)) or edge_waypoints.get((v, u))
        if pts and len(pts) >= 2:
            edge_phys_path = [(p['lat'], p['lon']) for p in pts]

    # ── Part 1: 水系线距离检查（沿 waypoint 路径采样）──
    dist_m = haversine_distance(n1['lat'], n1['lon'], n2['lat'], n2['lon'])
    if dist_m < 1000:
        sample_count = 1
        threshold = WATERWAY_PROXIMITY_M
    elif dist_m < LONG_EDGE_STRICT_THRESHOLD_M:
        sample_count = 3
        threshold = WATERWAY_PROXIMITY_M
    else:
        sample_count = 5
        threshold = min(WATERWAY_PROXIMITY_M, 500)

    for i in range(sample_count):
        if edge_phys_path and len(edge_phys_path) >= 2:
            n_pts = len(edge_phys_path)
            pt_idx = int(round(i / max(sample_count - 1, 1) * (n_pts - 1)))
            lat, lon = edge_phys_path[pt_idx]
        else:
            t = i / max(sample_count - 1, 1)
            lat = n1['lat'] + t * (n2['lat'] - n1['lat'])
            lon = n1['lon'] + t * (n2['lon'] - n1['lon'])

        grid_size = 0.02
        col = int(lon / grid_size)
        row = int(lat / grid_size)
        nearby_indices = _waterway_grid.get((col, row), [])

        if not nearby_indices:
            for dc in range(-1, 2):
                for dr in range(-1, 2):
                    if dc == 0 and dr == 0:
                        continue
                    nearby_indices.extend(_waterway_grid.get((col + dc, row + dr), []))
            nearby_indices = list(set(nearby_indices))

        if not nearby_indices:
            return False

        point = Point(lon, lat)
        min_dist = float('inf')
        for idx in nearby_indices:
            geom = _waterway_geoms[idx]
            try:
                dist = point.distance(geom)
                if dist < min_dist:
                    min_dist = dist
            except Exception:
                continue

        min_dist_m = min_dist * 111000
        if min_dist_m > threshold:
            return False

    # ── Part 2: 水域面(polygon)穿陆检查（仅长边，沿 waypoint 全段采样）──
    # 修复（2026-06-15）：用 waypoint 全段采样替代旧的 10 点直线采样
    # 保持只对 >= 2000m 长边生效，避免对短边过度过滤
    if dist_m >= LONG_EDGE_STRICT_THRESHOLD_M and _water_polygon_geoms:
        grid_size = 0.02
        col1 = int(n1['lon'] / grid_size)
        row1 = int(n1['lat'] / grid_size)
        col2 = int(n2['lon'] / grid_size)
        row2 = int(n2['lat'] / grid_size)
        has_nearby_poly1 = bool(_water_polygon_grid.get((col1, row1)))
        has_nearby_poly2 = bool(_water_polygon_grid.get((col2, row2)))
        if has_nearby_poly1 and has_nearby_poly2:
            # 有 waypoint 时沿全段采样；无 waypoint 时沿直线采样
            poly_sample_count = max(10, len(edge_phys_path)) if edge_phys_path else 10
            max_consecutive_land = 0
            consecutive_land = 0
            for i in range(poly_sample_count):
                if edge_phys_path and len(edge_phys_path) >= 2:
                    n_pts = len(edge_phys_path)
                    pt_idx = int(round(i / max(poly_sample_count - 1, 1) * (n_pts - 1)))
                    lat, lon = edge_phys_path[pt_idx]
                else:
                    t = i / max(poly_sample_count - 1, 1)
                    lat = n1['lat'] + t * (n2['lat'] - n1['lat'])
                    lon = n1['lon'] + t * (n2['lon'] - n1['lon'])
                if not _is_point_in_water(lat, lon):
                    consecutive_land += 1
                    max_consecutive_land = max(max_consecutive_land, consecutive_land)
                else:
                    consecutive_land = 0
            # 自适应阈值：采样点越多容忍度越高（水域面边界有 50-100m 不确定度）
            land_threshold = max(5, poly_sample_count // 3)
            if max_consecutive_land >= land_threshold:
                return False

    return True


def _filter_edges():
    global graph_edges
    _load_waterways()

    original_count = len(graph_edges)

    # 备份所有边，用于后续连通性恢复
    all_edges_backup = dict(graph_edges)

    import networkx as nx
    raw_G = nx.Graph()
    for nid, attrs in nodes_data.items():
        raw_G.add_node(nid, **attrs)
    for (u, v), attrs in graph_edges.items():
        raw_G.add_edge(u, v, weight=attrs.get('weight', 1))
    raw_degrees = dict(raw_G.degree())
    raw_bridges = set(nx.bridges(raw_G))
    raw_bridge_count = len(raw_bridges)
    print(f"\n原始图拓扑分析:")
    print(f"  - 节点数: {raw_G.number_of_nodes()}, 边数: {raw_G.number_of_edges()}")
    print(f"  - 桥接边: {raw_bridge_count} ({raw_bridge_count / raw_G.number_of_edges() * 100:.1f}%)")

    HIGH_FREQ_THRESHOLD = 500
    high_freq_nodes = {nid for nid, attrs in nodes_data.items() if attrs.get('frequency', 0) >= HIGH_FREQ_THRESHOLD}
    hf_edge_dist = {}
    for (u, v), attrs in graph_edges.items():
        if u in high_freq_nodes or v in high_freq_nodes:
            n1 = nodes_data.get(u)
            n2 = nodes_data.get(v)
            if n1 and n2:
                d = haversine_distance(n1['lat'], n1['lon'], n2['lat'], n2['lon'])
                if u in high_freq_nodes:
                    hf_edge_dist.setdefault(u, []).append(((u, v), d))
                if v in high_freq_nodes:
                    hf_edge_dist.setdefault(v, []).append(((u, v), d))
    for nid in hf_edge_dist:
        hf_edge_dist[nid].sort(key=lambda x: x[1])

    length_removed = 0
    water_removed = 0
    bridge_saved = 0
    hf_saved = 0
    passed = {}
    length_examples = []
    water_examples = []

    for (u, v), attrs in graph_edges.items():
        n1 = nodes_data.get(u)
        n2 = nodes_data.get(v)
        if not n1 or not n2:
            passed[(u, v)] = attrs
            continue

        is_bridge = (u, v) in raw_bridges or (v, u) in raw_bridges
        if is_bridge and _is_edge_near_waterway(u, v):
            passed[(u, v)] = attrs
            bridge_saved += 1
            continue

        u_is_hf = u in high_freq_nodes
        v_is_hf = v in high_freq_nodes
        if (u_is_hf or v_is_hf) and _is_edge_near_waterway(u, v):
            target_nid = u if u_is_hf else v
            shortest_edges = hf_edge_dist.get(target_nid, [])
            if shortest_edges and (u, v) in [e[0] for e in shortest_edges[:2]]:
                passed[(u, v)] = attrs
                hf_saved += 1
                continue

        dist = haversine_distance(n1['lat'], n1['lon'], n2['lat'], n2['lon'])

        if dist > MAX_EDGE_LENGTH_M:
            length_removed += 1
            if len(length_examples) < 5:
                length_examples.append((u, v, dist))
            continue

        if not _is_edge_near_waterway(u, v):
            water_removed += 1
            if len(water_examples) < 5:
                water_examples.append((u, v, dist))
            continue

        passed[(u, v)] = attrs

    graph_edges = passed

    # 连通性恢复：被水域面检测过滤的边，如果过滤后两端点不可达，则恢复
    if _water_polygon_geoms:
        import networkx as nx
        restore_G = nx.Graph()
        for nid in nodes_data:
            restore_G.add_node(nid)
        for (u, v) in graph_edges:
            restore_G.add_edge(u, v)
        restored_count = 0
        # 保存水域面数据用于临时清除
        _saved_poly = _water_polygon_geoms
        _saved_pgrid = _water_polygon_grid
        for (u, v), attrs in all_edges_backup.items():
            if (u, v) in graph_edges:
                continue
            # 只恢复因水域面检测被过滤的边（通过水系线距离检测但未通过水域面检测的边）
            n1 = nodes_data.get(u)
            n2 = nodes_data.get(v)
            if not n1 or not n2:
                continue
            dist = haversine_distance(n1['lat'], n1['lon'], n2['lat'], n2['lon'])
            if dist > MAX_EDGE_LENGTH_M:
                continue
            # 临时清除水域面数据检查是否通过水系线距离检测
            _water_polygon_geoms.clear()
            _water_polygon_grid.clear()
            passes_waterway = _is_edge_near_waterway(u, v)
            _water_polygon_geoms.extend(_saved_poly)
            _water_polygon_grid.update(_saved_pgrid)
            if not passes_waterway:
                continue
            # 检查过滤后两端点是否不可达
            if not nx.has_path(restore_G, u, v):
                graph_edges[(u, v)] = attrs
                restore_G.add_edge(u, v)
                restored_count += 1
        if restored_count > 0:
            print(f"  - 恢复穿陆地但维持连通性的边: {restored_count}")

    # ── 连通性恢复 Round 2（2026-06-15 新增）──
    # 对所有被移除的边，如果移除后两端点在不同分量中（即这条边的缺失导致了碎片化），
    # 按距离从小到大贪心恢复。这解决了 Round 1 只能恢复 polygon 失败边的局限性，
    # 覆盖了因水系线距离、超长等原因被移除但对连通性至关重要的边。
    import networkx as nx
    restore_G2 = nx.Graph()
    for nid in nodes_data:
        restore_G2.add_node(nid)
    for (u, v) in graph_edges:
        restore_G2.add_edge(u, v)

    # 收集所有被移除且连接不同分量的边
    candidates = []
    for (u, v), attrs in all_edges_backup.items():
        if (u, v) in graph_edges:
            continue
        n1 = nodes_data.get(u)
        n2 = nodes_data.get(v)
        if not n1 or not n2:
            continue
        if not nx.has_path(restore_G2, u, v):
            dist = haversine_distance(n1['lat'], n1['lon'], n2['lat'], n2['lon'])
            candidates.append((dist, u, v, attrs))

    # 按距离排序，贪心恢复（最短优先 → 优先用短边重建连通性）
    candidates.sort(key=lambda x: x[0])
    restored_round2 = 0
    for dist, u, v, attrs in candidates:
        if not nx.has_path(restore_G2, u, v):
            graph_edges[(u, v)] = attrs
            restore_G2.add_edge(u, v)
            restored_round2 += 1

    if restored_round2 > 0:
        # 报告恢复统计
        comps_after = list(nx.connected_components(restore_G2))
        comps_after.sort(key=len, reverse=True)
        main_pct = len(comps_after[0]) / len(nodes_data) * 100 if nodes_data else 0
        print(f"  - Round 2 连通性恢复: {restored_round2} 条边, {len(comps_after)} 分量, 主分量 {main_pct:.1f}%")

    total_removed = original_count - len(graph_edges)
    print(f"\n边过滤完成:")
    print(f"  - 原始边数: {original_count}")
    print(f"  - 过滤后边数: {len(graph_edges)}")
    print(f"  - 保留桥接边: {bridge_saved}")
    print(f"  - 保留高频节点连接: {hf_saved}")
    print(f"  - 移除超长边 (> {MAX_EDGE_LENGTH_M/1000:.1f}km): {length_removed}")
    print(f"  - 移除远离水系边 (> {WATERWAY_PROXIMITY_M}m): {water_removed}")
    if length_examples:
        print(f"  - 移除超长边示例:")
        for u, v, d in length_examples:
            print(f"      {u} -> {v}: {d/1000:.2f} km")
    if water_examples:
        print(f"  - 移除远离水系边示例:")
        for u, v, d in water_examples:
            n1, n2 = nodes_data[u], nodes_data[v]
            print(f"      {u} -> {v}: {d:.0f}m, 节点位置 ({n1['lat']:.4f},{n1['lon']:.4f})~({n2['lat']:.4f},{n2['lon']:.4f})")



def _compute_graph_topology():
    global graph_degrees, main_component
    import networkx as nx
    G = nx.Graph()
    for nid, attrs in nodes_data.items():
        G.add_node(nid, **attrs)
    for (u, v), attrs in graph_edges.items():
        G.add_edge(u, v, weight=attrs.get('weight', 1))
    graph_degrees = dict(G.degree())
    components = list(nx.connected_components(G))
    components.sort(key=len, reverse=True)
    main_component = components[0] if components else set()


# 船舶类型模板（基于 shipxy/CCS 309 艘真实数据按船型中位数聚合，2026-06-10 更新）
# height/tonnage 无真实源，保留合理推断值
# 2026-06-17 新增：执法船/挖泥船/液体散货/游艇/翼船/其他 六种模板，覆盖 CSV 全部船型
# 2026-06-18 修正：大型集装箱船/大型油轮参数取 ship_characteristics_db.csv 对应船型 max 值
SHIP_TEMPLATES = {
    '小型货船': {'length': 53, 'width': 11, 'draft': 2.6, 'height': 15, 'tonnage': 3000, 'max_speed': 8},
    '中型货船': {'length': 63, 'width': 13, 'draft': 3.2, 'height': 15, 'tonnage': 994, 'max_speed': 8},
    '大型货船': {'length': 77, 'width': 16, 'draft': 3.7, 'height': 15, 'tonnage': 3000, 'max_speed': 8},
    '集装箱船': {'length': 49, 'width': 13, 'draft': 2.4, 'height': 15, 'tonnage': 3000, 'max_speed': 8},
    '大型集装箱船': {'length': 70, 'width': 18, 'draft': 3.8, 'height': 15, 'tonnage': 3000, 'max_speed': 9.2},
    '油轮': {'length': 63, 'width': 13, 'draft': 3.31, 'height': 15, 'tonnage': 1187, 'max_speed': 8},
    '大型油轮': {'length': 96, 'width': 16, 'draft': 5.894, 'height': 15, 'tonnage': 3572, 'max_speed': 11},
    '客船': {'length': 44, 'width': 9, 'draft': 1.85, 'height': 15, 'tonnage': 488, 'max_speed': 8},
    '渔船': {'length': 17, 'width': 4, 'draft': 1.5, 'height': 8, 'tonnage': 200, 'max_speed': 6},
    '拖船': {'length': 31, 'width': 10, 'draft': 2.2, 'height': 10, 'tonnage': 300, 'max_speed': 8},
    '执法船': {'length': 50, 'width': 8, 'draft': 2.0, 'height': 15, 'tonnage': 500, 'max_speed': 18},
    '挖泥船': {'length': 80, 'width': 16, 'draft': 4.5, 'height': 20, 'tonnage': 3000, 'max_speed': 6},
    '液体散货': {'length': 70, 'width': 12, 'draft': 3.5, 'height': 15, 'tonnage': 2000, 'max_speed': 8},
    '游艇': {'length': 30, 'width': 6, 'draft': 1.5, 'height': 10, 'tonnage': 100, 'max_speed': 20},
    '翼船': {'length': 25, 'width': 8, 'draft': 1.0, 'height': 8, 'tonnage': 50, 'max_speed': 30},
    '其他': {'length': 50, 'width': 10, 'draft': 2.5, 'height': 12, 'tonnage': 1000, 'max_speed': 8},
}


def build_graph():
    import networkx as nx
    G = nx.Graph()
    for nid, attrs in nodes_data.items():
        G.add_node(nid, **attrs)
    for (u, v), attrs in graph_edges.items():
        G.add_edge(u, v, weight=attrs.get('weight', 1))
    return G


def find_nearest_node(lat, lon, max_search_radius=5000):
    """
    距离优先 + 频次降权的最近节点匹配。
    历史版本只用频次硬过滤 (>=50 算"水"), 会出现"1km 内低频真实节点全被忽略,
    硬拉到 994m 外高频节点"的 bug(参见 2026-06-06 复盘 §8.4)。
    现改为: score = dist + max(0, 50 - freq) * 5, 取最小 score。
    频次每少 1 次, 距离惩罚 +5m, 频次 = 0 时额外 +250m, 高频节点不惩罚。
    """
    candidates = []
    for nid, attrs in nodes_data.items():
        dist = haversine_distance(lat, lon, attrs['lat'], attrs['lon'])
        if dist < max_search_radius:
            freq = attrs.get('frequency', 1)
            candidates.append((nid, dist, freq))

    if not candidates:
        return None, float('inf')

    # 距离优先, 频次作软惩罚; 频次>=50 时惩罚=0, 等价于纯距离最近
    def _score(c):
        return c[1] + max(0, 50 - c[2]) * 5.0

    best = min(candidates, key=_score)
    return best[0], best[1]


def calculate_edge_cost(u, v, ship, cost_type='distance'):
    key = (u, v)
    feat = edge_features.get(key) or edge_features.get((v, u), {})

    if cost_type == 'distance':
        if feat:
            return feat['avg_distance']
        u_attrs = nodes_data.get(u)
        v_attrs = nodes_data.get(v)
        if u_attrs and v_attrs:
            return haversine_distance(u_attrs['lat'], u_attrs['lon'], v_attrs['lat'], v_attrs['lon']) / 1000

    elif cost_type == 'time':
        if feat:
            speed = feat.get('avg_actual_speed', 8)
            dist = feat.get('avg_distance', 1)
            if speed > 0:
                return dist / (speed * 1.852)
            return feat.get('avg_travel_time', 0.5)
        return 0.5

    return 1.0


def plan_paths(start_node, end_node, ship_type='中型货船', ship_name=None, max_paths=3):
    from datetime import datetime as _dt
    _init_navigator()

    # 2026-06-17 改进：规范化 ship_type + NaN 安全的真实参数优先
    if ship_name and ship_name in ship_db:
        real = ship_db[ship_name]
        resolved_type = real.get('ship_type', ship_type)
        tpl = SHIP_TEMPLATES.get(resolved_type, SHIP_TEMPLATES['中型货船'])
        ship = ShipCharacteristics(
            ship_name=ship_name,
            ship_type=resolved_type,
            length=_safe_float(real.get('length'), tpl['length']),
            width=_safe_float(real.get('width'), tpl['width']),
            draft=_safe_float(real.get('draft'), tpl['draft']),
            height=_safe_float(real.get('height'), tpl['height']),
            tonnage=_safe_float(real.get('tonnage'), tpl['tonnage']),
            max_speed=_safe_float(real.get('max_speed'), tpl['max_speed']),
        )
        ship_tpl = {
            'length': ship.length, 'width': ship.width, 'draft': ship.draft,
            'height': ship.height, 'tonnage': ship.tonnage, 'max_speed': ship.max_speed,
        }
    else:
        ship_tpl = SHIP_TEMPLATES.get(ship_type, SHIP_TEMPLATES['中型货船'])
        ship = ShipCharacteristics(
            ship_name=f'模板_{ship_type}' if not ship_name else ship_name,
            ship_type=ship_type,
            **ship_tpl
        )

    # 使用多目标导航器进行路径规划
    result_paths = _navigator.find_paths(
        start_node, end_node, ship, hour=None, max_paths=max_paths
    )

    if not result_paths:
        return {'success': False, 'message': '起终点之间无可通行路径'}

    path_type_labels = {
        PathType.SAFEST: '安全优先',
        PathType.FASTEST: '时间最短',
        PathType.BALANCED: '综合最优',
        PathType.FREQUENT: '通航频次最高',
        PathType.SHORTEST: '距离最短',
        PathType.RELAXED: '约束放宽路径',
    }

    routes = []
    for p in result_paths:
        # 转换 waypoints 单位（导航器用米/秒，API 输出用公里/分钟）
        waypoints = []
        for wp in p.waypoint_details:
            waypoints.append({
                'sequence': wp['sequence'],
                'from_node': wp['from_node'],
                'to_node': wp['to_node'],
                'distance': round(wp['distance'] / 1000, 1),
                'time': round(wp['time'] / 60, 1),
                'waterway_type': wp.get('waterway_type', 'open'),
            })

        route = {
            'nodes': p.nodes,
            'edges': p.edges,
            'total_distance_km': round(p.total_distance / 1000, 2),
            'total_time_min': round(p.total_time / 60, 2),
            'avg_speed_knots': round(p.avg_speed, 2),
            'risk_score': round(p.risk_score, 1),
            'safety_score': round(p.safety_score, 1),
            'constraints_met': p.constraints_met,
            'waypoint_count': len(waypoints),
            'waypoints': waypoints,
            'type': path_type_labels.get(p.path_type, '未知'),
        }
        routes.append(route)

    # 选择推荐路径：按综合评分排序（安全+时间+距离）
    if len(routes) > 1:
        max_time = max(r['total_time_min'] for r in routes)
        max_dist = max(r['total_distance_km'] for r in routes)
        max_time = max(max_time, 1)
        max_dist = max(max_dist, 1)

        def route_score(r):
            safety_norm = r['safety_score'] / 100
            time_norm = 1 - (r['total_time_min'] / max_time)
            dist_norm = 1 - (r['total_distance_km'] / max_dist)
            return safety_norm * 0.40 + time_norm * 0.30 + dist_norm * 0.30

        routes.sort(key=route_score, reverse=True)
        recommended = routes[0]
    else:
        recommended = routes[0]

    return {
        'success': True,
        'timestamp': str(_dt.now()),
        'departure_time': str(_dt.now()),
        'ship': {
            'name': ship_name if ship_name else f'模板_{ship_type}',
            'type': ship_type,
            'data_source': ship_db.get(ship_name, {}).get('data_source', 'template') if ship_name else 'template',
            **ship_tpl,
        },
        'start_node': start_node,
        'end_node': end_node,
        'recommended_path': recommended,
        'alternative_paths': [r for r in routes if r != recommended],
    }


def build_route_geojson(path_info, path_index):
    FOLD_RATIO = 2.0
    coordinates = []
    waypoints_list = []
    path_nodes = path_info.get('nodes', [])
    # 记录每个坐标是否为节点锚点（简化时必须保留）
    _anchor_flags = []

    for i, node_id in enumerate(path_nodes):
        node = nodes_data.get(node_id)
        if not node:
            continue

        coordinates.append([node['lon'], node['lat']])
        _anchor_flags.append(True)  # 节点锚点，不可删除
        waypoints_list.append({
            'node_id': node_id,
            'lat': node['lat'],
            'lon': node['lon'],
            'type': node.get('type', 'unknown'),
            'frequency': node.get('frequency', 0)
        })

        if i < len(path_nodes) - 1:
            next_node_id = path_nodes[i + 1]
            next_node = nodes_data.get(next_node_id)
            if not next_node:
                continue

            edge_key = (node_id, next_node_id)
            wp_list = edge_waypoints.get(edge_key)
            is_reverse = False
            if wp_list is None:
                edge_key_rev = (next_node_id, node_id)
                wp_list = edge_waypoints.get(edge_key_rev)
                is_reverse = wp_list is not None
            if is_reverse and wp_list:
                wp_list = list(reversed(wp_list))

            if wp_list:
                # 1) RDP 简化（per-edge）：eps=200m，过滤 GPS 锚地残留 fold-back
                #    修复(2026-06-15): 从 100m 提高到 200m，更多 waypoint 噪声被消除
                wp_lonlats = [(wp['lon'], wp['lat']) for wp in wp_list]
                wp_lonlats = _rdp_simplify(wp_lonlats, eps_deg=200 / 111000)
                # 2) 最小距离过滤：相邻坐标 < 50m 跳过
                MIN_SEG_M = 50
                for wp_lon, wp_lat in wp_lonlats:
                    if coordinates:
                        last_lon, last_lat = coordinates[-1]
                        d = haversine_distance(last_lat, last_lon, wp_lat, wp_lon)
                        if d < MIN_SEG_M:
                            continue
                    coordinates.append([wp_lon, wp_lat])
                    _anchor_flags.append(False)  # waypoint 插入点，可被简化

        # 3) Per-edge 收敛循环：删除当前 edge 内的 fold-back 顶点
        changed = True
        while changed and len(coordinates) >= 3:
            changed = False
            j = 1
            while j < len(coordinates) - 1:
                if _anchor_flags[j]:
                    j += 1
                    continue
                a_lon, a_lat = coordinates[j-1]
                b_lon, b_lat = coordinates[j]
                c_lon, c_lat = coordinates[j+1]
                d_ab = haversine_distance(a_lat, a_lon, b_lat, b_lon)
                d_bc = haversine_distance(b_lat, b_lon, c_lat, c_lon)
                d_ac = haversine_distance(a_lat, a_lon, c_lat, c_lon)
                if d_ac > 10 and (d_ab + d_bc) / d_ac > FOLD_RATIO:
                    del coordinates[j]
                    del _anchor_flags[j]
                    changed = True
                    continue
                j += 1

    # ── 保存密集坐标参考（用于后续跨陆地检测恢复原始航道弯折）──
    _dense_coords = [c[:] for c in coordinates]

    # ── 全局后处理（2026-06-15 新增）──
    # 修复跨 edge 的多余转折：per-edge 简化无法捕获跨 edge 边界的 fold-back
    # 和 GPS 锚地残留导致的 zigzag，需要在全局层面再做一轮简化。

    if len(coordinates) > 4:
        # 4) 全局 RDP 简化：eps=300m
        #    对所有非锚点坐标进行简化，保留偏离 > 300m 的真实航道弯折。
        #    节点锚点（_anchor_flags=True）始终保留。
        GLOBAL_EPS_M = 300
        global_eps_deg = GLOBAL_EPS_M / 111000

        # 分段简化：以节点锚点为分段边界，对每段内的 waypoint 独立做 RDP
        segments = []  # [(start_idx, end_idx), ...]
        seg_start = 0
        for k in range(1, len(coordinates)):
            if _anchor_flags[k]:
                if k > seg_start + 1:  # 段内有 waypoint 可简化
                    segments.append((seg_start, k))
                seg_start = k

        # 对每段执行 RDP（保持段端点不动）
        to_keep = [True] * len(coordinates)
        for seg_s, seg_e in segments:
            seg_pts = [tuple(coordinates[k]) for k in range(seg_s, seg_e + 1)]
            if len(seg_pts) < 3:
                continue
            simplified = _rdp_simplify(seg_pts, eps_deg=global_eps_deg)
            simplified_set = set(simplified)
            for k in range(seg_s + 1, seg_e):  # 不改段端点
                if not _anchor_flags[k] and tuple(coordinates[k]) not in simplified_set:
                    to_keep[k] = False

        # 重建 coordinates（保留锚点 + 通过全局 RDP 的 waypoint）
        new_coords = []
        new_flags = []
        for k in range(len(coordinates)):
            if to_keep[k]:
                new_coords.append(coordinates[k])
                new_flags.append(_anchor_flags[k])
        coordinates = new_coords
        _anchor_flags = new_flags

    if len(coordinates) > 4:
        # 5) 全局 Fold-back 收敛循环
        #    检测跨 edge 的 fold-back（如 GPS 锚地造成的 N-S 来回）
        #    仅删除非锚点坐标
        changed = True
        max_iterations = 10  # 防止无限循环
        iteration = 0
        while changed and len(coordinates) >= 3 and iteration < max_iterations:
            changed = False
            iteration += 1
            j = 1
            while j < len(coordinates) - 1:
                if _anchor_flags[j]:
                    j += 1
                    continue
                a_lon, a_lat = coordinates[j-1]
                b_lon, b_lat = coordinates[j]
                c_lon, c_lat = coordinates[j+1]
                d_ab = haversine_distance(a_lat, a_lon, b_lat, b_lon)
                d_bc = haversine_distance(b_lat, b_lon, c_lat, c_lon)
                d_ac = haversine_distance(a_lat, a_lon, c_lat, c_lon)
                if d_ac > 10 and (d_ab + d_bc) / d_ac > FOLD_RATIO:
                    del coordinates[j]
                    del _anchor_flags[j]
                    changed = True
                    continue
                j += 1

    if len(coordinates) > 4:
        # 5.5) 增强锯齿过滤（2026-06-16 v2 重写）
        #      原 v1 过滤器的 d_ab/d_bc < 1000m 距离约束太严格，
        #      导致 d_ab=1588m/turn=176.9° 等明显折返被漏掉。
        #      v2 策略：分两轮，先抓尖锐折返（角度驱动），再抓中等锯齿（ratio 驱动）。
        import math as _math

        def _calc_turn(j):
            """计算 coordinates[j] 处的转折角度和距离指标"""
            a_lon, a_lat = coordinates[j-1]
            b_lon, b_lat = coordinates[j]
            c_lon, c_lat = coordinates[j+1]
            d_ab = haversine_distance(a_lat, a_lon, b_lat, b_lon)
            d_bc = haversine_distance(b_lat, b_lon, c_lat, c_lon)
            d_ac = haversine_distance(a_lat, a_lon, c_lat, c_lon)
            v1x, v1y = a_lon - b_lon, a_lat - b_lat
            v2x, v2y = c_lon - b_lon, c_lat - b_lat
            dot = v1x * v2x + v1y * v2y
            mag1 = _math.sqrt(v1x**2 + v1y**2)
            mag2 = _math.sqrt(v2x**2 + v2y**2)
            if mag1 * mag2 > 1e-12:
                cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                turn_deg = 180.0 - _math.degrees(_math.acos(cos_a))
            else:
                turn_deg = 0
            ratio = (d_ab + d_bc) / d_ac if d_ac > 10 else 1.0
            return turn_deg, ratio, d_ab, d_bc, d_ac

        # Pass 5.5a: 尖锐折返 — turn > 140° 且 ratio > 1.3（不限距离）
        #   接近掉头的转折，无论段长多长都应消除
        changed = True
        while changed and len(coordinates) >= 3:
            changed = False
            j = 1
            while j < len(coordinates) - 1:
                if _anchor_flags[j]:
                    j += 1
                    continue
                turn_deg, ratio, d_ab, d_bc, d_ac = _calc_turn(j)
                if turn_deg > 140 and ratio > 1.3:
                    del coordinates[j]
                    del _anchor_flags[j]
                    changed = True
                    continue
                j += 1

        # Pass 5.5b: 中等锯齿 — ratio > 1.25 且 turn > 70°（不限距离）
        #   绕行比 > 1.25 表示走了至少 25% 的冤枉路，配合 70°+ 转折判定为锯齿
        #   ratio 条件本身已足够保护真实航道弯折（大弯道 ratio 通常 < 1.15）
        changed = True
        while changed and len(coordinates) >= 3:
            changed = False
            j = 1
            while j < len(coordinates) - 1:
                if _anchor_flags[j]:
                    j += 1
                    continue
                turn_deg, ratio, d_ab, d_bc, d_ac = _calc_turn(j)
                if ratio > 1.25 and turn_deg > 70:
                    del coordinates[j]
                    del _anchor_flags[j]
                    changed = True
                    continue
                j += 1

        # Pass 5.5c: 不必要转弯消除 — turn > 100° 且 ratio > 1.1
        #   针对简化后残留的尖锐转弯：即使绕行比不高，
        #   超过 100° 的转弯在视觉上仍显突兀，且通常不是航道必需弯折
        changed = True
        while changed and len(coordinates) >= 3:
            changed = False
            j = 1
            while j < len(coordinates) - 1:
                if _anchor_flags[j]:
                    j += 1
                    continue
                turn_deg, ratio, d_ab, d_bc, d_ac = _calc_turn(j)
                if turn_deg > 100 and ratio > 1.1:
                    del coordinates[j]
                    del _anchor_flags[j]
                    changed = True
                    continue
                j += 1

    if len(coordinates) > 4:
        # 6) Near-U-turn 消除：收敛循环检测角度 > 120° 且前后段 < 800m 的转折点
        #    修复(2026-06-16 v2): 角度从 130° 降至 120°，距离从 600m 升至 800m
        #    v3: 改为收敛循环，一次 pass 可能暴露新的 U-turn
        changed = True
        while changed and len(coordinates) >= 3:
            changed = False
            j = 1
            while j < len(coordinates) - 1:
                if _anchor_flags[j]:
                    j += 1
                    continue
                a_lon, a_lat = coordinates[j-1]
                b_lon, b_lat = coordinates[j]
                c_lon, c_lat = coordinates[j+1]
                d_ab = haversine_distance(a_lat, a_lon, b_lat, b_lon)
                d_bc = haversine_distance(b_lat, b_lon, c_lat, c_lon)
                v1x, v1y = a_lon - b_lon, a_lat - b_lat
                v2x, v2y = c_lon - b_lon, c_lat - b_lat
                dot = v1x * v2x + v1y * v2y
                mag1 = _math.sqrt(v1x**2 + v1y**2)
                mag2 = _math.sqrt(v2x**2 + v2y**2)
                if mag1 * mag2 > 1e-12:
                    cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                    turn_deg = 180.0 - _math.degrees(_math.acos(cos_a))
                else:
                    turn_deg = 0

                if turn_deg > 120 and d_ab < 800 and d_bc < 800:
                    del coordinates[j]
                    del _anchor_flags[j]
                    changed = True
                    continue
                j += 1

    # ── Pass 7: 跨陆地检测与修复（2026-06-16 新增）──
    # 简化可能导致相邻坐标之间的直线穿过陆地（如河流弯道处直线跨越半岛）。
    # 检测方式：沿简化后线段多点采样，用 _is_point_in_water() 判断是否在水域内。
    # 修复方式：若检测到穿陆，从密集坐标参考（全局简化前的航道路径）中恢复中间点。
    if _water_polygon_geoms and len(coordinates) >= 3:
        def _restore_segment(si, ei, depth=0):
            """递归检测并修复 _dense_coords[si:ei+1] 范围内的跨陆线段"""
            if ei - si <= 1:
                return [_dense_coords[ei]]

            # 沿直线均匀采样 5 个点（不含端点）
            n_samples = 5
            land_detected = False
            for k in range(1, n_samples):
                t = k / n_samples
                lat = _dense_coords[si][1] + t * (_dense_coords[ei][1] - _dense_coords[si][1])
                lon = _dense_coords[si][0] + t * (_dense_coords[ei][0] - _dense_coords[si][0])
                if not _is_point_in_water(lat, lon):
                    land_detected = True
                    break

            if not land_detected:
                return [_dense_coords[ei]]  # 直线段安全，无需恢复

            if depth >= 3:
                # 达到递归上限，恢复全段密集坐标保证安全
                return [_dense_coords[k] for k in range(si + 1, ei + 1)]

            mid = (si + ei) // 2
            left = _restore_segment(si, mid, depth + 1)
            right = _restore_segment(mid, ei, depth + 1)
            return left + right

        new_coords = [coordinates[0]]
        new_flags = [_anchor_flags[0]]
        seg_len_threshold = 300  # 仅检查 > 300m 的线段

        for i in range(len(coordinates) - 1):
            lon1, lat1 = coordinates[i]
            lon2, lat2 = coordinates[i + 1]
            seg_m = haversine_distance(lat1, lon1, lat2, lon2)

            if seg_m <= seg_len_threshold:
                new_coords.append(coordinates[i + 1])
                new_flags.append(_anchor_flags[i + 1])
                continue

            # 快速中点检测
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            if _is_point_in_water(mid_lat, mid_lon):
                new_coords.append(coordinates[i + 1])
                new_flags.append(_anchor_flags[i + 1])
                continue

            # 中点不在水域 → 尝试从密集坐标恢复
            si_match = None
            ei_match = None
            for k in range(len(_dense_coords)):
                if _dense_coords[k] == coordinates[i]:
                    si_match = k
                if _dense_coords[k] == coordinates[i + 1]:
                    ei_match = k
                    break

            if si_match is not None and ei_match is not None and ei_match > si_match + 1:
                restored = _restore_segment(si_match, ei_match)
                for pt in restored:
                    new_coords.append(pt)
                    new_flags.append(False)
            else:
                new_coords.append(coordinates[i + 1])
                new_flags.append(_anchor_flags[i + 1])

        coordinates = new_coords
        _anchor_flags = new_flags

    # ── Pass 8: 锚点拐角平滑 v3（2026-06-16）──
    # 拓扑节点（锚点）位置可能形成尖锐转折（如 163.7° U-turn），
    # 因锚点不可删除，前面所有简化器无法处理。
    # 删除锚点后可能产生新的 >140° 转折，因此 >140° 处理需迭代。
    # 策略：
    #   >140° 近 U-turn：删除中间点（A→C 直连），锚点和自由点都处理
    #   90°-140° 尖角（仅锚点）：中点切角（A→mid_AB→mid_BC→C）
    SMOOTH_UTURN_THRESHOLD = 140  # deg
    SMOOTH_SHARP_THRESHOLD = 90   # deg
    if len(coordinates) >= 3:
        import math as _math  # fix: 确保 _math 在 Pass 8 作用域内可用
        # 迭代删除 >140° 转折（删除一个可能暴露相邻新转折）
        for _iter in range(5):
            uturn_indices = []
            for j in range(1, len(coordinates) - 1):
                a_lon, a_lat = coordinates[j-1]
                b_lon, b_lat = coordinates[j]
                c_lon, c_lat = coordinates[j+1]
                v1x, v1y = a_lon - b_lon, a_lat - b_lat
                v2x, v2y = c_lon - b_lon, c_lat - b_lat
                dot = v1x * v2x + v1y * v2y
                mag1 = _math.sqrt(v1x**2 + v1y**2)
                mag2 = _math.sqrt(v2x**2 + v2y**2)
                if mag1 * mag2 > 1e-12:
                    cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                    turn_deg = 180.0 - _math.degrees(_math.acos(cos_a))
                else:
                    continue
                if turn_deg > SMOOTH_UTURN_THRESHOLD:
                    uturn_indices.append(j)
            if not uturn_indices:
                break
            for j in reversed(uturn_indices):
                del coordinates[j]
                del _anchor_flags[j]
        # 对残留锚点尖角做中点切角
        sharp_indices = []
        for j in range(1, len(coordinates) - 1):
            if not _anchor_flags[j]:
                continue
            a_lon, a_lat = coordinates[j-1]
            b_lon, b_lat = coordinates[j]
            c_lon, c_lat = coordinates[j+1]
            v1x, v1y = a_lon - b_lon, a_lat - b_lat
            v2x, v2y = c_lon - b_lon, c_lat - b_lat
            dot = v1x * v2x + v1y * v2y
            mag1 = _math.sqrt(v1x**2 + v1y**2)
            mag2 = _math.sqrt(v2x**2 + v2y**2)
            if mag1 * mag2 > 1e-12:
                cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
                turn_deg = 180.0 - _math.degrees(_math.acos(cos_a))
            else:
                continue
            if turn_deg > SMOOTH_SHARP_THRESHOLD:
                sharp_indices.append((j, turn_deg))
        for j, turn_deg in reversed(sharp_indices):
            a_lon, a_lat = coordinates[j-1]
            b_lon, b_lat = coordinates[j]
            c_lon, c_lat = coordinates[j+1]
            p1_lon = (a_lon + b_lon) / 2.0
            p1_lat = (a_lat + b_lat) / 2.0
            p2_lon = (b_lon + c_lon) / 2.0
            p2_lat = (b_lat + c_lat) / 2.0
            coordinates[j] = [p1_lon, p1_lat]
            _anchor_flags[j] = False
            coordinates.insert(j + 1, [p2_lon, p2_lat])
            _anchor_flags.insert(j + 1, False)

    return {
        'path_id': path_index + 1,
        'path_name': path_info.get('type', ''),
        'coordinates': coordinates,
        'waypoints': waypoints_list,
        'statistics': {
            'total_distance_km': path_info.get('total_distance_km', 0),
            'total_time_min': path_info.get('total_time_min', 0),
            'avg_speed_knots': path_info.get('avg_speed_knots', 0),
            'safety_score': path_info.get('safety_score', 0),
            'risk_score': path_info.get('risk_score', 0),
        },
        'warning': path_info.get('warning'),
    }


@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


@app.route('/api/ship_types', methods=['GET'])
def get_ship_types():
    # 返回 ship_db 中实际存在的船型（已规范化），同时包含所有模板类型
    db_types = sorted(set(d['ship_type'] for d in ship_db.values()))
    tpl_types = list(SHIP_TEMPLATES.keys())
    # 合并去重，保持模板类型在前
    all_types = list(dict.fromkeys(tpl_types + db_types))
    return jsonify({'success': True, 'data': all_types})


@app.route('/api/ships', methods=['GET'])
def get_ships():
    """返回船舶列表，支持按船型过滤和关键词搜索"""
    ship_type_filter = request.args.get('ship_type', '')
    keyword = request.args.get('keyword', '')

    results = []
    for name, data in ship_db.items():
        if ship_type_filter and data['ship_type'] != ship_type_filter:
            continue
        if keyword and keyword.lower() not in name.lower():
            continue
        results.append({
            'ship_name': name,
            'ship_type': data['ship_type'],
            'length': _safe_float(data.get('length')),
            'width': _safe_float(data.get('width')),
            'draft': _safe_float(data.get('draft')),
            'tonnage': _safe_float(data.get('tonnage')),
            'data_source': data.get('data_source', ''),
        })

    results.sort(key=lambda x: (x['ship_type'], x['ship_name']))
    return jsonify({'success': True, 'data': results, 'total': len(results)})


@app.route('/api/topology_nodes', methods=['GET'])
def get_topology_nodes():
    nodes_list = []
    for nid, attrs in nodes_data.items():
        nodes_list.append({
            'node_id': nid,
            'lat': attrs['lat'],
            'lon': attrs['lon'],
            'type': attrs.get('type', 'unknown'),
            'frequency': attrs.get('frequency', 0)
        })
    return jsonify({'success': True, 'data': nodes_list})


@app.route('/api/trajectory_sample')
def trajectory_sample():
    """返回采样轨迹数据用于回放动画
    若前端传入 OD bbox（min_lat/max_lat/min_lon/max_lon），
    则优先返回 bbox 内的轨迹；否则返回全数据集的随机采样。
    """
    try:
        cleaned_path = os.path.join(OUTPUT_DIR, 'cleaned_data.csv')
        if not os.path.exists(cleaned_path):
            return jsonify({'success': False, 'error': '轨迹数据未就绪，请先运行 main.py'})

        try:
            min_lat = float(request.args.get('min_lat', ''))
            max_lat = float(request.args.get('max_lat', ''))
            min_lon = float(request.args.get('min_lon', ''))
            max_lon = float(request.args.get('max_lon', ''))
            has_bbox = True
        except (TypeError, ValueError):
            has_bbox = False

        df = pd.read_csv(cleaned_path, usecols=['船舶英文名称', '纬度', '经度'], nrows=200000)

        if has_bbox:
            df = df[(df['纬度'] >= min_lat) & (df['纬度'] <= max_lat) &
                    (df['经度'] >= min_lon) & (df['经度'] <= max_lon)]

        trajectories = []
        for mmsi, group in df.groupby('船舶英文名称'):
            if len(group) >= 30:
                raw_pts = group.head(300)[['纬度', '经度']].values.tolist()
                pts = [list(wgs84_to_gcj02(p[0], p[1])) for p in raw_pts]
                trajectories.append({'mmsi': str(mmsi), 'points': pts})
            if len(trajectories) >= 12:
                break
        return jsonify({'success': True, 'trajectories': trajectories, 'count': len(trajectories)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/plan', methods=['POST'])
def plan_route():
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'message': '请提供请求数据'}), 400

    start_lat = data.get('start_lat')
    start_lon = data.get('start_lon')
    end_lat = data.get('end_lat')
    end_lon = data.get('end_lon')
    ship_type = data.get('ship_type', '中型货船')
    ship_name = data.get('ship_name', '')

    # 如果有 ship_name，从 DB 查真实 ship_type
    if ship_name and ship_name in ship_db:
        ship_type = ship_db[ship_name].get('ship_type', ship_type)

    if None in [start_lat, start_lon, end_lat, end_lon]:
        return jsonify({'success': False, 'message': '请提供完整的起终点GPS坐标'}), 400

    try:
        slat, slon = float(start_lat), float(start_lon)
        elat, elon = float(end_lat), float(end_lon)
    except (ValueError, TypeError):
        return jsonify({'success': False, 'message': 'GPS坐标格式不正确'}), 400

    try:
        start_node, start_dist = find_nearest_node(slat, slon)
        end_node, end_dist = find_nearest_node(elat, elon)

        if start_node is None or end_node is None:
            return jsonify({'success': False, 'message': '未找到匹配的航道节点，请点击靠近水域的位置'}), 400

        if start_node == end_node:
            return jsonify({'success': False, 'message': '起点和终点匹配到同一航道节点，请拉开距离重新选择'}), 400

        start_info = nodes_data[start_node]
        end_info = nodes_data[end_node]

        start_dist_km = round(start_dist / 1000, 2)
        end_dist_km = round(end_dist / 1000, 2)
        start_freq = nodes_data[start_node].get('frequency', 0)
        end_freq = nodes_data[end_node].get('frequency', 0)
        warnings_list = []

        # 硬上限: 1.5km 外直接拒绝规划, 防止"陆上长直线连接"被误判为穿楼路径
        # (2026-06-06 复盘 §8.7, 原阈值 5km 过宽, 994m 都不警告)
        HARD_SNAP_LIMIT_M = 1500
        if start_dist > HARD_SNAP_LIMIT_M or end_dist > HARD_SNAP_LIMIT_M:
            too_far = []
            if start_dist > HARD_SNAP_LIMIT_M:
                too_far.append(f'起点 {start_dist_km}km')
            if end_dist > HARD_SNAP_LIMIT_M:
                too_far.append(f'终点 {end_dist_km}km')
            return jsonify({
                'success': False,
                'message': f'{"、".join(too_far)}距离最近航道节点超过 1.5km，'
                           f'请贴近水面重新选择起/终点',
                'snap_distance_m': {'start': round(start_dist, 1), 'end': round(end_dist, 1)},
                'snap_limit_m': HARD_SNAP_LIMIT_M,
            }), 400

        # 软警告: 阈值从 2km/5km 降到 0.3km/1km, 让 1km 内的中等距离 snap 也能被用户感知
        if start_dist_km > 1.0:
            warnings_list.append(f'起点距离最近航道节点 {start_dist_km}km（较远），建议靠近水域选择')
        elif start_dist_km > 0.3:
            warnings_list.append(f'起点距离最近航道节点 {start_dist_km}km，建议更靠近水面')
        if end_dist_km > 1.0:
            warnings_list.append(f'终点距离最近航道节点 {end_dist_km}km（较远），建议靠近水域选择')
        elif end_dist_km > 0.3:
            warnings_list.append(f'终点距离最近航道节点 {end_dist_km}km，建议更靠近水面')
        if start_freq < 10:
            warnings_list.append(f'起点匹配的航道节点通航频次较低({start_freq}次)，匹配精度有限')
        if end_freq < 10:
            warnings_list.append(f'终点匹配的航道节点通航频次较低({end_freq}次)，匹配精度有限')

        result = plan_paths(start_node, end_node, ship_type, ship_name=ship_name, max_paths=4)

        if not result.get('success'):
            base_msg = result.get('message', '路径规划失败')
            hint = f'（起点节点{start_node} → 终点节点{end_node}，距离水面分别约 {start_dist_km}km / {end_dist_km}km）'
            return jsonify({
                'success': False,
                'message': base_msg,
                'hint': hint,
                'debug_start_node': start_node,
                'debug_end_node': end_node
            }), 400

        recommended = result.get('recommended_path', {})
        alternatives = result.get('alternative_paths', [])

        all_routes = [recommended] + alternatives
        routes_output = []

        for i, path_info in enumerate(all_routes):
            if path_info:
                route = build_route_geojson(path_info, i)
                routes_output.append(route)

        return jsonify({
            'success': True,
            'data': {
                'start': {
                    'input_lat': slat, 'input_lon': slon,
                    'matched_node': start_node,
                    'matched_lat': start_info['lat'], 'matched_lon': start_info['lon'],
                    'match_distance_km': start_dist_km,
                    'frequency': start_freq,
                    'ship_count': start_info.get('ship_count', 0),
                },
                'end': {
                    'input_lat': elat, 'input_lon': elon,
                    'matched_node': end_node,
                    'matched_lat': end_info['lat'], 'matched_lon': end_info['lon'],
                    'match_distance_km': end_dist_km,
                    'frequency': end_freq,
                    'ship_count': end_info.get('ship_count', 0),
                },
                'warnings': warnings_list,
                'ship': result.get('ship', {}),
                'routes': routes_output
            }
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'路径规划出错: {str(e)}'}), 500


if __name__ == '__main__':
    print("正在加载拓扑数据...")
    load_data()
    print(f"水上航道智能路径规划系统启动中...")
    print(f"访问地址: http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
