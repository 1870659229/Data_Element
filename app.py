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
from math import radians, cos, sin, asin, sqrt
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='static', static_url_path='/static')

OUTPUT_DIR = 'output'
nodes_data = {}
graph_edges = []


def haversine_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return c * 6371000


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
    with open(edges_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            graph_edges.append({
                'from': int(row['from_node']),
                'to': int(row['to_node']),
                'weight': float(row.get('weight', 1))
            })

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

    print(f"数据加载完成: {len(nodes_data)} 个节点, {len(graph_edges)} 条边, {len(edge_features)} 条边特征")


edge_features = {}

SHIP_TEMPLATES = {
    '小型货船': {'length': 80, 'width': 12, 'draft': 4.5, 'height': 15, 'tonnage': 3000, 'max_speed': 12},
    '中型货船': {'length': 150, 'width': 22, 'draft': 7.5, 'height': 25, 'tonnage': 15000, 'max_speed': 14},
    '大型货船': {'length': 250, 'width': 32, 'draft': 11.0, 'height': 30, 'tonnage': 50000, 'max_speed': 15},
    '集装箱船': {'length': 200, 'width': 30, 'draft': 10.0, 'height': 40, 'tonnage': 35000, 'max_speed': 18},
    '大型集装箱船': {'length': 350, 'width': 45, 'draft': 14.0, 'height': 50, 'tonnage': 100000, 'max_speed': 22},
    '油轮': {'length': 180, 'width': 28, 'draft': 9.0, 'height': 20, 'tonnage': 25000, 'max_speed': 14},
    '大型油轮': {'length': 300, 'width': 50, 'draft': 15.0, 'height': 25, 'tonnage': 120000, 'max_speed': 15},
    '客船': {'length': 100, 'width': 18, 'draft': 5.0, 'height': 30, 'tonnage': 8000, 'max_speed': 20},
    '渔船': {'length': 30, 'width': 6, 'draft': 2.5, 'height': 8, 'tonnage': 200, 'max_speed': 10},
    '拖船': {'length': 25, 'width': 8, 'draft': 3.0, 'height': 10, 'tonnage': 300, 'max_speed': 12},
}


def build_graph():
    import networkx as nx
    G = nx.DiGraph()
    for nid, attrs in nodes_data.items():
        G.add_node(nid, **attrs)
    for edge in graph_edges:
        G.add_edge(edge['from'], edge['to'], weight=edge['weight'])
    return G


def find_nearest_node(lat, lon):
    min_dist = float('inf')
    nearest = None
    for nid, attrs in nodes_data.items():
        dist = haversine_distance(lat, lon, attrs['lat'], attrs['lon'])
        if dist < min_dist:
            min_dist = dist
            nearest = nid
    return nearest, min_dist


def calculate_edge_cost(u, v, ship, cost_type='distance'):
    key = (u, v)
    feat = edge_features.get(key, {})

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


def plan_paths(start_node, end_node, ship_type='中型货船', max_paths=3):
    import networkx as nx
    from collections import defaultdict

    ship = SHIP_TEMPLATES.get(ship_type, SHIP_TEMPLATES['中型货船'])
    G = build_graph()

    if start_node not in G or end_node not in G:
        return {'success': False, 'message': '起点或终点不在拓扑网络中'}

    if not nx.has_path(G, start_node, end_node):
        return {'success': False, 'message': '起点和终点之间无可用路径'}

    def compute_path_stats(nodes, path_type):
        edges_list = []
        total_dist_km = 0
        total_time_min = 0
        total_risk = 0
        waypoints = []

        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            edges_list.append([u, v])

            key = (u, v)
            feat = edge_features.get(key, {})
            dist_km = feat.get('avg_distance', 0.5)
            time_min = feat.get('avg_travel_time', 0.5) if feat.get('avg_travel_time') else (dist_km / (feat.get('avg_actual_speed', 8) * 1.852) * 60)
            seg_count = feat.get('segment_count', 0)
            wtype = feat.get('waterway_type', 'open')

            edge_risk = 30
            if wtype == 'narrow':
                edge_risk += 15
            if seg_count < 3:
                edge_risk += 10

            total_dist_km += dist_km
            total_time_min += time_min
            total_risk += edge_risk

            waypoints.append({
                'sequence': i + 1,
                'from_node': u,
                'to_node': v,
                'distance': round(dist_km * 1000, 2),
                'time': round(time_min * 60, 2),
                'risk': round(edge_risk, 1),
                'waterway_type': wtype
            })

        n = len(edges_list)
        avg_risk = total_risk / n if n > 0 else 0
        safety_score = max(0, 100 - avg_risk)
        avg_speed = (total_dist_km / (total_time_min / 60)) / 1.852 if total_time_min > 0 else 8

        return {
            'type': path_type,
            'nodes': nodes,
            'edges': edges_list,
            'total_distance_km': round(total_dist_km, 2),
            'total_time_min': round(total_time_min, 2),
            'avg_speed_knots': round(avg_speed, 2),
            'risk_score': round(avg_risk, 1),
            'safety_score': round(safety_score, 1),
            'constraints_met': True,
            'waypoint_count': len(waypoints),
            'waypoints': waypoints
        }

    all_paths = []

    try:
        raw_paths = list(nx.all_simple_paths(G, start_node, end_node, cutoff=12))
    except Exception:
        raw_paths = []

    if len(raw_paths) >= 2:
        scored = []
        for nodes in raw_paths:
            edges_count = len(nodes) - 1
            blocked = 0
            for i in range(edges_count):
                key = (nodes[i], nodes[i + 1])
                feat = edge_features.get(key, {})
                if feat.get('waterway_type') == 'narrow' and ship['draft'] > 8:
                    blocked += 1
            scored.append((nodes, edges_count, blocked))

        scored.sort(key=lambda x: (x[2], -x[1]))

        type_map = {0: '综合最优', 1: '时间最短', 2: '距离最短'}
        seen_seqs = set()
        for i, (nodes, _, _) in enumerate(scored):
            if len(all_paths) >= max_paths:
                break
            seq = tuple(nodes)
            if seq in seen_seqs:
                continue
            seen_seqs.add(seq)
            path_type = type_map.get(i, '通航频次最高')
            all_paths.append(compute_path_stats(nodes, path_type))

        if len(all_paths) >= max_paths:
            result = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'departure_time': datetime.now().isoformat(),
                'ship': {
                    'name': f'模板_{ship_type}',
                    'type': ship_type,
                    **ship
                },
                'start_node': start_node,
                'end_node': end_node,
                'recommended_path': all_paths[0],
                'alternative_paths': all_paths[1:]
            }
            return result

    dist_path = nx.dijkstra_path(G, start_node, end_node, weight=lambda u, v, d: calculate_edge_cost(u, v, ship, 'distance'))
    if dist_path is None:
        return {'success': False, 'message': '未找到路径'}

    time_path = None
    try:
        time_path = nx.dijkstra_path(G, start_node, end_node, weight=lambda u, v, d: calculate_edge_cost(u, v, ship, 'time'))
    except Exception:
        time_path = dist_path

    if not time_path:
        time_path = dist_path

    dist_seq = tuple(dist_path)
    time_seq = tuple(time_path)

    if dist_seq == time_seq:
        all_paths.append(compute_path_stats(dist_path, '综合最优'))
    else:
        all_paths.append(compute_path_stats(dist_path, '距离最短'))
        if len(all_paths) < max_paths and time_seq != dist_seq:
            all_paths.append(compute_path_stats(time_path, '时间最短'))

    if len(all_paths) < max_paths:
        try:
            for edge_idx, _ in enumerate(zip(dist_path, dist_path[1:])):
                if len(all_paths) >= max_paths:
                    break
                temp_G = G.copy()
                u, v = dist_path[edge_idx], dist_path[edge_idx + 1]
                if temp_G.has_edge(u, v):
                    temp_G.remove_edge(u, v)
                if nx.has_path(temp_G, start_node, end_node):
                    alt_path = nx.dijkstra_path(temp_G, start_node, end_node)
                    alt_seq = tuple(alt_path)
                    if alt_seq != dist_seq and alt_seq != time_seq:
                        seen = {tuple(p['nodes']) for p in all_paths}
                        if alt_seq not in seen:
                            all_paths.append(compute_path_stats(alt_path, '综合最优'))
        except Exception:
            pass

    if len(all_paths) == 1:
        p = all_paths[0]
        p['type'] = '综合最优'
        all_paths[0] = p

    result = {
        'success': True,
        'timestamp': datetime.now().isoformat(),
        'departure_time': datetime.now().isoformat(),
        'ship': {
            'name': f'模板_{ship_type}',
            'type': ship_type,
            **ship
        },
        'start_node': start_node,
        'end_node': end_node,
        'recommended_path': all_paths[0],
        'alternative_paths': all_paths[1:]
    }
    return result


def build_route_geojson(path_info, path_index):
    coordinates = []
    waypoints_list = []
    path_nodes = path_info.get('nodes', [])

    for node_id in path_nodes:
        node = nodes_data.get(node_id)
        if node:
            coordinates.append([node['lon'], node['lat']])
            waypoints_list.append({
                'node_id': node_id,
                'lat': node['lat'],
                'lon': node['lon'],
                'type': node.get('type', 'unknown'),
                'frequency': node.get('frequency', 0)
            })

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
        }
    }


@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


@app.route('/api/ship_types', methods=['GET'])
def get_ship_types():
    types = list(SHIP_TEMPLATES.keys())
    return jsonify({'success': True, 'data': types})


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
            return jsonify({'success': False, 'message': '未找到匹配的航道节点'}), 400

        if start_node == end_node:
            return jsonify({'success': False, 'message': '起点和终点匹配到同一节点'}), 400

        start_info = nodes_data[start_node]
        end_info = nodes_data[end_node]

        result = plan_paths(start_node, end_node, ship_type, max_paths=3)

        if not result.get('success'):
            return jsonify({'success': False, 'message': result.get('message', '路径规划失败')}), 400

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
                    'match_distance_km': round(start_dist / 1000, 2)
                },
                'end': {
                    'input_lat': elat, 'input_lon': elon,
                    'matched_node': end_node,
                    'matched_lat': end_info['lat'], 'matched_lon': end_info['lon'],
                    'match_distance_km': round(end_dist / 1000, 2)
                },
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
