"""
航道拓扑节点网络提取系统 - 拓扑网络构建模块（优化版）
功能：构建标准化的水上航道拓扑网络数据集

优化内容：
1. HMM地图匹配（概率化轨迹-节点匹配，替代最近邻）
2. 双向图（替代强制有向，保留通行方向统计）
3. 边形状点（存储航道曲线几何）
4. 时间依赖图（24小时时段权重）
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import networkx as nx
import json
import logging

from config import TOPOLOGY_CONFIG
from utils import haversine_distance, calculate_bearing

logger = logging.getLogger(__name__)


class TopologyBuilder:
    """拓扑网络构建器（优化版）"""

    def __init__(self, config: Dict = None):
        self.config = config if config else TOPOLOGY_CONFIG
        self.graph = nx.Graph()  # 优化：使用无向图作为基础，方向信息存储在边属性中

    def build_topology(self, nodes: List[Dict], trajectories_df: pd.DataFrame) -> nx.Graph:
        """构建航道拓扑网络（优化版）"""
        logger.info("开始构建拓扑网络...")

        self._add_nodes_to_graph(nodes)
        edges = self._extract_trajectory_edges_hmm(nodes, trajectories_df)
        self._add_edges_to_graph(edges)
        self._filter_low_weight_edges()

        if self.config['merge_similar_nodes']:
            self._merge_similar_nodes()

        stats = self._calculate_network_stats()
        logger.info("拓扑网络: %d 节点, %d 边, 聚类系数 %.4f",
                     self.graph.number_of_nodes(), self.graph.number_of_edges(),
                     stats.get('avg_clustering', 0))
        return self.graph

    def _add_nodes_to_graph(self, nodes: List[Dict]):
        for node in nodes:
            self.graph.add_node(
                node['node_id'], lat=node['lat'], lon=node['lon'],
                node_type=node.get('final_type', node['type']),
                frequency=node['frequency'], ship_count=node.get('ship_count', 0),
                type_distribution=node.get('type_distribution', {}))

    def _extract_trajectory_edges_hmm(self, nodes: List[Dict], trajectories_df: pd.DataFrame) -> List[Dict]:
        """使用HMM地图匹配提取轨迹边（优化版）"""
        if not nodes or trajectories_df.empty:
            return []

        node_index = self._build_spatial_index(nodes)
        edges_dict = defaultdict(lambda: {
            'count': 0, 'ships': set(), 'total_distance': 0,
            'total_time': 0, 'speeds': [], 'courses': [],
            'shape_points': [], 'hourly_counts': defaultdict(int),
            'direction_counts': {'forward': 0, 'backward': 0}
        })

        # 动态计算搜索半径：取节点间距的75分位数
        if len(nodes) > 1:
            node_coords = np.array([[n['lat'], n['lon']] for n in nodes])
            from scipy.spatial.distance import pdist
            distances = pdist(node_coords, metric=lambda u, v: haversine_distance(u[0], u[1], v[0], v[1]))
            dynamic_radius = float(np.percentile(distances, 75)) if len(distances) > 0 else 500
            search_radius = max(dynamic_radius * 0.5, 200)  # 至少200米
        else:
            search_radius = 500

        n_ships = trajectories_df['船舶名称'].nunique()
        for idx, (ship_name, group) in enumerate(trajectories_df.groupby('船舶名称')):
            if (idx + 1) % 50 == 0:
                logger.info("边提取进度: %d/%d", idx + 1, n_ships)
            group = group.sort_values('时间').reset_index(drop=True)

            # HMM 地图匹配（使用动态搜索半径）
            visited = self._hmm_map_matching(group, node_index, nodes, search_radius=search_radius)

            for i in range(len(visited) - 1):
                nf, nt = visited[i], visited[i + 1]
                if nf['node_id'] == nt['node_id']:
                    continue

                key = tuple(sorted([nf['node_id'], nt['node_id']]))  # 无向边
                dist = haversine_distance(nf['lat'], nf['lon'], nt['lat'], nt['lon'])
                time_diff = (nt['time'] - nf['time']).total_seconds()

                e = edges_dict[key]
                e['count'] += 1
                e['ships'].add(ship_name)
                e['total_distance'] += dist
                e['total_time'] += time_diff
                e['speeds'].append(nf.get('speed', 0))
                e['courses'].append(nf.get('course', 0))

                # 收集形状点
                if i < len(visited) - 2:
                    e['shape_points'].append((nt['lat'], nt['lon']))

                # 时段统计
                hour = nf['time'].hour if hasattr(nf['time'], 'hour') else 0
                e['hourly_counts'][hour] += 1

                # 方向统计（基于节点ID顺序）
                if nf['node_id'] < nt['node_id']:
                    e['direction_counts']['forward'] += 1
                else:
                    e['direction_counts']['backward'] += 1

        edges = []
        for (n1, n2), a in edges_dict.items():
            # 计算24小时时间依赖权重
            hourly_weights = dict(a['hourly_counts'])
            total_hourly = sum(hourly_weights.values())
            predicted_times = {}
            if total_hourly > 0:
                avg_time = a['total_time'] / a['count'] if a['count'] else 30
                for h in range(24):
                    # 时段因子：高频时段时间更可靠
                    hour_freq = hourly_weights.get(h, 0)
                    reliability = min(1.0, hour_freq / max(total_hourly / 24, 1))
                    predicted_times[h] = avg_time * (1.0 - reliability * 0.2)

            edges.append({
                'from_node': n1, 'to_node': n2, 'count': a['count'],
                'ship_count': len(a['ships']),
                'ships': a['ships'],
                'avg_distance': a['total_distance'] / a['count'] if a['count'] else 0,
                'avg_time': a['total_time'] / a['count'] if a['count'] else 0,
                'avg_speed': np.mean(a['speeds']) if a['speeds'] else 0,
                'avg_course': np.mean(a['courses']) if a['courses'] else 0,
                'shape_points': list(set(a['shape_points'])),
                'predicted_times': predicted_times,
                'hourly_counts': hourly_weights,
                'direction_counts': dict(a['direction_counts']),
                'is_bidirectional': a['direction_counts']['forward'] > 0 and a['direction_counts']['backward'] > 0
            })

        logger.info("提取边: %d 条", len(edges))
        return edges

    def _hmm_map_matching(self, trajectory: pd.DataFrame, node_index: Dict, nodes: List[Dict],
                           search_radius: float = None) -> List[Dict]:
        """HMM 地图匹配：概率化轨迹-节点匹配"""
        if len(trajectory) < 2:
            return []

        # 为每个轨迹点找到候选节点（半径内）
        if search_radius is None:
            search_radius = self.config.get('search_radius', 500)  # 默认500米
        candidate_nodes_per_point = []

        for _, row in trajectory.iterrows():
            lat, lon = row['纬度'], row['经度']
            gl, gn = int(lat / 0.001), int(lon / 0.001)
            nearby = []
            for dl in [-1, 0, 1]:
                for dn in [-1, 0, 1]:
                    nearby.extend(node_index.get((gl + dl, gn + dn), []))

            candidates = []
            for node in nearby:
                dist = haversine_distance(lat, lon, node['lat'], node['lon'])
                if dist < search_radius:
                    # 发射概率：高斯分布
                    emission_prob = np.exp(-dist**2 / (2 * 100**2))
                    candidates.append({
                        'node_id': node['node_id'],
                        'lat': lat, 'lon': lon,
                        'time': row['时间'],
                        'speed': row['航速'],
                        'course': row['航向'],
                        'emission_prob': emission_prob,
                        'dist': dist
                    })

            if not candidates:
                # 无候选节点，使用最近节点
                if nearby:
                    nearest = min(nearby, key=lambda n: haversine_distance(lat, lon, n['lat'], n['lon']))
                    candidates.append({
                        'node_id': nearest['node_id'],
                        'lat': lat, 'lon': lon,
                        'time': row['时间'],
                        'speed': row['航速'],
                        'course': row['航向'],
                        'emission_prob': 0.1,
                        'dist': haversine_distance(lat, lon, nearest['lat'], nearest['lon'])
                    })
                else:
                    # 极端情况：全局无节点，跳过该轨迹点
                    continue

            candidate_nodes_per_point.append(candidates)

        # 确保至少有两个点才能进行Viterbi解码
        if len(candidate_nodes_per_point) < 2:
            return []

        # Viterbi 解码找最优节点序列
        return self._viterbi_decode(candidate_nodes_per_point, trajectory, nodes)

    def _viterbi_decode(self, candidates_per_point: List[List[Dict]], trajectory: pd.DataFrame,
                        nodes: List[Dict]) -> List[Dict]:
        """Viterbi 算法解码最优节点序列"""
        if not candidates_per_point or len(candidates_per_point) < 2:
            return []

        T = len(candidates_per_point)

        # 过滤空候选列表
        valid_points = [(i, cands) for i, cands in enumerate(candidates_per_point) if cands]
        if len(valid_points) < 2:
            return []

        # 重新映射候选列表
        filtered_candidates = [cands for _, cands in valid_points]
        T = len(filtered_candidates)

        # 动态规划表
        dp = []
        backtrack = []

        # 初始化
        dp.append({i: c['emission_prob'] for i, c in enumerate(filtered_candidates[0])})
        backtrack.append({i: None for i in range(len(filtered_candidates[0]))})

        # 递推
        for t in range(1, T):
            curr_dp = {}
            curr_bt = {}
            prev_candidates = filtered_candidates[t - 1]
            curr_candidates = filtered_candidates[t]

            for j, curr_cand in enumerate(curr_candidates):
                max_prob = -1
                best_prev = None

                for i, prev_cand in enumerate(prev_candidates):
                    # 转移概率：基于节点间距离和轨迹一致性
                    if prev_cand['node_id'] == curr_cand['node_id']:
                        trans_prob = 0.9  # 停留在同一节点
                    else:
                        node_dist = haversine_distance(
                            nodes[prev_cand['node_id']]['lat'], nodes[prev_cand['node_id']]['lon'],
                            nodes[curr_cand['node_id']]['lat'], nodes[curr_cand['node_id']]['lon']
                        )
                        # 轨迹点间实际距离
                        actual_dist = haversine_distance(
                            prev_cand['lat'], prev_cand['lon'],
                            curr_cand['lat'], curr_cand['lon']
                        )
                        # 转移概率：节点距离应与实际距离接近
                        dist_diff = abs(node_dist - actual_dist)
                        trans_prob = np.exp(-dist_diff / max(actual_dist, 100))

                    prob = dp[t - 1][i] * trans_prob * curr_cand['emission_prob']
                    if prob > max_prob:
                        max_prob = prob
                        best_prev = i

                curr_dp[j] = max_prob
                curr_bt[j] = best_prev

            dp.append(curr_dp)
            backtrack.append(curr_bt)

        # 确保dp表不为空
        if not dp or not dp[-1]:
            return []

        # 回溯找最优路径
        best_final = max(dp[-1], key=dp[-1].get)
        path = [best_final]
        for t in range(T - 1, 0, -1):
            prev_idx = backtrack[t].get(path[-1])
            if prev_idx is None:
                break
            path.append(prev_idx)
        path.reverse()

        # 去重连续相同节点
        result = []
        last_node_id = None
        for i, idx in enumerate(path):
            t = i  # t对应path中的位置
            if t >= len(filtered_candidates):
                break
            if idx >= len(filtered_candidates[t]):
                continue
            cand = filtered_candidates[t][idx]
            if cand['node_id'] != last_node_id:
                result.append(cand)
                last_node_id = cand['node_id']

        return result

    def _build_spatial_index(self, nodes: List[Dict], grid_size: float = 0.001) -> Dict:
        index = defaultdict(list)
        for node in nodes:
            index[(int(node['lat'] / grid_size), int(node['lon'] / grid_size))].append(node)
        return index

    def _add_edges_to_graph(self, edges: List[Dict]):
        for e in edges:
            self.graph.add_edge(e['from_node'], e['to_node'],
                                weight=e['count'],
                                ship_count=e['ship_count'],
                                ships=e.get('ships', set()),
                                avg_distance=e['avg_distance'],
                                avg_time=e['avg_time'],
                                avg_speed=e['avg_speed'],
                                avg_course=e['avg_course'],
                                shape_points=e.get('shape_points', []),
                                predicted_times=e.get('predicted_times', {}),
                                hourly_counts=e.get('hourly_counts', {}),
                                direction_counts=e.get('direction_counts', {}),
                                is_bidirectional=e.get('is_bidirectional', False))

    def _filter_low_weight_edges(self):
        min_w = self.config['min_edge_weight']
        to_remove = [(u, v) for u, v, d in self.graph.edges(data=True) if d['weight'] < min_w]
        self.graph.remove_edges_from(to_remove)
        isolated = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolated)
        logger.info("过滤: 删除 %d 低权重边, %d 孤立节点", len(to_remove), len(isolated))

    def _merge_similar_nodes(self):
        merge_dist = self.config['node_merge_distance']
        nodes = list(self.graph.nodes(data=True))
        merged = set()
        merge_groups = []

        for i, (n1id, n1d) in enumerate(nodes):
            if n1id in merged:
                continue
            group = [n1id]
            for j, (n2id, n2d) in enumerate(nodes[i+1:], i+1):
                if n2id not in merged and haversine_distance(
                        n1d['lat'], n1d['lon'], n2d['lat'], n2d['lon']) < merge_dist:
                    group.append(n2id)
                    merged.add(n2id)
            if len(group) > 1:
                merge_groups.append(group)

        for group in merge_groups:
            main = max(group, key=lambda n: self.graph.nodes[n]['frequency'])
            for nid in group:
                if nid == main:
                    continue
                for neighbor in self.graph.neighbors(nid):
                    if neighbor != main:
                        ed = self.graph.edges[nid, neighbor]
                        if self.graph.has_edge(main, neighbor):
                            existing = self.graph.edges[main, neighbor]
                            existing['weight'] += ed['weight']
                            merged_ships = existing.get('ships', set()) | ed.get('ships', set())
                            existing['ships'] = merged_ships
                            existing['ship_count'] = len(merged_ships)
                        else:
                            self.graph.add_edge(main, neighbor, **ed)
                self.graph.remove_node(nid)
        logger.info("合并节点组: %d", len(merge_groups))

    def _calculate_network_stats(self) -> Dict:
        stats = {}
        try:
            stats['avg_clustering'] = nx.average_clustering(self.graph)
        except Exception:
            stats['avg_clustering'] = 0
        stats['connected_components'] = nx.number_connected_components(self.graph)
        components = list(nx.connected_components(self.graph))
        if components:
            stats['largest_component_size'] = len(max(components, key=len))
        else:
            stats['largest_component_size'] = 0
        degrees = [d for _, d in self.graph.degree()]
        stats['avg_degree'] = np.mean(degrees) if degrees else 0
        return stats

    def export_to_json(self, output_path: str):
        nodes_data = [{'id': int(n), 'lat': float(d['lat']), 'lon': float(d['lon']),
                        'type': d['node_type'], 'frequency': int(d['frequency']),
                        'ship_count': int(d.get('ship_count', 0))}
                       for n, d in self.graph.nodes(data=True)]
        edges_data = []
        for u, v, d in self.graph.edges(data=True):
            edge_data = {
                'from': int(u), 'to': int(v),
                'weight': int(d['weight']),
                'ship_count': int(d['ship_count']),
                'avg_speed': float(d.get('avg_speed', 0)),
                'avg_distance': float(d.get('avg_distance', 0)),
                'avg_time': float(d.get('avg_time', 0)),
                'is_bidirectional': bool(d.get('is_bidirectional', False)),
                'predicted_times': {str(k): float(v) for k, v in d.get('predicted_times', {}).items()},
                'shape_points': d.get('shape_points', [])
            }
            edges_data.append(edge_data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({'nodes': nodes_data, 'edges': edges_data,
                        'metadata': {'node_count': len(nodes_data), 'edge_count': len(edges_data)}},
                       f, ensure_ascii=False, indent=2)
        logger.info("已导出 JSON: %s", output_path)

    def export_to_csv(self, nodes_path: str, edges_path: str):
        nodes_data = [{'node_id': n, 'lat': d['lat'], 'lon': d['lon'],
                       'type': d['node_type'], 'frequency': d['frequency'],
                       'ship_count': d.get('ship_count', 0)}
                      for n, d in self.graph.nodes(data=True)]
        if nodes_data:
            nodes_df = pd.DataFrame(nodes_data)
            nodes_df.to_csv(nodes_path, index=False)
        else:
            # 写入空表头
            pd.DataFrame(columns=['node_id', 'lat', 'lon', 'type', 'frequency', 'ship_count']).to_csv(nodes_path, index=False)

        edges_data = [{
            'from_node': u, 'to_node': v, 'weight': d['weight'],
            'ship_count': d['ship_count'],
            'avg_speed': d.get('avg_speed', 0),
            'avg_distance': d.get('avg_distance', 0),
            'avg_time': d.get('avg_time', 0),
            'is_bidirectional': d.get('is_bidirectional', False)
        } for u, v, d in self.graph.edges(data=True)]

        if edges_data:
            edges_df = pd.DataFrame(edges_data)
            edges_df.to_csv(edges_path, index=False)
        else:
            pd.DataFrame(columns=['from_node', 'to_node', 'weight', 'ship_count',
                                   'avg_speed', 'avg_distance', 'avg_time', 'is_bidirectional']).to_csv(edges_path, index=False)

        logger.info("已导出 CSV: %s, %s", nodes_path, edges_path)
