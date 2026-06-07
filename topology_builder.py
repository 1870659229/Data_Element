# -*- coding: utf-8 -*-
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
from scipy.spatial import cKDTree

from config import TOPOLOGY_CONFIG, EMISSION_VARIANCE_CONFIG, KD_TREE_CONFIG
from utils import haversine_distance, calculate_bearing, calculate_angle_difference

logger = logging.getLogger(__name__)


def _process_single_ship(args):
    """模块顶层工作函数（用于多进程）- 每个worker自己构建KD-Tree"""
    ship_name, group, nodes_data, search_radius = args
    
    # 每个worker自己构建KD-Tree（节点只有1195个，构建很快）
    node_list = nodes_data
    N = len(node_list)
    coords_rad = np.zeros((N, 2))
    for i, n in enumerate(node_list):
        coords_rad[i, 0] = np.radians(n['lat'])
        coords_rad[i, 1] = np.radians(n['lon'])
    tree = cKDTree(coords_rad)
    node_index = (tree, coords_rad, node_list)
    
    # 创建临时builder实例
    builder = TopologyBuilder()
    visited = builder._hmm_map_matching(group, node_index, search_radius=search_radius)
    
    edges = []
    for i in range(len(visited) - 1):
        nf, nt = visited[i], visited[i + 1]
        if nf['node_id'] == nt['node_id']:
            continue
        key = tuple(sorted([nf['node_id'], nt['node_id']]))
        dist = haversine_distance(nf['lat'], nf['lon'], nt['lat'], nt['lon'])
        time_diff = (nt['time'] - nf['time']).total_seconds()
        hour = nf['time'].hour if hasattr(nf['time'], 'hour') else 0
        forward = 1 if nf['node_id'] < nt['node_id'] else 0
        edges.append({
            'key': key, 'ship_name': ship_name,
            'dist': dist, 'time_diff': time_diff,
            'speed': nf.get('speed', 0), 'course': nf.get('course', 0),
            'shape_point': (nt['lat'], nt['lon']) if i < len(visited) - 2 else None,
            'hour': hour, 'forward': forward
        })
    return edges


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

        # 动态计算搜索半径：取节点间距的75分位数，至少覆盖大部分区域
        if len(nodes) > 1:
            node_coords = np.array([[n['lat'], n['lon']] for n in nodes])
            from scipy.spatial.distance import pdist
            distances = pdist(node_coords, metric=lambda u, v: haversine_distance(u[0], u[1], v[0], v[1]))
            dynamic_radius = float(np.percentile(distances, 75)) if len(distances) > 0 else 500
            multiplier = KD_TREE_CONFIG.get('search_radius_multiplier', 2.0)
            search_radius = max(dynamic_radius * multiplier, KD_TREE_CONFIG['min_search_radius'])
            search_radius = min(search_radius, KD_TREE_CONFIG['max_search_radius'])
        else:
            search_radius = KD_TREE_CONFIG['min_search_radius']

        n_ships = trajectories_df['船舶名称'].nunique()
        
        # 准备船舶数据列表（用于并行处理）
        ship_data = []
        for ship_name, group in trajectories_df.groupby('船舶名称'):
            group = group.sort_values('时间').reset_index(drop=True)
            ship_data.append((ship_name, group))
        
        # 准备参数列表（传nodes数据而非KD-Tree，让每个worker自己构建KD-Tree）
        nodes_data = [{'node_id': n['node_id'], 'lat': n['lat'], 'lon': n['lon']} for n in nodes]
        args_list = [(name, group, nodes_data, search_radius) for name, group in ship_data]
        
        # 使用进程池（每个worker自己构建KD-Tree，绕过GIL）
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing
        
        n_workers = min(multiprocessing.cpu_count(), 8)  # 最多用8个核心
        logger.info("并行处理: %d 艘船, %d 个工作进程", n_ships, n_workers)
        
        all_edges = []
        completed = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_process_single_ship, args) for args in args_list]
            for future in as_completed(futures):
                try:
                    ship_edges = future.result()
                    all_edges.extend(ship_edges)
                    completed += 1
                    if completed % 50 == 0:
                        logger.info("边提取进度: %d/%d", completed, n_ships)
                except Exception as e:
                    logger.error("处理船舶时出错: %s", e)
        
        # 聚合边数据
        for edge_info in all_edges:
            key = edge_info['key']
            e = edges_dict[key]
            e['count'] += 1
            e['ships'].add(edge_info['ship_name'])
            e['total_distance'] += edge_info['dist']
            e['total_time'] += edge_info['time_diff']
            e['speeds'].append(edge_info['speed'])
            e['courses'].append(edge_info['course'])
            if edge_info['shape_point']:
                e['shape_points'].append(edge_info['shape_point'])
            e['hourly_counts'][edge_info['hour']] += 1
            if edge_info['forward']:
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

    def _hmm_map_matching(self, trajectory: pd.DataFrame, node_index: Tuple,
                           search_radius: float = None) -> List[Dict]:
        """HMM 地图匹配（KD-Tree 批量优化版）"""
        if len(trajectory) < 2:
            return []

        # 超长轨迹分段处理（每段最多5000点，有重叠）
        segment_size = 5000
        if len(trajectory) > segment_size:
            return self._hmm_map_matching_segmented(trajectory, node_index, search_radius, segment_size)

        return self._hmm_map_matching_single(trajectory, node_index, search_radius)

    def _hmm_map_matching_segmented(self, trajectory: pd.DataFrame, node_index: Tuple,
                                     search_radius: float = None, segment_size: int = 5000) -> List[Dict]:
        """超长轨迹分段处理（每段独立做Viterbi，最后合并）"""
        n = len(trajectory)
        overlap = 500  # 重叠点数，确保边界处连续性
        
        logger.info("超长轨迹分段处理: %d 点, 分段大小 %d", n, segment_size)
        
        all_visited = []
        last_node_id = None
        
        for start in range(0, n, segment_size - overlap):
            end = min(start + segment_size, n)
            segment = trajectory.iloc[start:end].reset_index(drop=True)
            
            # 对每段做完整的 HMM 匹配
            visited = self._hmm_map_matching_single(segment, node_index, search_radius)
            
            if not visited:
                continue
            
            # 合并时去掉重叠部分的重复节点
            if last_node_id is not None:
                # 跳过与上一段末尾相同的节点
                visited = [v for v in visited if v['node_id'] != last_node_id]
            
            if visited:
                all_visited.extend(visited)
                last_node_id = visited[-1]['node_id']
            
            if end == n:
                break
        
        return all_visited

    def _hmm_map_matching_single(self, trajectory: pd.DataFrame, node_index: Tuple,
                                  search_radius: float = None) -> List[Dict]:
        """单段轨迹的HMM匹配（不含分段逻辑）"""
        if len(trajectory) < 2:
            return []

        tree, node_coords_rad, node_list = node_index

        if search_radius is None:
            search_radius = self.config.get('search_radius', 500)

        # 批量查询所有轨迹点的候选节点
        lats = trajectory['纬度'].values
        lons = trajectory['经度'].values
        times = trajectory['时间'].values
        speeds = trajectory['航速'].values
        courses = trajectory['航向'].values
        
        # 确保时间是 pandas Timestamp（支持 .total_seconds()）
        times = pd.to_datetime(times)
        
        pts_rad = np.column_stack([np.radians(lats), np.radians(lons)])
        radius_rad = search_radius / 6371000.0
        all_nearby = tree.query_ball_point(pts_rad, radius_rad)
        
        # 限制每个点的最大候选数（取最近的10个）
        max_candidates = 10
        
        candidate_nodes_per_point = []
        for i in range(len(lats)):
            nearby_idxs = all_nearby[i]
            if not nearby_idxs:
                # 无候选，找最近节点
                nearest_dist, nearest_idx = tree.query(pts_rad[i], k=1)
                nearest = node_list[nearest_idx]
                candidates = [{
                    'node_id': nearest['node_id'],
                    'lat': lats[i], 'lon': lons[i],
                    'time': times[i], 'speed': speeds[i], 'course': courses[i],
                    'emission_prob': 0.1,
                    'dist': haversine_distance(lats[i], lons[i], nearest['lat'], nearest['lon'])
                }]
            else:
                # 批量计算距离（向量化版本，更快）
                nearby_lats = np.array([node_list[j]['lat'] for j in nearby_idxs])
                nearby_lons = np.array([node_list[j]['lon'] for j in nearby_idxs])
                
                # 向量化 Haversine 距离计算
                lat1_rad = np.radians(lats[i])
                lon1_rad = np.radians(lons[i])
                lat2_rad = np.radians(nearby_lats)
                lon2_rad = np.radians(nearby_lons)
                
                dlat = lat2_rad - lat1_rad
                dlon = lon2_rad - lon1_rad
                
                a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
                dists = 2 * 6371000 * np.arcsin(np.sqrt(a))
                
                # 限制候选数：按距离排序取最近的K个
                if len(dists) > max_candidates:
                    top_k = np.argpartition(dists, max_candidates)[:max_candidates]
                    nearby_idxs = [nearby_idxs[j] for j in top_k]
                    dists = dists[top_k]
                
                # 自适应方差
                adaptive_var = self._compute_adaptive_variance_for_density(len(nearby_idxs), search_radius)
                
                # 批量计算发射概率
                emission_probs = np.exp(-dists**2 / (2 * adaptive_var))
                
                candidates = []
                for j, idx in enumerate(nearby_idxs):
                    node = node_list[idx]
                    candidates.append({
                        'node_id': node['node_id'],
                        'lat': lats[i], 'lon': lons[i],
                        'time': times[i], 'speed': speeds[i], 'course': courses[i],
                        'emission_prob': emission_probs[j],
                        'dist': dists[j]
                    })
            
            candidate_nodes_per_point.append(candidates)

        if len(candidate_nodes_per_point) < 2:
            return []

        return self._viterbi_decode(candidate_nodes_per_point, trajectory, node_list)

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
                    # 转移概率：方向感知的距离一致性
                    if prev_cand['node_id'] == curr_cand['node_id']:
                        trans_prob = 0.9
                    else:
                        prev_node = nodes[prev_cand['node_id']]
                        curr_node = nodes[curr_cand['node_id']]
                        trans_prob = self._compute_transition_probability(
                            prev_node, curr_node, prev_cand, curr_cand)

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

    def _build_spatial_index(self, nodes: List[Dict]) -> Tuple:
        """构建 KD-Tree 空间索引（替代原 Grid 索引）
        
        Returns:
            (tree, coords_rad, node_list): cKDTree, 弧度坐标数组, 节点列表
        """
        node_list = list(nodes)
        N = len(node_list)
        coords_rad = np.zeros((N, 2))
        for i, n in enumerate(node_list):
            coords_rad[i, 0] = np.radians(n['lat'])
            coords_rad[i, 1] = np.radians(n['lon'])
        tree = cKDTree(coords_rad)
        return tree, coords_rad, node_list

    def _compute_adaptive_variance(self, candidate_nodes, lat, lon, search_radius):
        """根据局部节点密度自适应计算发射概率方差
        密集区 → 小方差（严格匹配）；稀疏区 → 大方差（宽松匹配）
        """
        base_var = EMISSION_VARIANCE_CONFIG.get('base_variance', 100.0 ** 2)
        min_var = EMISSION_VARIANCE_CONFIG.get('min_variance', 30.0 ** 2)
        max_var = EMISSION_VARIANCE_CONFIG.get('max_variance', 300.0 ** 2)

        if not candidate_nodes:
            return base_var

        # 计算局部节点密度：候选节点间的平均最近邻距离
        if len(candidate_nodes) >= 2:
            coords = np.array([[n['lat'], n['lon']] for n in candidate_nodes])
            distances = []
            for i in range(len(coords)):
                dists = [haversine_distance(coords[i][0], coords[i][1],
                                            coords[j][0], coords[j][1])
                         for j in range(len(coords)) if j != i]
                if dists:
                    distances.append(min(dists))
            avg_nearest_dist = np.mean(distances) if distances else search_radius
        else:
            avg_nearest_dist = search_radius

        # 方差与平均最近邻距离成正比
        density_factor = avg_nearest_dist / max(search_radius, 1)
        adaptive_var = base_var * np.clip(density_factor, 0.3, 3.0)
        adaptive_var = np.clip(adaptive_var, min_var, max_var)

        return float(adaptive_var)

    def _compute_adaptive_variance_for_density(self, n_candidates: int, search_radius: float) -> float:
        """基于候选节点数量自适应计算发射概率方差（轻量版）
        
        候选少 → 大方差（宽松匹配，让远距离节点也有机会被匹配）
        候选多 → 小方差（严格匹配，只匹配近距离节点）
        """
        base_var = EMISSION_VARIANCE_CONFIG.get('base_variance', 100.0 ** 2)
        min_var = EMISSION_VARIANCE_CONFIG.get('min_variance', 30.0 ** 2)
        max_var = EMISSION_VARIANCE_CONFIG.get('max_variance', 300.0 ** 2)

        if n_candidates <= 0:
            return max_var  # 无候选时使用最大方差，匹配对距离不敏感

        density_ratio = n_candidates / max(5, 1)
        density_factor = np.clip(1.0 / density_ratio, 0.3, 3.0)
        adaptive_var = base_var * density_factor
        return float(np.clip(adaptive_var, min_var, max_var))

    def _compute_transition_probability(self, prev_node, curr_node, prev_point, curr_point):
        """方向感知的转移概率：距离一致性 + 方向一致性"""
        node_dist = haversine_distance(
            prev_node['lat'], prev_node['lon'],
            curr_node['lat'], curr_node['lon']
        )
        actual_dist = haversine_distance(
            prev_point['lat'], prev_point['lon'],
            curr_point['lat'], curr_point['lon']
        )

        # 距离一致性
        dist_diff = abs(node_dist - actual_dist)
        dist_score = np.exp(-dist_diff / max(actual_dist, 100))

        # 方向一致性
        node_bearing = calculate_bearing(
            prev_node['lat'], prev_node['lon'],
            curr_node['lat'], curr_node['lon']
        )
        traj_bearing = calculate_bearing(
            prev_point['lat'], prev_point['lon'],
            curr_point['lat'], curr_point['lon']
        )
        angle_diff = calculate_angle_difference(node_bearing, traj_bearing)
        direction_score = max(0.0, 1.0 - angle_diff / 90.0)

        # 综合：距离权重 0.6，方向权重 0.4
        combined_score = 0.6 * dist_score + 0.4 * direction_score

        return float(combined_score)

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
        min_ships = self.config.get('min_ship_count', 0)
        to_remove = [(u, v) for u, v, d in self.graph.edges(data=True)
                     if d['weight'] < min_w or d.get('ship_count', 0) < min_ships]
        self.graph.remove_edges_from(to_remove)
        isolated = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolated)
        logger.info("过滤: 删除 %d 低权重/低船舶边, %d 孤立节点", len(to_remove), len(isolated))

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

        # 导出 edge_waypoints.csv（从 shape_points 生成，按沿实际水道的最近邻顺序排序）
        # 说明：单点投影参数 t 对弯曲水道不可靠（不同方向的轨迹点会交错），
        # 改用：从 u 出发，按最近邻贪心串联所有 shape_points，最后逼近 v
        waypoints_path = edges_path.replace('topology_edges.csv', 'edge_waypoints.csv')
        wp_rows = []
        for u, v, d in self.graph.edges(data=True):
            u_data = self.graph.nodes[u]
            v_data = self.graph.nodes[v]
            ux, uy = u_data['lon'], u_data['lat']
            vx, vy = v_data['lon'], v_data['lat']
            # 收集有效点（粗筛：到 u-v 连线的垂距不能太大，否则属于另一条边）
            dx, dy = vx - ux, vy - uy
            L2 = dx * dx + dy * dy
            if L2 == 0:
                continue
            candidates = []
            for pt in d.get('shape_points', []):
                if not (isinstance(pt, (list, tuple)) and len(pt) >= 2):
                    continue
                plat, plon = pt[0], pt[1]
                # 投影参数 t∈[0,1]（粗筛）
                t = ((plon - ux) * dx + (plat - uy) * dy) / L2
                if -0.1 <= t <= 1.1:
                    candidates.append((plat, plon))
            if not candidates:
                continue
            # 进一步用更严格的容差：垂距 / |u-v| < 0.5（防止明显跨边的点混入）
            edge_len = L2 ** 0.5
            # 距离 u 也用平方加速
            def dist2(p1, p2):
                return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2
            def perp_dist(p):
                # 点 p 到直线 u-v 的垂距（用 lat/lon 当平面坐标，足够）
                if edge_len == 0:
                    return 0
                num = abs(dx*(uy-p[0]) - dy*(ux-p[1]))
                return num / edge_len
            candidates = [p for p in candidates if perp_dist(p) < 0.5 * edge_len]
            if not candidates:
                continue
            # 最近邻贪心：从 u 出发
            remaining = candidates[:]
            current = (uy, ux)  # lat, lon
            ordered = []
            while remaining:
                # 找最近
                nearest_idx = min(range(len(remaining)), key=lambda i: dist2(remaining[i], current))
                nxt = remaining.pop(nearest_idx)
                ordered.append(nxt)
                current = nxt
            # 去重（极近距离视为同一点）
            dedup = [ordered[0]]
            for p in ordered[1:]:
                if dist2(p, dedup[-1]) > 1e-8:
                    dedup.append(p)
            for seq, pt in enumerate(dedup):
                wp_rows.append({'from_node': u, 'to_node': v, 'sequence': seq,
                                'lat': pt[0], 'lon': pt[1]})
        if wp_rows:
            pd.DataFrame(wp_rows).to_csv(waypoints_path, index=False)
            logger.info("已导出 edge_waypoints: %s (%d 点)", waypoints_path, len(wp_rows))

        logger.info("已导出 CSV: %s, %s", nodes_path, edges_path)
