"""
航道拓扑节点网络提取系统 - 节点提取模块（优化版）
功能：从轨迹中提取关键节点（拐点、分岔点、汇合点）

优化内容：
1. 滑动窗口 Douglas-Peucker（长轨迹分段压缩）
2. 曲率计算（转向点判定更精确）
3. DBSCAN空间聚类提取停泊点（替代简单低速持续）
4. 轨迹段方向分析优化（8邻域分箱→均值漂移方向聚类）
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from collections import defaultdict
import logging

from config import NODE_EXTRACTION_CONFIG
from utils import haversine_distance, calculate_bearing, calculate_angle_difference, douglas_peucker_indices

logger = logging.getLogger(__name__)


class NodeExtractor:
    """节点提取器（优化版）"""

    def __init__(self, config: Dict = None):
        self.config = config if config else NODE_EXTRACTION_CONFIG

    def extract_nodes(self, df: pd.DataFrame) -> List[Dict]:
        """从轨迹数据中提取关键节点"""
        logger.info("开始提取关键节点...")

        feature_points = self._extract_feature_points(df)
        stop_points = self._extract_stop_points_dbscan(df)
        all_points = feature_points + stop_points

        unique_nodes = self._deduplicate_and_count(all_points)
        nodes_with_type = self._classify_node_types(unique_nodes, df)

        logger.info("节点提取完成: %d 个", len(nodes_with_type))
        return nodes_with_type

    def _extract_feature_points(self, df: pd.DataFrame) -> List[Dict]:
        """提取轨迹特征点（滑动窗口DP + 曲率计算）"""
        feature_points = []
        grouped = df.groupby('船舶名称')
        window_size = 200  # 滑动窗口大小

        for idx, (ship_name, group) in enumerate(grouped):
            if (idx + 1) % 50 == 0:
                logger.info("特征点提取进度: %d/%d", idx + 1, len(grouped))

            group = group.sort_values('时间').reset_index(drop=True)
            if len(group) < 3:
                continue

            points = list(zip(group['纬度'].values, group['经度'].values))

            # 滑动窗口 Douglas-Peucker：长轨迹分段压缩
            if len(points) > window_size:
                simplified_indices = self._sliding_window_dp(points, window_size)
            else:
                simplified_indices = douglas_peucker_indices(
                    points, self.config['douglas_peucker_tolerance'])

            simplified_set = set(simplified_indices)

            for i in simplified_set:
                row = group.loc[i]
                # 使用曲率计算判断转向点（更精确）
                is_turn = self._is_turn_point_curvature(group, i)
                feature_points.append({
                    'lat': row['纬度'], 'lon': row['经度'],
                    'ship_name': ship_name,
                    'type': 'turn_point' if is_turn else 'waypoint',
                    'speed': row['航速'], 'course': row['航向'], 'time': row['时间']
                })

        logger.info("特征点: %d 个", len(feature_points))
        return feature_points

    def _sliding_window_dp(self, points: List[tuple], window_size: int) -> List[int]:
        """滑动窗口 Douglas-Peucker：将长轨迹切分为窗口分别压缩"""
        all_indices = []
        step = window_size // 2  # 50% 重叠避免边界丢失

        for start in range(0, len(points), step):
            end = min(start + window_size, len(points))
            window_points = points[start:end]
            window_indices = douglas_peucker_indices(
                window_points, self.config['douglas_peucker_tolerance'])
            # 映射回全局索引
            global_indices = [start + idx for idx in window_indices]
            all_indices.extend(global_indices)

            if end == len(points):
                break

        return sorted(set(all_indices))

    def _is_turn_point_curvature(self, group: pd.DataFrame, idx: int) -> bool:
        """基于曲率计算判断转向点（比简单航向变化更精确）"""
        if idx < 2 or idx >= len(group) - 2:
            return False

        # 取前后各2个点计算曲率
        lats = group.loc[idx-2:idx+2, '纬度'].values
        lons = group.loc[idx-2:idx+2, '经度'].values

        # 计算三点曲率（使用前后各1点）
        curvature = self._calculate_curvature(
            lats[1], lons[1],  # prev
            lats[2], lons[2],  # curr
            lats[3], lons[3]   # next
        )

        # 同时检查航向变化作为辅助
        b1 = calculate_bearing(lats[1], lons[1], lats[2], lons[2])
        b2 = calculate_bearing(lats[2], lons[2], lats[3], lons[3])
        angle_diff = calculate_angle_difference(b1, b2)

        # 曲率或航向变化任一超过阈值即判定为转向点
        curvature_threshold = 0.001  # 曲率阈值（1/米）
        return curvature > curvature_threshold or angle_diff > self.config['direction_change_threshold']

    def _calculate_curvature(self, lat1, lon1, lat2, lon2, lat3, lon3) -> float:
        """计算三点曲率（1/R）"""
        # 将经纬度转换为米（近似）
        x1, y1 = lon1 * 111000 * np.cos(np.radians(lat1)), lat1 * 111000
        x2, y2 = lon2 * 111000 * np.cos(np.radians(lat2)), lat2 * 111000
        x3, y3 = lon3 * 111000 * np.cos(np.radians(lat3)), lat3 * 111000

        # 使用叉积计算曲率
        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x3 - x2, y3 - y2

        cross = abs(dx1 * dy2 - dy1 * dx2)
        dist1 = np.sqrt(dx1**2 + dy1**2)
        dist2 = np.sqrt(dx2**2 + dy2**2)
        dist3 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)

        if dist1 < 1e-6 or dist2 < 1e-6 or dist3 < 1e-6:
            return 0.0

        # 曲率 = 4 * 三角形面积 / (三边乘积)
        area = 0.5 * cross
        curvature = 4 * area / (dist1 * dist2 * dist3)
        return curvature

    def _extract_stop_points_dbscan(self, df: pd.DataFrame) -> List[Dict]:
        """使用 DBSCAN 空间聚类提取停泊点（替代简单低速持续）"""
        stop_points = []
        speed_threshold = self.config['speed_change_threshold']
        min_duration = self.config['stop_point_min_duration']

        for ship_name, group in df.groupby('船舶名称'):
            group = group.sort_values('时间').reset_index(drop=True)
            low_speed = group['航速'] < speed_threshold

            # 先找出所有低速段
            segments = []
            i = 0
            while i < len(group):
                if low_speed.iloc[i]:
                    start = i
                    while i < len(group) and low_speed.iloc[i]:
                        i += 1
                    duration = (group.loc[i-1, '时间'] - group.loc[start, '时间']).total_seconds()
                    if duration > min_duration:
                        segment = group.loc[start:i-1]
                        segments.append(segment)
                else:
                    i += 1

            # 对每个低速段用 DBSCAN 聚类找停泊中心
            for segment in segments:
                if len(segment) < 3:
                    # 点太少，直接取平均
                    stop_points.append({
                        'lat': segment['纬度'].mean(),
                        'lon': segment['经度'].mean(),
                        'ship_name': ship_name, 'type': 'stop_point',
                        'duration': (segment['时间'].iloc[-1] - segment['时间'].iloc[0]).total_seconds(),
                        'speed': 0,
                        'course': segment['航向'].iloc[0],
                        'time': segment['时间'].iloc[0]
                    })
                    continue

                # DBSCAN 空间聚类
                coords = segment[['纬度', '经度']].values
                try:
                    from sklearn.cluster import DBSCAN
                    # eps=50米（约0.00045度），min_samples=3
                    eps_deg = 50 / 111000
                    clustering = DBSCAN(eps=eps_deg, min_samples=3).fit(coords)
                    labels = clustering.labels_

                    # 提取每个聚类的中心作为停泊点
                    unique_labels = set(labels) - {-1}
                    if not unique_labels:
                        # 无聚类，取平均
                        unique_labels = {0}
                        labels = [0] * len(coords)

                    for label in unique_labels:
                        mask = labels == label
                        cluster_points = segment[mask]
                        stop_points.append({
                            'lat': cluster_points['纬度'].mean(),
                            'lon': cluster_points['经度'].mean(),
                            'ship_name': ship_name, 'type': 'stop_point',
                            'duration': (cluster_points['时间'].iloc[-1] - cluster_points['时间'].iloc[0]).total_seconds(),
                            'speed': 0,
                            'course': cluster_points['航向'].iloc[0],
                            'time': cluster_points['时间'].iloc[0]
                        })

                except ImportError:
                    # 回退到简单平均
                    stop_points.append({
                        'lat': segment['纬度'].mean(),
                        'lon': segment['经度'].mean(),
                        'ship_name': ship_name, 'type': 'stop_point',
                        'duration': (segment['时间'].iloc[-1] - segment['时间'].iloc[0]).total_seconds(),
                        'speed': 0,
                        'course': segment['航向'].iloc[0],
                        'time': segment['时间'].iloc[0]
                    })

        logger.info("停泊点(DBSCAN聚类): %d 个", len(stop_points))
        return stop_points

    def _deduplicate_and_count(self, feature_points: List[Dict]) -> List[Dict]:
        """去重并统计节点频率"""
        grid_size = 0.0005
        node_grid = defaultdict(list)

        for p in feature_points:
            node_grid[(int(p['lat'] / grid_size), int(p['lon'] / grid_size))].append(p)

        unique_nodes = []
        for node_id, (_, points) in enumerate(sorted(node_grid.items())):
            types = defaultdict(int)
            ships = set()
            for p in points:
                types[p['type']] += 1
                ships.add(p['ship_name'])

            unique_nodes.append({
                'node_id': node_id,
                'lat': np.mean([p['lat'] for p in points]),
                'lon': np.mean([p['lon'] for p in points]),
                'type': max(types.items(), key=lambda x: x[1])[0],
                'frequency': len(points),
                'ship_count': len(ships),
                'type_distribution': dict(types)
            })

        unique_nodes = self._merge_adjacent_grid_nodes(unique_nodes, grid_size)

        unique_nodes.sort(key=lambda x: x['frequency'], reverse=True)
        for idx, node in enumerate(unique_nodes):
            node['node_id'] = idx

        logger.info("去重后: %d 个节点", len(unique_nodes))
        return unique_nodes

    def _merge_adjacent_grid_nodes(self, nodes: List[Dict], grid_size: float) -> List[Dict]:
        """合并相邻网格中距离过近的节点"""
        grid_dict = defaultdict(list)
        for node in nodes:
            key = (int(node['lat'] / grid_size), int(node['lon'] / grid_size))
            grid_dict[key].append(node)

        merged_set = set()
        merge_distance = 30
        result = []

        for node in nodes:
            if id(node) in merged_set:
                continue

            gl = int(node['lat'] / grid_size)
            gn = int(node['lon'] / grid_size)
            group = [node]

            for dl in [-1, 0, 1]:
                for dn in [-1, 0, 1]:
                    if dl == 0 and dn == 0:
                        continue
                    key = (gl + dl, gn + dn)
                    if key in grid_dict:
                        for neighbor in grid_dict[key]:
                            if id(neighbor) not in merged_set:
                                dist = haversine_distance(
                                    node['lat'], node['lon'],
                                    neighbor['lat'], neighbor['lon'])
                                if dist < merge_distance:
                                    group.append(neighbor)
                                    merged_set.add(id(neighbor))

            if len(group) > 1:
                total_freq = sum(n['frequency'] for n in group)
                merged_types = defaultdict(int)
                merged_ships = set()
                for n in group:
                    for t, c in n.get('type_distribution', {}).items():
                        merged_types[t] += c
                    merged_ships.add(n.get('ship_count', 1))

                result.append({
                    'node_id': node['node_id'],
                    'lat': np.mean([n['lat'] for n in group]),
                    'lon': np.mean([n['lon'] for n in group]),
                    'type': max(merged_types.items(), key=lambda x: x[1])[0],
                    'frequency': total_freq,
                    'ship_count': len(merged_ships),
                    'type_distribution': dict(merged_types)
                })
            else:
                result.append(node)

        return result

    def _classify_node_types(self, nodes: List[Dict], df: pd.DataFrame) -> List[Dict]:
        """识别节点类型（优化版：均值漂移方向聚类替代8分箱）"""
        trajectory_data = self._build_trajectory_directions(df)

        for node in nodes:
            directions = self._get_node_directions(node, trajectory_data)
            if len(directions) >= 3:
                node['detailed_type'] = self._analyze_directions_mean_shift(directions)
            else:
                node['detailed_type'] = node['type']

        type_counts = defaultdict(int)
        for n in nodes:
            type_counts[n.get('detailed_type', n['type'])] += 1
        logger.info("节点类型分布: %s", dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True)))
        return nodes

    def _build_trajectory_directions(self, df: pd.DataFrame) -> Dict:
        """构建轨迹段方向索引"""
        grid_size = 0.001
        grid_index = defaultdict(set)
        all_segments = []

        for idx, (ship_name, group) in enumerate(df.groupby('船舶名称')):
            if (idx + 1) % 50 == 0:
                logger.info("索引进度: %d/%d", idx + 1, len(df.groupby('船舶名称')))
            group = group.sort_values('时间').reset_index(drop=True)

            for i in range(len(group) - 1):
                lat1, lon1 = group.loc[i, '纬度'], group.loc[i, '经度']
                lat2, lon2 = group.loc[i+1, '纬度'], group.loc[i+1, '经度']

                seg_idx = len(all_segments)
                all_segments.append({
                    'start_lat': lat1, 'start_lon': lon1,
                    'end_lat': lat2, 'end_lon': lon2,
                    'bearing': calculate_bearing(lat1, lon1, lat2, lon2),
                    'distance': haversine_distance(lat1, lon1, lat2, lon2),
                    'ship_name': ship_name
                })

                for lat, lon in [(lat1, lon1), (lat2, lon2)]:
                    grid_index[(int(lat / grid_size), int(lon / grid_size))].add(seg_idx)
                self._fill_intermediate_grids(grid_index, seg_idx, lat1, lon1, lat2, lon2, grid_size)

        return {'grid_index': grid_index, 'segments': all_segments, 'grid_size': grid_size}

    def _fill_intermediate_grids(self, grid_index, seg_idx, lat1, lon1, lat2, lon2, grid_size):
        """填充轨迹段经过的中间网格"""
        gl1, gn1 = int(lat1 / grid_size), int(lon1 / grid_size)
        gl2, gn2 = int(lat2 / grid_size), int(lon2 / grid_size)
        if abs(gl2 - gl1) <= 1 and abs(gn2 - gn1) <= 1:
            return
        for gl in range(min(gl1, gl2), max(gl1, gl2) + 1):
            for gn in range(min(gn1, gn2), max(gn1, gn2) + 1):
                grid_index[(gl, gn)].add(seg_idx)

    def _get_node_directions(self, node: Dict, trajectory_data: Dict) -> List[Dict]:
        """获取经过节点的轨迹方向"""
        directions = []
        grid_index = trajectory_data['grid_index']
        segments = trajectory_data['segments']
        gs = trajectory_data['grid_size']
        radius = 100

        gl, gn = int(node['lat'] / gs), int(node['lon'] / gs)
        nearby = set()
        for dl in [-1, 0, 1]:
            for dn in [-1, 0, 1]:
                key = (gl + dl, gn + dn)
                if key in grid_index:
                    nearby.update(grid_index[key])

        for si in nearby:
            seg = segments[si]
            d_start = haversine_distance(node['lat'], node['lon'], seg['start_lat'], seg['start_lon'])
            d_end = haversine_distance(node['lat'], node['lon'], seg['end_lat'], seg['end_lon'])
            if d_start < radius or d_end < radius:
                directions.append({'bearing': seg['bearing'], 'ship_name': seg['ship_name']})
        return directions

    def _analyze_directions_mean_shift(self, directions: List[Dict]) -> str:
        """使用均值漂移方向聚类分析节点类型（替代8分箱）"""
        if not directions:
            return 'unknown'

        bearings = np.array([d['bearing'] for d in directions])

        # 将角度转换为二维单位向量（处理360度环绕）
        angles_rad = np.radians(bearings)
        vectors = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])

        # 使用 DBSCAN 对方向向量聚类
        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=3).fit(vectors)
            labels = clustering.labels_
        except ImportError:
            # 回退到简单分箱
            return self._analyze_directions_simple(bearings)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        if n_clusters >= 3:
            # 检查是否有相反方向（汇合点特征）
            cluster_centers = []
            for label in set(labels) - {-1}:
                mask = labels == label
                center_vec = vectors[mask].mean(axis=0)
                center_angle = np.degrees(np.arctan2(center_vec[1], center_vec[0])) % 360
                cluster_centers.append(center_angle)

            # 检查是否有相反方向对
            has_opposite = False
            for i in range(len(cluster_centers)):
                for j in range(i + 1, len(cluster_centers)):
                    if calculate_angle_difference(cluster_centers[i], cluster_centers[j]) > 150:
                        has_opposite = True
                        break

            return 'merge_point' if has_opposite else 'bifurcation_point'

        # 计算方向标准差（圆周标准差）
        mean_vec = vectors.mean(axis=0)
        r = np.sqrt(mean_vec[0]**2 + mean_vec[1]**2)
        circ_std = np.degrees(np.sqrt(-2 * np.log(r))) if r > 0 else 180

        return 'turn_point' if circ_std > 45 else 'waypoint'

    def _analyze_directions_simple(self, bearings: List[float]) -> str:
        """简化版方向分析（8分箱，作为回退）"""
        bins = defaultdict(int)
        for b in bearings:
            bins[int(b / 45) % 8] += 1

        main = [k for k, v in bins.items() if v >= len(bearings) * 0.15]
        if len(main) >= 3:
            has_opposite = any((d + 4) % 8 in main for d in main)
            return 'merge_point' if has_opposite else 'bifurcation_point'

        return 'turn_point' if np.std(bearings) > 60 else 'waypoint'
