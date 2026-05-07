"""
航道拓扑节点网络提取系统 - 节点提取模块（性能优化版）
功能：从轨迹中提取关键节点（拐点、分岔点、汇合点）

优化内容：
1. 单次遍历：合并特征点、停泊点、方向索引为一次遍历
2. 向量化计算：曲率和方向计算使用 numpy 向量化
3. 优化索引：使用 KDTree 替代网格索引加速空间查询
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import logging

from config import NODE_EXTRACTION_CONFIG
from utils import haversine_distance, calculate_bearing, calculate_angle_difference

logger = logging.getLogger(__name__)


class NodeExtractor:
    """节点提取器（性能优化版）"""

    def __init__(self, config: Dict = None):
        self.config = config if config else NODE_EXTRACTION_CONFIG

    def extract_nodes(self, df: pd.DataFrame) -> List[Dict]:
        """从轨迹数据中提取关键节点"""
        logger.info("开始提取关键节点...")

        all_points, trajectory_segments = self._extract_all_in_one(df)
        unique_nodes = self._deduplicate_and_count(all_points)
        nodes_with_type = self._classify_node_types_fast(unique_nodes, trajectory_segments)

        logger.info("节点提取完成: %d 个", len(nodes_with_type))
        return nodes_with_type

    def _extract_all_in_one(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """
        单次遍历：同时提取特征点、停泊点、轨迹段
        """
        feature_points = []
        stop_points = []
        trajectory_segments = []
        
        grouped = df.groupby('船舶名称')
        total_ships = len(grouped)
        
        speed_threshold = self.config['speed_change_threshold']
        min_duration = self.config['stop_point_min_duration']
        dp_tolerance = self.config['douglas_peucker_tolerance']
        angle_threshold = self.config['direction_change_threshold']

        for idx, (ship_name, group) in enumerate(grouped):
            if (idx + 1) % 100 == 0:
                logger.info("提取进度: %d/%d", idx + 1, total_ships)

            group = group.sort_values('时间').reset_index(drop=True)
            n = len(group)
            if n < 3:
                continue

            lats = group['纬度'].values
            lons = group['经度'].values
            speeds = group['航速'].values
            courses = group['航向'].values
            times = group['时间'].values

            points = list(zip(lats, lons))

            if len(points) > 200:
                simplified_indices = self._sliding_window_dp(points, 200, dp_tolerance)
            else:
                simplified_indices = self._douglas_peucker_indices(points, dp_tolerance)

            simplified_set = set(simplified_indices)

            bearings = np.zeros(n - 1)
            for i in range(n - 1):
                bearings[i] = calculate_bearing(lats[i], lons[i], lats[i+1], lons[i+1])

            for i in simplified_set:
                is_turn = False
                if 2 <= i < n - 2:
                    curvature = self._calculate_curvature_fast(
                        lats[i-2:i+3], lons[i-2:i+3]
                    )
                    if i < n - 1:
                        angle_diff = calculate_angle_difference(bearings[i-1] if i > 0 else 0, bearings[i])
                    else:
                        angle_diff = 0
                    is_turn = curvature > 0.001 or angle_diff > angle_threshold

                feature_points.append({
                    'lat': lats[i], 'lon': lons[i],
                    'ship_name': ship_name,
                    'type': 'turn_point' if is_turn else 'waypoint',
                    'speed': speeds[i], 'course': courses[i], 'time': times[i]
                })

            low_speed = speeds < speed_threshold
            i = 0
            while i < n:
                if low_speed[i]:
                    start = i
                    while i < n and low_speed[i]:
                        i += 1
                    if i > start:
                        duration = (pd.Timestamp(times[i-1]) - pd.Timestamp(times[start])).total_seconds()
                        if duration > min_duration:
                            seg_lats = lats[start:i]
                            seg_lons = lons[start:i]
                            center_lat = seg_lats.mean()
                            center_lon = seg_lons.mean()
                            stop_points.append({
                                'lat': center_lat, 'lon': center_lon,
                                'ship_name': ship_name, 'type': 'stop_point',
                                'duration': duration,
                                'speed': 0, 'course': courses[start], 'time': times[start]
                            })
                else:
                    i += 1

            for i in range(n - 1):
                trajectory_segments.append({
                    'start_lat': lats[i], 'start_lon': lons[i],
                    'end_lat': lats[i+1], 'end_lon': lons[i+1],
                    'bearing': bearings[i],
                    'distance': haversine_distance(lats[i], lons[i], lats[i+1], lons[i+1]),
                    'ship_name': ship_name
                })

        logger.info("特征点: %d 个, 停泊点: %d 个, 轨迹段: %d 个", 
                    len(feature_points), len(stop_points), len(trajectory_segments))

        stop_points = self._merge_stop_points(stop_points)
        logger.info("合并后停泊点: %d 个", len(stop_points))

        all_points = feature_points + stop_points
        return all_points, trajectory_segments

    def _merge_stop_points(self, stop_points: List[Dict]) -> List[Dict]:
        """使用网格合并停泊点（简单高效）"""
        if len(stop_points) < 2:
            return stop_points

        grid_size = 50 / 111000
        grid_dict = defaultdict(list)

        for p in stop_points:
            key = (int(p['lat'] / grid_size), int(p['lon'] / grid_size))
            grid_dict[key].append(p)

        result = []
        for points in grid_dict.values():
            if len(points) == 1:
                result.append(points[0])
            else:
                total_duration = sum(p.get('duration', 0) for p in points)
                result.append({
                    'lat': np.mean([p['lat'] for p in points]),
                    'lon': np.mean([p['lon'] for p in points]),
                    'ship_name': points[0]['ship_name'],
                    'type': 'stop_point',
                    'duration': total_duration,
                    'speed': 0,
                    'course': points[0]['course'],
                    'time': points[0]['time'],
                    'merged_count': len(points)
                })

        return result

    def _douglas_peucker_indices(self, points: List[tuple], tolerance: float) -> List[int]:
        """Douglas-Peucker 算法提取关键点索引"""
        if len(points) <= 2:
            return list(range(len(points)))

        indices = [0, len(points) - 1]
        self._dp_recursive(points, 0, len(points) - 1, tolerance, indices)
        return sorted(indices)

    def _dp_recursive(self, points: List[tuple], start: int, end: int, 
                      tolerance: float, indices: List[int]):
        """Douglas-Peucker 递归实现"""
        if end - start <= 1:
            return

        max_dist = 0
        max_idx = start

        p1 = points[start]
        p2 = points[end]

        for i in range(start + 1, end):
            dist = self._point_line_distance(points[i], p1, p2)
            if dist > max_dist:
                max_dist = dist
                max_idx = i

        if max_dist > tolerance:
            indices.append(max_idx)
            self._dp_recursive(points, start, max_idx, tolerance, indices)
            self._dp_recursive(points, max_idx, end, tolerance, indices)

    def _point_line_distance(self, point: tuple, p1: tuple, p2: tuple) -> float:
        """计算点到线段的距离（米）"""
        lat, lon = point
        lat1, lon1 = p1
        lat2, lon2 = p2

        lat1_m = lat1 * 111000
        lat2_m = lat2 * 111000
        lat_m = lat * 111000
        lon1_m = lon1 * 111000 * np.cos(np.radians(lat1))
        lon2_m = lon2 * 111000 * np.cos(np.radians(lat2))
        lon_m = lon * 111000 * np.cos(np.radians(lat))

        dx = lon2_m - lon1_m
        dy = lat2_m - lat1_m

        if dx == 0 and dy == 0:
            return np.sqrt((lon_m - lon1_m)**2 + (lat_m - lat1_m)**2)

        t = max(0, min(1, ((lon_m - lon1_m) * dx + (lat_m - lat1_m) * dy) / (dx**2 + dy**2)))
        proj_lon = lon1_m + t * dx
        proj_lat = lat1_m + t * dy

        return np.sqrt((lon_m - proj_lon)**2 + (lat_m - proj_lat)**2)

    def _sliding_window_dp(self, points: List[tuple], window_size: int, 
                           tolerance: float) -> List[int]:
        """滑动窗口 Douglas-Peucker"""
        all_indices = []
        step = window_size // 2

        for start in range(0, len(points), step):
            end = min(start + window_size, len(points))
            window_points = points[start:end]
            window_indices = self._douglas_peucker_indices(window_points, tolerance)
            global_indices = [start + idx for idx in window_indices]
            all_indices.extend(global_indices)
            if end == len(points):
                break

        return sorted(set(all_indices))

    def _calculate_curvature_fast(self, lats: np.ndarray, lons: np.ndarray) -> float:
        """向量化曲率计算"""
        if len(lats) < 5:
            return 0.0

        x = lons * 111000 * np.cos(np.radians(lats))
        y = lats * 111000

        dx1 = x[2] - x[1]
        dy1 = y[2] - y[1]
        dx2 = x[3] - x[2]
        dy2 = y[3] - y[2]

        cross = abs(dx1 * dy2 - dy1 * dx2)
        dist1 = np.sqrt(dx1**2 + dy1**2)
        dist2 = np.sqrt(dx2**2 + dy2**2)
        dist3 = np.sqrt((x[3] - x[1])**2 + (y[3] - y[1])**2)

        if dist1 < 1e-6 or dist2 < 1e-6 or dist3 < 1e-6:
            return 0.0

        area = 0.5 * cross
        curvature = 4 * area / (dist1 * dist2 * dist3)
        return curvature

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

            courses = [p['course'] for p in points if p.get('course') is not None]
            if courses:
                angles_rad = np.radians(courses)
                sin_mean = np.mean(np.sin(angles_rad))
                cos_mean = np.mean(np.cos(angles_rad))
                heading = np.degrees(np.arctan2(sin_mean, cos_mean)) % 360
                heading_concentration = np.sqrt(sin_mean**2 + cos_mean**2)
            else:
                heading = 0.0
                heading_concentration = 0.0

            unique_nodes.append({
                'node_id': node_id,
                'lat': np.mean([p['lat'] for p in points]),
                'lon': np.mean([p['lon'] for p in points]),
                'type': max(types.items(), key=lambda x: x[1])[0],
                'frequency': len(points),
                'ship_count': len(ships),
                'type_distribution': dict(types),
                'heading': heading,
                'heading_concentration': heading_concentration
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
                    if 'ship_count' in n:
                        merged_ships.add(n['ship_count'])

                sin_sum = sum(n['frequency'] * np.sin(np.radians(n.get('heading', 0))) for n in group)
                cos_sum = sum(n['frequency'] * np.cos(np.radians(n.get('heading', 0))) for n in group)
                merged_heading = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
                merged_R = np.sqrt(sin_sum**2 + cos_sum**2) / total_freq if total_freq > 0 else 0

                result.append({
                    'node_id': node['node_id'],
                    'lat': np.mean([n['lat'] for n in group]),
                    'lon': np.mean([n['lon'] for n in group]),
                    'type': max(merged_types.items(), key=lambda x: x[1])[0],
                    'frequency': total_freq,
                    'ship_count': len(merged_ships) if merged_ships else 1,
                    'type_distribution': dict(merged_types),
                    'heading': merged_heading,
                    'heading_concentration': merged_R
                })
            else:
                result.append(node)

        return result

    def _classify_node_types_fast(self, nodes: List[Dict], 
                                   trajectory_segments: List[Dict]) -> List[Dict]:
        """快速节点类型分类（使用 KDTree 加速空间查询）"""
        try:
            from scipy.spatial import cKDTree
            HAS_KDTREE = True
        except ImportError:
            HAS_KDTREE = False

        if not trajectory_segments:
            return nodes

        seg_lats = np.array([(s['start_lat'] + s['end_lat']) / 2 for s in trajectory_segments])
        seg_lons = np.array([(s['start_lon'] + s['end_lon']) / 2 for s in trajectory_segments])
        seg_bearings = np.array([s['bearing'] for s in trajectory_segments])

        seg_coords = np.column_stack([seg_lats, seg_lons])

        if HAS_KDTREE:
            tree = cKDTree(seg_coords)
            radius_deg = 100 / 111000

        radius = 100

        for node in nodes:
            node_lat = node['lat']
            node_lon = node['lon']

            if HAS_KDTREE:
                nearby_indices = tree.query_ball_point([node_lat, node_lon], radius_deg)
            else:
                distances = np.sqrt((seg_lats - node_lat)**2 + (seg_lons - node_lon)**2)
                nearby_indices = np.where(distances < radius_deg)[0]

            if len(nearby_indices) >= 3:
                bearings = seg_bearings[nearby_indices]
                node['detailed_type'] = self._analyze_directions_fast(bearings)
            else:
                node['detailed_type'] = node['type']

        type_counts = defaultdict(int)
        for n in nodes:
            type_counts[n.get('detailed_type', n['type'])] += 1
        logger.info("节点类型分布: %s", dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True)))
        return nodes

    def _analyze_directions_fast(self, bearings: np.ndarray) -> str:
        """快速方向分析"""
        if len(bearings) < 3:
            return 'waypoint'

        angles_rad = np.radians(bearings)
        vectors = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])

        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=3).fit(vectors)
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            if n_clusters >= 3:
                cluster_centers = []
                for label in set(labels) - {-1}:
                    mask = labels == label
                    center_vec = vectors[mask].mean(axis=0)
                    center_angle = np.degrees(np.arctan2(center_vec[1], center_vec[0])) % 360
                    cluster_centers.append(center_angle)

                has_opposite = False
                for i in range(len(cluster_centers)):
                    for j in range(i + 1, len(cluster_centers)):
                        if calculate_angle_difference(cluster_centers[i], cluster_centers[j]) > 150:
                            has_opposite = True
                            break
                    if has_opposite:
                        break

                return 'merge_point' if has_opposite else 'bifurcation_point'

        except ImportError:
            pass

        mean_vec = vectors.mean(axis=0)
        r = np.sqrt(mean_vec[0]**2 + mean_vec[1]**2)
        circ_std = np.degrees(np.sqrt(-2 * np.log(max(r, 1e-10)))) if r > 0 else 180

        return 'turn_point' if circ_std > 45 else 'waypoint'
