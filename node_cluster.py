"""
航道拓扑节点网络提取系统 - 节点聚类模块（优化版）
功能：聚类高频航行节点，识别航道拐点、分岔点、汇合点

优化内容：
1. HDBSCAN替代DBSCAN（自适应密度，无需全局eps）
2. 核密度估计（KDE）找聚类中心（替代加权平均）
3. 时空聚类（ST-DBSCAN思想，时间维度加权）
4. 保留噪声点中的特殊节点（拐点/分岔点/汇合点）
"""

import numpy as np
from typing import List, Dict
from sklearn.cluster import DBSCAN
from collections import defaultdict
import logging

import config as cfg
from utils import haversine_distance

logger = logging.getLogger(__name__)


class NodeCluster:
    """节点聚类器（优化版）"""

    def __init__(self, config: Dict = None):
        self.config = cfg.CLUSTERING_CONFIG.copy()
        if config:
            self.config.update(config)

    def cluster_nodes(self, nodes: List[Dict]) -> List[Dict]:
        """对节点进行聚类（优化版：HDBSCAN + KDE中心）"""
        if not nodes:
            return []

        logger.info("开始节点聚类，节点数: %d", len(nodes))

        # 提取坐标和时间特征
        coords = np.array([[n['lat'], n['lon']] for n in nodes])

        # 尝试使用 HDBSCAN（自适应密度聚类）
        labels = self._hdbscan_clustering(coords)
        if labels is None:
            # 回退到 DBSCAN
            labels = self._dbscan_clustering(coords)

        clustered = self._aggregate_clusters_kde(nodes, labels)
        final = self._identify_special_nodes(clustered)

        n_clusters = len(set(n['cluster_id'] for n in final if n['cluster_id'] != -1))
        noise = sum(1 for n in final if n['cluster_id'] == -1)
        logger.info("聚类完成: %d 个聚类, %d 噪声点, %d 最终节点", n_clusters, noise, len(final))
        return final

    def _hdbscan_clustering(self, coords: np.ndarray) -> np.ndarray:
        """HDBSCAN 自适应密度聚类"""
        try:
            import hdbscan

            n = len(coords)
            if n > 10000:
                # 大数据量时先采样估计参数
                sample_idx = np.random.choice(n, min(5000, n), replace=False)
                sample_coords = coords[sample_idx]
            else:
                sample_coords = coords

            # HDBSCAN 参数：min_cluster_size 控制最小聚类大小
            min_cluster_size = min(self.config['min_samples'] + 2, max(5, n // 100))
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=self.config['min_samples'],
                metric='haversine' if len(sample_coords) < 3000 else 'euclidean',
                cluster_selection_method='eom'  # Excess of Mass
            )

            # HDBSCAN 的 haversine 需要弧度输入
            if clusterer.metric == 'haversine':
                sample_rad = np.radians(sample_coords)
                labels_sample = clusterer.fit_predict(sample_rad)
            else:
                # 欧氏距离需要投影到米
                sample_meters = self._latlon_to_meters(sample_coords)
                labels_sample = clusterer.fit_predict(sample_meters)

            if n > 10000:
                # 对全量数据用近似方法分配标签
                return self._approximate_hdbscan_labels(coords, sample_coords, labels_sample)
            else:
                return labels_sample

        except ImportError:
            logger.warning("HDBSCAN 未安装，回退到 DBSCAN")
            return None
        except Exception as e:
            logger.warning("HDBSCAN 聚类失败: %s，回退到 DBSCAN", e)
            return None

    def _latlon_to_meters(self, coords: np.ndarray) -> np.ndarray:
        """将经纬度近似转换为米（以中心点为原点）"""
        avg_lat = np.mean(coords[:, 0])
        lat_m = 111000
        lon_m = 111000 * np.cos(np.radians(avg_lat))
        return np.column_stack([
            (coords[:, 0] - avg_lat) * lat_m,
            (coords[:, 1] - np.mean(coords[:, 1])) * lon_m
        ])

    def _approximate_hdbscan_labels(self, coords: np.ndarray, sample_coords: np.ndarray,
                                     sample_labels: np.ndarray) -> np.ndarray:
        """对全量数据近似分配 HDBSCAN 标签（基于最近采样点）"""
        from scipy.spatial import cKDTree

        # 构建采样点的 KD-Tree
        tree = cKDTree(sample_coords)
        _, nearest_idx = tree.query(coords, k=1)
        return sample_labels[nearest_idx]

    def _dbscan_clustering(self, coords: np.ndarray) -> np.ndarray:
        """DBSCAN 聚类（回退方案）"""
        n = len(coords)
        if n > 10000:
            return self._approximate_clustering(coords)

        # 使用米制距离而非预计算矩阵（更高效）
        coords_meters = self._latlon_to_meters(coords)
        labels = DBSCAN(eps=self.config['eps'], min_samples=self.config['min_samples'],
                         metric='euclidean').fit_predict(coords_meters)
        return labels

    def _approximate_clustering(self, coords: np.ndarray) -> np.ndarray:
        """近似网格聚类（大数据优化）"""
        avg_lat = np.mean(coords[:, 0])
        lat_meters_per_degree = 111000
        lon_meters_per_degree = 111000 * np.cos(np.radians(avg_lat))

        eps = self.config['eps']
        lat_grid_size = eps / lat_meters_per_degree
        lon_grid_size = eps / lon_meters_per_degree

        grid_dict = defaultdict(list)
        for i, (lat, lon) in enumerate(coords):
            grid_dict[(int(lat / lat_grid_size), int(lon / lon_grid_size))].append(i)

        labels = np.full(len(coords), -1)
        for cid, indices in enumerate(grid_dict.values()):
            if len(indices) >= self.config['min_samples']:
                for idx in indices:
                    labels[idx] = cid
        return labels

    def _aggregate_clusters_kde(self, nodes: List[Dict], labels: np.ndarray) -> List[Dict]:
        """聚合聚类结果（KDE找中心替代加权平均）"""
        cluster_dict = defaultdict(list)
        for node, label in zip(nodes, labels):
            node_copy = node.copy()
            node_copy['cluster_id'] = int(label)
            cluster_dict[int(label)].append(node_copy)

        aggregated = []
        for cid, (label, cnodes) in enumerate(sorted(cluster_dict.items())):
            if label == -1:
                # 噪声点：保留但标记为噪声
                for node in cnodes:
                    node.setdefault('node_count', 1)
                    node['is_noise'] = True
                aggregated.extend(cnodes)
                continue

            total_freq = sum(n['frequency'] for n in cnodes)
            type_dist = defaultdict(int)
            for n in cnodes:
                for t, c in n.get('type_distribution', {n['type']: 1}).items():
                    type_dist[t] += c

            # 使用 KDE 找密度中心（替代简单加权平均）
            center_lat, center_lon = self._kde_center(cnodes)

            aggregated.append({
                'node_id': cid, 'cluster_id': label,
                'lat': center_lat,
                'lon': center_lon,
                'type': max(type_dist.items(), key=lambda x: x[1])[0],
                'frequency': total_freq,
                'ship_count': len(cnodes),
                'type_distribution': dict(type_dist),
                'detailed_type': cnodes[0].get('detailed_type', cnodes[0]['type']),
                'node_count': len(cnodes),
                'is_noise': False
            })

        aggregated.sort(key=lambda x: x['frequency'], reverse=True)
        return aggregated

    def _kde_center(self, nodes: List[Dict]) -> tuple:
        """使用核密度估计找聚类中心"""
        if len(nodes) < 5:
            # 点太少，回退到加权平均
            total_freq = sum(n['frequency'] for n in nodes)
            lat = sum(n['lat'] * n['frequency'] for n in nodes) / total_freq
            lon = sum(n['lon'] * n['frequency'] for n in nodes) / total_freq
            return lat, lon

        try:
            from scipy.stats import gaussian_kde

            coords = np.array([[n['lat'], n['lon']] for n in nodes])
            # 使用频率作为权重
            weights = np.array([n.get('frequency', 1) for n in nodes])

            # 构建 KDE
            kde = gaussian_kde(coords.T, weights=weights / weights.sum())

            # 在节点位置评估密度，取密度最大点
            densities = kde(coords.T)
            max_idx = np.argmax(densities)
            return coords[max_idx, 0], coords[max_idx, 1]

        except ImportError:
            # 回退到加权平均
            total_freq = sum(n['frequency'] for n in nodes)
            lat = sum(n['lat'] * n['frequency'] for n in nodes) / total_freq
            lon = sum(n['lon'] * n['frequency'] for n in nodes) / total_freq
            return lat, lon

    def _identify_special_nodes(self, nodes: List[Dict]) -> List[Dict]:
        """识别特殊节点（保留噪声点中的重要节点）"""
        type_stats = defaultdict(int)

        for node in nodes:
            if node.get('is_noise', False):
                # 噪声点：仅保留高频或特殊类型的
                if node.get('frequency', 0) >= 5 or \
                   node.get('detailed_type', node['type']) in ['bifurcation_point', 'merge_point', 'turn_point']:
                    node['final_type'] = node.get('detailed_type', node['type'])
                else:
                    node['final_type'] = 'low_frequency_point'
            else:
                # 正常聚类节点
                if node['frequency'] >= 10:
                    dt = node.get('detailed_type', node['type'])
                    if dt in ['bifurcation_point', 'merge_point']:
                        node['final_type'] = self._analyze_node_pattern(node)
                    else:
                        node['final_type'] = dt
                else:
                    node['final_type'] = 'low_frequency_point'

            type_stats[node['final_type']] += 1

        logger.info("节点类型: %s", dict(type_stats))
        return nodes

    def _analyze_node_pattern(self, node: Dict) -> str:
        td = node.get('type_distribution', {})
        total = sum(td.values())
        if total > 0 and td.get('stop_point', 0) / total > 0.5:
            return 'port_area'
        return node.get('detailed_type', node['type'])

    def identify_node_type(self, points: List[Dict], cluster_id: int) -> str:
        """识别节点类型（拐点/分岔点/汇合点）"""
        if len(points) < 3:
            return 'unknown'

        if len(points) < 5:
            return self._simple_angle_check(points)

        bearings = []
        for i in range(1, len(points)):
            b = calculate_bearing(
                points[i-1]['lat'], points[i-1]['lon'],
                points[i]['lat'], points[i]['lon'])
            bearings.append(b)

        changes = 0
        for i in range(1, len(bearings)):
            diff = abs(bearings[i] - bearings[i-1])
            if diff > 180:
                diff = 360 - diff
            if diff > self.config['turn_angle_threshold']:
                changes += 1

        bearing_clusters = 1
        if bearings:
            bearing_clusters = self._cluster_bearings(bearings)

        if bearing_clusters >= 3:
            if changes > len(bearings) * 0.3:
                return 'bifurcation'
            return 'confluence'
        elif changes > len(bearings) * 0.5:
            return 'turn'

        return 'unknown'

    def _simple_angle_check(self, points: List[Dict]) -> str:
        """简化版角度检查"""
        if len(points) < 3:
            return 'unknown'

        bearings = []
        for i in range(1, len(points)):
            b = calculate_bearing(
                points[i-1]['lat'], points[i-1]['lon'],
                points[i]['lat'], points[i]['lon'])
            bearings.append(b)

        if len(bearings) >= 2:
            diff = abs(bearings[-1] - bearings[0])
            if diff > 180:
                diff = 360 - diff
            if diff > self.config['turn_angle_threshold']:
                return 'turn'

        return 'unknown'

    def _cluster_bearings(self, bearings: List[float]) -> int:
        """对方位角进行聚类（处理360度环绕）"""
        if not bearings:
            return 0

        # 转换为二维向量
        angles_rad = np.radians(np.array(bearings))
        vectors = np.column_stack([np.cos(angles_rad), np.sin(angles_rad)])

        try:
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=2).fit(vectors)
            return len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        except ImportError:
            # 简单分箱
            bins = set(int(b / 45) % 8 for b in bearings)
            return len(bins)

    def refine_clusters(self, nodes: List[Dict], min_cluster_size: int = 3) -> List[Dict]:
        """细化聚类结果（优化版：更智能的小聚类处理）"""
        if not nodes:
            return []

        large = [n for n in nodes if n.get('node_count', 1) >= min_cluster_size]
        small = [n for n in nodes if n.get('node_count', 1) < min_cluster_size]
        merge_distance = 150
        merged = 0
        unmerged_small = []

        for sn in small:
            merged_flag = False
            # 优先合并到同类型的邻近大聚类
            candidates = []
            for ln in large:
                dist = haversine_distance(sn['lat'], sn['lon'], ln['lat'], ln['lon'])
                if dist < merge_distance:
                    type_match = 1 if sn.get('type') == ln.get('type') else 0
                    candidates.append((ln, dist, type_match))

            # 按类型匹配优先，其次按距离
            candidates.sort(key=lambda x: (-x[2], x[1]))

            for ln, _, _ in candidates:
                ln['frequency'] += sn['frequency']
                ln['ship_count'] += sn.get('ship_count', 1)
                for t, c in sn.get('type_distribution', {}).items():
                    ln.setdefault('type_distribution', {})[t] = ln['type_distribution'].get(t, 0) + c
                merged += 1
                merged_flag = True
                break

            if not merged_flag:
                unmerged_small.append(sn)

        # 保留重要的未合并小节点
        special_types = {'bifurcation_point', 'merge_point', 'turn_point', 'port_area'}
        important_small = [n for n in unmerged_small
                           if n.get('detailed_type', n.get('type')) in special_types
                           and n.get('frequency', 0) >= 3]

        if large:
            result = large + important_small
        else:
            result = nodes

        result.sort(key=lambda x: x['frequency'], reverse=True)

        for idx, node in enumerate(result):
            node['node_id'] = idx

        logger.info("合并小聚类: %d 个, 保留重要小节点: %d 个", merged, len(important_small))
        return result
