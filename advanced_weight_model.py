"""
航道拓扑节点网络提取系统 - 多算法对比的动态路段耗时权重建模模块

支持的算法：
1. XGBoost - 梯度提升树
2. LightGBM - 轻量级梯度提升机
3. Random Forest - 随机森林
4. MLP - 多层感知机
5. GNN - 图神经网络 (PyTorch Geometric)

功能：
- 多算法训练与对比
- 自动选择最优模型
- 模型评估报告
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass
import time
import logging
import warnings
import os
import pickle
import json

# 抑制 LightGBM feature names 警告（已通过 numpy array 转换处理）
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from config import TOPOLOGY_CONFIG
from utils import haversine_distance, calculate_bearing, calculate_angle_difference

# 机器学习库
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# 可选依赖
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
    torch.set_num_threads(os.cpu_count() or 4)
except ImportError:
    HAS_TORCH = False

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

# 模块级 GNN 模型定义（确保可 pickle 序列化）
if HAS_TORCH and HAS_PYG:
    class EdgeGNN(nn.Module):
        def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=2, dropout=0.2):
            super().__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.convs.append(GATConv(node_dim, hidden_dim, heads=2, concat=False))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=2, concat=False))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
            
            self.dropout = nn.Dropout(dropout)
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
        
        def forward(self, x, edge_index, edge_attr, num_target_edges):
            for conv, bn in zip(self.convs, self.bns):
                x = F.relu(bn(conv(x, edge_index)))
                x = self.dropout(x)
            
            # 只对前 num_target_edges 条边做预测（原始方向）
            row = edge_index[0][:num_target_edges]
            col = edge_index[1][:num_target_edges]
            edge_input = torch.cat([x[row], x[col], edge_attr], dim=1)
            return self.edge_mlp(edge_input).squeeze()

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """模型训练结果"""
    model_name: str
    train_time: float
    mae: float
    rmse: float
    r2: float
    mape: float  # 平均绝对百分比误差
    model: object
    predictions: np.ndarray = None
    r2_log: float = None  # log空间R²（仅log变换模型有）
    use_log_transform: bool = True  # 模型是否使用log1p变换


class AdvancedWeightModel:
    """
    多算法对比的动态路段耗时权重模型
    """
    
    def __init__(self, config: Dict = None):
        """初始化"""
        self.config = config if config else TOPOLOGY_CONFIG
        self.edge_features = {}
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
        # 时段划分
        self.time_periods = {
            'night': (0, 6),
            'morning': (6, 10),
            'midday': (10, 14),
            'afternoon': (14, 18),
            'evening': (18, 22),
            'late_night': (22, 24)
        }
        self.peak_hours = set(range(6, 10)) | set(range(17, 20))
        
        # 特征名称（边×时段聚合级别，与预测时输入对齐）
        # 去除 avg_actual_speed（目标泄漏：distance/time_diff ≈ 确定性关系）
        # 去除 avg_speed_diff（由泄漏特征派生）
        # 新增 theoretical_time（物理先验：distance/reported_speed，预测时可用默认速度填充）
        self.feature_names = [
            'avg_reported_speed', 'std_reported_speed', 'speed_cv',
            'bearing', 'bearing_sin', 'bearing_cos', 'avg_course_change',
            'std_course_change', 'course_change_x_narrow',
            'is_peak_hour', 'is_weekend', 'waterway_type',
            'hour', 'hour_sin', 'hour_cos',
            'node_degree_from', 'node_degree_to', 'edge_betweenness',
            'sample_count', 'log_sample_count',
            'narrow_x_peak', 'course_change_x_peak'
        ]
        
        # 检查依赖
        missing = [k for k, v in {
            'XGBoost': HAS_XGBOOST, 'LightGBM': HAS_LIGHTGBM,
            'PyTorch': HAS_TORCH, 'PyG': HAS_PYG, 'Optuna': HAS_OPTUNA
        }.items() if not v]
        if missing:
            logger.info("未安装: %s", ', '.join(missing))
    
    def build_weights_with_comparison(self, graph, trajectories_df: pd.DataFrame,
                                       models_to_compare: List[str] = None,
                                       use_grid_search: bool = True) -> Dict:
        """
        使用多种算法构建权重并对比效果
        
        Args:
            graph: 拓扑网络图
            trajectories_df: 轨迹数据
            models_to_compare: 要对比的模型列表，如 ['xgboost', 'lightgbm', 'rf', 'mlp', 'gnn']
            use_grid_search: 是否使用网格搜索调参（默认启用）
        
        Returns:
            边权重字典
        """
        logger.info("多算法对比 - 动态路段耗时权重建模")
        
        # 默认对比所有可用模型（MLP 效果差且耗时长，默认不启用）
        if models_to_compare is None:
            models_to_compare = []
            if HAS_XGBOOST:
                models_to_compare.append('xgboost')
            if HAS_LIGHTGBM:
                models_to_compare.append('lightgbm')
                models_to_compare.append('lightgbm_tweedie')
            models_to_compare.append('random_forest')
            if HAS_PYG:
                models_to_compare.append('gnn')
        
        logger.info("对比模型: %s, 超参搜索: %s (%s)", models_to_compare, 
                     '启用' if use_grid_search else '禁用',
                     '贝叶斯(Optuna)' if (use_grid_search and HAS_OPTUNA) else '随机搜索' if use_grid_search else '默认参数')
        
        # 1. 提取特征
        logger.info("提取轨迹段特征...")
        segment_features = self._extract_segment_features(trajectories_df)
        
        # 2. 计算水域类型
        logger.info("计算水域类型...")
        self._compute_waterway_types(segment_features, graph)
        
        # 3. 计算网络特征
        logger.info("计算网络拓扑特征...")
        self._compute_network_features(graph)
        
        # 4. 映射到边
        logger.info("映射轨迹段到网络边...")
        edge_segments = self._map_segments_to_edges(graph, segment_features)
        
        # 5. 构建训练数据（边×时段聚合，与预测对齐）
        logger.info("构建训练数据集（边×时段聚合，目标=time_ratio）...")
        X, y_ratio, y_time, theoretical_times, edge_period_info = self._build_training_data(edge_segments, graph)
        self.edge_period_info = edge_period_info
        self._edge_theoretical_times_full = theoretical_times

        print(f"  time_ratio: min={y_ratio.min():.3f}, max={y_ratio.max():.3f}, "
              f"mean={y_ratio.mean():.3f}, std={y_ratio.std():.3f}")

        logger.info("训练并对比模型...")
        results = self._train_and_compare_models(X, y_ratio, y_time, theoretical_times,
                                                  models_to_compare, graph, edge_segments, use_grid_search)

        self._select_best_model(results)

        logger.info("预测边权重...")
        self._predict_all_weights(edge_segments, graph)
        
        # 10. 更新图
        self._update_graph_edges(graph)
        
        logger.info("建模完成，共 %d 条边", len(self.edge_features))
        
        return self.edge_features
    
    # ==================== 特征提取 ====================
    
    def _extract_segment_features(self, df: pd.DataFrame) -> List[Dict]:
        """提取轨迹段特征（矢量化优化）"""
        segments = []
        grouped = df.groupby('船舶名称')
        total = len(grouped)
        
        for idx, (ship_name, group) in enumerate(grouped):
            if (idx + 1) % 50 == 0:
                print(f"  处理进度: {idx+1}/{total} 艘船舶")
            
            if len(group) < 2:
                continue
            
            group = group.sort_values('时间').reset_index(drop=True)
            n = len(group) - 1
            
            lat1 = group['纬度'].values[:n]
            lat2 = group['纬度'].values[1:]
            lon1 = group['经度'].values[:n]
            lon2 = group['经度'].values[1:]
            
            # 矢量化计算距离（弧度）
            lat1_r = np.radians(lat1)
            lat2_r = np.radians(lat2)
            dlat_r = np.radians(lat2 - lat1)
            dlon_r = np.radians(lon2 - lon1)
            a = np.sin(dlat_r / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon_r / 2) ** 2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distances = 6371000 * c
            
            # 时间差（秒）
            times = group['时间'].values
            time_diffs_sec = np.array([(times[i + 1] - times[i]) / np.timedelta64(1, 's') for i in range(n)])
            
            # 计算速度用于过滤（停泊/锚泊数据：速度<0.5节）
            actual_speeds_all = np.zeros(n)
            mask_pos = (time_diffs_sec > 0) & (distances >= 10)
            actual_speeds_all[mask_pos] = distances[mask_pos] / time_diffs_sec[mask_pos] * 1.944  # m/s → 节
            
            # 过滤无效段：时间差>0且<24小时，距离>=10米，速度>=0.5节（排除停泊数据）
            valid = (time_diffs_sec > 0) & (time_diffs_sec < 86400) & (distances >= 10) & (actual_speeds_all >= 0.5)
            valid_idx = np.where(valid)[0]
            
            if len(valid_idx) == 0:
                continue
            
            # 矢量化计算航向
            lat1_v = lat1[valid_idx]
            lat2_v = lat2[valid_idx]
            dlon_v = lon2[valid_idx] - lon1[valid_idx]
            y_v = np.sin(np.radians(dlon_v)) * np.cos(np.radians(lat2_v))
            x_v = (np.cos(np.radians(lat1_v)) * np.sin(np.radians(lat2_v)) -
                   np.sin(np.radians(lat1_v)) * np.cos(np.radians(lat2_v)) * np.cos(np.radians(dlon_v)))
            bearings = (np.degrees(np.arctan2(y_v, x_v)) + 360) % 360
            
            # 矢量化计算航向差
            course1 = group['航向'].values[:n][valid_idx]
            course2 = group['航向'].values[1:][valid_idx]
            diff = np.abs(course1 - course2)
            course_changes = np.where(diff > 180, 360 - diff, diff)
            
            # 速度计算
            td = time_diffs_sec[valid_idx]
            dv = distances[valid_idx]
            actual_speeds = dv / td * 1.944
            reported_speeds = (group['航速'].values[:n][valid_idx] + group['航速'].values[1:][valid_idx]) / 2
            speed_diffs = np.abs(actual_speeds - reported_speeds)
            
            # 时间特征
            valid_times = times[valid_idx]
            hours = pd.to_datetime(valid_times).hour
            is_peak = np.isin(hours, self.peak_hours).astype(int)
            weekdays = pd.to_datetime(valid_times).weekday
            is_weekend = (weekdays >= 5).astype(int)
            
            for j in range(len(valid_idx)):
                i = valid_idx[j]
                segments.append({
                    'ship_name': ship_name,
                    'start_lat': float(lat1[i]),
                    'start_lon': float(lon1[i]),
                    'end_lat': float(lat2[i]),
                    'end_lon': float(lon2[i]),
                    'distance': float(distances[i]),
                    'time_diff': float(time_diffs_sec[i]),
                    'actual_speed': float(actual_speeds[j]),
                    'reported_speed': float(reported_speeds[j]),
                    'speed_diff': float(speed_diffs[j]),
                    'bearing': float(bearings[j]),
                    'course_change': float(course_changes[j]),
                    'hour': int(hours[j]),
                    'is_peak_hour': int(is_peak[j]),
                    'is_weekend': int(is_weekend[j]),
                    'time_period': self._get_time_period(int(hours[j]))
                })
        
        print(f"  提取轨迹段: {len(segments):,} 个")
        return segments
    
    def _get_time_period(self, hour: int) -> str:
        for period_name, (start, end) in self.time_periods.items():
            if start <= hour < end:
                return period_name
        return 'night'
    
    def _compute_waterway_types(self, segments: List[Dict], graph):
        """计算水域类型：基于网格密度，分别计算节点级别和边级别
        
        使用航道边上密度的P75作为阈值（而非全局网格密度），
        确保narrow/open在航道内部也能有效区分。
        """
        grid_size = 0.005
        density_grid = defaultdict(int)
        
        for seg in segments:
            mid_lat = (seg['start_lat'] + seg['end_lat']) / 2
            mid_lon = (seg['start_lon'] + seg['end_lon']) / 2
            grid_key = (int(mid_lat / grid_size), int(mid_lon / grid_size))
            density_grid[grid_key] += 1
        
        # 保存密度网格，供动态查询使用
        self._density_grid = density_grid
        self._density_grid_size = grid_size
        
        # 计算阈值：只基于航道边端点附近的密度（而非全部网格）
        # 这样确保阈值是航道内部的区分，而非航道vs外围
        edge_densities = []
        for node_id, attrs in graph.nodes(data=True):
            avg_density = self._get_density_at(attrs['lat'], attrs['lon'])
            edge_densities.append(avg_density)
        
        density_threshold = np.percentile(edge_densities, 75) if edge_densities else 0
        self._density_threshold = density_threshold
        
        # 节点级别（保留，供 GNN 等需要节点特征的场景使用）
        self.node_waterway_types = {}
        for node_id, attrs in graph.nodes(data=True):
            avg_density = self._get_density_at(attrs['lat'], attrs['lon'])
            self.node_waterway_types[node_id] = 1 if avg_density >= density_threshold else 0
        
        # 边级别：基于边中点所在网格密度判断，避免 max 传播导致全 narrow
        self.edge_waterway_types = {}
        for u, v in graph.edges():
            u_attr = graph.nodes[u]
            v_attr = graph.nodes[v]
            mid_lat = (u_attr['lat'] + v_attr['lat']) / 2
            mid_lon = (u_attr['lon'] + v_attr['lon']) / 2
            avg_density = self._get_density_at(mid_lat, mid_lon)
            self.edge_waterway_types[(u, v)] = 1 if avg_density >= density_threshold else 0
    
    def _get_density_at(self, lat: float, lon: float) -> float:
        """获取指定位置附近的平均轨迹密度"""
        grid_key = (int(lat / self._density_grid_size), int(lon / self._density_grid_size))
        nearby_density = 0
        count = 0
        for dlat in [-1, 0, 1]:
            for dlon in [-1, 0, 1]:
                key = (grid_key[0] + dlat, grid_key[1] + dlon)
                if key in self._density_grid:
                    nearby_density += self._density_grid[key]
                    count += 1
        return nearby_density / count if count > 0 else 0
    
    def _get_edge_waterway_type(self, from_node: int, to_node: int, graph) -> int:
        """获取边的水域类型，基于边中点密度计算（避免 max 传播）"""
        # 优先使用预计算的边级别结果
        edge_key = (from_node, to_node)
        if edge_key in self.edge_waterway_types:
            return self.edge_waterway_types[edge_key]
        
        # 动态计算：基于边中点密度
        if hasattr(self, '_density_grid'):
            try:
                u_attr = graph.nodes[from_node]
                v_attr = graph.nodes[to_node]
                mid_lat = (u_attr['lat'] + v_attr['lat']) / 2
                mid_lon = (u_attr['lon'] + v_attr['lon']) / 2
                avg_density = self._get_density_at(mid_lat, mid_lon)
                return 1 if avg_density >= self._density_threshold else 0
            except (KeyError, AttributeError):
                pass
        
        # 最终 fallback：取两端节点的平均值（而非 max）
        return 1 if (self.node_waterway_types.get(from_node, 0) + 
                     self.node_waterway_types.get(to_node, 0)) >= 1.5 else 0
    
    def _compute_network_features(self, graph):
        """计算网络拓扑特征"""
        import networkx as nx
        
        # 节点度
        self.node_degrees = dict(graph.degree())
        
        # 边介数中心性（大图用采样近似）
        try:
            n_nodes = graph.number_of_nodes()
            if n_nodes > 500:
                betweenness = nx.edge_betweenness_centrality(graph, k=min(200, n_nodes))
            else:
                betweenness = nx.edge_betweenness_centrality(graph)
            self.edge_betweenness = betweenness
        except:
            self.edge_betweenness = {}
    
    def _map_segments_to_edges(self, graph, segments: List[Dict]) -> Dict:
        """映射轨迹段到网络边"""
        grid_size = 0.001
        node_grid = defaultdict(list)
        
        for node_id, attrs in graph.nodes(data=True):
            grid_lat = int(attrs['lat'] / grid_size)
            grid_lon = int(attrs['lon'] / grid_size)
            node_grid[(grid_lat, grid_lon)].append({
                'node_id': node_id,
                'lat': attrs['lat'],
                'lon': attrs['lon']
            })
        
        edge_segments = defaultdict(list)
        search_radius = 500  # Expanded from 200 to 500 to improve trajectory mapping
        
        for segment in segments:
            start_node = self._find_nearest_node(
                segment['start_lat'], segment['start_lon'],
                node_grid, grid_size, search_radius
            )
            end_node = self._find_nearest_node(
                segment['end_lat'], segment['end_lon'],
                node_grid, grid_size, search_radius
            )
            
            if start_node is not None and end_node is not None and start_node != end_node:
                edge_segments[(start_node, end_node)].append(segment)
        
        print(f"  映射边数量: {len(edge_segments):,}")
        return edge_segments
    
    def _find_nearest_node(self, lat: float, lon: float,
                           node_grid: Dict, grid_size: float,
                           search_radius: float) -> Optional[int]:
        grid_lat = int(lat / grid_size)
        grid_lon = int(lon / grid_size)
        
        min_dist = float('inf')
        nearest_node = None
        
        # Search 3x3 grid cells (expanded from default)
        for dlat in [-1, 0, 1]:
            for dlon in [-1, 0, 1]:
                key = (grid_lat + dlat, grid_lon + dlon)
                for node in node_grid.get(key, []):
                    dist = haversine_distance(lat, lon, node['lat'], node['lon'])
                    if dist < min_dist and dist < search_radius:
                        min_dist = dist
                        nearest_node = node['node_id']
        
        return nearest_node
    
    def _build_training_data(self, edge_segments: Dict, graph) -> Tuple:
        """
        构建边×时段级别的训练数据

        核心改造：目标变量改为 time_ratio = actual_time / theoretical_time，
        归一化掉距离/速度主效应，让模型专注学习动态偏差。
        移除泄漏特征（theoretical_time, distance），
        新增交互特征（speed_cv, narrow_x_peak, course_change_x_peak）。

        Returns:
            X: 特征矩阵
            y_ratio: time_ratio 目标值
            y_time: 原始 travel_time（用于评估）
            theoretical_times: 每个样本对应的 theoretical_time
            edge_period_info: 边×时段信息
        """
        X_list = []
        y_ratio_list = []
        y_time_list = []
        tt_list = []
        edge_period_info = {}
        self._edge_theoretical_times = {}

        for edge_key, segments in edge_segments.items():
            if len(segments) < 2:
                continue

            from_node, to_node = edge_key
            waterway_type = self._get_edge_waterway_type(from_node, to_node, graph)
            node_degree_from = self.node_degrees.get(from_node, 0)
            node_degree_to = self.node_degrees.get(to_node, 0)
            betweenness = self.edge_betweenness.get(edge_key, 0)

            distances = [s['distance'] for s in segments]
            avg_distance = np.mean(distances)
            avg_bearing = np.mean([s['bearing'] for s in segments])
            bearing_rad = np.deg2rad(avg_bearing)
            avg_course_change = np.mean([s['course_change'] for s in segments])

            reported_speeds_all = [s['reported_speed'] for s in segments]
            avg_reported_speed_edge = np.mean(reported_speeds_all)
            speed_ms = max(avg_reported_speed_edge, 0.5) * 0.5144
            theoretical_time = avg_distance / speed_ms
            self._edge_theoretical_times[edge_key] = theoretical_time

            period_groups = defaultdict(list)
            for seg in segments:
                period_groups[seg['time_period']].append(seg)

            for period_name, period_segs in period_groups.items():
                time_diffs = [s['time_diff'] for s in period_segs]
                reported_speeds = [s['reported_speed'] for s in period_segs]
                course_changes = [s['course_change'] for s in period_segs]

                avg_reported_speed = np.mean(reported_speeds)
                std_reported_speed = np.std(reported_speeds) if len(reported_speeds) > 1 else 0
                avg_travel_time = np.mean(time_diffs)
                avg_travel_time = min(avg_travel_time, 3600)

                time_ratio = avg_travel_time / max(theoretical_time, 1.0)
                time_ratio = np.clip(time_ratio, 0.1, 20.0)

                speed_cv = std_reported_speed / max(avg_reported_speed, 0.1)
                std_course_change = np.std(course_changes) if len(course_changes) > 1 else 0
                course_change_x_narrow = avg_course_change * waterway_type

                hours = [s['hour'] for s in period_segs]
                rep_hour = int(np.median(hours))
                is_peak = 1 if rep_hour in self.peak_hours else 0
                is_weekend_mode = 1 if sum(s['is_weekend'] for s in period_segs) > len(period_segs) / 2 else 0
                hour_rad = np.deg2rad(rep_hour * 15)

                narrow_x_peak = waterway_type * is_peak
                course_change_x_peak = avg_course_change * is_peak

                sample_count = len(period_segs)

                features = [
                    avg_reported_speed,
                    std_reported_speed,
                    speed_cv,
                    avg_bearing,
                    np.sin(bearing_rad),
                    np.cos(bearing_rad),
                    avg_course_change,
                    std_course_change,
                    course_change_x_narrow,
                    is_peak,
                    is_weekend_mode,
                    waterway_type,
                    rep_hour,
                    np.sin(hour_rad),
                    np.cos(hour_rad),
                    node_degree_from,
                    node_degree_to,
                    betweenness,
                    sample_count,
                    np.log1p(sample_count),
                    narrow_x_peak,
                    course_change_x_peak
                ]

                X_list.append(features)
                y_ratio_list.append(time_ratio)
                y_time_list.append(avg_travel_time)
                tt_list.append(theoretical_time)

                edge_period_info[(edge_key, period_name)] = {
                    'hour': rep_hour,
                    'sample_count': sample_count,
                    'avg_travel_time': avg_travel_time,
                    'time_ratio': time_ratio,
                    'theoretical_time': theoretical_time,
                    'time_period': period_name,
                }

        X = np.array(X_list)
        y_ratio = np.array(y_ratio_list)
        y_time = np.array(y_time_list)
        theoretical_times = np.array(tt_list)

        print(f"  训练样本数: {len(X):,} (边×时段聚合)")
        print(f"  特征维度: {X.shape[1]}")
        print(f"  time_ratio: min={y_ratio.min():.3f}, max={y_ratio.max():.3f}, "
              f"mean={y_ratio.mean():.3f}, std={y_ratio.std():.3f}")

        return X, y_ratio, y_time, theoretical_times, edge_period_info
    
    # ==================== 模型训练与对比 ====================
    
    def _train_and_compare_models(self, X: np.ndarray, y_ratio: np.ndarray,
                                   y_time: np.ndarray, theoretical_times: np.ndarray,
                                   models: List[str], graph, edge_segments,
                                   use_grid_search: bool = True) -> Dict[str, ModelResult]:
        """训练并对比多个模型（统一使用 time_ratio 作为目标）"""
        results = {}

        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y_ratio, 'values'):
            y_ratio = y_ratio.values

        X_train, X_test, y_ratio_train, y_ratio_test = train_test_split(
            X, y_ratio, test_size=0.2, random_state=42
        )
        _, _, y_time_train, y_time_test = train_test_split(
            X, y_time, test_size=0.2, random_state=42
        )
        _, _, tt_train, tt_test = train_test_split(
            X, theoretical_times, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scaler = scaler
        self.use_grid_search = use_grid_search
        self._tt_test = tt_test
        self._y_time_test = y_time_test

        print(f"\n  训练集: {len(X_train):,} 样本")
        print(f"  测试集: {len(X_test):,} 样本")
        print(f"  网格搜索: {'启用' if use_grid_search else '禁用'}")
        print("\n  " + "-"*60)

        if 'xgboost' in models and HAS_XGBOOST:
            result = self._train_xgboost(X_train, X_test, y_ratio_train, y_ratio_test)
            results['xgboost'] = result
            self._print_result(result)

        if 'lightgbm' in models and HAS_LIGHTGBM:
            result = self._train_lightgbm(X_train, X_test, y_ratio_train, y_ratio_test)
            results['lightgbm'] = result
            self._print_result(result)

        if 'lightgbm_tweedie' in models and HAS_LIGHTGBM:
            result = self._train_lightgbm_tweedie(X_train, X_test, y_ratio_train, y_ratio_test)
            results['lightgbm_tweedie'] = result
            self._print_result(result)

        if 'random_forest' in models:
            result = self._train_random_forest(X_train, X_test, y_ratio_train, y_ratio_test)
            results['random_forest'] = result
            self._print_result(result)

        if 'mlp' in models and HAS_TORCH:
            result = self._train_mlp(X_train_scaled, X_test_scaled, y_ratio_train, y_ratio_test)
            results['mlp'] = result
            self._print_result(result)

        if 'gnn' in models and HAS_PYG:
            all_indices = np.arange(len(X))
            train_indices, test_indices = train_test_split(
                all_indices, test_size=0.2, random_state=42
            )
            result = self._train_gnn(X, y_ratio, graph, edge_segments)
            results['gnn'] = result
            self._print_result(result)

        self._print_comparison_table(results)
        self._model_results = results

        return results
    
    def _train_xgboost(self, X_train, X_test, y_train, y_test) -> ModelResult:
        """训练 XGBoost（支持贝叶斯调参 / 默认参数）"""
        start_time = time.time()
        
        if self.use_grid_search and HAS_OPTUNA:
            print(f"\n    [XGBoost] 执行贝叶斯调参（Optuna）...")
            from sklearn.model_selection import cross_val_score

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                }
                candidate = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
                scores = cross_val_score(candidate, X_train, y_train, cv=3,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
                return scores.mean()

            study = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=20, show_progress_bar=HAS_TQDM)
            best_params = study.best_params
            print(f"    最佳参数: {best_params}")
            print(f"    最佳 CV 分数: {study.best_value:.4f}")

            model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train, verbose=False)
        elif self.use_grid_search:
            # optuna 未安装，回退到随机搜索
            from sklearn.model_selection import ParameterSampler, cross_val_score
            print(f"\n    [XGBoost] Optuna 未安装，回退到随机搜索...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.2],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
            param_samples = list(ParameterSampler(param_grid, n_iter=20, random_state=42))
            best_score = -np.inf
            best_params = None
            for params in tqdm(param_samples, desc="    XGBoost 搜索", unit="组", disable=not HAS_TQDM):
                candidate = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1)
                scores = cross_val_score(candidate, X_train, y_train, cv=3,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_params = params
            model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train, verbose=False)
            print(f"    最佳参数: {best_params}")
        else:
            model = xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train, verbose=False)
        
        train_time = time.time() - start_time
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0.1, 20.0)

        self.feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        
        return self._evaluate_model('XGBoost', model, y_test, y_pred, train_time)
    
    def _train_lightgbm(self, X_train, X_test, y_train, y_test) -> ModelResult:
        """训练 LightGBM"""
        start_time = time.time()
        
        # 用 DataFrame 包装，确保训练/预测时特征名一致，避免 sklearn 警告
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=self.feature_names)
            X_test = pd.DataFrame(X_test, columns=self.feature_names)
        
        if self.use_grid_search and HAS_OPTUNA:
            print(f"\n    [LightGBM] 执行贝叶斯调参（Optuna）...")
            from sklearn.model_selection import cross_val_score

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 7, 63),
                    'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
                    'min_split_gain': trial.suggest_float('min_split_gain', 0.01, 0.5, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': 5,
                }
                candidate = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
                scores = cross_val_score(candidate, X_train, y_train, cv=5,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
                return scores.mean()

            study = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=30, show_progress_bar=HAS_TQDM)
            best_params = study.best_params
            best_params['bagging_freq'] = 5
            print(f"    最佳参数: {best_params}")
            print(f"    最佳 CV 分数: {study.best_value:.4f}")

            model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X_train, y_train)
        elif self.use_grid_search:
            # optuna 未安装，回退到随机搜索
            from sklearn.model_selection import ParameterSampler, cross_val_score
            print(f"\n    [LightGBM] Optuna 未安装，回退到随机搜索...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.02, 0.05],
                'num_leaves': [15, 31, 63],
                'min_child_samples': [20, 50, 80],
                'reg_alpha': [0.5, 1.0, 2.0],
                'reg_lambda': [0.5, 1.0, 2.0],
                'min_split_gain': [0.05, 0.1, 0.2],
                'feature_fraction': [0.6, 0.8, 1.0],
                'bagging_fraction': [0.6, 0.8, 1.0],
                'bagging_freq': [5],
            }
            param_samples = list(ParameterSampler(param_grid, n_iter=25, random_state=42))
            best_score = -np.inf
            best_params = None
            for params in tqdm(param_samples, desc="    LGBM 搜索", unit="组", disable=not HAS_TQDM):
                candidate = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
                scores = cross_val_score(candidate, X_train, y_train, cv=5,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_params = params
            model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X_train, y_train)
            print(f"    最佳参数: {best_params}")
        else:
            model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.02,
                num_leaves=31, min_child_samples=50,
                reg_alpha=1.0, reg_lambda=1.0, min_split_gain=0.1,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                random_state=42, n_jobs=-1, verbose=-1
            )
            model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0.1, 20.0)

        self.feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        
        return self._evaluate_model('LightGBM', model, y_test, y_pred, train_time)
    
    def _train_lightgbm_tweedie(self, X_train, X_test, y_train, y_test) -> ModelResult:
        """训练 LightGBM Tweedie 回归（适合正偏分布的 ratio 目标）"""
        start_time = time.time()
        
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=self.feature_names)
            X_test = pd.DataFrame(X_test, columns=self.feature_names)
        
        y_train_pos = np.maximum(y_train, 0.1)
        
        if self.use_grid_search and HAS_OPTUNA:
            print(f"\n    [LightGBM-Tweedie] 执行贝叶斯调参（Optuna）...")
            from sklearn.model_selection import cross_val_score

            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 7, 63),
                    'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
                    'min_split_gain': trial.suggest_float('min_split_gain', 0.01, 0.5, log=True),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
                    'bagging_freq': 5,
                    'objective': 'tweedie',
                    'tweedie_variance_power': trial.suggest_float('tweedie_variance_power', 1.1, 1.9),
                }
                candidate = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
                scores = cross_val_score(candidate, X_train, y_train_pos, cv=5,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
                return scores.mean()

            study = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=30, show_progress_bar=HAS_TQDM)
            best_params = study.best_params
            best_params['bagging_freq'] = 5
            best_params['objective'] = 'tweedie'
            print(f"    最佳参数: {best_params}")
            print(f"    最佳 CV 分数: {study.best_value:.4f}")

            model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X_train, y_train_pos)
        else:
            # 默认 Tweedie 参数
            model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.02,
                num_leaves=31, min_child_samples=50,
                reg_alpha=1.0, reg_lambda=1.0, min_split_gain=0.1,
                feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
                objective='tweedie', tweedie_variance_power=1.5,
                random_state=42, n_jobs=-1, verbose=-1
            )
            model.fit(X_train, y_train_pos)
        
        train_time = time.time() - start_time
        y_pred = model.predict(X_test)
        
        y_pred = np.clip(y_pred, 0.1, 20.0)
        
        if not hasattr(self, 'feature_importance'):
            self.feature_importance = {}
        self.feature_importance_tweedie = dict(zip(self.feature_names, model.feature_importances_))
        
        return self._evaluate_model('LightGBM-Tweedie', model, y_test, y_pred, train_time,
                                     use_log_transform=False)
    
    def _train_random_forest(self, X_train, X_test, y_train, y_test) -> ModelResult:
        """训练随机森林（支持贝叶斯调参 / 默认参数）"""
        start_time = time.time()
        
        # 转换为 numpy array
        if hasattr(X_train, 'values'):
            X_train = X_train.values
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        
        if self.use_grid_search and HAS_OPTUNA:
            print(f"\n    [RandomForest] 执行贝叶斯调参（Optuna）...")
            from sklearn.model_selection import cross_val_score

            def objective(trial):
                max_depth = trial.suggest_int('max_depth', 3, 20)
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': max_depth,
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.6, 0.8, 1.0]),
                }
                candidate = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                scores = cross_val_score(candidate, X_train, y_train, cv=3,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
                return scores.mean()

            study = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective, n_trials=20, show_progress_bar=HAS_TQDM)
            best_params = study.best_params
            print(f"    最佳参数: {best_params}")
            print(f"    最佳 CV 分数: {study.best_value:.4f}")

            model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
        elif self.use_grid_search:
            # optuna 未安装，回退到随机搜索
            from sklearn.model_selection import ParameterSampler, cross_val_score
            print(f"\n    [RandomForest] Optuna 未安装，回退到随机搜索...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [6, 8, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.8]
            }
            param_samples = list(ParameterSampler(param_grid, n_iter=20, random_state=42))
            best_score = -np.inf
            best_params = None
            for params in tqdm(param_samples, desc="    RF 搜索", unit="组", disable=not HAS_TQDM):
                candidate = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
                scores = cross_val_score(candidate, X_train, y_train, cv=3,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_params = params
            model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            print(f"    最佳参数: {best_params}")
        else:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=10,
                random_state=42, n_jobs=-1
            )
            model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        y_pred = model.predict(X_test)
        
        # 保存特征重要性
        self.feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        
        return self._evaluate_model('RandomForest', model, y_test, y_pred, train_time)
    
    def _train_mlp(self, X_train, X_test, y_train, y_test) -> ModelResult:
        """训练 MLP（支持网格搜索，简化结构，增加 dropout）"""
        start_time = time.time()
        
        # 获取输出的合理范围
        y_min, y_max = y_train.min(), y_train.max()
        self.mlp_y_range = (y_min, y_max)
        
        # 定义简化模型 - 更少层数，更强正则化
        class MLP(nn.Module):
            def __init__(self, input_dim, hidden_layers, dropout):
                super().__init__()
                layers = []
                prev_dim = input_dim
                
                for hidden_dim in hidden_layers:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)  # 强 dropout
                    ])
                    prev_dim = hidden_dim
                
                layers.append(nn.Linear(prev_dim, 1))
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        # 定义超参数搜索空间 - 简化结构，增加 dropout
        param_combinations = [
            {'hidden_layers': [32, 16], 'lr': 0.001, 'dropout': 0.5},  # 简化 + 强 dropout
            {'hidden_layers': [64, 32], 'lr': 0.001, 'dropout': 0.5},  # 简化 + 强 dropout
            {'hidden_layers': [64, 32, 16], 'lr': 0.0005, 'dropout': 0.4},
            {'hidden_layers': [48, 24], 'lr': 0.001, 'dropout': 0.5},
            {'hidden_layers': [32], 'lr': 0.001, 'dropout': 0.5},  # 最简单结构
        ]
        
        # 转换为 Tensor
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
        X_test_t = torch.FloatTensor(X_test)
        
        best_model_state = None
        best_val_loss = float('inf')
        best_params = None
        
        if self.use_grid_search:
            print(f"\n    [MLP] 执行网格搜索（简化结构，强正则化）...")
            
            # 使用 K-Fold 交叉验证
            from sklearn.model_selection import KFold
            kfold = KFold(n_splits=5, shuffle=True, random_state=42)
            
            for i, params in enumerate(param_combinations):
                cv_losses = []
                
                for fold, (train_idx, val_idx) in tqdm(enumerate(kfold.split(X_train_t)),
                    total=kfold.get_n_splits(), desc=f"      参数组 {i+1}/{len(param_combinations)}",
                    leave=False, disable=not HAS_TQDM):
                    X_tr, X_val = X_train_t[train_idx], X_train_t[val_idx]
                    y_tr, y_val = y_train_t[train_idx], y_train_t[val_idx]
                    
                    model = MLP(X_train.shape[1], params['hidden_layers'], params['dropout'])
                    criterion = nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-3)  # 强 L2 正则化
                    
                    # 早停训练
                    model.train()
                    best_fold_loss = float('inf')
                    patience = 0
                    
                    for epoch in range(200):
                        optimizer.zero_grad()
                        outputs = model(X_tr)
                        loss = criterion(outputs, y_tr)
                        loss.backward()
                        optimizer.step()
                        
                        # 验证
                        model.eval()
                        with torch.no_grad():
                            val_loss = criterion(model(X_val), y_val).item()
                        model.train()
                        
                        if val_loss < best_fold_loss:
                            best_fold_loss = val_loss
                            patience = 0
                        else:
                            patience += 1
                            if patience > 15:
                                break
                    
                    cv_losses.append(best_fold_loss)
                
                avg_cv_loss = np.mean(cv_losses)
                std_cv_loss = np.std(cv_losses)
                
                print(f"      参数组 {i+1}/{len(param_combinations)}: CV_loss={avg_cv_loss:.4f} (±{std_cv_loss:.4f})")
                
                if avg_cv_loss < best_val_loss:
                    best_val_loss = avg_cv_loss
                    best_params = params
            
            print(f"    最佳参数: {best_params}")
            
            # 使用最佳参数重新训练完整模型
            model = MLP(X_train.shape[1], best_params['hidden_layers'], best_params['dropout'])
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        else:
            # 默认使用简化结构
            model = MLP(X_train.shape[1], [64, 32], 0.5)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # 完整训练 - 使用早停
        model.train()
        best_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in tqdm(range(300), desc="    MLP 训练", unit="epoch", disable=not HAS_TQDM):
            optimizer.zero_grad()
            outputs = model(X_train_t)
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            if hasattr(scheduler, 'step'):
                scheduler.step(loss.item())
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > 25:  # 早停
                    break
        
        # 恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        train_time = time.time() - start_time
        
        # 预测
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_t).numpy().flatten()
        
        # 裁剪到合理范围
        y_pred = np.clip(y_pred, max(1.0, y_min), y_max)
        
        return self._evaluate_model('MLP', model, y_test, y_pred, train_time)
    
    def _train_gnn(self, X, y, graph, edge_segments) -> ModelResult:
        """训练图神经网络（支持网格搜索，丰富边特征+训练测试划分）"""
        start_time = time.time()
        
        # 先收集有效边（有足够轨迹数据的边），扩展边特征
        valid_edges = []
        edge_features_list = []
        edge_targets_ratio = []
        edge_targets_original_list = []
        edge_theoretical_times_list = []
        
        for edge_key, segments in edge_segments.items():
            if len(segments) < 2:
                continue
            
            from_node, to_node = edge_key
            avg_time = np.mean([s['time_diff'] for s in segments])
            avg_reported_speed = np.mean([s['reported_speed'] for s in segments])
            distance = np.mean([s['distance'] for s in segments])
            avg_bearing = np.mean([s['bearing'] for s in segments])
            avg_course_change = np.mean([s['course_change'] for s in segments])
            waterway_type = self._get_edge_waterway_type(from_node, to_node, graph)
            node_degree_from = self.node_degrees.get(from_node, 0)
            node_degree_to = self.node_degrees.get(to_node, 0)
            betweenness = self.edge_betweenness.get(edge_key, 0)
            
            speed_ms = max(avg_reported_speed, 0.5) * 0.5144
            theoretical_time = distance / speed_ms
            
            valid_edges.append((from_node, to_node))
            bearing_rad = np.deg2rad(avg_bearing)
            edge_features_list.append([
                distance, theoretical_time, avg_reported_speed,
                np.sin(bearing_rad), np.cos(bearing_rad), avg_course_change,
                waterway_type, node_degree_from, node_degree_to, betweenness
            ])
            edge_targets_ratio.append(avg_time / max(theoretical_time, 1e-6))
            edge_targets_original_list.append(avg_time)
            edge_theoretical_times_list.append(theoretical_time)
        
        if len(valid_edges) == 0:
            return ModelResult('GNN', 0, float('inf'), float('inf'), 0, 100, None)
        
        n_edges = len(valid_edges)
        print(f"\n    [GNN] 有效边数: {n_edges}, 边特征维度: {len(edge_features_list[0])}")
        
        edge_targets_ratio = np.array(edge_targets_ratio, dtype=np.float64)
        edge_targets_original = np.array(edge_targets_original_list, dtype=np.float64)
        edge_theoretical_times = np.array(edge_theoretical_times_list, dtype=np.float64)
        
        # 收集有效边涉及的节点
        valid_nodes = set()
        for u, v in valid_edges:
            valid_nodes.add(u)
            valid_nodes.add(v)
        
        # 构建节点特征（扩展：增加度中心性、聚类系数等）
        node_features = []
        node_id_to_idx = {}
        
        try:
            import networkx as nx
            clustering = nx.clustering(graph)
        except:
            clustering = {}
        
        for idx, node_id in enumerate(sorted(valid_nodes)):
            node_id_to_idx[node_id] = idx
            attrs = graph.nodes[node_id]
            degree = self.node_degrees.get(node_id, 0)
            waterway = self.node_waterway_types.get(node_id, 0)
            cluster_coeff = clustering.get(node_id, 0)
            node_features.append([degree, waterway, attrs['lat'], attrs['lon'], cluster_coeff])
        
        node_features = torch.FloatTensor(node_features)
        
        # 边索引（双向：加入反向边使消息传递更充分）
        edge_index_forward = []
        for u, v in valid_edges:
            edge_index_forward.append([node_id_to_idx[u], node_id_to_idx[v]])
        # 加入反向边
        edge_index_bidir = edge_index_forward + [[v, u] for u, v in edge_index_forward]
        edge_index = torch.LongTensor(edge_index_bidir).t().contiguous()
        
        edge_features = torch.FloatTensor(edge_features_list)
        edge_targets_original_t = torch.FloatTensor(edge_targets_original)
        edge_targets = torch.FloatTensor(edge_targets_ratio)
        
        # 标准化边特征
        edge_scaler = StandardScaler()
        edge_features_np = edge_scaler.fit_transform(edge_features.numpy())
        edge_features = torch.FloatTensor(edge_features_np)
        self.gnn_edge_scaler = edge_scaler
        
        # 标准化节点特征
        node_scaler = StandardScaler()
        node_features_np = node_scaler.fit_transform(node_features.numpy())
        node_features = torch.FloatTensor(node_features_np)
        self.gnn_node_scaler = node_scaler
        
        # 训练/测试划分（按边索引 80/20）
        n_train = int(n_edges * 0.8)
        perm = torch.randperm(n_edges)
        train_mask = torch.zeros(n_edges, dtype=torch.bool)
        train_mask[perm[:n_train]] = True
        test_mask = ~train_mask
        print(f"    训练边: {train_mask.sum().item()}, 测试边: {test_mask.sum().item()}")
        
        # EdgeGNN 已移至模块级别（确保可 pickle 序列化）
        num_target_edges = n_edges  # 前 n_edges 条是原始方向
        
        # 超参数搜索空间
        param_combinations = [
            {'hidden_dim': 64, 'num_layers': 2, 'lr': 0.005, 'dropout': 0.2},
            {'hidden_dim': 128, 'num_layers': 2, 'lr': 0.003, 'dropout': 0.3},
            {'hidden_dim': 64, 'num_layers': 3, 'lr': 0.003, 'dropout': 0.2},
            {'hidden_dim': 128, 'num_layers': 3, 'lr': 0.001, 'dropout': 0.3},
        ]
        
        best_model_state = None
        best_val_loss = float('inf')
        best_params = None
        
        if self.use_grid_search:
            print(f"    [GNN] 执行网格搜索（{len(param_combinations)} 组参数）...")
            
            for i, params in enumerate(param_combinations):
                model = EdgeGNN(
                    node_features.shape[1], 
                    edge_features.shape[1],
                    hidden_dim=params['hidden_dim'],
                    num_layers=params['num_layers'],
                    dropout=params['dropout']
                )
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-4)
                
                # 训练（在训练集上训练，验证集上选最优）
                model.train()
                best_fold_val = float('inf')
                patience = 0
                for epoch in tqdm(range(200), desc=f"      参数组 {i+1}/{len(param_combinations)}",
                                  leave=False, disable=not HAS_TQDM):
                    optimizer.zero_grad()
                    outputs = model(node_features, edge_index, edge_features, num_target_edges)
                    loss = criterion(outputs[train_mask], edge_targets[train_mask])
                    loss.backward()
                    optimizer.step()
                    
                    # 验证
                    model.eval()
                    with torch.no_grad():
                        val_loss = criterion(outputs, edge_targets).item()
                        val_loss_masked = criterion(outputs[test_mask], edge_targets[test_mask]).item()
                    model.train()
                    
                    if val_loss_masked < best_fold_val:
                        best_fold_val = val_loss_masked
                        patience = 0
                    else:
                        patience += 1
                        if patience > 30:
                            break
                
                print(f"      参数组 {i+1}/{len(param_combinations)}: val_loss={best_fold_val:.4f}")
                
                if best_fold_val < best_val_loss:
                    best_val_loss = best_fold_val
                    best_params = params
            
            print(f"    最佳参数: {best_params}")
            
            # 使用最佳参数重新训练完整模型
            model = EdgeGNN(
                node_features.shape[1],
                edge_features.shape[1],
                hidden_dim=best_params['hidden_dim'],
                num_layers=best_params['num_layers'],
                dropout=best_params['dropout']
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
        else:
            model = EdgeGNN(node_features.shape[1], edge_features.shape[1], hidden_dim=64, num_layers=2, dropout=0.2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
        
        criterion = nn.HuberLoss(delta=2.0)
        
        # 完整训练（带早停）
        model.train()
        best_train_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in tqdm(range(500), desc="    GNN 训练", unit="epoch", disable=not HAS_TQDM):
            optimizer.zero_grad()
            outputs = model(node_features, edge_index, edge_features, num_target_edges)
            loss = criterion(outputs[train_mask], edge_targets[train_mask])
            loss.backward()
            optimizer.step()
            
            if hasattr(scheduler, 'step'):
                scheduler.step(loss.item())
            
            if loss.item() < best_train_loss:
                best_train_loss = loss.item()
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > 50:
                    break
        
        # 恢复最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        train_time = time.time() - start_time
        
        # 在测试集上评估
        model.eval()
        with torch.no_grad():
            y_pred_ratio = model(node_features, edge_index, edge_features, num_target_edges).numpy()
        
        y_true_test = edge_targets[test_mask].numpy()
        y_pred_test = y_pred_ratio[test_mask.numpy()]
        y_pred_test = np.clip(y_pred_test, 0.1, 20.0)
        
        y_true_time_test = edge_targets_original_t[test_mask].numpy()
        gnn_tt_test = edge_theoretical_times[test_mask.numpy()]
        
        print(f"    GNN 训练集 loss: {best_train_loss:.4f}, 测试边数: {test_mask.sum().item()}")
        
        return self._evaluate_model('GNN', model, y_true_test, y_pred_test, train_time,
                                     y_true_time=y_true_time_test,
                                     theoretical_times_test=gnn_tt_test)
    
    def _compute_duan_smearing_factor(self, X: np.ndarray, y_log: np.ndarray):
        """已弃用：time_ratio 目标无需 Duan smearing 校正"""
        self._duan_smearing_factor = 1.0
    
    def _evaluate_model(self, name: str, model, y_true, y_pred, train_time: float,
                         use_log_transform: bool = False,
                         y_true_time: np.ndarray = None,
                         theoretical_times_test: np.ndarray = None) -> ModelResult:
        """
        评估模型（在 ratio 空间和原始时间空间同时计算指标）
        
        y_true/y_pred 均为 time_ratio 空间。
        通过 theoretical_times_test 和 y_true_time 转换到原始时间空间评估。
        """
        if HAS_TORCH and isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if HAS_TORCH and isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        y_true = np.asarray(y_true, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        r2_ratio = r2_score(y_true, y_pred)
        mae_ratio = mean_absolute_error(y_true, y_pred)
        
        tt_test = theoretical_times_test if theoretical_times_test is not None else getattr(self, '_tt_test', None)
        y_time_test = y_true_time if y_true_time is not None else getattr(self, '_y_time_test', None)
        
        if tt_test is not None and y_time_test is not None and len(tt_test) == len(y_true):
            y_pred_time = y_pred * tt_test
            y_true_time = y_time_test
            
            mae = mean_absolute_error(y_true_time, y_pred_time)
            rmse = np.sqrt(mean_squared_error(y_true_time, y_pred_time))
            r2 = r2_score(y_true_time, y_pred_time)
            
            mask = y_true_time != 0
            mape = np.mean(np.abs((y_true_time[mask] - y_pred_time[mask]) / y_true_time[mask])) * 100 if mask.any() else 0
        else:
            mae = mae_ratio
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_ratio
            mask = y_true != 0
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
        
        return ModelResult(
            model_name=name,
            train_time=train_time,
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            model=model,
            predictions=y_pred,
            r2_log=r2_ratio,
            use_log_transform=use_log_transform
        )
    
    def _print_result(self, result: ModelResult):
        """打印单个模型结果"""
        print(f"\n  {result.model_name}:")
        print(f"    训练时间: {result.train_time:.2f}秒")
        print(f"    MAE: {result.mae:.2f}秒")
        print(f"    RMSE: {result.rmse:.2f}秒")
        print(f"    R2 (原始空间): {result.r2:.4f}")
        if result.r2_log is not None:
            print(f"    R2 (log空间):   {result.r2_log:.4f}")
        print(f"    MAPE: {result.mape:.2f}%")
    
    def _print_comparison_table(self, results: Dict[str, ModelResult]):
        """打印对比表"""
        print("\n  " + "="*85)
        print("  模型对比结果")
        print("  " + "="*85)
        print(f"  {'模型':<20} {'训练时间':>10} {'MAE':>10} {'RMSE':>10} {'R2(原始)':>10} {'R2(log)':>10} {'MAPE':>10}")
        print("  " + "-"*85)
        
        for name, result in sorted(results.items(), key=lambda x: x[1].r2, reverse=True):
            r2_log_str = f"{result.r2_log:.4f}" if result.r2_log is not None else "N/A"
            print(f"  {result.model_name:<20} {result.train_time:>8.2f}s {result.mae:>10.2f} "
                  f"{result.rmse:>10.2f} {result.r2:>10.4f} {r2_log_str:>10} {result.mape:>9.2f}%")
        
        print("  " + "="*85)
    
    def _select_best_model(self, results: Dict[str, ModelResult]):
        """选择最优模型"""
        if not results:
            raise ValueError("没有可用的模型")
        
        best = max(results.items(), key=lambda x: x[1].r2)
        self.best_model_name = best[0]
        self.best_model = best[1].model
        
        print(f"\n  最优模型: {best[1].model_name}")
        print(f"  R2 得分: {best[1].r2:.4f}")
    
    # ==================== 预测与更新 ====================
    
    def _predict_all_weights(self, edge_segments: Dict, graph):
        """
        使用经验统计+模型插值混合策略预测边的动态耗时权重
        
        策略：
        - 密集时段（>=2样本）：直接使用经验均值
        - 稀疏时段（<2样本）：使用训练好的模型插值
        - 无数据边：使用模型外推（基于边结构特征）
        
        关键：模型训练在边×时段聚合级别，与预测时输入对齐。
        """
        print(f"\n  使用 [{self.best_model_name}] 模型 + 经验统计混合预测动态耗时...")
        
        # GNN 模型需要特殊处理
        if self.best_model_name == 'gnn':
            self._predict_with_gnn(edge_segments, graph)
            return
        
        # 收集图中所有边
        all_edges = set(graph.edges())
        edges_with_data = set(edge_segments.keys())
        edges_without_data = all_edges - edges_with_data
        
        if edges_without_data:
            print(f"  无轨迹数据边: {len(edges_without_data)} 条，将使用模型外推")
        
        # ===== 阶段1：为所有需要模型预测的边×小时构建特征 =====
        model_predict_features = []  # 需要模型预测的 (edge_key, hour) 列表
        model_predict_keys = []      # 对应的 (edge_key, hour) 标识
        
        # 经验数据缓存
        edge_empirical = {}  # {edge_key: {hour: (avg_time, count)}}
        
        n_empirical_hours = 0
        n_model_hours = 0
        
        for edge_key, segments in edge_segments.items():
            if len(segments) == 0:
                continue
            
            from_node, to_node = edge_key
            waterway_type = self._get_edge_waterway_type(from_node, to_node, graph)
            node_degree_from = self.node_degrees.get(from_node, 0)
            node_degree_to = self.node_degrees.get(to_node, 0)
            betweenness = self.edge_betweenness.get(edge_key, 0)
            
            # 边级静态特征
            distances = [s['distance'] for s in segments]
            avg_distance = np.mean(distances)
            avg_bearing = np.mean([s['bearing'] for s in segments])
            bearing_rad = np.deg2rad(avg_bearing)
            avg_course_change = np.mean([s['course_change'] for s in segments])
            avg_reported_speed = np.mean([s['reported_speed'] for s in segments])
            std_reported_speed = np.std([s['reported_speed'] for s in segments]) if len(segments) > 1 else 0
            speed_cv = std_reported_speed / max(avg_reported_speed, 0.1)
            total_sample_count = len(segments)
            speed_ms = max(avg_reported_speed, 0.5) * 0.5144
            theoretical_time = avg_distance / speed_ms
            
            # 按小时聚合经验数据
            hourly_data = defaultdict(list)
            period_data = defaultdict(list)
            for seg in segments:
                hourly_data[seg['hour']].append(seg['time_diff'])
                period_data[seg['time_period']].append(seg['time_diff'])
            
            overall_avg = np.mean([s['time_diff'] for s in segments])
            
            edge_empirical[edge_key] = {
                'hourly': {},
                'period': {},
                'overall_avg': overall_avg,
                'theoretical_time': theoretical_time,
                'static_features': {
                    'avg_reported_speed': avg_reported_speed,
                    'std_reported_speed': std_reported_speed,
                    'speed_cv': speed_cv,
                    'avg_bearing': avg_bearing,
                    'bearing_sin': np.sin(bearing_rad),
                    'bearing_cos': np.cos(bearing_rad),
                    'avg_course_change': avg_course_change,
                    'waterway_type': waterway_type,
                    'node_degree_from': node_degree_from,
                    'node_degree_to': node_degree_to,
                    'edge_betweenness': betweenness,
                    'sample_count': total_sample_count,
                }
            }
            
            for hour, times in hourly_data.items():
                edge_empirical[edge_key]['hourly'][hour] = (np.mean(times), len(times))
            
            for period, times in period_data.items():
                edge_empirical[edge_key]['period'][period] = (np.mean(times), len(times))
            
            # 判断哪些小时需要模型插值
            for hour in range(24):
                if hour in hourly_data and len(hourly_data[hour]) >= 2:
                    # 密集时段：用经验值
                    n_empirical_hours += 1
                else:
                    # 稀疏时段：构建特征让模型预测
                    period = self._get_time_period(hour)
                    is_peak = 1 if hour in self.peak_hours else 0
                    # 判断是否周末（用该时段的大部分数据判断）
                    weekend_segs = [s for s in segments if s['is_weekend'] == 1 and s['time_period'] == period]
                    period_segs = [s for s in segments if s['time_period'] == period]
                    is_weekend = 1 if len(weekend_segs) > len(period_segs) / 2 else 0
                    
                    sf = edge_empirical[edge_key]['static_features']
                    hour_rad = np.deg2rad(hour * 15)
                    features = [
                        sf['avg_reported_speed'],
                        sf['std_reported_speed'],
                        sf['speed_cv'],
                        sf['avg_bearing'],
                        sf['bearing_sin'],
                        sf['bearing_cos'],
                        sf['avg_course_change'],
                        0,
                        sf['avg_course_change'] * sf['waterway_type'],
                        is_peak,
                        is_weekend,
                        sf['waterway_type'],
                        hour,
                        np.sin(hour_rad),
                        np.cos(hour_rad),
                        sf['node_degree_from'],
                        sf['node_degree_to'],
                        sf['edge_betweenness'],
                        sf['sample_count'],
                        np.log1p(sf['sample_count']),
                        sf['waterway_type'] * is_peak,
                        sf['avg_course_change'] * is_peak
                    ]
                    model_predict_features.append(features)
                    model_predict_keys.append((edge_key, hour))
                    n_model_hours += 1
        
        # 为无数据边构建24小时预测特征
        for from_node, to_node in edges_without_data:
            waterway_type = self._get_edge_waterway_type(from_node, to_node, graph)
            node_degree_from = self.node_degrees.get(from_node, 0)
            node_degree_to = self.node_degrees.get(to_node, 0)
            betweenness = self.edge_betweenness.get((from_node, to_node), 0)
            
            u_attr = graph.nodes[from_node]
            v_attr = graph.nodes[to_node]
            dist = haversine_distance(u_attr['lat'], u_attr['lon'], v_attr['lat'], v_attr['lon'])
            avg_bearing = calculate_bearing(u_attr['lat'], u_attr['lon'], v_attr['lat'], v_attr['lon'])
            bearing_rad = np.deg2rad(avg_bearing)
            
            for hour in range(24):
                is_peak = 1 if hour in self.peak_hours else 0
                default_speed_ms = 2.57
                theoretical_time_no_data = dist / default_speed_ms
                hour_rad = np.deg2rad(hour * 15)
                features = [
                    5.0,
                    0.0,
                    0.0,
                    avg_bearing,
                    np.sin(bearing_rad),
                    np.cos(bearing_rad),
                    0,
                    0,
                    0,
                    is_peak,
                    0,
                    waterway_type,
                    hour,
                    np.sin(hour_rad),
                    np.cos(hour_rad),
                    node_degree_from,
                    node_degree_to,
                    betweenness,
                    0,
                    0,
                    waterway_type * is_peak,
                    0
                ]
                model_predict_features.append(features)
                model_predict_keys.append(((from_node, to_node), hour))
                n_model_hours += 1
        
        print(f"  经验覆盖: {n_empirical_hours} 边×小时 (>=2样本)")
        print(f"  模型插值: {n_model_hours} 边×小时 (稀疏/无数据)")
        
        # ===== 阶段2：批量模型预测 =====
        model_predictions = {}
        if len(model_predict_features) > 0:
            X_model = np.array(model_predict_features, dtype=np.float64)
            
            if self.best_model_name == 'mlp':
                X_scaled = self.scaler.transform(X_model)
                with torch.no_grad():
                    preds = self.best_model(torch.FloatTensor(X_scaled)).numpy().flatten()
            elif self.best_model_name in ('lightgbm', 'lightgbm_tweedie'):
                X_df = pd.DataFrame(X_model, columns=self.feature_names)
                preds = self.best_model.predict(X_df)
            else:
                preds = self.best_model.predict(X_model)
            
            preds = np.clip(preds, 0.1, 20.0)
            
            # Convert ratio predictions to travel time: predicted_time = ratio * theoretical_time
            for i, (edge_key, hour) in enumerate(model_predict_keys):
                tt = edge_empirical.get(edge_key, {}).get('theoretical_time')
                if tt is None:
                    from_node, to_node = edge_key
                    if graph.has_edge(from_node, to_node):
                        u_attr = graph.nodes[from_node]
                        v_attr = graph.nodes[to_node]
                        dist = haversine_distance(u_attr['lat'], u_attr['lon'], v_attr['lat'], v_attr['lon'])
                        tt = dist / 2.57
                    else:
                        tt = 100.0
                model_predictions[(edge_key, hour)] = float(preds[i]) * tt
            
            pred_times = np.array(list(model_predictions.values()))
            print(f"  模型预测完成，范围: [{pred_times.min():.1f}, {pred_times.max():.1f}]秒")
        
        # ===== 阶段3：组装边特征 =====
        n_edges = 0
        n_model_used = 0
        
        for edge_key, segments in edge_segments.items():
            if len(segments) == 0:
                continue
            
            n_edges += 1
            from_node, to_node = edge_key
            waterway_type = self._get_edge_waterway_type(from_node, to_node, graph)
            
            time_diffs = [s['time_diff'] for s in segments]
            actual_speeds = [s['actual_speed'] for s in segments]
            reported_speeds = [s['reported_speed'] for s in segments]
            bearings = [s['bearing'] for s in segments]
            
            avg_distance = np.mean([s['distance'] for s in segments])
            avg_actual_speed = np.mean(actual_speeds)
            avg_reported_speed = np.mean(reported_speeds)
            avg_bearing = np.mean(bearings)
            avg_course_change = np.mean([s['course_change'] for s in segments])
            std_bearing = np.std(bearings) if len(bearings) > 1 else 0
            node_degree_from = self.node_degrees.get(from_node, 0)
            node_degree_to = self.node_degrees.get(to_node, 0)
            betweenness = self.edge_betweenness.get(edge_key, 0)
            
            overall_avg = np.mean(time_diffs)
            emp = edge_empirical.get(edge_key, {})
            
            # 混合策略：经验 + 模型插值
            predicted_times = {}
            for hour in range(24):
                key = (edge_key, hour)
                if key in model_predictions:
                    # 模型插值
                    predicted_times[hour] = model_predictions[key]
                    n_model_used += 1
                elif hour in emp.get('hourly', {}):
                    # 经验值
                    predicted_times[hour] = emp['hourly'][hour][0]
                else:
                    period = self._get_time_period(hour)
                    if period in emp.get('period', {}):
                        predicted_times[hour] = emp['period'][period][0]
                    else:
                        predicted_times[hour] = overall_avg
            
            time_period_weights = self._compute_time_period_weights_for_edge(segments)
            direction_features = {
                'avg_bearing': avg_bearing,
                'std_bearing': std_bearing,
                'avg_course_change': avg_course_change,
                'direction_distribution': self._compute_direction_distribution(bearings),
                'is_bidirectional': self._check_bidirectional(bearings)
            }

            self.edge_features[edge_key] = {
                'segment_count': len(segments),
                'from_node': from_node,
                'to_node': to_node,
                'avg_distance': avg_distance,
                'avg_travel_time': overall_avg,
                'std_travel_time': np.std(time_diffs) if len(time_diffs) > 1 else 0,
                'min_travel_time': np.min(time_diffs),
                'max_travel_time': np.max(time_diffs),
                'median_travel_time': np.median(time_diffs),
                'avg_actual_speed': avg_actual_speed,
                'std_actual_speed': np.std(actual_speeds) if len(actual_speeds) > 1 else 0,
                'avg_reported_speed': avg_reported_speed,
                'speed_reliability': max(0, 1 - (np.mean(np.abs(np.array(actual_speeds) - np.array(reported_speeds))) / max(avg_reported_speed, avg_actual_speed, 1e-6))),
                'waterway_type': 'narrow' if waterway_type == 1 else 'open',
                'waterway_type_code': waterway_type,
                'predicted_times': predicted_times,
                'predicted_time_morning': predicted_times.get(8, overall_avg),
                'predicted_time_midday': predicted_times.get(12, overall_avg),
                'predicted_time_evening': predicted_times.get(18, overall_avg),
                'predicted_time_night': predicted_times.get(0, overall_avg),
                'time_period_weights': time_period_weights,
                'direction_features': direction_features,
                'node_degree_from': node_degree_from,
                'node_degree_to': node_degree_to,
                'edge_betweenness': betweenness,
                'model_used': 'hybrid'
            }
        
        # 为无轨迹数据的边用模型外推
        for from_node, to_node in edges_without_data:
            waterway_type = self._get_edge_waterway_type(from_node, to_node, graph)
            node_degree_from = self.node_degrees.get(from_node, 0)
            node_degree_to = self.node_degrees.get(to_node, 0)
            betweenness = self.edge_betweenness.get((from_node, to_node), 0)
            
            u_attr = graph.nodes[from_node]
            v_attr = graph.nodes[to_node]
            dist = haversine_distance(u_attr['lat'], u_attr['lon'], v_attr['lat'], v_attr['lon'])
            avg_bearing = calculate_bearing(u_attr['lat'], u_attr['lon'], v_attr['lat'], v_attr['lon'])
            
            predicted_times = {}
            for hour in range(24):
                key = ((from_node, to_node), hour)
                if key in model_predictions:
                    predicted_times[hour] = model_predictions[key]
                else:
                    # fallback：距离/默认速度
                    is_peak = 1 if hour in self.peak_hours else 0
                    peak_factor = 1.2 if is_peak else 1.0
                    predicted_times[hour] = (dist / 2.57) * peak_factor
            
            default_time = dist / 2.57
            
            self.edge_features[(from_node, to_node)] = {
                'segment_count': 0,
                'from_node': from_node,
                'to_node': to_node,
                'avg_distance': dist,
                'avg_travel_time': default_time,
                'std_travel_time': default_time * 0.2,
                'min_travel_time': default_time * 0.8,
                'max_travel_time': default_time * 1.2,
                'median_travel_time': default_time,
                'avg_actual_speed': 5.0,
                'std_actual_speed': 1.0,
                'avg_reported_speed': 5.0,
                'speed_reliability': 0.5,
                'waterway_type': 'narrow' if waterway_type == 1 else 'open',
                'waterway_type_code': waterway_type,
                'predicted_times': predicted_times,
                'predicted_time_morning': predicted_times.get(8, default_time),
                'predicted_time_midday': predicted_times.get(12, default_time),
                'predicted_time_evening': predicted_times.get(18, default_time),
                'predicted_time_night': predicted_times.get(0, default_time),
                'time_period_weights': {},
                'direction_features': {
                    'avg_bearing': avg_bearing,
                    'std_bearing': 0,
                    'avg_course_change': 0,
                    'direction_distribution': {},
                    'is_bidirectional': False
                },
                'node_degree_from': node_degree_from,
                'node_degree_to': node_degree_to,
                'edge_betweenness': betweenness,
                'model_used': 'model_extrapolation'
            }
        
        print(f"  完成: {n_edges} 条有数据边 + {len(edges_without_data)} 条无数据边")
        print(f"  模型插值使用: {n_model_used} 个边×小时槽位")
    
    def _predict_with_gnn(self, edge_segments: Dict, graph):
        """使用 GNN 模型预测（混合策略：经验 + 模型插值）"""
        # 重新构建图数据用于预测
        valid_edges = []
        edge_features_list = []
        
        for edge_key, segments in edge_segments.items():
            if len(segments) == 0:
                continue
            
            from_node, to_node = edge_key
            avg_reported_speed = np.mean([s['reported_speed'] for s in segments])
            distance = np.mean([s['distance'] for s in segments])
            avg_bearing = np.mean([s['bearing'] for s in segments])
            avg_course_change = np.mean([s['course_change'] for s in segments])
            waterway_type = self._get_edge_waterway_type(from_node, to_node, graph)
            node_degree_from = self.node_degrees.get(from_node, 0)
            node_degree_to = self.node_degrees.get(to_node, 0)
            betweenness = self.edge_betweenness.get(edge_key, 0)
            
            # 物理先验：理论耗时
            speed_ms = max(avg_reported_speed, 0.5) * 0.5144
            theoretical_time = distance / speed_ms
            
            bearing_rad = np.deg2rad(avg_bearing)
            valid_edges.append((from_node, to_node, segments))
            edge_features_list.append([
                distance, theoretical_time, avg_reported_speed,
                np.sin(bearing_rad), np.cos(bearing_rad), avg_course_change,
                waterway_type, node_degree_from, node_degree_to, betweenness
            ])
        
        if len(valid_edges) == 0:
            return
        
        # 构建节点特征：使用图中所有节点，确保edge_index不会越界
        try:
            import networkx as nx
            clustering = nx.clustering(graph)
        except:
            clustering = {}
        
        node_features = []
        node_id_to_idx = {}
        
        # 使用图中所有节点，而不是只有数据边的节点
        all_nodes = sorted(graph.nodes())
        for idx, node_id in enumerate(all_nodes):
            node_id_to_idx[node_id] = idx
            attrs = graph.nodes[node_id]
            degree = self.node_degrees.get(node_id, 0)
            waterway = self.node_waterway_types.get(node_id, 0)
            cluster_coeff = clustering.get(node_id, 0)
            node_features.append([degree, waterway, attrs['lat'], attrs['lon'], cluster_coeff])
        
        node_features_np = np.array(node_features)
        if hasattr(self, 'gnn_node_scaler'):
            node_features_np = self.gnn_node_scaler.transform(node_features_np)
        node_features = torch.FloatTensor(node_features_np)
        
        # 边索引（双向）
        n_edges = len(valid_edges)
        edge_index_forward = []
        for u, v, _ in valid_edges:
            edge_index_forward.append([node_id_to_idx[u], node_id_to_idx[v]])
        edge_index_bidir = edge_index_forward + [[v, u] for u, v in edge_index_forward]
        edge_index = torch.LongTensor(edge_index_bidir).t().contiguous()
        
        edge_features_np = np.array(edge_features_list)
        if hasattr(self, 'gnn_edge_scaler'):
            edge_features_np = self.gnn_edge_scaler.transform(edge_features_np)
        edge_features_tensor = torch.FloatTensor(edge_features_np)
        
        # 预测
        self.best_model.eval()
        with torch.no_grad():
            predictions = self.best_model(node_features, edge_index, edge_features_tensor, n_edges).numpy()
        
        # Convert ratio predictions to travel time
        predictions = np.maximum(predictions, 0.1)
        
        # Get theoretical times for each edge
        for i, (u, v, segments) in enumerate(valid_edges):
            edge_key = (u, v)
            tt = self._edge_theoretical_times.get(edge_key, 100.0)
            predictions[i] = predictions[i] * tt
        
        # 存储结果（混合策略）
        n_model_used = 0
        for i, (u, v, segments) in enumerate(valid_edges):
            edge_key = (u, v)
            gnn_pred_time = float(predictions[i])
            
            time_diffs = [s['time_diff'] for s in segments]
            actual_speeds = [s['actual_speed'] for s in segments]
            overall_avg = np.mean(time_diffs)
            
            waterway_type = self._get_edge_waterway_type(u, v, graph)
            
            # 混合策略：经验值 + GNN 预测
            hourly_data = defaultdict(list)
            period_data = defaultdict(list)
            for seg in segments:
                hourly_data[seg['hour']].append(seg['time_diff'])
                period_data[seg['time_period']].append(seg['time_diff'])
            
            predicted_times = {}
            for hour in range(24):
                if hour in hourly_data and len(hourly_data[hour]) >= 2:
                    predicted_times[hour] = np.mean(hourly_data[hour])
                elif hour in hourly_data and len(hourly_data[hour]) >= 1:
                    # 只有1个样本：用经验值和GNN预测的加权平均
                    empirical = np.mean(hourly_data[hour])
                    predicted_times[hour] = 0.7 * empirical + 0.3 * gnn_pred_time
                    n_model_used += 1
                else:
                    period = self._get_time_period(hour)
                    if period in period_data and len(period_data[period]) >= 2:
                        predicted_times[hour] = np.mean(period_data[period])
                    else:
                        # 无数据时段：用GNN预测
                        predicted_times[hour] = gnn_pred_time
                        n_model_used += 1
            
            self.edge_features[edge_key] = {
                'segment_count': len(segments),
                'from_node': u,
                'to_node': v,
                'avg_distance': np.mean([s['distance'] for s in segments]),
                'avg_travel_time': overall_avg,
                'std_travel_time': np.std(time_diffs) if len(time_diffs) > 1 else 0,
                'min_travel_time': np.min(time_diffs),
                'max_travel_time': np.max(time_diffs),
                'median_travel_time': np.median(time_diffs),
                'avg_actual_speed': np.mean(actual_speeds),
                'std_actual_speed': np.std(actual_speeds) if len(actual_speeds) > 1 else 0,
                'avg_reported_speed': np.mean([s['reported_speed'] for s in segments]),
                'speed_reliability': max(0, 1 - np.mean([s['speed_diff'] for s in segments]) / max(np.mean([s['reported_speed'] for s in segments]), np.mean(actual_speeds), 1e-6)),
                'waterway_type': 'narrow' if waterway_type == 1 else 'open',
                'waterway_type_code': waterway_type,
                'predicted_times': predicted_times,
                'predicted_time_morning': predicted_times.get(8, overall_avg),
                'predicted_time_midday': predicted_times.get(12, overall_avg),
                'predicted_time_evening': predicted_times.get(18, overall_avg),
                'predicted_time_night': predicted_times.get(0, overall_avg),
                'time_period_weights': self._compute_time_period_weights_for_edge(segments),
                'direction_features': {
                    'avg_bearing': np.mean([s['bearing'] for s in segments]),
                    'std_bearing': np.std([s['bearing'] for s in segments]) if len(segments) > 1 else 0,
                    'avg_course_change': np.mean([s['course_change'] for s in segments]),
                    'direction_distribution': self._compute_direction_distribution([s['bearing'] for s in segments]),
                    'is_bidirectional': self._check_bidirectional([s['bearing'] for s in segments])
                },
                'node_degree_from': self.node_degrees.get(u, 0),
                'node_degree_to': self.node_degrees.get(v, 0),
                'edge_betweenness': self.edge_betweenness.get(edge_key, 0),
                'model_used': 'hybrid_gnn'
            }
        
        print(f"  GNN 混合预测: {n_model_used} 个稀疏时段使用模型插值")
    
    def _compute_time_period_weights_for_edge(self, segments: List[Dict]) -> Dict:
        """计算边的时段权重"""
        period_data = defaultdict(list)
        for seg in segments:
            period_data[seg['time_period']].append(seg['time_diff'])
        
        base_time = np.mean([s['time_diff'] for s in segments])
        time_period_weights = {}
        
        for period, times in period_data.items():
            if times:
                avg_time = np.mean(times)
                weight_ratio = avg_time / base_time if base_time > 0 else 1.0
                time_period_weights[period] = {
                    'avg_travel_time': avg_time,
                    'std_travel_time': np.std(times) if len(times) > 1 else 0,
                    'sample_count': len(times),
                    'weight_ratio': weight_ratio
                }
        
        return time_period_weights
    
    def _compute_direction_distribution(self, bearings: List[float]) -> Dict[int, int]:
        """计算方向分布"""
        direction_bins = defaultdict(int)
        for bearing in bearings:
            bin_idx = int(bearing / 45) % 8
            direction_bins[bin_idx] += 1
        return dict(direction_bins)
    
    def _check_bidirectional(self, bearings: List[float]) -> bool:
        """检查是否为双向航道"""
        direction_bins = self._compute_direction_distribution(bearings)
        for d1 in direction_bins:
            opposite = (d1 + 4) % 8
            if opposite in direction_bins:
                ratio = min(direction_bins[d1], direction_bins[opposite]) / \
                        max(direction_bins[d1], direction_bins[opposite])
                if ratio > 0.3:
                    return True
        return False
    
    def _update_graph_edges(self, graph):
        """更新图边属性"""
        for (from_node, to_node), features in self.edge_features.items():
            if graph.has_edge(from_node, to_node):
                graph[from_node][to_node].update({
                    'avg_travel_time': features['avg_travel_time'],
                    'predicted_times': features.get('predicted_times', {}),
                    'waterway_type': features['waterway_type'],
                    'model': self.best_model_name
                })
    
    # ==================== 导出 ====================
    
    def export_results(self, output_dir: str):
        """
        导出带动态耗时标签的高质量路段特征数据集
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n导出带动态耗时标签的路段特征数据集...")
        
        # 1. 导出完整的边特征数据集
        rows = []
        for (from_node, to_node), features in self.edge_features.items():
            row = {
                'from_node': from_node,
                'to_node': to_node,
                'model_used': features.get('model_used', self.best_model_name),
                'segment_count': features['segment_count'],
                'avg_distance': features['avg_distance'],
                
                # 实际耗时统计
                'avg_travel_time': features['avg_travel_time'],
                'std_travel_time': features['std_travel_time'],
                'min_travel_time': features.get('min_travel_time', features['avg_travel_time']),
                'max_travel_time': features.get('max_travel_time', features['avg_travel_time']),
                'median_travel_time': features.get('median_travel_time', features['avg_travel_time']),
                
                # 速度特征
                'avg_actual_speed': features['avg_actual_speed'],
                'std_actual_speed': features.get('std_actual_speed', 0),
                'avg_reported_speed': features.get('avg_reported_speed', 0),
                'speed_reliability': features.get('speed_reliability', 0),
                
                # 水域类型
                'waterway_type': features['waterway_type'],
                'waterway_type_code': features.get('waterway_type_code', 0),
                
                # 网络拓扑特征
                'node_degree_from': features.get('node_degree_from', 0),
                'node_degree_to': features.get('node_degree_to', 0),
                'edge_betweenness': features.get('edge_betweenness', 0),
            }
            
            # 动态耗时预测 - 24小时完整数据
            if 'predicted_times' in features:
                for hour in range(24):
                    row[f'predicted_time_h{hour:02d}'] = features['predicted_times'].get(hour, features['avg_travel_time'])
            
            # 关键时段预测值（便于分析）
            row['predicted_time_morning'] = features.get('predicted_time_morning', 0)
            row['predicted_time_midday'] = features.get('predicted_time_midday', 0)
            row['predicted_time_evening'] = features.get('predicted_time_evening', 0)
            row['predicted_time_night'] = features.get('predicted_time_night', 0)
            
            # 时段权重
            time_period_weights = features.get('time_period_weights', {})
            for period in ['night', 'morning', 'midday', 'afternoon', 'evening', 'late_night']:
                period_data = time_period_weights.get(period, {})
                row[f'{period}_avg_time'] = period_data.get('avg_travel_time', 0)
                row[f'{period}_weight_ratio'] = period_data.get('weight_ratio', 1.0)
                row[f'{period}_sample_count'] = period_data.get('sample_count', 0)
            
            # 方向特征
            direction_features = features.get('direction_features', {})
            row['avg_bearing'] = direction_features.get('avg_bearing', 0)
            row['std_bearing'] = direction_features.get('std_bearing', 0)
            row['avg_course_change'] = direction_features.get('avg_course_change', 0)
            row['is_bidirectional'] = int(direction_features.get('is_bidirectional', False))
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        output_path = f"{output_dir}/edge_features_dynamic_weights.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n  主数据集: {output_path}")
        print(f"  边数量: {len(rows):,}")
        print(f"  特征列数: {len(df.columns)}")
        
        # 2. 导出24小时动态耗时矩阵（便于热力图分析）
        time_matrix_path = f"{output_dir}/dynamic_time_matrix.csv"
        time_rows = []
        for (from_node, to_node), features in self.edge_features.items():
            if 'predicted_times' in features:
                time_row = {
                    'edge_id': f"{from_node}_{to_node}",
                    'from_node': from_node,
                    'to_node': to_node
                }
                for hour in range(24):
                    time_row[f'h{hour:02d}'] = features['predicted_times'].get(hour, features['avg_travel_time'])
                time_rows.append(time_row)
        
        time_df = pd.DataFrame(time_rows)
        time_df.to_csv(time_matrix_path, index=False, encoding='utf-8-sig')
        print(f"  动态耗时矩阵: {time_matrix_path}")
        
        # 3. 导出模型评估报告
        report_path = f"{output_dir}/model_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("动态路段耗时权重建模报告\n")
            f.write("="*80 + "\n\n")
            f.write(f"最优模型: {self.best_model_name}\n")
            f.write(f"处理边数: {len(self.edge_features):,}\n\n")
            
            # 统计信息
            if self.edge_features:
                avg_times = [f['avg_travel_time'] for f in self.edge_features.values()]
                f.write(f"平均耗时: {np.mean(avg_times):.2f} 秒\n")
                f.write(f"耗时标准差: {np.std(avg_times):.2f} 秒\n")
                f.write(f"最小耗时: {np.min(avg_times):.2f} 秒\n")
                f.write(f"最大耗时: {np.max(avg_times):.2f} 秒\n\n")
                
                # 水域类型分布
                narrow_count = sum(1 for f in self.edge_features.values() if f.get('waterway_type') == 'narrow')
                f.write(f"狭窄水道边数: {narrow_count} ({narrow_count/len(self.edge_features)*100:.1f}%)\n")
                f.write(f"开阔海面边数: {len(self.edge_features) - narrow_count} ({(1-narrow_count/len(self.edge_features))*100:.1f}%)\n\n")
            
            # 多模型对比详情
            if hasattr(self, '_model_results') and self._model_results:
                f.write("-"*80 + "\n")
                f.write("多模型对比评估\n")
                f.write("-"*80 + "\n\n")
                f.write(f"{'模型':<20s} {'MAE':>10s} {'RMSE':>10s} {'R²':>10s} {'MAPE(%)':>10s} {'训练时间(s)':>12s}\n")
                f.write("-"*72 + "\n")
                for name, result in self._model_results.items():
                    best_mark = " ★" if name == self.best_model_name else ""
                    f.write(f"{name+best_mark:<20s} {result.mae:>10.4f} {result.rmse:>10.4f} {result.r2:>10.4f} {result.mape:>10.2f} {result.train_time:>12.2f}\n")
                f.write("-"*72 + "\n")
                f.write("★ 标记为最优模型\n\n")
            
            # 特征重要性 Top10
            if hasattr(self, 'feature_importance') and self.feature_importance:
                f.write("-"*80 + "\n")
                f.write("特征重要性 Top10\n")
                f.write("-"*80 + "\n\n")
                sorted_fi = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                f.write(f"{'排名':<6s} {'特征名':<30s} {'重要性':>10s}\n")
                f.write("-"*48 + "\n")
                for rank, (feat, imp) in enumerate(sorted_fi, 1):
                    f.write(f"{rank:<6d} {feat:<30s} {imp:>10.6f}\n")
        
        print(f"  模型报告: {report_path}")
        
        # 4. 导出特征重要性（如果可用）
        if hasattr(self, 'feature_importance') and self.feature_importance:
            importance_path = f"{output_dir}/feature_importance.csv"
            importance_df = pd.DataFrame([
                {'feature': k, 'importance': v} 
                for k, v in sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            ])
            importance_df.to_csv(importance_path, index=False, encoding='utf-8-sig')
            print(f"  特征重要性: {importance_path}")
        
        print(f"\n数据集导出完成!")
        return df

    # ==================== 模型保存与加载 ====================
    
    def save_model(self, output_dir: str, filename: str = None) -> str:
        """
        保存训练好的模型及相关配置
        
        Args:
            output_dir: 输出目录
            filename: 文件名（可选，默认自动生成）
        
        Returns:
            保存的文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if filename is None:
            filename = f"weight_model_{self.best_model_name}.pkl"
        
        filepath = os.path.join(output_dir, filename)
        
        # 清理 GNN 模型对象（model_results 中的 GNN 不保存模型对象）
        model_results_clean = {}
        if hasattr(self, '_model_results') and self._model_results:
            for name, result in self._model_results.items():
                if name == 'gnn':
                    # 对于 GNN，只保存元数据，不保存模型对象
                    model_results_clean[name] = ModelResult(
                        model_name=result.model_name,
                        train_time=result.train_time,
                        mae=result.mae,
                        rmse=result.rmse,
                        r2=result.r2,
                        r2_log=getattr(result, 'r2_log', None),
                        mape=result.mape,
                        use_log_transform=getattr(result, 'use_log_transform', True),
                        model=None,  # 不保存模型对象
                        predictions=result.predictions
                    )
                else:
                    model_results_clean[name] = result

        # GNN 最佳模型单独用 torch.save 保存 state_dict
        best_model_to_save = self.best_model
        gnn_state_path = None
        gnn_model_config = None
        if self.best_model_name == 'gnn' and self.best_model is not None:
            best_model_to_save = None  # pkl 中不保存 GNN 模型对象
            gnn_state_path = filepath.replace('.pkl', '_gnn_state.pt')
            torch.save(self.best_model.state_dict(), gnn_state_path)
            # 保存模型结构参数以便加载时重建
            gnn_model_config = {
                'node_dim': self.best_model.convs[0].in_channels,
                'edge_dim': self.best_model.edge_mlp[0].in_features - self.best_model.convs[-1].out_channels * 2,
                'hidden_dim': self.best_model.convs[0].out_channels,
                'num_layers': len(self.best_model.convs),
                'dropout': self.best_model.dropout.p,
            }

        # 构建保存字典
        model_data = {
            'best_model': best_model_to_save,
            'best_model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'scaler': getattr(self, 'scaler', None),
            'y_original_stats': None,
            'duan_smearing_factor': getattr(self, '_duan_smearing_factor', 1.0),
            'edge_theoretical_times': getattr(self, '_edge_theoretical_times', {}),
            'node_degrees': getattr(self, 'node_degrees', None),
            'edge_betweenness': getattr(self, 'edge_betweenness', None),
            'node_waterway_types': getattr(self, 'node_waterway_types', None),
            'edge_waterway_types': getattr(self, 'edge_waterway_types', None),
            'density_grid': getattr(self, '_density_grid', None),
            'density_grid_size': getattr(self, '_density_grid_size', None),
            'density_threshold': getattr(self, '_density_threshold', None),
            'feature_importance': getattr(self, 'feature_importance', None),
            'time_periods': self.time_periods,
            'peak_hours': self.peak_hours,
            'use_grid_search': getattr(self, 'use_grid_search', True),
            'model_results': model_results_clean,
        }
        
        # GNN 模型特殊处理
        if self.best_model_name == 'gnn':
            model_data['gnn_node_scaler'] = getattr(self, 'gnn_node_scaler', None)
            model_data['gnn_edge_scaler'] = getattr(self, 'gnn_edge_scaler', None)
            model_data['gnn_model_config'] = gnn_model_config
            model_data['gnn_use_log_transform'] = getattr(self, '_gnn_use_log_transform', True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        total_size = os.path.getsize(filepath)
        if gnn_state_path and os.path.exists(gnn_state_path):
            total_size += os.path.getsize(gnn_state_path)
        
        print(f"\n模型已保存: {filepath}")
        print(f"  模型类型: {self.best_model_name}")
        print(f"  文件大小: {total_size / 1024:.1f} KB")
        
        return filepath
    
    def load_model(self, filepath: str):
        """加载已保存的模型"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.feature_names = model_data['feature_names']
        self.scaler = model_data.get('scaler')
        self._duan_smearing_factor = model_data.get('duan_smearing_factor', 1.0)
        self._edge_theoretical_times = model_data.get('edge_theoretical_times', {})
        self.node_degrees = model_data.get('node_degrees', {})
        self.edge_betweenness = model_data.get('edge_betweenness', {})
        self.node_waterway_types = model_data.get('node_waterway_types', {})
        self.edge_waterway_types = model_data.get('edge_waterway_types', {})
        self._density_grid = model_data.get('density_grid')
        self._density_grid_size = model_data.get('density_grid_size')
        self._density_threshold = model_data.get('density_threshold')
        self.feature_importance = model_data.get('feature_importance')
        self.time_periods = model_data.get('time_periods', self.time_periods)
        self.peak_hours = model_data.get('peak_hours', self.peak_hours)
        self.use_grid_search = model_data.get('use_grid_search', True)
        self._model_results = model_data.get('model_results')
        
        if self.best_model_name == 'gnn':
            self.gnn_node_scaler = model_data.get('gnn_node_scaler')
            self.gnn_edge_scaler = model_data.get('gnn_edge_scaler')
            self._gnn_use_log_transform = model_data.get('gnn_use_log_transform', True)
            # 从 state_dict 重建 GNN 模型
            gnn_config = model_data.get('gnn_model_config')
            gnn_state_path = filepath.replace('.pkl', '_gnn_state.pt')
            if gnn_config and os.path.exists(gnn_state_path):
                self.best_model = EdgeGNN(**gnn_config)
                self.best_model.load_state_dict(torch.load(gnn_state_path, map_location='cpu', weights_only=True))
                self.best_model.eval()
                print(f"  GNN 模型已从 state_dict 重建")
        
        print(f"\n模型已加载: {filepath}")
        print(f"  模型类型: {self.best_model_name}")
    
    def predict_with_loaded_model(self, graph, trajectories_df: pd.DataFrame) -> Dict:
        """使用已加载的模型预测边权重（跳过训练）"""
        if self.best_model is None:
            raise ValueError("请先加载模型")
        
        logger.info("使用已加载模型 [%s] 预测边权重", self.best_model_name)
        
        segment_features = self._extract_segment_features(trajectories_df)
        self._compute_waterway_types(segment_features, graph)
        self._compute_network_features(graph)
        edge_segments = self._map_segments_to_edges(graph, segment_features)
        
        if self.best_model_name == 'gnn':
            self._predict_with_gnn(edge_segments, graph)
        else:
            self._predict_all_weights(edge_segments, graph)
        
        self._update_graph_edges(graph)
        return self.edge_features
    
    def export_model_metadata(self, output_dir: str):
        """
        导出模型元数据（JSON 格式，便于查看）
        """
        os.makedirs(output_dir, exist_ok=True)
        
        metadata = {
            'model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names),
            'time_periods': self.time_periods,
            'peak_hours': sorted(list(self.peak_hours)),
            'use_grid_search': getattr(self, 'use_grid_search', True),
            'training_timestamp': datetime.now().isoformat(),
        }
        
        if hasattr(self, 'feature_importance') and self.feature_importance:
            metadata['feature_importance'] = dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
        
        if hasattr(self, '_model_results') and self._model_results:
            metadata['model_comparison'] = {
                name: {
                    'mae': result.mae,
                    'rmse': result.rmse,
                    'r2': result.r2,
                    'mape': result.mape,
                    'train_time': result.train_time,
                }
                for name, result in self._model_results.items()
            }
        
        metadata_path = os.path.join(output_dir, 'model_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"  模型元数据: {metadata_path}")

    # ==================== 动态权重查询接口 ====================
    
    def get_dynamic_weight(self, from_node: int, to_node: int, 
                           hour: int = None, 
                           time_period: str = None) -> Optional[float]:
        """
        获取动态耗时权重
        
        Args:
            from_node: 起始节点
            to_node: 目标节点
            hour: 出发小时（0-23），优先使用
            time_period: 时段名称（night/morning/midday/afternoon/evening/late_night）
        
        Returns:
            预估耗时（秒）
        """
        edge_key = (from_node, to_node)
        
        if edge_key not in self.edge_features:
            return None
        
        features = self.edge_features[edge_key]
        
        # 使用混合策略（经验+模型插值）的24小时动态耗时
        if hour is not None and 'predicted_times' in features:
            return features['predicted_times'].get(hour, features['avg_travel_time'])
        
        # 使用时段权重
        if time_period is not None:
            period_weights = features.get('time_period_weights', {})
            if time_period in period_weights:
                return period_weights[time_period].get('avg_travel_time', features['avg_travel_time'])
        
        # 返回平均耗时
        return features['avg_travel_time']
    
    def get_edge_info(self, from_node: int, to_node: int) -> Optional[Dict]:
        """
        获取边的完整信息
        
        Args:
            from_node: 起始节点
            to_node: 目标节点
        
        Returns:
            边特征字典
        """
        return self.edge_features.get((from_node, to_node))
    
    def get_peak_off_peak_ratio(self, from_node: int, to_node: int) -> Optional[float]:
        """
        获取高峰/非高峰耗时比率
        
        Returns:
            比率（>1 表示高峰时段耗时更长）
        """
        edge_key = (from_node, to_node)
        
        if edge_key not in self.edge_features:
            return None
        
        features = self.edge_features[edge_key]
        
        if 'predicted_times' not in features:
            return None
        
        # 早高峰平均耗时
        peak_times = [features['predicted_times'].get(h, 0) for h in range(7, 10)]
        peak_avg = np.mean(peak_times) if peak_times else 0
        
        # 深夜平均耗时
        off_peak_times = [features['predicted_times'].get(h, 0) for h in range(0, 5)]
        off_peak_avg = np.mean(off_peak_times) if off_peak_times else 0
        
        if off_peak_avg > 0:
            return peak_avg / off_peak_avg
        
        return None


if __name__ == '__main__':
    import sys
    import os
    import networkx as nx
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from data_preprocessor import DataPreprocessor
    
    print("=" * 60)
    print("Task5: 动态路段耗时权重建模 (time_ratio 改进版)")
    print("=" * 60)
    
    preprocessor = DataPreprocessor()

    cleaned_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'cleaned_data.csv')
    if os.path.exists(cleaned_path):
        print(f"Loading cleaned data from {cleaned_path}...")
        cleaned_df = pd.read_csv(cleaned_path)
        cleaned_df['时间'] = pd.to_datetime(cleaned_df['时间'])
        print(f"Loaded {len(cleaned_df)} rows")
    else:
        print("Cleaned data not found, running preprocessing...")
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Data')
        file_paths = [
            os.path.join(data_dir, '基于海量轨迹数据的船舶智能导航路径规划数据集构建与应用1_20260401204631.xlsx'),
            os.path.join(data_dir, '基于海量轨迹数据的船舶智能导航路径规划数据集构建与应用2_20260401204651.xlsx')
        ]
        cleaned_df = preprocessor.load_data(file_paths)
        if cleaned_df is not None:
            cleaned_df = preprocessor.preprocess()
    
    topo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'waterway_topology.json')
    if not os.path.exists(topo_path):
        print(f"ERROR: {topo_path} not found. Run topology builder first.")
        sys.exit(1)

    print(f"Loading topology from {topo_path}...")
    with open(topo_path, 'r', encoding='utf-8') as f:
        topo_data = json.load(f)

    graph = nx.DiGraph()
    for node in topo_data['nodes']:
        graph.add_node(node['id'], **{k: v for k, v in node.items() if k != 'id'})
    for edge in topo_data['edges']:
        graph.add_edge(edge['from'], edge['to'], **{k: v for k, v in edge.items() if k not in ('from', 'to')})
    print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    model = AdvancedWeightModel()
    edge_features = model.build_weights_with_comparison(graph, cleaned_df, use_grid_search=True)
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    model.export_results(output_dir)
    model.save_model(output_dir)
    model.export_model_metadata(output_dir)
    
    print("\n" + "=" * 60)
    print("特征重要性分布:")
    if hasattr(model, 'feature_importance') and model.feature_importance:
        total = sum(model.feature_importance.values())
        for feat, imp in sorted(model.feature_importance.items(), key=lambda x: x[1], reverse=True):
            pct = imp / total * 100 if total > 0 else 0
            bar = '#' * int(pct)
            print(f"  {feat:<25} {pct:>6.1f}% {bar}")
    print("=" * 60)
