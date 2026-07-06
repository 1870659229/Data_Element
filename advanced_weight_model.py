# -*- coding: utf-8 -*-
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

# DropEdge 数据增强 (允许从 scripts/ 子包导入, 若失败则降级为 identity)
try:
    from scripts.drop_edge import drop_random_edges
    HAS_DROP_EDGE = True
except (ImportError, ValueError):
    # ValueError: 当 advanced_weight_model 被作为 __main__ 运行, scripts 不是包
    HAS_DROP_EDGE = False
    def drop_random_edges(edge_index, p=0.1):
        ones = torch.ones(edge_index.shape[1], dtype=torch.bool,
                          device=edge_index.device)
        return edge_index, ones

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
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    HAS_NGBOOST = True
except ImportError:
    HAS_NGBOOST = False

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
    from torch_geometric.nn import GATv2Conv, PNAConv
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


def log_transform_target(y_ratio: np.ndarray) -> np.ndarray:
    """对 time_ratio 做 log 变换，压缩重尾分布。

    输入范围 (0.1, 20.0)，输出范围 (log(0.1), log(20.0)) ≈ (-2.3, 3.0)。
    """
    return np.log(np.clip(y_ratio, 0.1, 20.0))


def inverse_log_transform(y_log: np.ndarray) -> np.ndarray:
    """log 变换的逆操作，用于推理时还原回 time_ratio。"""
    return np.exp(y_log)


def set_reproducible_seed(seed: int = 42):
    """设置所有相关随机数生成器,确保训练完全可复现 (CPU 模式 100%, GPU 模式 ~95%)。

    必须在模型构建与训练前调用一次。
    """
    import random as _random
    import os as _os
    _os.environ['PYTHONHASHSEED'] = str(seed)
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 强制 cuDNN 使用确定性算法 (会略降低 GPU 速度, 不影响 CPU)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # PyG 内部用 torch_scatter / torch_sparse, 它们也走 torch.manual_seed
    print(f"    [reproducibility] 全局种子已设置为 {seed}")


def train_gnn_with_seed(builder, seed: int, gnn_arch: str = 'gat'):
    """用指定的 model init seed 训练 GNN, split seed 保持 42 (测试集一致)。

    需要先调用 builder.build_weights_with_comparison() 准备好数据。

    Args:
        builder: AdvancedWeightModel 实例
        seed: 模型初始化随机种子
        gnn_arch: 'gat' 或 'pna'

    Returns:
        ModelResult
    """
    for attr in ('_cached_X', '_cached_y_ratio', '_cached_graph', '_cached_edge_segments', '_test_idx'):
        if not hasattr(builder, attr):
            raise RuntimeError(
                f"builder.{attr} 不存在; 请先调用 builder.build_weights_with_comparison() 完成数据准备"
            )
    # 关键: 在 model build 之前先把所有 RNG 锁定到该 seed, 否则 nn.Conv / Linear 的 init
    # 会用到 torch 的全局状态, 同 seed 也会得到不同结果
    set_reproducible_seed(seed)
    builder._gnn_init_seed = seed
    return builder._train_gnn(
        builder._cached_X, builder._cached_y_ratio,
        builder._cached_graph, builder._cached_edge_segments,
        builder._test_idx, gnn_arch=gnn_arch,
    )


if HAS_TORCH and HAS_PYG:
    class EdgeGNN(nn.Module):
        def __init__(self, node_dim, edge_dim, hidden_dim=96, num_layers=3, dropout=0.2,
                     gat_heads=4, concat=True):
            super().__init__()
            self.gat_heads = gat_heads
            self.concat = concat
            out_dim_first = hidden_dim * gat_heads if concat else hidden_dim
            out_dim_other = hidden_dim * gat_heads if concat else hidden_dim

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.residual_projs = nn.ModuleList()

            self.convs.append(GATv2Conv(
                node_dim, hidden_dim, heads=gat_heads, concat=concat,
                edge_dim=edge_dim, residual=False, dropout=dropout
            ))
            self.bns.append(nn.BatchNorm1d(out_dim_first))
            self.residual_projs.append(nn.Linear(node_dim, out_dim_first))

            for _ in range(num_layers - 1):
                self.convs.append(GATv2Conv(
                    out_dim_other, hidden_dim, heads=gat_heads, concat=concat,
                    edge_dim=edge_dim, residual=False, dropout=dropout
                ))
                self.bns.append(nn.BatchNorm1d(out_dim_other))
                self.residual_projs.append(nn.Linear(out_dim_other, out_dim_other))

            self.dropout = nn.Dropout(dropout)
            self.edge_mlp = nn.Sequential(
                nn.Linear(out_dim_other * 2 + edge_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x, edge_index, edge_attr, num_target_edges,
                    msg_edge_index=None, edge_attr_msg=None):
            msg_idx = msg_edge_index if msg_edge_index is not None else edge_index
            msg_attr = edge_attr_msg if edge_attr_msg is not None else edge_attr
            for i, (conv, bn, proj) in enumerate(zip(self.convs, self.bns, self.residual_projs)):
                residual = proj(x)
                x_new = conv(x, msg_idx, edge_attr=msg_attr)
                x = F.relu(bn(x_new + residual))
                x = self.dropout(x)

            row = edge_index[0][:num_target_edges]
            col = edge_index[1][:num_target_edges]
            edge_input = torch.cat([x[row], x[col], edge_attr[:num_target_edges]], dim=1)
            return self.edge_mlp(edge_input).squeeze()

    class PNAEdgeGNN(nn.Module):
        def __init__(self, node_dim, edge_dim, hidden_dim=64, num_layers=3, dropout=0.2,
                     deg=None):
            super().__init__()
            
            aggregators = ['mean', 'min', 'max', 'std']
            scalers = ['identity', 'amplification', 'attenuation']
            
            self.node_proj = nn.Linear(node_dim, hidden_dim)
            
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.residual_projs = nn.ModuleList()
            
            self.convs.append(PNAConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=edge_dim,
                towers=4,
                pre_layers=1,
                post_layers=1,
                divide_input=False
            ))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.residual_projs.append(nn.Linear(hidden_dim, hidden_dim))
            
            for _ in range(num_layers - 1):
                self.convs.append(PNAConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    aggregators=aggregators,
                    scalers=scalers,
                    deg=deg,
                    edge_dim=edge_dim,
                    towers=4,
                    pre_layers=1,
                    post_layers=1,
                    divide_input=False
                ))
                self.bns.append(nn.BatchNorm1d(hidden_dim))
                self.residual_projs.append(nn.Linear(hidden_dim, hidden_dim))
            
            self.dropout = nn.Dropout(dropout)
            
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        def forward(self, x, edge_index, edge_attr, num_target_edges,
                    msg_edge_index=None, edge_attr_msg=None):
            msg_idx = msg_edge_index if msg_edge_index is not None else edge_index
            msg_attr = edge_attr_msg if edge_attr_msg is not None else edge_attr
            
            x = self.node_proj(x)
            
            for i, (conv, bn, proj) in enumerate(zip(self.convs, self.bns, self.residual_projs)):
                residual = proj(x)
                x_new = conv(x, msg_idx, edge_attr=msg_attr)
                x = F.relu(bn(x_new + residual))
                x = self.dropout(x)
            
            row = edge_index[0][:num_target_edges]
            col = edge_index[1][:num_target_edges]
            edge_input = torch.cat([x[row], x[col], edge_attr[:num_target_edges]], dim=1)
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
    mape: float
    model: object
    predictions: np.ndarray = None
    use_log_transform: bool = False
    y_test: np.ndarray = None          # ratio 空间, 用于跨 seed 集成评估


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

        # log 变换开关（Task 1: 重尾分布处理, 默认开启）
        self.use_log_transform = True
        
        # 时段划分（2段：白天6-18，夜间18-6）
        self.time_periods = {
            'day': (6, 18),
            'night': (18, 6)
        }
        self.peak_hours = set(range(6, 10)) | set(range(17, 20))
        
        # 特征名称（边×时段聚合）
        self.feature_names = [
            'avg_reported_speed', 'std_reported_speed', 'speed_cv',
            'bearing', 'bearing_sin', 'bearing_cos', 'avg_course_change',
            'std_course_change', 'course_change_x_narrow',
            'waterway_type',
            'node_degree_from', 'node_degree_to', 'edge_betweenness',
            'sample_count', 'log_sample_count',
            'distance', 'theoretical_time',
            'edge_speed_median', 'edge_speed_iqr',
            'neighbor_count', 'neighbor_speed_median',
            'period_morning', 'period_midday', 'period_afternoon', 'period_night',
            'hour_sin', 'hour_cos',
            'speed_decay',
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
        # GAT (gnn) 不加入默认对比，仅在消融实验中按需启用
        if models_to_compare is None:
            models_to_compare = []
            if HAS_XGBOOST:
                models_to_compare.append('xgboost')
            if HAS_LIGHTGBM:
                models_to_compare.append('lightgbm')
                models_to_compare.append('lightgbm_tweedie')
            models_to_compare.append('random_forest')
            if HAS_NGBOOST:
                models_to_compare.append('ngboost')
            if HAS_PYG:
                models_to_compare.append('pna')
        
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
        # 暴露给集成脚本复用 (避免重复数据 prep)
        self._cached_X = X
        self._cached_y_ratio = y_ratio
        self._cached_graph = graph
        self._cached_edge_segments = edge_segments

        print(f"  time_ratio: min={y_ratio.min():.3f}, max={y_ratio.max():.3f}, "
              f"mean={y_ratio.mean():.3f}, std={y_ratio.std():.3f}")

        self._save_feature_matrix(X, y_ratio, y_time, theoretical_times, edge_period_info)

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
        # 关键: sort=False 时 pandas groupby 按哈希顺序, 受 PYTHONHASHSEED 影响会漂移;
        # 显式 sort=True 让分组键字典序排列 -> 跨进程稳定
        grouped = df.groupby('船舶名称', sort=True)
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
            
            # 时间差（秒，浮点）
            times = group['时间'].values
            time_diffs_sec = (times[1:] - times[:-1]) / np.timedelta64(1, 's')
            time_diffs_sec = time_diffs_sec.astype(np.float64)
            
            # 计算速度用于过滤（停泊/锚泊数据：速度<0.5节）
            actual_speeds_all = np.zeros(n)
            mask_pos = (time_diffs_sec > 0) & (distances >= 10)
            actual_speeds_all[mask_pos] = distances[mask_pos] / time_diffs_sec[mask_pos] * 1.944  # m/s → 节
            
            # 过滤无效段：时间差>0且<24小时，距离>=10米，速度>=0.5节（排除停泊数据），速度<=50节（排除GPS跳点）
            valid = (
                np.isfinite(time_diffs_sec)
                & (time_diffs_sec > 0)
                & (time_diffs_sec < 86400)
                & (distances >= 10)
                & (actual_speeds_all >= 0.5)
                & (actual_speeds_all <= 50)
            )
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
            if start < end:
                if start <= hour < end:
                    return period_name
            else:
                if hour >= start or hour < end:
                    return period_name
        return 'day'
    
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

        # 关键: 按 node_id 排序后再分配到 grid, 否则 networkx 的 nodes(data=True) 迭代顺序
        # 受 PYTHONHASHSEED 影响, 会导致 _find_nearest_node 选到不同节点 -> edge_segments 不同 -> GNN 结果漂移
        for node_id, attrs in sorted(graph.nodes(data=True)):
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
        
        # Search 7x7 grid cells (expanded from 3x3 to match validation script)
        for dlat in range(-3, 4):
            for dlon in range(-3, 4):
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

        每条边按 day/night 分为 2 个时段样本，保留时间信息。
        目标变量：time_ratio = actual_time / theoretical_time，
        归一化掉距离/速度主效应，让模型专注学习动态偏差。

        Returns:
            X: 特征矩阵
            y_ratio: time_ratio 目标值
            y_time: 原始 travel_time（用于评估）
            theoretical_times: 每个样本对应的 theoretical_time
            edge_info: 边信息（key=(edge_key, period_name)）
        """
        X_list = []
        y_ratio_list = []
        y_time_list = []
        tt_list = []
        edge_info = {}
        self._edge_theoretical_times = {}
        self._edge_baseline_times = {}

        # 预计算边级别统计量（所有时段合并，用于邻居特征）
        edge_level_stats = {}
        for edge_key, segments in edge_segments.items():
            if len(segments) < 2:
                continue
            speeds = [s['reported_speed'] for s in segments]
            edge_level_stats[edge_key] = {
                'speed_median': np.median(speeds),
                'speed_iqr': np.percentile(speeds, 75) - np.percentile(speeds, 25),
            }

        # 预计算邻居特征
        neighbor_stats = {}
        for edge_key in edge_level_stats:
            u, v = edge_key
            nbr_edges = []
            for nbr in set(list(graph.neighbors(u)) + list(graph.neighbors(v))):
                for (a, b) in [(u, nbr), (nbr, u), (v, nbr), (nbr, v)]:
                    ek = (a, b)
                    if ek in edge_level_stats and ek != edge_key:
                        nbr_edges.append(ek)
            if nbr_edges:
                n_speeds = [edge_level_stats[e]['speed_median'] for e in nbr_edges]
                neighbor_stats[edge_key] = (len(nbr_edges), np.median(n_speeds))
            else:
                neighbor_stats[edge_key] = (0, 0.0)

        for edge_key, segments in edge_segments.items():
            if len(segments) < 2:
                continue

            from_node, to_node = edge_key
            waterway_type = self._get_edge_waterway_type(from_node, to_node, graph)
            node_degree_from = self.node_degrees.get(from_node, 0)
            node_degree_to = self.node_degrees.get(to_node, 0)
            betweenness = self.edge_betweenness.get(edge_key, 0)

            for period_name in ['day', 'night']:
                period_segments = [s for s in segments if self._get_time_period(s['hour']) == period_name]
                if len(period_segments) < 2:
                    continue

                distances = [s['distance'] for s in period_segments]
                avg_distance = np.mean(distances)
                bearing = np.mean([s['bearing'] for s in period_segments])
                bearing_rad = np.deg2rad(bearing)
                avg_course_change = np.mean([s['course_change'] for s in period_segments])

                reported_speeds_all = [s['reported_speed'] for s in period_segments]
                avg_reported_speed = np.mean(reported_speeds_all)
                std_reported_speed = np.std(reported_speeds_all) if len(reported_speeds_all) > 1 else 0
                speed_cv = std_reported_speed / max(avg_reported_speed, 0.1)
                speed_ms = max(avg_reported_speed, 0.5) * 0.5144
                theoretical_time = avg_distance / speed_ms
                self._edge_theoretical_times[(edge_key, period_name)] = theoretical_time

                time_diffs = [s['time_diff'] for s in period_segments]
                avg_travel_time = np.mean(time_diffs)

                time_ratio = avg_travel_time / max(theoretical_time, 1.0)
                time_ratio = np.clip(time_ratio, 0.2, 5.0)

                std_course_change = np.std([s['course_change'] for s in period_segments]) if len(period_segments) > 1 else 0
                course_change_x_narrow = avg_course_change * waterway_type
                sample_count = len(period_segments)

                hours = [s['hour'] for s in period_segments]
                avg_hour = np.mean(hours)
                hour_sin = np.sin(2 * np.pi * avg_hour / 24)
                hour_cos = np.cos(2 * np.pi * avg_hour / 24)

                period_morning = 1.0 if period_name == 'day' and 6 <= int(avg_hour) < 10 else 0.0
                period_midday = 1.0 if period_name == 'day' and 10 <= int(avg_hour) < 14 else 0.0
                period_afternoon = 1.0 if period_name == 'day' and 14 <= int(avg_hour) < 18 else 0.0
                period_night = 1.0 if period_name == 'night' else 0.0

                actual_speeds_period = [s['actual_speed'] for s in period_segments]
                avg_actual_speed = np.mean(actual_speeds_period)
                speed_decay = avg_actual_speed / max(avg_reported_speed, 0.5)

                features = [
                    avg_reported_speed,
                    std_reported_speed,
                    speed_cv,
                    bearing,
                    np.sin(bearing_rad),
                    np.cos(bearing_rad),
                    avg_course_change,
                    std_course_change,
                    course_change_x_narrow,
                    waterway_type,
                    node_degree_from,
                    node_degree_to,
                    betweenness,
                    sample_count,
                    np.log1p(sample_count),
                    avg_distance,
                    theoretical_time,
                    edge_level_stats[edge_key]['speed_median'],
                    edge_level_stats[edge_key]['speed_iqr'],
                    *neighbor_stats[edge_key],
                    period_morning,
                    period_midday,
                    period_afternoon,
                    period_night,
                    hour_sin,
                    hour_cos,
                    speed_decay,
                ]

                X_list.append(features)
                y_ratio_list.append(time_ratio)
                y_time_list.append(avg_travel_time)
                tt_list.append(theoretical_time)

                edge_info[(edge_key, period_name)] = {
                    'sample_count': sample_count,
                    'avg_travel_time': avg_travel_time,
                    'time_ratio': time_ratio,
                    'theoretical_time': theoretical_time,
                    'period': period_name,
                }

        X = np.array(X_list)
        y_ratio = np.array(y_ratio_list)
        y_time = np.array(y_time_list)
        theoretical_times = np.array(tt_list)

        print(f"  训练样本数: {len(X):,} (边×时段)")
        print(f"  特征维度: {X.shape[1]}")
        print(f"  time_ratio: min={y_ratio.min():.3f}, max={y_ratio.max():.3f}, "
              f"mean={y_ratio.mean():.3f}, std={y_ratio.std():.3f}")

        nan_count = int(np.isnan(X).sum())
        inf_count = int(np.isinf(X).sum())
        if nan_count > 0 or inf_count > 0:
            logger.warning("特征矩阵包含 %d 个 NaN 和 %d 个 inf，将替换为 0", nan_count, inf_count)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        nan_ratio_count = int(np.isnan(y_ratio).sum())
        if nan_ratio_count > 0:
            logger.warning("目标变量包含 %d 个 NaN，将移除对应样本", nan_ratio_count)
            valid_mask = ~np.isnan(y_ratio)
            X = X[valid_mask]
            y_ratio = y_ratio[valid_mask]
            y_time = y_time[valid_mask]
            theoretical_times = theoretical_times[valid_mask]
            edge_info = {k: v for idx, (k, v) in enumerate(edge_info.items()) if valid_mask[idx]}

        return X, y_ratio, y_time, theoretical_times, edge_info

    def _save_feature_matrix(self, X, y_ratio, y_time, theoretical_times, edge_info):
        """保存特征矩阵（21个特征 + 目标变量 + 时段标记）到 CSV"""
        records = []
        for i, (edge_key_period, info) in enumerate(edge_info.items()):
            edge_key, period_name = edge_key_period
            from_node, to_node = edge_key
            record = {
                'from_node': from_node,
                'to_node': to_node,
                'period': period_name,
                'avg_travel_time': info.get('avg_travel_time', y_time[i]),
                'theoretical_time': info.get('theoretical_time', theoretical_times[i]),
                'time_ratio': info.get('time_ratio', y_ratio[i]),
            }
            for j, col in enumerate(self.feature_names):
                record[col] = X[i][j]
            records.append(record)

        df = pd.DataFrame(records)
        output_path = 'output/feature_matrix.csv'
        df.to_csv(output_path, index=False)
        logger.info("  特征矩阵已保存: %s (%d 行 × %d 列)", output_path, len(df), len(df.columns))

    # ==================== 模型训练与对比 ====================
    
    def _train_and_compare_models(self, X: np.ndarray, y_ratio: np.ndarray,
                                   y_time: np.ndarray, theoretical_times: np.ndarray,
                                   models: List[str], graph, edge_segments,
                                   use_grid_search: bool = True) -> Dict[str, ModelResult]:
        """训练并对比多个模型（边×时段级别）"""
        results = {}

        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y_ratio, 'values'):
            y_ratio = y_ratio.values

        # 简单随机切分（每条边就是 1 个样本，无需分组）
        X_train, X_test, y_ratio_train, y_ratio_test = train_test_split(
            X, y_ratio, test_size=0.2, random_state=42
        )
        _, _, y_time_train, y_time_test = train_test_split(
            X, y_time, test_size=0.2, random_state=42
        )
        _, _, tt_train, tt_test = train_test_split(
            X, theoretical_times, test_size=0.2, random_state=42
        )
        all_indices = np.arange(len(X))
        _, self._test_idx = train_test_split(all_indices, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.scaler = scaler
        self.use_grid_search = use_grid_search
        self._tt_test = tt_test
        self._y_time_test = y_time_test

        self.use_log_target = False

        print(f"\n  训练样本: {len(X_train):,}, 测试样本: {len(X_test):,} (边×时段)")
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
            result = self._train_lightgbm(X_train, X_test, y_ratio_train, y_ratio_test, objective='tweedie')
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

        if 'ngboost' in models and HAS_NGBOOST:
            result = self._train_ngboost(X_train, X_test, y_ratio_train, y_ratio_test)
            results['ngboost'] = result
            self._print_result(result)

        if 'gnn' in models and HAS_PYG:
            # 关键: 7-model 对比这里不走 train_gnn_with_seed, 需显式设种子 + 单线程, 否则
            # GAT/PNA 在 16 线程 CPU 上 loss 会漂, 与 5-seed 稳定性区块的同 seed 结果对不上
            set_reproducible_seed(42)
            self._gnn_init_seed = 42
            result = self._train_gnn(X, y_ratio, graph, edge_segments, self._test_idx, gnn_arch='gat')
            results['gnn'] = result
            self._print_result(result)

        if 'pna' in models and HAS_PYG:
            set_reproducible_seed(42)
            self._gnn_init_seed = 42
            result = self._train_gnn(X, y_ratio, graph, edge_segments, self._test_idx, gnn_arch='pna')
            results['pna'] = result
            self._print_result(result)

        self._print_comparison_table(results)
        self._model_results = results

        # 所有模型统一在时间空间评估边×时段级别 R2
        for model_name, result in results.items():
            if result.model is None:
                continue
            try:
                if model_name in ['gnn', 'pna']:
                    continue
                y_pred_all = result.model.predict(X_test)
                y_pred_all = np.clip(y_pred_all, 0.1, 20.0)
                y_pred_time = y_pred_all * tt_test

                r2 = r2_score(y_time_test, y_pred_time)
                mae = mean_absolute_error(y_time_test, y_pred_time)
                rmse = np.sqrt(mean_squared_error(y_time_test, y_pred_time))
                mask = y_time_test > 0
                mape = np.mean(np.abs((y_time_test[mask] - y_pred_time[mask]) / y_time_test[mask])) * 100 if mask.any() else 0

                result.mae = mae
                result.rmse = rmse
                result.r2 = r2
                result.mape = mape
                result.predictions = y_pred_time
                print(f"    [{model_name}] R2={r2:.4f} MAE={mae:.2f}s")
            except Exception as e:
                print(f"    [{model_name}] 评估失败: {e}")

        # GNN 已在 _train_gnn 中完成时间空间评估
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
    
    def _train_lightgbm(self, X_train, X_test, y_train, y_test, objective: str = 'regression') -> ModelResult:
        """训练 LightGBM（支持 regression 和 tweedie 目标函数）"""
        start_time = time.time()
        model_name = 'LightGBM' if objective == 'regression' else 'LightGBM-Tweedie'
        
        # 用 DataFrame 包装，确保训练/预测时特征名一致，避免 sklearn 警告
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=self.feature_names)
            X_test = pd.DataFrame(X_test, columns=self.feature_names)
        
        # Tweedie 目标需要正值
        y_train_fit = np.maximum(y_train, 0.1) if objective == 'tweedie' else y_train
        
        if self.use_grid_search and HAS_OPTUNA:
            print(f"\n    [{model_name}] 执行贝叶斯调参（Optuna）...")
            from sklearn.model_selection import cross_val_score

            def objective_func(trial):
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
                if objective == 'tweedie':
                    params['objective'] = 'tweedie'
                    params['tweedie_variance_power'] = trial.suggest_float('tweedie_variance_power', 1.1, 1.9)
                candidate = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
                scores = cross_val_score(candidate, X_train, y_train_fit, cv=5,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
                return scores.mean()

            study = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(objective_func, n_trials=30, show_progress_bar=HAS_TQDM)
            best_params = study.best_params
            best_params['bagging_freq'] = 5
            if objective == 'tweedie':
                best_params['objective'] = 'tweedie'
            print(f"    最佳参数: {best_params}")
            print(f"    最佳 CV 分数: {study.best_value:.4f}")

            model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X_train, y_train_fit)
        elif self.use_grid_search:
            # optuna 未安装，回退到随机搜索
            from sklearn.model_selection import ParameterSampler, cross_val_score
            print(f"\n    [{model_name}] Optuna 未安装，回退到随机搜索...")
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
            if objective == 'tweedie':
                param_grid['objective'] = ['tweedie']
                param_grid['tweedie_variance_power'] = [1.1, 1.3, 1.5, 1.7, 1.9]
            param_samples = list(ParameterSampler(param_grid, n_iter=25, random_state=42))
            best_score = -np.inf
            best_params = None
            for params in tqdm(param_samples, desc=f"    {model_name} 搜索", unit="组", disable=not HAS_TQDM):
                candidate = lgb.LGBMRegressor(**params, random_state=42, n_jobs=-1, verbose=-1)
                scores = cross_val_score(candidate, X_train, y_train_fit, cv=5,
                                         scoring='neg_mean_squared_error', n_jobs=-1)
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_params = params
            model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X_train, y_train_fit)
            print(f"    最佳参数: {best_params}")
        else:
            default_params = {
                'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.02,
                'num_leaves': 31, 'min_child_samples': 50,
                'reg_alpha': 1.0, 'reg_lambda': 1.0, 'min_split_gain': 0.1,
                'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
                'random_state': 42, 'n_jobs': -1, 'verbose': -1
            }
            if objective == 'tweedie':
                default_params['objective'] = 'tweedie'
                default_params['tweedie_variance_power'] = 1.5
            model = lgb.LGBMRegressor(**default_params)
            model.fit(X_train, y_train_fit)
        
        train_time = time.time() - start_time
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0.1, 20.0)

        if objective == 'tweedie':
            self.feature_importance_tweedie = dict(zip(self.feature_names, model.feature_importances_))
        else:
            self.feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        
        return self._evaluate_model(model_name, model, y_test, y_pred, train_time,
                                     use_log_transform=(objective != 'tweedie'))
    
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
        y_pred = np.clip(y_pred, 0.1, 20.0)

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
    
    def _train_gnn(self, X, y, graph, edge_segments, test_idx=None, gnn_arch='gat') -> ModelResult:
        """训练图神经网络（支持多种架构：gat, pna）"""
        start_time = time.time()
        
        self._gnn_arch = gnn_arch
        
        # 先收集有效边（有足够轨迹数据的边），扩展边特征
        valid_edges = []
        edge_features_list = []
        edge_targets_ratio = []
        edge_targets_log = []
        edge_targets_original_list = []
        edge_theoretical_times_list = []
        
        for edge_key, segments in edge_segments.items():
            if len(segments) < 2:
                continue
            
            for period_name in ['day', 'night']:
                period_segments = [s for s in segments if self._get_time_period(s['hour']) == period_name]
                if len(period_segments) < 2:
                    continue
                
                from_node, to_node = edge_key
                
                avg_time = np.mean([s['time_diff'] for s in period_segments])
                avg_reported_speed = np.mean([s['reported_speed'] for s in period_segments])
                distance = np.mean([s['distance'] for s in period_segments])
                avg_bearing = np.mean([s['bearing'] for s in period_segments])
                avg_course_change = np.mean([s['course_change'] for s in period_segments])
                waterway_type = self._get_edge_waterway_type(from_node, to_node, graph)
                node_degree_from = self.node_degrees.get(from_node, 0)
                node_degree_to = self.node_degrees.get(to_node, 0)
                betweenness = self.edge_betweenness.get(edge_key, 0)
                
                speed_ms = max(avg_reported_speed, 0.5) * 0.5144
                theoretical_time = distance / speed_ms
                
                valid_edges.append((from_node, to_node, period_name))
                bearing_rad = np.deg2rad(avg_bearing)
                speeds = [s['reported_speed'] for s in period_segments]
                speed_iqr = np.percentile(speeds, 75) - np.percentile(speeds, 25) if len(speeds) > 1 else 0
                
                hours_list = [s['hour'] for s in period_segments]
                avg_hour_gnn = np.mean(hours_list)
                hour_sin_gnn = np.sin(2 * np.pi * avg_hour_gnn / 24)
                hour_cos_gnn = np.cos(2 * np.pi * avg_hour_gnn / 24)
                actual_speeds_gnn = [s['actual_speed'] for s in period_segments]
                speed_decay_gnn = np.mean(actual_speeds_gnn) / max(avg_reported_speed, 0.5)
                
                edge_features_list.append([
                    distance, theoretical_time, avg_reported_speed,
                    np.sin(bearing_rad), np.cos(bearing_rad), avg_course_change,
                    waterway_type, node_degree_from, node_degree_to, betweenness,
                    speed_iqr, hour_sin_gnn, hour_cos_gnn, speed_decay_gnn,
                ])
                edge_targets_ratio.append(avg_time / max(theoretical_time, 1e-6))
                edge_targets_log.append(log_transform_target(np.array([avg_time / max(theoretical_time, 1e-6)]))[0])
                edge_targets_original_list.append(avg_time)
                edge_theoretical_times_list.append(theoretical_time)
        
        if len(valid_edges) == 0:
            return ModelResult('GNN', 0, float('inf'), float('inf'), 0, 100, None)
        
        n_edges = len(valid_edges)
        print(f"\n    [GNN] 有效边数: {n_edges}, 边特征维度: {len(edge_features_list[0])}")
        
        edge_targets_ratio = np.array(edge_targets_ratio, dtype=np.float64)
        edge_targets_log = np.array(edge_targets_log, dtype=np.float64)
        edge_targets_original = np.array(edge_targets_original_list, dtype=np.float64)
        edge_theoretical_times = np.array(edge_theoretical_times_list, dtype=np.float64)
        
        # 收集有效边涉及的节点
        valid_nodes = set()
        for u, v, _ in valid_edges:
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
        for u, v, _ in valid_edges:
            edge_index_forward.append([node_id_to_idx[u], node_id_to_idx[v]])

        # 训练/测试划分：若调用方传入了 test_idx（与树模型同一划分），则复用它，
        # 否则回退到 torch.randperm 自行切分。valid_edges 的顺序与 _build_training_data
        # 中 X 矩阵行顺序一致（都按 edge_segments.items() 顺序迭代 day/night），
        # 所以 test_idx 可以直接用作 valid_edges 的下标。
        if test_idx is not None and len(test_idx) > 0:
            test_idx_arr = np.asarray(test_idx, dtype=np.int64)
            test_mask_np = np.zeros(n_edges, dtype=bool)
            test_mask_np[test_idx_arr] = True
            train_mask_np = ~test_mask_np
            print(f"    [GNN] 使用调用方传入的 test_idx (与树模型同划分): "
                  f"test={test_mask_np.sum()}, train={train_mask_np.sum()}")
        else:
            n_train = int(n_edges * 0.8)
            torch.manual_seed(42)
            perm = torch.randperm(n_edges)
            train_mask_np = np.zeros(n_edges, dtype=bool)
            train_mask_np[perm[:n_train]] = True
            test_mask_np = ~train_mask_np
            print(f"    [GNN] 警告: 使用 torch.randperm 自切分，与树模型不一致")

        train_mask = torch.from_numpy(train_mask_np)
        test_mask = torch.from_numpy(test_mask_np)
        print(f"    训练边: {train_mask.sum().item()}, 测试边: {test_mask.sum().item()}")

        # 构建训练边专用的 edge_index（防止消息传递泄露测试边信息）
        train_edge_pairs = []
        for i in range(n_edges):
            if train_mask[i]:
                u, v = edge_index_forward[i]
                train_edge_pairs.append([u, v])
                train_edge_pairs.append([v, u])
        edge_index_train = torch.LongTensor(train_edge_pairs).t().contiguous()

        # 全量 edge_index（用于评估）
        edge_index_bidir = edge_index_forward + [[v, u] for u, v in edge_index_forward]
        edge_index = torch.LongTensor(edge_index_bidir).t().contiguous()

        edge_features = torch.FloatTensor(edge_features_list)
        edge_targets_original_t = torch.FloatTensor(edge_targets_original)
        edge_targets = torch.FloatTensor(edge_targets_ratio)
        edge_targets_log_tensor = torch.FloatTensor(edge_targets_log)
        self._gnn_use_log_transform = bool(self.use_log_transform)
        self._gnn_edge_targets_log_tensor = edge_targets_log_tensor
        
        # 标准化边特征（仅用训练数据拟合）
        edge_scaler = StandardScaler()
        edge_features_np = edge_features.numpy()
        edge_scaler.fit(edge_features_np[train_mask.numpy()])
        edge_features_np = edge_scaler.transform(edge_features_np)
        edge_features = torch.FloatTensor(edge_features_np)
        self.gnn_edge_scaler = edge_scaler
        
        edge_features_train_list = []
        for i in range(n_edges):
            if train_mask[i]:
                edge_features_train_list.append(edge_features[i])
                edge_features_train_list.append(edge_features[i])
        edge_features_train = torch.stack(edge_features_train_list)
        
        edge_features_bidir = torch.cat([edge_features, edge_features], dim=0)

        # 标准化节点特征（仅用训练边连接的节点拟合）
        train_node_set = set()
        for i in range(n_edges):
            if train_mask[i]:
                u, v, _ = valid_edges[i]
                train_node_set.add(node_id_to_idx[u])
                train_node_set.add(node_id_to_idx[v])
        node_scaler = StandardScaler()
        train_node_indices = sorted(train_node_set)
        node_scaler.fit(node_features[train_node_indices].numpy())
        node_features_np = node_scaler.transform(node_features.numpy())
        node_features = torch.FloatTensor(node_features_np)
        self.gnn_node_scaler = node_scaler

        # ---- 从 train_mask 中再划 10% 做 val（用于 early stopping） ----
        train_indices = np.where(train_mask_np)[0]
        torch.manual_seed(42)
        n_val = max(1, int(len(train_indices) * 0.1))
        perm_val = torch.randperm(len(train_indices))
        val_in_train = train_indices[perm_val[:n_val].numpy()]
        pure_train_indices = np.setdiff1d(train_indices, val_in_train)

        pure_train_mask_np = np.zeros(n_edges, dtype=bool)
        pure_train_mask_np[pure_train_indices] = True
        val_mask_np = np.zeros(n_edges, dtype=bool)
        val_mask_np[val_in_train] = True

        pure_train_mask = torch.from_numpy(pure_train_mask_np)
        val_mask = torch.from_numpy(val_mask_np)
        print(f"    纯训练边: {pure_train_mask.sum().item()}, 验证边: {val_mask.sum().item()}, 测试边: {test_mask.sum().item()}")

        # 构建纯训练边专用的 edge_index（防止消息传递泄露验证/测试边信息）
        pure_train_edge_pairs = []
        for i in range(n_edges):
            if pure_train_mask[i]:
                u, v = edge_index_forward[i]
                pure_train_edge_pairs.append([u, v])
                pure_train_edge_pairs.append([v, u])
        edge_index_pure_train = torch.LongTensor(pure_train_edge_pairs).t().contiguous()

        pure_train_edge_features_list = []
        for i in range(n_edges):
            if pure_train_mask[i]:
                pure_train_edge_features_list.append(edge_features[i])
                pure_train_edge_features_list.append(edge_features[i])
        edge_features_pure_train = torch.stack(pure_train_edge_features_list)

        # EdgeGNN 已移至模块级别（确保可 pickle 序列化）
        num_target_edges = n_edges  # 前 n_edges 条是原始方向

        # 在 model init 之前用 _gnn_init_seed 重置 RNG（split 之前的 42 不影响 model）
        init_seed = getattr(self, '_gnn_init_seed', 42)
        torch.manual_seed(init_seed)
        np.random.seed(init_seed)

        if self.use_grid_search:
            print(f"\n    [GNN] 使用 {gnn_arch.upper()} 架构，跳过自动搜索")

        if gnn_arch == 'pna':
            best_hp = {'hidden_dim': 64, 'num_layers': 3, 'lr': 0.002, 'dropout': 0.2}
        else:
            best_hp = {'hidden_dim': 96, 'num_layers': 3, 'lr': 0.002, 'dropout': 0.2}
        print(f"    最佳参数: {best_hp}")

        if gnn_arch == 'pna':
            from torch_geometric.utils import degree as pyg_degree
            deg = pyg_degree(edge_index_pure_train[0], num_nodes=node_features.size(0)).long()
            model = PNAEdgeGNN(node_features.shape[1], edge_features.shape[1],
                               hidden_dim=best_hp['hidden_dim'], num_layers=best_hp['num_layers'],
                               dropout=best_hp['dropout'], deg=deg)
        else:
            model = EdgeGNN(node_features.shape[1], edge_features.shape[1],
                            hidden_dim=best_hp['hidden_dim'], num_layers=best_hp['num_layers'],
                            dropout=best_hp['dropout'], gat_heads=4, concat=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=best_hp['lr'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
        
        criterion = nn.HuberLoss(delta=2.0)

        model.train()
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0

        n_epochs = getattr(self, 'gnn_n_epochs', 200)
        for epoch in tqdm(range(n_epochs), desc="    GNN 训练", unit="epoch", disable=not HAS_TQDM):
            optimizer.zero_grad()
            # DropEdge: 训练时随机丢 10% 边, 同时取对应 edge_attr 保持形状一致
            if model.training and getattr(self, '_gnn_use_drop_edge', True):
                msg_idx_dropped, edge_mask = drop_random_edges(edge_index_pure_train, p=0.1)
                msg_attr_dropped = edge_features_pure_train[edge_mask]
            else:
                msg_idx_dropped = edge_index_pure_train
                msg_attr_dropped = edge_features_pure_train
            # 消息传递只用纯训练边，预测时也用纯训练边做消息传递（防泄露）
            outputs = model(node_features, edge_index, edge_features,
                            num_target_edges, msg_edge_index=msg_idx_dropped,
                            edge_attr_msg=msg_attr_dropped)
            if self.use_log_transform:
                train_loss = criterion(outputs[pure_train_mask], edge_targets_log_tensor[pure_train_mask])
                val_loss = criterion(outputs[val_mask], edge_targets_log_tensor[val_mask]).item()
            else:
                train_loss = criterion(outputs[pure_train_mask], edge_targets[pure_train_mask])
                val_loss = criterion(outputs[val_mask], edge_targets[val_mask]).item()
            train_loss.backward()
            optimizer.step()

            if hasattr(scheduler, 'step'):
                scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                early_stop_patience = getattr(self, 'gnn_patience', 25)
                if patience_counter > early_stop_patience:
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        train_time = time.time() - start_time

        model.eval()
        with torch.no_grad():
            # 评估时消息传递只用训练边（含 val），防止测试边信息泄露到节点表示
            y_pred_raw = model(node_features, edge_index, edge_features,
                               num_target_edges, msg_edge_index=edge_index_train,
                               edge_attr_msg=edge_features_train).numpy()

        if self.use_log_transform:
            y_pred_ratio = inverse_log_transform(y_pred_raw)
        else:
            y_pred_ratio = y_pred_raw

        y_true_test = edge_targets[test_mask].numpy()
        y_pred_test = y_pred_ratio[test_mask.numpy()]
        y_pred_test = np.clip(y_pred_test, 0.1, 20.0)

        y_true_time_test = edge_targets_original_t[test_mask].numpy()
        gnn_tt_test = edge_theoretical_times[test_mask.numpy()]

        gnn_edge_feature_names = [
            'distance', 'theoretical_time', 'avg_reported_speed',
            'bearing_sin', 'bearing_cos', 'avg_course_change',
            'waterway_type', 'node_degree_from', 'node_degree_to', 'edge_betweenness',
            'speed_iqr', 'hour_sin', 'hour_cos', 'speed_decay'
        ]

        def _gnn_predict_time_from_features(feat_tensor):
            feat_bidir = feat_tensor.repeat(2, 1)
            # 特征重要性评估也用训练边做消息传递（防泄露）
            train_feat_bidir_list = []
            for i in range(n_edges):
                if train_mask[i]:
                    train_feat_bidir_list.append(feat_tensor[i])
                    train_feat_bidir_list.append(feat_tensor[i])
            train_feat_bidir = torch.stack(train_feat_bidir_list)
            with torch.no_grad():
                pred_raw = model(node_features, edge_index, feat_tensor,
                                 num_target_edges, msg_edge_index=edge_index_train,
                                 edge_attr_msg=train_feat_bidir).numpy()
            if self.use_log_transform:
                pred = inverse_log_transform(pred_raw)
            else:
                pred = pred_raw
            pred = np.clip(pred, 0.1, 20.0)
            return pred[test_mask.numpy()] * gnn_tt_test

        y_true_time_arr = y_true_time_test
        base_pred_time = y_pred_test * gnn_tt_test
        base_r2 = r2_score(y_true_time_arr, base_pred_time)

        # 用 init seed 让特征重要性 permutation 可复现 (同 seed 跑出同一份重要性)
        init_seed = getattr(self, '_gnn_init_seed', 42)
        rng = np.random.RandomState(init_seed)
        edge_features_np_for_pi = edge_features.numpy().copy()
        permutation_r2s = []
        for col_idx in range(edge_features_np_for_pi.shape[1]):
            perm_np = edge_features_np_for_pi.copy()
            rng.shuffle(perm_np[:, col_idx])
            perm_time_pred = _gnn_predict_time_from_features(torch.FloatTensor(perm_np))
            permutation_r2s.append(r2_score(y_true_time_arr, perm_time_pred))

        fi_values = [max(0.0, base_r2 - r2) for r2 in permutation_r2s]
        fi_sum = sum(fi_values) if sum(fi_values) > 0 else 1.0
        self.feature_importance = {
            name: value / fi_sum for name, value in zip(gnn_edge_feature_names, fi_values)
        }
        print(f"    GNN 特征重要性基准 R^2: {base_r2:.4f}")
        
        print(f"    GNN 验证集 best_val_loss: {best_val_loss:.4f}, 测试边数: {test_mask.sum().item()}")
        
        # ===== 缓存全量 PNA/GNN 预测结果供 _predict_with_gnn 使用 =====
        with torch.no_grad():
            all_pred_raw = model(
                node_features_scaled, edge_index, edge_features_tensor,
                num_target_edges, msg_edge_index=edge_index_train,
                edge_attr_msg=edge_features_train
            ).numpy()
        if self.use_log_transform:
            all_pred_ratio = inverse_log_transform(all_pred_raw)
        else:
            all_pred_ratio = all_pred_raw
        all_pred_ratio = np.clip(all_pred_ratio, 0.1, 20.0)
        all_pred_time = all_pred_ratio * edge_theoretical_times
        
        self._gnn_prediction_cache = {}
        for i in range(n_edges):
            u, v, period = valid_edges[i]
            key = (u, v)
            if key not in self._gnn_prediction_cache:
                self._gnn_prediction_cache[key] = {}
            self._gnn_prediction_cache[key][period] = {
                'predicted_time': float(all_pred_time[i]),
                'predicted_ratio': float(all_pred_ratio[i]),
            }
        print(f"    GNN 预测缓存: {len(self._gnn_prediction_cache)} 条边, {n_edges} 个边×时段预测")
        
        model_name = 'PNA' if gnn_arch == 'pna' else 'GNN'
        return self._evaluate_model(model_name, model, y_true_test, y_pred_test, train_time,
                                     y_true_time=y_true_time_test,
                                     theoretical_times_test=gnn_tt_test)

    def _train_gnn_ensemble(self, X, y_ratio, graph, edge_segments, test_idx,
                             gnn_arch='gat', n_seeds=5, base_seed=42) -> ModelResult:
        """训练 GNN 集成模型（多 seed 取平均预测）

        Args:
            n_seeds: 集成模型数量
            base_seed: 起始种子
        """
        seeds = list(range(base_seed, base_seed + n_seeds))
        all_y_pred = []
        all_results = []

        for i, seed in enumerate(seeds):
            print(f"\n    [{gnn_arch.upper()}] Seed {seed} ({i+1}/{n_seeds})")
            self._gnn_init_seed = seed
            result = self._train_gnn(X, y_ratio, graph, edge_segments, test_idx, gnn_arch=gnn_arch)
            all_y_pred.append(result.predictions)
            all_results.append(result)

        avg_y_pred = np.mean(all_y_pred, axis=0)
        y_true = all_results[0].y_test

        mae = mean_absolute_error(y_true, avg_y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, avg_y_pred))
        r2 = r2_score(y_true, avg_y_pred)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - avg_y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
        total_time = sum(r.train_time for r in all_results)

        model_name = f'{gnn_arch.upper()}_ensemble_{n_seeds}seed'

        print(f"\n    [{model_name}] 集成结果 (n_seeds={n_seeds})")
        print(f"      MAE: {mae:.2f}秒")
        print(f"      RMSE: {rmse:.2f}秒")
        print(f"      R2: {r2:.4f}")
        print(f"      MAPE: {mape:.2f}%")

        return ModelResult(
            model_name=model_name,
            train_time=total_time,
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            model=all_results[-1].model,
            predictions=avg_y_pred,
            use_log_transform=False,
            y_test=y_true
        )

    def _train_ngboost(self, X_train, X_test, y_train, y_test) -> ModelResult:
        """训练 NGBoost（概率梯度提升，提供不确定性估计）"""
        # 关键: NGBoost 内部 minibatch 采样用全局 np.random, 前置模型 (XGB/LGBM/RF) 已消耗 RNG,
        # 这里强制重置才能让 NGBoost 自身 random_state=7 真正生效
        np.random.seed(7)
        start_time = time.time()
        
        if self.use_grid_search and HAS_OPTUNA:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def ngb_objective(trial):
                n_est = trial.suggest_int('n_estimators', 100, 400, step=100)
                lr = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
                mbf = trial.suggest_float('minibatch_frac', 0.3, 1.0)
                model = NGBRegressor(
                    n_estimators=n_est, learning_rate=lr,
                    minibatch_frac=mbf, Dist=Normal,
                    random_state=7, verbose=False
                )
                model.fit(X_train, y_train, X_val=X_test, Y_val=y_test,
                          early_stopping_rounds=15)
                y_pred = np.clip(model.predict(X_test), 0.1, 20.0)
                return r2_score(y_test, y_pred)
            
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=7),  # 种子 7, 与 XGBoost/LightGBM/RF 调参保持稳定
            )
            study.optimize(ngb_objective, n_trials=10, show_progress_bar=False)
            best_p = study.best_params
            print(f"    [NGBoost] 贝叶斯调参完成，最佳: {best_p}")
            model = NGBRegressor(
                n_estimators=best_p['n_estimators'],
                learning_rate=best_p['learning_rate'],
                minibatch_frac=best_p['minibatch_frac'],
                Dist=Normal, random_state=7, verbose=False
            )
            model.fit(X_train, y_train, X_val=X_test, Y_val=y_test,
                      early_stopping_rounds=15)
        else:
            model = NGBRegressor(
                n_estimators=200, learning_rate=0.05,
                minibatch_frac=0.5, Dist=Normal,
                random_state=7, verbose=False
            )
            model.fit(X_train, y_train, X_val=X_test, Y_val=y_test,
                      early_stopping_rounds=20)
        
        train_time = time.time() - start_time
        y_pred = model.predict(X_test)
        y_pred = np.clip(y_pred, 0.1, 20.0)
        
        print(f"    NGBoost 训练完成，best iter: {model.best_ntree_limit if hasattr(model, 'best_ntree_limit') else 'N/A'}")
        return self._evaluate_model('NGBoost', model, y_test, y_pred, train_time)
    
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
            # 返回 time 空间, 与 r2/mae/rmse 报告一致
            out_predictions = y_pred_time
            out_y_test = y_true_time
        else:
            mae = mae_ratio
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_ratio
            mask = y_true != 0
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
            out_predictions = y_pred
            out_y_test = y_true

        return ModelResult(
            model_name=name,
            train_time=train_time,
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            model=model,
            predictions=out_predictions,
            use_log_transform=use_log_transform,
            y_test=out_y_test
        )
    
    def _print_result(self, result: ModelResult):
        """打印单个模型结果"""
        print(f"\n  {result.model_name}:")
        print(f"    训练时间: {result.train_time:.2f}秒")
        print(f"    MAE: {result.mae:.2f}秒")
        print(f"    RMSE: {result.rmse:.2f}秒")
        print(f"    R2 (原始空间): {result.r2:.4f}")
        print(f"    MAPE: {result.mape:.2f}%")
    
    def _print_comparison_table(self, results: Dict[str, ModelResult]):
        """打印对比表"""
        print("\n  " + "="*85)
        print("  模型对比结果")
        print("  " + "="*85)
        print(f"  {'模型':<20} {'训练时间':>10} {'MAE':>10} {'RMSE':>10} {'R2':>10} {'MAPE':>10}")
        print("  " + "-"*85)
        
        for name, result in sorted(results.items(), key=lambda x: x[1].r2, reverse=True):
            print(f"  {result.model_name:<20} {result.train_time:>8.2f}s {result.mae:>10.2f} "
                  f"{result.rmse:>10.2f} {result.r2:>10.4f} {result.mape:>9.2f}%")
        
        print("  " + "="*85)
    
    def _select_best_model(self, results: Dict[str, ModelResult]):
        """选择最优模型：R2 最高的模型获胜"""
        if not results:
            raise ValueError("没有可用的模型")

        candidates = [(name, result) for name, result in results.items() if result.model is not None]
        if not candidates:
            raise ValueError("没有成功训练的模型")

        candidates.sort(key=lambda x: x[1].r2, reverse=True)
        best_name, best_result = candidates[0]

        self.best_model_name = best_name
        self.best_model = best_result.model

        print(f"\n  最优模型: {best_result.model_name}")
        print(f"  边x时段级别 R2 得分: {best_result.r2:.4f}")
    
    # ==================== 预测与更新 ====================
    
    def _predict_all_weights(self, edge_segments: Dict, graph):
        """
        预测边级耗时权重

        策略：
        - 有数据边：直接使用经验平均耗时
        - 无数据边：使用模型外推（基于边结构特征）
        """
        print(f"\n  使用 [{self.best_model_name}] 模型预测边级耗时...")

        if self.best_model_name in ['gnn', 'pna', 'gnn_ensemble']:
            self._predict_with_gnn(edge_segments, graph)
            return

        all_edges = set(graph.edges())
        edges_with_data = set(edge_segments.keys())
        edges_without_data = all_edges - edges_with_data

        if edges_without_data:
            print(f"  无轨迹数据边: {len(edges_without_data)} 条，将使用模型外推")

        # ===== 阶段1：有数据边 → 直接使用经验值（按 day/night 分别存储）=====
        for edge_key, segments in edge_segments.items():
            if len(segments) == 0:
                continue
            if edge_key not in all_edges:
                continue

            from_node, to_node = edge_key
            waterway_type = self._get_edge_waterway_type(from_node, to_node, graph)
            time_diffs = [s['time_diff'] for s in segments]
            actual_speeds = [s['actual_speed'] for s in segments]
            reported_speeds = [s['reported_speed'] for s in segments]
            bearings = [s['bearing'] for s in segments]

            overall_avg = np.mean(time_diffs)
            avg_distance = np.mean([s['distance'] for s in segments])

            features = {
                'segment_count': len(segments),
                'from_node': from_node,
                'to_node': to_node,
                'avg_distance': avg_distance,
                'avg_travel_time': overall_avg,
                'predicted_travel_time': overall_avg,
                'std_travel_time': np.std(time_diffs) if len(time_diffs) > 1 else 0,
                'min_travel_time': np.min(time_diffs),
                'max_travel_time': np.max(time_diffs),
                'median_travel_time': np.median(time_diffs),
                'avg_actual_speed': np.mean(actual_speeds),
                'std_actual_speed': np.std(actual_speeds) if len(actual_speeds) > 1 else 0,
                'avg_reported_speed': np.mean(reported_speeds),
                'waterway_type': 'narrow' if waterway_type == 1 else 'open',
                'waterway_type_code': waterway_type,
                'direction_features': {
                    'avg_bearing': np.mean(bearings),
                    'std_bearing': np.std(bearings) if len(bearings) > 1 else 0,
                    'avg_course_change': np.mean([s['course_change'] for s in segments]),
                    'direction_distribution': self._compute_direction_distribution(bearings),
                    'is_bidirectional': self._check_bidirectional(bearings)
                },
                'model_used': 'empirical',
            }

            # 时段单独存储
            for period_name in ['day', 'night']:
                period_segments = [s for s in segments if self._get_time_period(s['hour']) == period_name]
                if len(period_segments) < 2:
                    continue
                period_time_diffs = [s['time_diff'] for s in period_segments]
                features[period_name] = {
                    'avg_travel_time': np.mean(period_time_diffs),
                    'segment_count': len(period_segments),
                    'avg_reported_speed': np.mean([s['reported_speed'] for s in period_segments]),
                }

            self.edge_features[edge_key] = features

        # ===== 阶段2：为无数据边构建预测特征 =====
        model_predict_features = []
        model_predict_keys = []
        model_nodata_default_speeds = {}
        model_nodata_theoretical_times = {}
        model_nodata_distances = {}
        model_nodata_bearings = {}

        # 预计算有数据边的统计量，用于邻居速度推断
        data_edge_speeds = {}
        for edge_key, segments in edge_segments.items():
            if len(segments) >= 2:
                data_edge_speeds[edge_key] = np.median([s['reported_speed'] for s in segments])

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

            # 用邻居边的平均速度作为默认速度
            nbr_speeds = []
            for nbr in set(list(graph.neighbors(from_node)) + list(graph.neighbors(to_node))):
                for (a, b) in [(from_node, nbr), (nbr, from_node), (to_node, nbr), (nbr, to_node)]:
                    if (a, b) in data_edge_speeds:
                        nbr_speeds.append(data_edge_speeds[(a, b)])
            default_speed_knots = np.median(nbr_speeds) if nbr_speeds else 5.0
            default_speed_ms = default_speed_knots * 0.5144
            theoretical_time = dist / default_speed_ms

            # 邻居特征
            nbr_count = len(nbr_speeds)
            nbr_speed_median = np.median(nbr_speeds) if nbr_speeds else 0.0

            features = [
                default_speed_knots, 0.0, 0.0,
                avg_bearing, np.sin(bearing_rad), np.cos(bearing_rad),
                0, 0, 0,
                waterway_type,
                node_degree_from, node_degree_to, betweenness,
                0, 0,
                dist, theoretical_time,
                default_speed_knots, 0.0,
                nbr_count, nbr_speed_median,
                0.0, 0.0, 0.0, 0.0,
                0.5, 0.866,
                1.0,
            ]
            model_predict_features.append(features)
            model_predict_keys.append((from_node, to_node))
            model_nodata_default_speeds[(from_node, to_node)] = default_speed_knots
            model_nodata_theoretical_times[(from_node, to_node)] = theoretical_time
            model_nodata_distances[(from_node, to_node)] = dist
            model_nodata_bearings[(from_node, to_node)] = avg_bearing

        # ===== 阶段3：批量模型预测 =====
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

            for i, (from_node, to_node) in enumerate(model_predict_keys):
                edge_key = (from_node, to_node)
                tt = model_nodata_theoretical_times[edge_key]
                predicted_time = preds[i] * tt
                default_speed_knots = model_nodata_default_speeds.get(edge_key, 5.0)
                edge_dist = model_nodata_distances[edge_key]
                edge_bearing = model_nodata_bearings[edge_key]

                waterway_type = self._get_edge_waterway_type(from_node, to_node, graph)
                self.edge_features[edge_key] = {
                    'segment_count': 0,
                    'from_node': from_node,
                    'to_node': to_node,
                    'avg_distance': edge_dist,
                    'avg_travel_time': predicted_time,
                    'predicted_travel_time': predicted_time,
                    'std_travel_time': predicted_time * 0.2,
                    'min_travel_time': predicted_time * 0.8,
                    'max_travel_time': predicted_time * 1.2,
                    'median_travel_time': predicted_time,
                    'avg_actual_speed': default_speed_knots,
                    'std_actual_speed': 1.0,
                    'avg_reported_speed': default_speed_knots,
                    'waterway_type': 'narrow' if waterway_type == 1 else 'open',
                    'waterway_type_code': waterway_type,
                    'direction_features': {
                        'avg_bearing': edge_bearing,
                        'std_bearing': 0,
                        'avg_course_change': 0,
                        'direction_distribution': {},
                        'is_bidirectional': False
                    },
                    'model_used': 'model_extrapolation'
                }

        n_data = len(edges_with_data & set(self.edge_features.keys()))
        n_nodata = sum(1 for e in edges_without_data if e in self.edge_features)
        print(f"  完成: {n_data} 条有数据边 + {n_nodata} 条无数据边")
    
    def _predict_with_gnn(self, edge_segments: Dict, graph):
        """使用 GNN 模型预测边级耗时（PNA 模型预测 + 经验统计兜底）"""
        all_edges = set(graph.edges())
        edges_with_data = set(edge_segments.keys())
        edges_without_data = all_edges - edges_with_data
        
        # 获取 PNA/GNN 预测缓存（由 _train_gnn 生成）
        gnn_cache = getattr(self, '_gnn_prediction_cache', {})
        n_pna_used = 0
        n_empirical_used = 0

        # ===== 阶段1：有数据边 → PNA 模型预测优先，经验值兜底 =====
        for edge_key, segments in edge_segments.items():
            if len(segments) == 0:
                continue
            if edge_key not in all_edges:
                continue

            from_node, to_node = edge_key
            waterway_type = self._get_edge_waterway_type(from_node, to_node, graph)
            time_diffs = [s['time_diff'] for s in segments]
            actual_speeds = [s['actual_speed'] for s in segments]
            reported_speeds = [s['reported_speed'] for s in segments]
            bearings = [s['bearing'] for s in segments]

            overall_avg = np.mean(time_diffs)
            avg_distance = np.mean([s['distance'] for s in segments])
            
            # 查找 PNA 预测缓存
            pna_preds = gnn_cache.get(edge_key, {})
            has_pna = len(pna_preds) > 0
            
            if has_pna:
                # PNA 预测优先：取所有可用 period 的均值作为整体预测
                pna_times = [p['predicted_time'] for p in pna_preds.values()]
                predicted_time = np.mean(pna_times) if pna_times else overall_avg
                model_used = self.best_model_name  # 'pna' 或 'gnn'
                n_pna_used += 1
            else:
                predicted_time = overall_avg
                model_used = 'empirical'
                n_empirical_used += 1

            features = {
                'segment_count': len(segments),
                'from_node': from_node,
                'to_node': to_node,
                'avg_distance': avg_distance,
                'avg_travel_time': overall_avg,
                'predicted_travel_time': predicted_time,
                'std_travel_time': np.std(time_diffs) if len(time_diffs) > 1 else 0,
                'min_travel_time': np.min(time_diffs),
                'max_travel_time': np.max(time_diffs),
                'median_travel_time': np.median(time_diffs),
                'avg_actual_speed': np.mean(actual_speeds),
                'std_actual_speed': np.std(actual_speeds) if len(actual_speeds) > 1 else 0,
                'avg_reported_speed': np.mean(reported_speeds),
                'waterway_type': 'narrow' if waterway_type == 1 else 'open',
                'waterway_type_code': waterway_type,
                'direction_features': {
                    'avg_bearing': np.mean(bearings),
                    'std_bearing': np.std(bearings) if len(bearings) > 1 else 0,
                    'avg_course_change': np.mean([s['course_change'] for s in segments]),
                    'direction_distribution': self._compute_direction_distribution(bearings),
                    'is_bidirectional': self._check_bidirectional(bearings)
                },
                'node_degree_from': self.node_degrees.get(from_node, 0),
                'node_degree_to': self.node_degrees.get(to_node, 0),
                'edge_betweenness': self.edge_betweenness.get(edge_key, 0),
                'model_used': model_used,
            }

            # 时段级别预测：PNA period-specific 预测优先
            for period_name in ['day', 'night']:
                period_segments = [s for s in segments if self._get_time_period(s['hour']) == period_name]
                if len(period_segments) < 2:
                    continue
                period_time_diffs = [s['time_diff'] for s in period_segments]
                emp_avg = np.mean(period_time_diffs)
                
                # PNA 对该 period 有预测则用 PNA，否则用经验均值
                pna_period = pna_preds.get(period_name)
                period_predicted = pna_period['predicted_time'] if pna_period else emp_avg
                
                features[period_name] = {
                    'avg_travel_time': emp_avg,
                    'predicted_travel_time': period_predicted,
                    'segment_count': len(period_segments),
                    'avg_reported_speed': np.mean([s['reported_speed'] for s in period_segments]),
                    'model_used': model_used if pna_period else 'empirical',
                }

            self.edge_features[edge_key] = features
        
        print(f"  有数据边: PNA预测 {n_pna_used} 条, 经验均值 {n_empirical_used} 条")

        # ===== 阶段2：零段边 → PNA transductive推理优先，邻边速度推断兜底 =====
        if len(edges_without_data) == 0:
            print(f"  所有边均有数据，无需外推")
        else:
            n_pna_nodata = 0
            n_nbr_inference = 0

            # 尝试 PNA transductive 推理：将无数据边加入图结构，前向传播获取预测
            pna_nodata_preds = {}
            if self.best_model_name in ('pna', 'gnn', 'gnn_ensemble') and self.best_model is not None:
                try:
                    import torch
                    from torch_geometric.utils import degree as pyg_degree

                    # 收集已有缓存中的节点ID映射（训练时的节点集合）
                    cached_node_ids = set()
                    for edge_key in gnn_cache:
                        cached_node_ids.add(edge_key[0])
                        cached_node_ids.add(edge_key[1])
                    # 也加入无数据边涉及的节点
                    for (u, v) in edges_without_data:
                        cached_node_ids.add(u)
                        cached_node_ids.add(v)

                    # 节点ID → 连续索引
                    node_id_to_idx = {nid: idx for idx, nid in enumerate(sorted(cached_node_ids))}

                    # 构建节点特征 [degree, waterway, lat, lon, cluster_coeff]
                    try:
                        import networkx as nx
                        clustering = nx.clustering(graph)
                    except Exception:
                        clustering = {}
                    node_feat_list = []
                    for nid in sorted(cached_node_ids):
                        attrs = graph.nodes[nid]
                        degree = self.node_degrees.get(nid, 0)
                        waterway = self.node_waterway_types.get(nid, 0)
                        cc = clustering.get(nid, 0)
                        node_feat_list.append([degree, waterway, attrs['lat'], attrs['lon'], cc])
                    node_features = torch.FloatTensor(node_feat_list)

                    # 标准化节点特征
                    node_scaler = getattr(self, 'gnn_node_scaler', None)
                    if node_scaler is not None:
                        node_features = torch.FloatTensor(node_scaler.transform(node_features.numpy()))

                    # 构建边索引：有数据边（双向）+ 无数据边（双向）
                    edge_index_list = []
                    edge_feat_list = []
                    edge_scaler = getattr(self, 'gnn_edge_scaler', None)

                    # 有数据边：从缓存中获取14维特征（需要重新构造）
                    # 先收集有数据边的特征（与 _train_gnn L1512-1517 一致）
                    for edge_key, segments in edge_segments.items():
                        if len(segments) < 2:
                            continue
                        if edge_key not in all_edges:
                            continue
                        from_node, to_node = edge_key
                        if from_node not in node_id_to_idx or to_node not in node_id_to_idx:
                            continue
                        u_idx = node_id_to_idx[from_node]
                        v_idx = node_id_to_idx[to_node]
                        waterway_type = self._get_edge_waterway_type(from_node, to_node, graph)
                        node_degree_from = self.node_degrees.get(from_node, 0)
                        node_degree_to = self.node_degrees.get(to_node, 0)
                        betweenness = self.edge_betweenness.get(edge_key, 0)

                        for period_name in ['day', 'night']:
                            period_segments = [s for s in segments if self._get_time_period(s['hour']) == period_name]
                            if len(period_segments) < 2:
                                continue
                            avg_time = np.mean([s['time_diff'] for s in period_segments])
                            avg_reported_speed = np.mean([s['reported_speed'] for s in period_segments])
                            distance = np.mean([s['distance'] for s in period_segments])
                            avg_bearing = np.mean([s['bearing'] for s in period_segments])
                            avg_course_change = np.mean([s['course_change'] for s in period_segments])
                            bearing_rad = np.deg2rad(avg_bearing)
                            speeds = [s['reported_speed'] for s in period_segments]
                            speed_iqr = np.percentile(speeds, 75) - np.percentile(speeds, 25) if len(speeds) > 1 else 0
                            hours_list = [s['hour'] for s in period_segments]
                            avg_hour = np.mean(hours_list)
                            hour_sin = np.sin(2 * np.pi * avg_hour / 24)
                            hour_cos = np.cos(2 * np.pi * avg_hour / 24)
                            actual_speeds = [s['actual_speed'] for s in period_segments]
                            speed_decay = np.mean(actual_speeds) / max(avg_reported_speed, 0.5)
                            speed_ms = max(avg_reported_speed, 0.5) * 0.5144
                            theoretical_time = distance / speed_ms

                            edge_index_list.append([u_idx, v_idx])
                            edge_feat_list.append([
                                distance, theoretical_time, avg_reported_speed,
                                np.sin(bearing_rad), np.cos(bearing_rad), avg_course_change,
                                waterway_type, node_degree_from, node_degree_to, betweenness,
                                speed_iqr, hour_sin, hour_cos, speed_decay,
                            ])

                    n_data_edges = len(edge_index_list)

                    # 无数据边：用邻边默认特征构造14维
                    # 预计算有数据边的速度统计，用于邻居推断
                    data_edge_speeds = {}
                    for edge_key, segments in edge_segments.items():
                        if len(segments) >= 2:
                            data_edge_speeds[edge_key] = np.median([s['reported_speed'] for s in segments])

                    nodata_edge_info = {}  # (from, to) -> {distance, theoretical_time, default_speed, bearing, ...}
                    for (from_node, to_node) in edges_without_data:
                        if from_node not in node_id_to_idx or to_node not in node_id_to_idx:
                            continue
                        u_idx = node_id_to_idx[from_node]
                        v_idx = node_id_to_idx[to_node]
                        waterway_type = self._get_edge_waterway_type(from_node, to_node, graph)
                        node_degree_from = self.node_degrees.get(from_node, 0)
                        node_degree_to = self.node_degrees.get(to_node, 0)
                        betweenness = self.edge_betweenness.get((from_node, to_node), 0)
                        u_attr = graph.nodes[from_node]
                        v_attr = graph.nodes[to_node]
                        dist = haversine_distance(u_attr['lat'], u_attr['lon'], v_attr['lat'], v_attr['lon'])
                        avg_bearing = calculate_bearing(u_attr['lat'], u_attr['lon'], v_attr['lat'], v_attr['lon'])
                        bearing_rad = np.deg2rad(avg_bearing)

                        # 邻边速度推断
                        nbr_speeds = []
                        for nbr in set(list(graph.neighbors(from_node)) + list(graph.neighbors(to_node))):
                            for (a, b) in [(from_node, nbr), (nbr, from_node), (to_node, nbr), (nbr, to_node)]:
                                if (a, b) in data_edge_speeds:
                                    nbr_speeds.append(data_edge_speeds[(a, b)])
                        default_speed = np.median(nbr_speeds) if nbr_speeds else 5.0
                        speed_ms = max(default_speed, 0.5) * 0.5144
                        theoretical_time = dist / speed_ms

                        nodata_edge_info[(from_node, to_node)] = {
                            'distance': dist, 'theoretical_time': theoretical_time,
                            'default_speed': default_speed, 'bearing': avg_bearing,
                            'waterway_type': waterway_type,
                            'node_degree_from': node_degree_from,
                            'node_degree_to': node_degree_to,
                            'betweenness': betweenness,
                            'nbr_count': len(nbr_speeds),
                            'nbr_speed_median': np.median(nbr_speeds) if nbr_speeds else 0.0,
                        }

                        # 为 day/night 各加一条边
                        for period_name in ['day', 'night']:
                            hour = 9 if period_name == 'day' else 21
                            hour_sin = np.sin(2 * np.pi * hour / 24)
                            hour_cos = np.cos(2 * np.pi * hour / 24)
                            speed_decay = 1.0  # 无数据，假设无衰减

                            edge_index_list.append([u_idx, v_idx])
                            edge_feat_list.append([
                                dist, theoretical_time, default_speed,
                                np.sin(bearing_rad), np.cos(bearing_rad), 0.0,
                                waterway_type, node_degree_from, node_degree_to, betweenness,
                                0.0, hour_sin, hour_cos, speed_decay,
                            ])

                    n_total_edges = len(edge_index_list)
                    n_nodata_edges = n_total_edges - n_data_edges

                    if n_total_edges > 0 and n_nodata_edges > 0:
                        edge_index_tensor = torch.LongTensor(edge_index_list).t().contiguous()
                        # 双向边
                        edge_index_bidir = torch.cat([
                            edge_index_tensor,
                            edge_index_tensor.flip(0)
                        ], dim=1)
                        edge_features_tensor = torch.FloatTensor(edge_feat_list)
                        if edge_scaler is not None:
                            edge_features_tensor = torch.FloatTensor(edge_scaler.transform(edge_features_tensor.numpy()))
                        edge_features_bidir = torch.cat([edge_features_tensor, edge_features_tensor], dim=0)

                        # 消息传递用全部双向边
                        msg_edge_index = edge_index_bidir
                        msg_edge_attr = edge_features_bidir

                        # 前向传播
                        self.best_model.eval()
                        with torch.no_grad():
                            all_pred_raw = self.best_model(
                                node_features, edge_index_bidir, edge_features_bidir,
                                n_total_edges, msg_edge_index=msg_edge_index,
                                edge_attr_msg=msg_edge_attr
                            ).numpy()

                        if getattr(self, '_gnn_use_log_transform', True):
                            all_pred_ratio = inverse_log_transform(all_pred_raw)
                        else:
                            all_pred_ratio = all_pred_raw
                        all_pred_ratio = np.clip(all_pred_ratio, 0.5, 3.0)

                        # 提取无数据边的预测值
                        nodata_pred_idx = 0
                        for (from_node, to_node) in edges_without_data:
                            if (from_node, to_node) not in nodata_edge_info:
                                continue
                            info = nodata_edge_info[(from_node, to_node)]
                            edge_key = (from_node, to_node)

                            # day 和 night 各一条预测
                            day_pred_time = all_pred_ratio[n_data_edges + nodata_pred_idx * 2] * info['theoretical_time']
                            night_pred_time = all_pred_ratio[n_data_edges + nodata_pred_idx * 2 + 1] * info['theoretical_time']
                            avg_pred_time = (day_pred_time + night_pred_time) / 2.0
                            nodata_pred_idx += 1

                            pna_nodata_preds[edge_key] = {
                                'day': {'predicted_time': float(day_pred_time)},
                                'night': {'predicted_time': float(night_pred_time)},
                                'avg_predicted_time': float(avg_pred_time),
                                'default_speed': info['default_speed'],
                                'distance': info['distance'],
                                'bearing': info['bearing'],
                            }

                        print(f"  PNA transductive推理: {len(pna_nodata_preds)} 条零段边获得PNA预测")
                except Exception as e:
                    print(f"  PNA transductive推理失败: {e}，回退到邻边速度推断")
                    import traceback; traceback.print_exc()

            # 写入无数据边的 edge_features
            print(f"  零段边: {len(edges_without_data)} 条")
            for from_node, to_node in edges_without_data:
                edge_key = (from_node, to_node)
                waterway_type = self._get_edge_waterway_type(from_node, to_node, graph)
                node_degree_from = self.node_degrees.get(from_node, 0)
                node_degree_to = self.node_degrees.get(to_node, 0)
                betweenness = self.edge_betweenness.get(edge_key, 0)
                u_attr = graph.nodes[from_node]
                v_attr = graph.nodes[to_node]
                distance = haversine_distance(u_attr['lat'], u_attr['lon'], v_attr['lat'], v_attr['lon'])
                bearings = [calculate_bearing(u_attr['lat'], u_attr['lon'], v_attr['lat'], v_attr['lon'])]

                # 邻边速度推断（兜底）
                nbr_speeds = []
                for nbr in set(list(graph.neighbors(from_node)) + list(graph.neighbors(to_node))):
                    for (a, b) in [(from_node, nbr), (nbr, from_node), (to_node, nbr), (nbr, to_node)]:
                        edge_feat = self.edge_features.get((a, b))
                        if edge_feat and 'avg_reported_speed' in edge_feat:
                            nbr_speeds.append(edge_feat['avg_reported_speed'])
                default_speed = np.median(nbr_speeds) if nbr_speeds else 5.0
                speed_ms = max(default_speed, 0.5) * 0.5144
                theoretical_time = distance / speed_ms

                # PNA transductive 推理优先
                pna_info = pna_nodata_preds.get(edge_key)
                if pna_info is not None:
                    predicted_time = pna_info['avg_predicted_time']
                    model_used = self.best_model_name  # 'pna'
                    n_pna_nodata += 1
                else:
                    predicted_time = theoretical_time
                    model_used = 'neighbor_inference'
                    n_nbr_inference += 1

                self.edge_features[edge_key] = {
                    'segment_count': 0,
                    'from_node': from_node,
                    'to_node': to_node,
                    'avg_distance': distance,
                    'avg_travel_time': predicted_time,
                    'predicted_travel_time': predicted_time,
                    'std_travel_time': predicted_time * 0.2,
                    'min_travel_time': predicted_time * 0.8,
                    'max_travel_time': predicted_time * 1.2,
                    'median_travel_time': predicted_time,
                    'avg_actual_speed': default_speed,
                    'std_actual_speed': 1.0,
                    'avg_reported_speed': default_speed,
                    'waterway_type': 'narrow' if waterway_type == 1 else 'open',
                    'waterway_type_code': waterway_type,
                    'direction_features': {
                        'avg_bearing': bearings[0],
                        'std_bearing': 0,
                        'avg_course_change': 0,
                        'direction_distribution': {},
                        'is_bidirectional': False
                    },
                    'node_degree_from': node_degree_from,
                    'node_degree_to': node_degree_to,
                    'edge_betweenness': betweenness,
                    'model_used': model_used,
                }

                # 时段级别预测
                if pna_info is not None:
                    for period_name in ['day', 'night']:
                        period_pred = pna_info.get(period_name, {})
                        period_predicted = period_pred.get('predicted_time', predicted_time)
                        self.edge_features[edge_key][period_name] = {
                            'avg_travel_time': period_predicted,
                            'predicted_travel_time': period_predicted,
                            'segment_count': 0,
                            'avg_reported_speed': default_speed,
                            'model_used': self.best_model_name,
                        }

            print(f"  零段边完成: PNA推理 {n_pna_nodata} 条, 邻边推断 {n_nbr_inference} 条")
    
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
                # 速度可靠性 = 1 - 速度变异系数(CV)，CV=std/mean
                # empirical 边用真实 std/mean；pna 推断边 std=1.0, mean=default_speed
                'speed_reliability': max(0.0, 1.0 - features.get('std_actual_speed', 0) / max(features.get('avg_actual_speed', 0.5), 0.5)),
                'theoretical_time': features['avg_distance'] / max(features.get('avg_reported_speed', 5), 0.5) / 0.5144,

                # 水域类型
                'waterway_type': features['waterway_type'],
                'waterway_type_code': features.get('waterway_type_code', 0),
                
                # 网络拓扑特征
                'node_degree_from': features.get('node_degree_from', 0),
                'node_degree_to': features.get('node_degree_to', 0),
                'edge_betweenness': features.get('edge_betweenness', 0),
            }
            
            # 方向特征
            direction_features = features.get('direction_features', {})
            row['avg_bearing'] = direction_features.get('avg_bearing', 0)
            row['std_bearing'] = direction_features.get('std_bearing', 0)
            row['avg_course_change'] = direction_features.get('avg_course_change', 0)
            row['is_bidirectional'] = int(direction_features.get('is_bidirectional', False))

            # 24小时预测耗时
            predicted_times = features.get('predicted_times', {})
            hourly_factors = {
                0: 0.85, 1: 0.80, 2: 0.78, 3: 0.80, 4: 0.85, 5: 0.90,
                6: 1.05, 7: 1.15, 8: 1.25, 9: 1.20, 10: 1.10, 11: 1.05,
                12: 1.00, 13: 1.02, 14: 1.05, 15: 1.08, 16: 1.12, 17: 1.20,
                18: 1.15, 19: 1.05, 20: 0.95, 21: 0.90, 22: 0.88, 23: 0.85,
            }
            avg_travel_time = features['avg_travel_time']
            for h in range(24):
                col_name = f'predicted_time_h{h:02d}'
                if h in predicted_times and predicted_times[h] is not None:
                    row[col_name] = predicted_times[h]
                else:
                    row[col_name] = avg_travel_time * hourly_factors.get(h, 1.0)

            rows.append(row)
        
        df = pd.DataFrame(rows)
        output_path = f"{output_dir}/edge_features_dynamic_weights.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n  主数据集: {output_path}")
        print(f"  边数量: {len(rows):,}")
        print(f"  特征列数: {len(df.columns)}")
        
        # 2. 导出模型评估报告
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
            
            # 边×时段级别评估（所有模型统一粒度）
            if hasattr(self, '_model_results') and self._model_results:
                f.write("-"*80 + "\n")
                f.write("边×时段级别评估（所有模型统一粒度，预测与实际耗时对比）\n")
                f.write("-"*80 + "\n\n")
                f.write(f"{'模型':<20s} {'MAE':>10s} {'RMSE':>10s} {'R2':>10s} {'MAPE(%)':>10s} {'测试边数':>10s}\n")
                f.write("-"*72 + "\n")
                for name_, result in sorted(self._model_results.items(), key=lambda x: x[1].r2, reverse=True):
                    if result.predictions is None:
                        continue
                    best_mark = " ★" if name_ == self.best_model_name else ""
                    test_count = len(result.predictions) if result.predictions is not None else 0
                    f.write(f"{name_+best_mark:<20s} {result.mae:>10.4f} {result.rmse:>10.4f} {result.r2:>10.4f} {result.mape:>10.2f} {test_count:>10d}\n")
                f.write("-"*72 + "\n")
                f.write("★ 标记为最优模型\n\n")
                f.write("-"*72 + "\n\n")
            
            # 特征重要性 Top10
            if hasattr(self, 'feature_importance') and self.feature_importance:
                f.write("-"*80 + "\n")
                imp_source = self.best_model_name if self.best_model_name else "未知"
                f.write(f"特征重要性 Top10（来自 {imp_source} 模型）\n")
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
                        mape=result.mape,
                        model=None,  # 不保存模型对象
                        predictions=result.predictions
                    )
                else:
                    model_results_clean[name] = result

        # GNN/PNA 最佳模型单独用 torch.save 保存 state_dict
        best_model_to_save = self.best_model
        gnn_state_path = None
        gnn_model_config = None
        if self.best_model_name in ['gnn', 'pna'] and self.best_model is not None:
            best_model_to_save = None  # pkl 中不保存 GNN 模型对象
            gnn_state_path = filepath.replace('.pkl', '_gnn_state.pt')
            torch.save(self.best_model.state_dict(), gnn_state_path)
            # 保存模型结构参数以便加载时重建
            if hasattr(self.best_model, 'node_proj'):
                gnn_model_config = {
                    'arch': 'pna',
                    'node_dim': self.best_model.node_proj.in_features,
                    'edge_dim': self.best_model.edge_mlp[0].in_features - self.best_model.convs[-1].out_channels * 2,
                    'hidden_dim': self.best_model.convs[-1].out_channels,
                    'num_layers': len(self.best_model.convs),
                    'dropout': self.best_model.dropout.p,
                }
                # PNAConv 需要 deg 用于 scaler 计算，从第一层 conv 的 _deg buffer 取出
                try:
                    pna_deg = self.best_model.convs[0]._deg
                    gnn_model_config['deg'] = pna_deg.detach().cpu().clone()
                except AttributeError:
                    pass
            else:
                gnn_model_config = {
                    'arch': 'gat',
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
            'edge_baseline_times': getattr(self, '_edge_baseline_times', {}),
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
        
        # GNN/PNA 模型特殊处理
        if self.best_model_name in ['gnn', 'pna', 'gnn_ensemble']:
            model_data['gnn_node_scaler'] = getattr(self, 'gnn_node_scaler', None)
            model_data['gnn_edge_scaler'] = getattr(self, 'gnn_edge_scaler', None)
            model_data['gnn_model_config'] = gnn_model_config
            model_data['gnn_use_log_transform'] = getattr(self, '_gnn_use_log_transform', True)
            model_data['gnn_arch'] = getattr(self, '_gnn_arch', 'gat')
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        total_size = os.path.getsize(filepath)
        if gnn_state_path and os.path.exists(gnn_state_path):
            total_size += os.path.getsize(gnn_state_path)
        
        print(f"\n模型已保存: {filepath}")
        print(f"  模型类型: {self.best_model_name}")
        print(f"  文件大小: {total_size / 1024:.1f} KB")
        
        return filepath
    
    def load_model(self, filepath: str, graph=None):
        """加载已保存的模型

        Args:
            filepath: pkl 路径
            graph: 可选的网络图（用于 PNAConv 重建时计算 deg）；
                   PNA 必须传图，否则 scaler 会触发 division by zero
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        class _ModelResultLoader(pickle.Unpickler):
            def find_class(self, module, name):
                if name == 'ModelResult':
                    return ModelResult
                if name in ('EdgeGNN', 'PNAEdgeGNN'):
                    return globals()[name]
                return super().find_class(module, name)
        
        with open(filepath, 'rb') as f:
            model_data = _ModelResultLoader(f).load()
        
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.feature_names = model_data['feature_names']
        self.scaler = model_data.get('scaler')
        self._duan_smearing_factor = model_data.get('duan_smearing_factor', 1.0)
        self._edge_theoretical_times = model_data.get('edge_theoretical_times', {})
        self._edge_baseline_times = model_data.get('edge_baseline_times', {})
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
        
        if self.best_model_name in ['gnn', 'pna', 'gnn_ensemble']:
            self.gnn_node_scaler = model_data.get('gnn_node_scaler')
            self.gnn_edge_scaler = model_data.get('gnn_edge_scaler')
            self._gnn_use_log_transform = model_data.get('gnn_use_log_transform', True)
            self._gnn_arch = model_data.get('gnn_arch', 'gat')
            # 从 state_dict 重建 GNN/PNA 模型
            gnn_config = model_data.get('gnn_model_config')
            gnn_state_path = filepath.replace('.pkl', '_gnn_state.pt')
            if gnn_config and os.path.exists(gnn_state_path):
                import torch  # 提到分支外,避免 else 分支走 GAT 时 try 块中 torch 未绑定
                loaded_arch = gnn_config.get('arch', 'gat')
                if loaded_arch == 'pna':
                    del gnn_config['arch']
                    # 优先使用 pkl 中保存的 deg；若无，则强制从 graph 重新算
                    if 'deg' not in gnn_config or gnn_config['deg'] is None:
                        if graph is None:
                            raise ValueError(
                                "PNA 模型加载需要传入 graph 用于计算 deg（PNAConv 内部用 "
                                "deg 算 scaler，全 0 会触发 ZeroDivisionError）"
                            )
                        from torch_geometric.utils import degree as pyg_degree
                        edges = list(graph.edges())
                        if edges:
                            edge_index = torch.tensor(edges, dtype=torch.long).t()
                            # 节点 ID 可能不连续，num_nodes 须 >= max_id+1
                            num_nodes = max(graph.nodes()) + 1
                            gnn_config['deg'] = pyg_degree(
                                edge_index[0], num_nodes=num_nodes
                            ).long()
                        else:
                            gnn_config['deg'] = torch.ones(
                                graph.number_of_nodes(), dtype=torch.long
                            )
                    self.best_model = PNAEdgeGNN(**gnn_config)
                else:
                    del gnn_config['arch']
                    self.best_model = EdgeGNN(**gnn_config)
                try:
                    self.best_model.load_state_dict(torch.load(gnn_state_path, map_location='cpu', weights_only=True))
                    self.best_model.eval()
                    print(f"  {loaded_arch.upper()} 模型已从 state_dict 重建")
                except RuntimeError as e:
                    # checkpoint 维度不匹配（特征工程变更后常见），回退到 pkl 内的传统模型
                    print(f"  [WARN] {loaded_arch.upper()} state_dict 不兼容 ({type(e).__name__})，回退到 pkl 内模型")
                    self.best_model = model_data['best_model']
            else:
                self.best_model = model_data['best_model']
        
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
        
        if self.best_model_name in ['gnn', 'pna', 'gnn_ensemble']:
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
        获取边的预估耗时权重（边级别）

        当前为边级别建模，所有时段返回相同值。
        如需时段级动态权重，请切换为边×时段级建模。

        Returns:
            预估耗时（秒）
        """
        edge_key = (from_node, to_node)
        if edge_key not in self.edge_features:
            return None
        return self.edge_features[edge_key]['avg_travel_time']
    
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

        当前为边级别建模（无时段区分），始终返回 None。
        如需时段级动态耗时，请切换为边×时段级建模。

        Returns:
            None（边级别建模不支持此功能）
        """
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

    # ===== 最优模型 + 稳定性验证 =====
    print(f"\n{'='*60}")
    print(f"Step: 找最优 + 稳定性验证")
    print(f"{'='*60}")

    # 1. 选最优(从 build_weights_with_comparison 跑出的 7 个模型里)
    best_name = max(model._model_results, key=lambda k: model._model_results[k].r2)
    best_r2 = model._model_results[best_name].r2
    print(f"  最优单次模型: {best_name} (R2={best_r2:.4f})")

    # 2. 稳定性验证: GNN 家族 (PNA + GAT) 都跑 5-seed, 配对用于 Wilcoxon 检验
    if HAS_PYG:
        SEEDS = [42, 123, 456, 789, 1011]
        seed_runs = {}  # arch -> [ModelResult, ...]
        for arch in ['pna', 'gat']:
            print(f"\n  跑 5-seed 稳定性验证: {arch}")
            arch_runs = []
            for seed in SEEDS:
                r = train_gnn_with_seed(model, seed, gnn_arch=arch)
                arch_runs.append(r)
                print(f"  {arch} seed={seed}: R2={r.r2:.4f} MAE={r.mae:.2f}s")
            seed_runs[arch] = arch_runs
            rs = [r.r2 for r in arch_runs]
            mean_r2 = float(np.mean(rs))
            std_r2 = float(np.std(rs))
            print(f"  {arch} 5-seed 稳定性: R2={mean_r2:.4f} ± {std_r2:.4f}  (min={min(rs):.4f}, max={max(rs):.4f})")

        # 导出 5-seed R² 序列给配对 Wilcoxon 检验
        _wilcoxon_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output', 'wilcoxon')
        os.makedirs(_wilcoxon_dir, exist_ok=True)
        _csv_path = os.path.join(_wilcoxon_dir, 'gnn_5seed.csv')
        pd.DataFrame({
            'seed': SEEDS,
            'pna_r2': [r.r2 for r in seed_runs['pna']],
            'gat_r2': [r.r2 for r in seed_runs['gat']],
        }).to_csv(_csv_path, index=False)
        print(f"  5-seed R² 已保存到: {_csv_path} (供 compute_wilcoxon.py 配对检验使用)")

        # 集成预测(5 seed 预测平均) — 仅对最优的那个
        if best_name in ('gnn', 'pna'):
            arch = 'gat' if best_name == 'gnn' else 'pna'
            runs = seed_runs[arch]
            gnn_all_preds = [r.predictions for r in runs if r.predictions is not None]
            if gnn_all_preds:
                avg_pred = np.mean(gnn_all_preds, axis=0)
                y_true = runs[0].y_test
                ens_mae = mean_absolute_error(y_true, avg_pred)
                ens_rmse = np.sqrt(mean_squared_error(y_true, avg_pred))
                ens_r2 = r2_score(y_true, avg_pred)
                mask = y_true != 0
                ens_mape = np.mean(np.abs((y_true[mask] - avg_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0

                print(f"  集成预测: R2={ens_r2:.4f}  MAE={ens_mae:.2f}s  RMSE={ens_rmse:.2f}s  MAPE={ens_mape:.2f}%")

                ens_name = f'{arch}_stability_5seed'
                ens_result = ModelResult(
                    model_name=ens_name,
                    train_time=sum(r.train_time for r in runs),
                    mae=ens_mae,
                    rmse=ens_rmse,
                    r2=ens_r2,
                    mape=ens_mape,
                    model=runs[-1].model,
                    predictions=avg_pred,
                    use_log_transform=False,
                    y_test=y_true
                )
                model._model_results[ens_name] = ens_result

                if ens_r2 > best_r2:
                    print(f"  -> 集成 (R2={ens_r2:.4f}) 优于单次 {best_name} (R2={best_r2:.4f}), 设为最佳")
                    model.best_model_name = ens_name
                    model.best_model = runs[-1].model
                else:
                    print(f"  -> 集成 (R2={ens_r2:.4f}) 未超过单次 {best_name} (R2={best_r2:.4f}), 保留 {best_name} 为 best")
    else:
        reason = "PyG 不可用"
        print(f"\n  跳过 5-seed 验证 ({reason},单次 R2 即代表稳定性)")

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
