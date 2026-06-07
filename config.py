# -*- coding: utf-8 -*-
"""
航道拓扑节点网络提取系统 - 配置文件
"""

import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据路径配置
DATA_CONFIG = {
    'data_dir': os.path.join(BASE_DIR, 'Data'),
    'output_dir': os.path.join(BASE_DIR, 'output'),
    'file1': '基于海量轨迹数据的船舶智能导航路径规划数据集构建与应用1_20260401204631.xlsx',
    'file2': '基于海量轨迹数据的船舶智能导航路径规划数据集构建与应用2_20260401204651.xlsx',
}

# 数据清洗参数
CLEANING_CONFIG = {
    'max_speed': 30.0,
    'min_speed': 0.1,
    'max_acceleration': 5.0,
    'max_time_gap': 3600,
    'max_distance_jump': 500,
    'min_trajectory_points': 10,
    'iforest_n_estimators': 100,
    'iforest_contamination': 0.01,
}

# 轨迹平滑参数
SMOOTHING_CONFIG = {
    'window_size': 5,
    'ema_alpha': 0.3,
    # 2D KF + RTS 参数
    'use_kalman': True,
    'kf_process_noise_pos': 0.01,
    'kf_process_noise_vel': 0.001,
    'kf_measurement_noise': 0.1,
    'use_rts_smoother': True,
}

# 节点提取参数
NODE_EXTRACTION_CONFIG = {
    'direction_change_threshold': 30.0,
    'speed_change_threshold': 3.0,
    'min_segment_length': 100.0,
    'douglas_peucker_tolerance': 30.0,  # 50 → 30（更精细的轨迹保留，特别是在开阔海域）
    'stop_point_radius': 50.0,
    'stop_point_min_duration': 300,
    # 自适应阈值参数（基于文献[慕志颖2026]）
    'use_adaptive_threshold': True,   # True=自适应阈值，False=固定阈值
    'adaptive_threshold_multiplier': 1.5,  # 标准差倍数
}

# 节点聚类参数
CLUSTERING_CONFIG = {
    'eps': 150.0,         # 100 → 150（扩大聚类半径，让稀疏区域的节点能聚到一起）
    'min_samples': 3,     # 5 → 3（降低聚类门槛，减少噪声丢弃）
    'heading_weight': 100,
    'bifurcation_angle_threshold': 45.0,
    'merge_angle_threshold': 45.0,
    'turn_angle_threshold': 60.0,
}

# 拓扑网络构建参数
TOPOLOGY_CONFIG = {
    'edge_connection_distance': 200.0,
    'min_edge_weight': 3,
    'min_ship_count': 3,
    'merge_similar_nodes': True,
    'node_merge_distance': 200.0,
}

# HMM发射概率自适应配置
EMISSION_VARIANCE_CONFIG = {
    'base_variance': 100.0 ** 2,
    'min_variance': 30.0 ** 2,
    'max_variance': 300.0 ** 2,
}

# KD-Tree 空间搜索配置（替代原 Grid 搜索）
KD_TREE_CONFIG = {
    'enabled': True,
    'search_radius_multiplier': 2.0,
    'min_search_radius': 200.0,
    'max_search_radius': 10000.0,
}

# 可视化参数
VISUALIZATION_CONFIG = {
    'figure_size': (16, 12),
    'dpi': 300,
    'node_size': 50,
    'edge_width': 1.5,
    'trajectory_alpha': 0.3,
    'show_node_labels': True,
}
