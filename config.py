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
}

# 轨迹平滑参数
SMOOTHING_CONFIG = {
    'window_size': 5,
    'process_noise': 0.01,
    'measurement_noise': 0.1,
    'use_kalman': True,
    'ema_alpha': 0.3,
}

# 节点提取参数
NODE_EXTRACTION_CONFIG = {
    'direction_change_threshold': 30.0,
    'speed_change_threshold': 3.0,
    'min_segment_length': 100.0,
    'douglas_peucker_tolerance': 50.0,
    'stop_point_radius': 50.0,
    'stop_point_min_duration': 300,
}

# 节点聚类参数
CLUSTERING_CONFIG = {
    'eps': 100.0,
    'min_samples': 5,
    'heading_weight': 30,
    'bifurcation_angle_threshold': 45.0,
    'merge_angle_threshold': 45.0,
    'turn_angle_threshold': 60.0,
}

# 拓扑网络构建参数
TOPOLOGY_CONFIG = {
    'edge_connection_distance': 200.0,
    'min_edge_weight': 3,
    'merge_similar_nodes': True,
    'node_merge_distance': 50.0,
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
