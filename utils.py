"""
航道拓扑节点网络提取系统 - 工具函数
"""

import numpy as np
import math
from typing import Tuple, List


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两个经纬度点之间的距离（米）
    
    Args:
        lat1, lon1: 第一个点的纬度和经度
        lat2, lon2: 第二个点的纬度和经度
    
    Returns:
        距离（米）
    """
    R = 6371000  # 地球半径（米）
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两点之间的方位角（度）
    
    Args:
        lat1, lon1: 起点纬度和经度
        lat2, lon2: 终点纬度和经度
    
    Returns:
        方位角（0-360度）
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon_rad = math.radians(lon2 - lon1)
    
    x = math.sin(delta_lon_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad)
    
    bearing = math.atan2(x, y)
    bearing_deg = math.degrees(bearing)
    
    return (bearing_deg + 360) % 360


def calculate_angle_difference(angle1: float, angle2: float) -> float:
    """
    计算两个角度之间的最小差值（度）
    
    Args:
        angle1, angle2: 两个角度（度）
    
    Returns:
        角度差值（0-180度）
    """
    diff = abs(angle1 - angle2)
    return min(diff, 360 - diff)


def douglas_peucker(points: List[Tuple[float, float]], tolerance: float) -> List[Tuple[float, float]]:
    """
    Douglas-Peucker算法简化轨迹
    
    Args:
        points: 轨迹点列表 [(lat, lon), ...]
        tolerance: 容差（米）
    
    Returns:
        简化后的轨迹点列表
    """
    indices = douglas_peucker_indices(points, tolerance)
    return [points[i] for i in indices]


def douglas_peucker_indices(points: List[Tuple[float, float]], tolerance: float) -> List[int]:
    """
    Douglas-Peucker算法简化轨迹，返回保留点的索引
    
    Args:
        points: 轨迹点列表 [(lat, lon), ...]
        tolerance: 容差（米）
    
    Returns:
        保留点的索引列表
    """
    if len(points) <= 2:
        return list(range(len(points)))
    
    max_dist = 0
    max_idx = 0
    
    for i in range(1, len(points) - 1):
        dist = point_to_line_distance(points[i], points[0], points[-1])
        if dist > max_dist:
            max_dist = dist
            max_idx = i
    
    if max_dist > tolerance:
        left = douglas_peucker_indices(points[:max_idx + 1], tolerance)
        right = douglas_peucker_indices(points[max_idx:], tolerance)
        right = [max_idx + i for i in right]
        result = left[:-1] + right
    else:
        result = [0, len(points) - 1]
    
    return sorted(set(result))


def point_to_line_distance(point: Tuple[float, float], 
                          line_start: Tuple[float, float], 
                          line_end: Tuple[float, float]) -> float:
    """
    计算点到线段的距离（米）
    
    Args:
        point: 点坐标 (lat, lon)
        line_start: 线段起点 (lat, lon)
        line_end: 线段终点 (lat, lon)
    
    Returns:
        点到线段的距离（米）
    """
    lat, lon = point
    lat1, lon1 = line_start
    lat2, lon2 = line_end
    
    # 计算向量
    dx = lon2 - lon1
    dy = lat2 - lat1
    
    # 如果线段长度为0，返回点到点的距离
    if dx == 0 and dy == 0:
        return haversine_distance(lat, lon, lat1, lon1)
    
    # 计算投影参数
    t = ((lon - lon1) * dx + (lat - lat1) * dy) / (dx * dx + dy * dy)
    
    # 限制t在[0, 1]范围内
    t = max(0, min(1, t))
    
    # 计算投影点
    proj_lon = lon1 + t * dx
    proj_lat = lat1 + t * dy
    
    return haversine_distance(lat, lon, proj_lat, proj_lon)


def calculate_trajectory_statistics(trajectory: np.ndarray) -> dict:
    """
    计算轨迹统计信息
    
    Args:
        trajectory: 轨迹数组，shape为(n, 4)，列为[lat, lon, speed, course]
    
    Returns:
        统计信息字典
    """
    if len(trajectory) < 2:
        return {}
    
    lats = trajectory[:, 0]
    lons = trajectory[:, 1]
    speeds = trajectory[:, 2]
    courses = trajectory[:, 3]
    
    # 计算总距离
    total_distance = 0
    for i in range(len(trajectory) - 1):
        total_distance += haversine_distance(lats[i], lons[i], lats[i+1], lons[i+1])
    
    # 计算方向变化
    course_changes = []
    for i in range(len(courses) - 1):
        course_changes.append(calculate_angle_difference(courses[i], courses[i+1]))
    
    return {
        'total_distance': total_distance,
        'mean_speed': np.mean(speeds),
        'max_speed': np.max(speeds),
        'mean_course_change': np.mean(course_changes) if course_changes else 0,
        'max_course_change': np.max(course_changes) if course_changes else 0,
    }
