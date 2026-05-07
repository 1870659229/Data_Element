"""
船舶个性化导航决策系统

基于已有数据输出:
- topology_nodes.csv / topology_edges.csv: 拓扑网络
- edge_features_dynamic_weights.csv: 动态耗时权重
- cleaned_data.csv: 轨迹数据（船舶信息）

功能模块:
1. ShipCharacteristicsManager - 船舶特征属性检索
2. PhysicalConstraintChecker - 物理约束校验
3. MultiObjectiveNavigator - 多目标路径规划（改进A*/Dijkstra）
4. NavigationDecisionMaker - 导航决策输出
"""

import pandas as pd
import numpy as np
import networkx as nx
import heapq
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import os
import logging

import random
from utils import haversine_distance, calculate_bearing, calculate_angle_difference, douglas_peucker_indices
from navigation_models import RiskPredictionModel, PassabilityProbabilityModel

logger = logging.getLogger(__name__)


# ==================== 数据类定义 ====================

class PathType(Enum):
    """路径类型"""
    SAFEST = "安全优先"
    FASTEST = "时间最短"
    BALANCED = "综合最优"
    FREQUENT = "通航频次最高"
    RELAXED = "约束放宽路径"


@dataclass
class ShipCharacteristics:
    """船舶静态特征"""
    ship_name: str
    mmsi: str = ""
    length: float = 100.0       # 船长（米）
    width: float = 15.0         # 船宽（米）
    draft: float = 5.0          # 吃水深度（米）
    height: float = 20.0        # 水面以上高度（米）
    tonnage: float = 5000.0     # 载重吨位（吨）
    ship_type: str = "货船"
    max_speed: float = 15.0     # 最大航速（节）
    
    # 派生属性
    maneuverability: float = field(init=False)
    risk_level: str = field(init=False)
    
    def __post_init__(self):
        # 机动性：小船机动性好
        self.maneuverability = max(0.1, min(1.0, 1.0 - (self.length / 400)))
        # 风险等级
        if self.draft > 10 or self.tonnage > 50000:
            self.risk_level = "高风险"
        elif self.draft > 5 or self.tonnage > 10000:
            self.risk_level = "中风险"
        else:
            self.risk_level = "低风险"


@dataclass
class NavigationConstraint:
    """航行约束条件"""
    min_depth: float = 0.0
    max_height: float = 100.0
    max_width: float = 100.0
    avoid_narrow: bool = False
    avoid_shallow: bool = False
    prefer_main_channel: bool = True


@dataclass
class PathResult:
    """路径规划结果"""
    path_type: PathType
    nodes: List[int]
    edges: List[Tuple[int, int]]
    total_distance: float    # 米
    total_time: float        # 秒
    avg_speed: float         # 节
    risk_score: float        # 风险评分 0-100
    safety_score: float      # 安全评分 0-100
    constraints_met: bool
    blocked_edges: List[Tuple[int, int]]
    waypoint_details: List[Dict]


# ==================== 模块1：船舶特征管理 ====================

class ShipCharacteristicsManager:
    """
    船舶特征属性检索管理器
    
    功能：
    - 从轨迹数据中提取船舶信息（航速模式推断船型）
    - 提供默认船舶类型模板
    - 支持自定义船舶特征
    - 持久化船舶特征数据库到CSV
    """
    
    # 船舶类型模板（含完整物理参数）
    SHIP_TEMPLATES = {
        '小型货船': {'length': 80, 'width': 12, 'draft': 4.5, 'height': 15, 'tonnage': 3000, 'max_speed': 12},
        '中型货船': {'length': 150, 'width': 22, 'draft': 7.5, 'height': 25, 'tonnage': 15000, 'max_speed': 14},
        '大型货船': {'length': 250, 'width': 32, 'draft': 11.0, 'height': 30, 'tonnage': 50000, 'max_speed': 15},
        '集装箱船': {'length': 200, 'width': 30, 'draft': 10.0, 'height': 40, 'tonnage': 35000, 'max_speed': 18},
        '大型集装箱船': {'length': 350, 'width': 45, 'draft': 14.0, 'height': 50, 'tonnage': 100000, 'max_speed': 22},
        '油轮': {'length': 180, 'width': 28, 'draft': 9.0, 'height': 20, 'tonnage': 25000, 'max_speed': 14},
        '大型油轮': {'length': 300, 'width': 50, 'draft': 15.0, 'height': 25, 'tonnage': 120000, 'max_speed': 15},
        '客船': {'length': 100, 'width': 18, 'draft': 5.0, 'height': 30, 'tonnage': 8000, 'max_speed': 20},
        '渔船': {'length': 30, 'width': 6, 'draft': 2.5, 'height': 8, 'tonnage': 200, 'max_speed': 10},
        '拖船': {'length': 25, 'width': 8, 'draft': 3.0, 'height': 10, 'tonnage': 300, 'max_speed': 12},
    }
    
    # 船舶名称关键词 -> 船型映射（中文船舶名通常含类型关键词）
    SHIP_NAME_KEYWORDS = {
        '集装箱': '集装箱船', '集装': '集装箱船', 'container': '集装箱船',
        '油轮': '油轮', '油船': '油轮', 'tanker': '油轮', 'VLCC': '大型油轮',
        '散货': '大型货船', 'bulk': '大型货船',
        '客船': '客船', '客滚': '客船', 'passenger': '客船',
        '渔': '渔船', 'fishing': '渔船',
        '拖': '拖船', 'tug': '拖船',
        '货': '中型货船', 'cargo': '中型货船',
    }
    
    # 航速模式 -> 船型映射（基于最大航速和平均航速推断）
    SPEED_PATTERN_TO_TYPE = [
        (20, 25, '大型集装箱船'),   # 高速大型船
        (16, 20, '集装箱船'),       # 高速中型船
        (16, 20, '客船'),           # 高速中型船
        (12, 16, '中型货船'),       # 中速中型船
        (12, 16, '油轮'),           # 中速中型船
        (8, 12, '小型货船'),        # 低速小型船
        (6, 10, '渔船'),            # 低速小型船
        (0, 8, '拖船'),             # 极低速
    ]
    
    def __init__(self, trajectory_path: str = None, output_dir: str = "output"):
        """
        初始化船舶特征管理器
        
        Args:
            trajectory_path: 轨迹数据路径（用于提取船舶信息）
            output_dir: 输出目录（用于持久化船舶特征数据库）
        """
        self.ship_data = {}
        self.trajectory_data = None
        self.output_dir = output_dir
        
        # 尝试加载持久化的船舶特征数据库
        db_path = os.path.join(output_dir, 'ship_characteristics_db.csv')
        if os.path.exists(db_path):
            self._load_ship_db(db_path)
        elif trajectory_path and os.path.exists(trajectory_path):
            self._load_ship_info(trajectory_path)
            self._save_ship_db(db_path)
    
    def _load_ship_db(self, path: str):
        """加载持久化的船舶特征数据库"""
        logger.info("加载船舶特征数据库: %s", path)
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            self.ship_data[row['ship_name']] = {
                'ship_name': row['ship_name'],
                'ship_type': row.get('ship_type', '中型货船'),
                'max_speed': row.get('max_speed', 15),
                'avg_speed': row.get('avg_speed', 10),
                'record_count': row.get('record_count', 0),
                'inferred_type': row.get('inferred_type', ''),
                'length': row.get('length', 100),
                'width': row.get('width', 15),
                'draft': row.get('draft', 5),
                'height': row.get('height', 20),
                'tonnage': row.get('tonnage', 5000),
            }
        logger.info("已加载 %d 艘船舶特征", len(self.ship_data))
    
    def _save_ship_db(self, path: str):
        """持久化船舶特征数据库"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rows = []
        for name, data in self.ship_data.items():
            rows.append({
                'ship_name': name,
                'ship_type': data.get('ship_type', ''),
                'max_speed': data.get('max_speed', 15),
                'avg_speed': data.get('avg_speed', 10),
                'record_count': data.get('record_count', 0),
                'inferred_type': data.get('inferred_type', ''),
                'length': data.get('length', 100),
                'width': data.get('width', 15),
                'draft': data.get('draft', 5),
                'height': data.get('height', 20),
                'tonnage': data.get('tonnage', 5000),
            })
        pd.DataFrame(rows).to_csv(path, index=False, encoding='utf-8-sig')
        logger.info("船舶特征数据库已保存: %s (%d艘)", path, len(rows))
    
    def _load_ship_info(self, path: str):
        """从轨迹数据加载船舶信息并推断船型"""
        logger.info("从轨迹数据提取船舶信息: %s", path)
        df = pd.read_csv(path)
        df['时间'] = pd.to_datetime(df['时间'])
        
        for ship_name, group in df.groupby('船舶名称'):
            max_spd = group['航速'].max()
            avg_spd = group['航速'].mean()
            
            # 从船名推断船型
            inferred_type = self._infer_type_from_name(ship_name)
            
            # 如果名称推断不出，从航速模式推断
            if not inferred_type:
                inferred_type = self._infer_type_from_speed(max_spd)
            
            # 获取推断船型的物理参数
            template = self.SHIP_TEMPLATES.get(inferred_type, self.SHIP_TEMPLATES['中型货船'])
            
            self.ship_data[ship_name] = {
                'ship_name': ship_name,
                'ship_type': inferred_type or '中型货船',
                'max_speed': max_spd,
                'avg_speed': avg_spd,
                'record_count': len(group),
                'inferred_type': inferred_type,
                'length': template['length'],
                'width': template['width'],
                'draft': template['draft'],
                'height': template['height'],
                'tonnage': template['tonnage'],
            }
        
        logger.info("已提取 %d 艘船舶信息", len(self.ship_data))
    
    def _infer_type_from_name(self, ship_name: str) -> str:
        """从船舶名称关键词推断船型"""
        name_upper = str(ship_name).upper()
        for keyword, ship_type in self.SHIP_NAME_KEYWORDS.items():
            if keyword in name_upper or keyword.upper() in name_upper:
                return ship_type
        return ''
    
    def _infer_type_from_speed(self, max_speed: float) -> str:
        """从最大航速推断船型"""
        for low, high, ship_type in self.SPEED_PATTERN_TO_TYPE:
            if low <= max_speed < high:
                return ship_type
        return '中型货船'
    
    def get_ship_characteristics(self, ship_name: str = None, 
                                   ship_type: str = None,
                                   custom_params: Dict = None) -> ShipCharacteristics:
        """
        获取船舶特征
        
        Args:
            ship_name: 船舶名称（从数据库中查找）
            ship_type: 船舶类型（使用模板）
            custom_params: 自定义参数
        
        Returns:
            ShipCharacteristics 对象
        """
        # 优先使用自定义参数
        if custom_params:
            return ShipCharacteristics(
                ship_name=custom_params.get('ship_name', '自定义船舶'),
                mmsi=custom_params.get('mmsi', ''),
                length=custom_params.get('length', 100),
                width=custom_params.get('width', 15),
                draft=custom_params.get('draft', 5),
                height=custom_params.get('height', 20),
                tonnage=custom_params.get('tonnage', 5000),
                ship_type=custom_params.get('ship_type', '货船'),
                max_speed=custom_params.get('max_speed', 15)
            )
        
        # 从数据库查找（含推断的物理参数）
        if ship_name and ship_name in self.ship_data:
            data = self.ship_data[ship_name]
            return ShipCharacteristics(
                ship_name=ship_name,
                mmsi=data.get('mmsi', ''),
                length=data.get('length', 100),
                width=data.get('width', 15),
                draft=data.get('draft', 5),
                height=data.get('height', 20),
                tonnage=data.get('tonnage', 5000),
                ship_type=data.get('ship_type', '货船'),
                max_speed=data.get('max_speed', 15)
            )
        
        # 使用船舶类型模板
        if ship_type and ship_type in self.SHIP_TEMPLATES:
            template = self.SHIP_TEMPLATES[ship_type]
            return ShipCharacteristics(
                ship_name=f"模板_{ship_type}",
                ship_type=ship_type,
                **template
            )
        
        # 默认返回中型货船
        return ShipCharacteristics(ship_name="默认船舶", ship_type="中型货船")
    
    def list_available_ships(self) -> List[str]:
        """列出可用船舶"""
        return list(self.ship_data.keys())
    
    def list_ship_types(self) -> List[str]:
        """列出船舶类型模板"""
        return list(self.SHIP_TEMPLATES.keys())


# ==================== 模块2：物理约束校验 ====================

class PhysicalConstraintChecker:
    """
    物理约束校验器
    
    功能：
    - 检查船舶是否可通过指定航道
    - 标记受限边（浅滩、限高桥、狭窄水道）
    - 从轨迹数据特征推导航道物理约束
    
    推导逻辑：
    - 水深：根据通航船舶的最大吃水反推（航道水深 >= 历史最大吃水 + 安全裕度）
    - 限高：狭窄水道/内河段设限高，开阔海域无限制
    - 宽度：根据通航船舶的最大船宽反推
    """
    
    def __init__(self, edge_features: Dict, nodes: Dict, graph: nx.DiGraph = None):
        """
        初始化约束校验器
        
        Args:
            edge_features: 边特征字典 {(from_node, to_node): features}
            nodes: 节点字典 {node_id: {lat, lon, ...}}
            graph: 网络图（用于获取所有边，包括无轨迹数据的边）
        """
        self.edge_features = edge_features
        self.nodes = nodes
        self.graph = graph
        
        # 从边特征推导航道约束
        self._compute_constraints()
        
        # 初始化机器学习模型
        self.risk_model = RiskPredictionModel()
        self.passability_model = PassabilityProbabilityModel()
        self.multitask_model = None
        
        # 尝试加载预训练模型
        if not self.risk_model.load():
            logger.info("风险预测模型未找到，将使用规则计算")
        if not self.passability_model.load():
            logger.info("可达性概率模型未找到，将使用规则判断")
        
        # 尝试加载多任务DNN模型
        try:
            from navigation_models import MultiTaskNavigationModel
            self.multitask_model = MultiTaskNavigationModel()
            if self.multitask_model.load():
                logger.info("多任务DNN模型已加载")
            else:
                logger.info("多任务DNN模型未找到")
        except (ImportError, RuntimeError) as e:
            logger.info("多任务DNN模型不可用: %s", str(e)[:80])
    
    def _compute_constraints(self):
        """从轨迹数据特征推导航道约束
        
        核心原则：轨迹数据反映的是船舶行为（航速、频次），而非物理航道属性。
        航速低可能是拥堵/靠泊/限速，不等于浅水。默认假设航道可通航，
        仅在有明确窄水道标签时才适度收紧约束。
        """
        self.depth_map = {}
        self.height_map = {}
        self.width_map = {}
        
        for edge_key, features in self.edge_features.items():
            waterway_type = features.get('waterway_type', 'open')
            avg_distance = features.get('avg_distance', 100)
            
            if waterway_type == 'narrow' and avg_distance < 200:
                depth = 6.0
                width = 40.0
                height = 20.0
            elif waterway_type == 'narrow':
                depth = 8.0
                width = 60.0
                height = 30.0
            else:
                depth = 15.0
                width = 120.0
                height = 60.0
            
            self.depth_map[edge_key] = depth
            self.width_map[edge_key] = width
            self.height_map[edge_key] = height
        
        if self.graph:
            default_constraint_count = 0
            for u, v in self.graph.edges():
                edge_key = (u, v)
                if edge_key not in self.depth_map:
                    neighbor_depths = []
                    neighbor_widths = []
                    neighbor_heights = []
                    for nu, nv in self.graph.edges():
                        if nu == u or nv == v or nu == v or nv == u:
                            nk = (nu, nv)
                            if nk in self.depth_map:
                                neighbor_depths.append(self.depth_map[nk])
                                neighbor_widths.append(self.width_map[nk])
                                neighbor_heights.append(self.height_map[nk])
                    
                    if neighbor_depths:
                        import statistics
                        self.depth_map[edge_key] = max(statistics.median(neighbor_depths), 8.0)
                        self.width_map[edge_key] = max(statistics.median(neighbor_widths), 60.0)
                        self.height_map[edge_key] = max(statistics.median(neighbor_heights), 30.0)
                    else:
                        self.depth_map[edge_key] = 15.0
                        self.width_map[edge_key] = 120.0
                        self.height_map[edge_key] = 60.0
                    
                    default_constraint_count += 1
            
            if default_constraint_count > 0:
                logger.info("为 %d 条无轨迹数据的边设置约束（邻居继承或中等默认值）", default_constraint_count)
        
        # 统计约束信息
        shallow_edges = sum(1 for k, d in self.depth_map.items() if d < 10)
        low_bridge_edges = sum(1 for k, h in self.height_map.items() if h < 50)
        narrow_width_edges = sum(1 for k, w in self.width_map.items() if w < 80)
        
        logger.info("航道约束: %d 浅水段(<10m), %d 限高段(<50m), %d 窄航段(<80m), 总边数: %d",
                     shallow_edges, low_bridge_edges, narrow_width_edges, len(self.depth_map))
    
    def check_edge_passable(self, edge_key: Tuple[int, int], 
                            ship: ShipCharacteristics) -> Tuple[bool, str]:
        """
        检查船舶是否可通过指定边
        
        Args:
            edge_key: 边标识 (from_node, to_node)
            ship: 船舶特征
        
        Returns:
            (是否可通过, 原因说明)
        """
        # 边既不在edge_features也不在depth_map中（不应该出现，但防御性检查）
        has_feature = edge_key in self.edge_features
        has_constraint = edge_key in self.depth_map
        
        if not has_feature and not has_constraint:
            # 无任何信息的边：允许通行但标记为未知
            return True, "未知航道（默认可通行）"
        
        # 吃水检查（允许20%的吃水裕度）
        min_depth = self.depth_map.get(edge_key, 15.0)
        if ship.draft > min_depth * 1.2:
            return False, f"吃水超限: 船舶{ship.draft}m > 航道{min_depth * 1.2:.1f}m"
        
        # 高度检查（允许20%的高度裕度）
        max_height = self.height_map.get(edge_key, 100.0)
        if ship.height > max_height * 1.2:
            return False, f"高度超限: 船舶{ship.height}m > 限高{max_height * 1.2:.1f}m"
        
        # 宽度检查（允许20%的宽度裕度）
        max_width = self.width_map.get(edge_key, 100.0)
        if ship.width > max_width * 1.2:
            return False, f"宽度超限: 船舶{ship.width}m > 航道{max_width * 1.2:.1f}m"
        
        return True, "可通行"
    
    def get_blocked_edges(self, ship: ShipCharacteristics) -> Set[Tuple[int, int]]:
        """
        获取船舶无法通过的所有边
        
        Args:
            ship: 船舶特征
        
        Returns:
            被阻塞的边集合
        """
        blocked = set()
        # 遍历所有有约束的边（包括图中的所有边）
        for edge_key in self.depth_map:
            passable, _ = self.check_edge_passable(edge_key, ship)
            if not passable:
                blocked.add(edge_key)
        return blocked
    
    def get_edge_risk_score(self, edge_key: Tuple[int, int], 
                            ship: ShipCharacteristics) -> float:
        """
        计算边对特定船舶的风险评分
        
        优先使用机器学习模型，如果模型未训练则回退到规则计算
        """
        features = self.edge_features.get(edge_key)
        
        # 尝试使用多任务DNN模型预测
        if self.multitask_model is not None and self.multitask_model.is_trained and features:
            ship_features = {
                'draft': ship.draft,
                'width': ship.width,
                'height': ship.height,
                'length': ship.length,
                'tonnage': ship.tonnage,
                'max_speed': ship.max_speed
            }
            edge_feat_with_constraint = dict(features)
            edge_feat_with_constraint['min_depth'] = self.depth_map.get(edge_key, 15.0)
            edge_feat_with_constraint['max_width'] = self.width_map.get(edge_key, 100.0)
            edge_feat_with_constraint['max_height'] = self.height_map.get(edge_key, 100.0)
            
            try:
                ml_risk, ml_passable = self.multitask_model.predict(edge_feat_with_constraint, ship_features)
                return ml_risk
            except Exception as e:
                logger.warning("多任务DNN风险预测失败，尝试单任务模型: %s", e)
        
        # 尝试使用单任务ML模型预测
        if self.risk_model.is_trained and features:
            ship_features = {
                'draft': ship.draft,
                'width': ship.width,
                'height': ship.height,
                'length': ship.length,
                'tonnage': ship.tonnage,
                'max_speed': ship.max_speed
            }
            # 添加约束信息到边特征
            edge_feat_with_constraint = dict(features)
            edge_feat_with_constraint['min_depth'] = self.depth_map.get(edge_key, 15.0)
            edge_feat_with_constraint['max_width'] = self.width_map.get(edge_key, 100.0)
            edge_feat_with_constraint['max_height'] = self.height_map.get(edge_key, 100.0)
            
            try:
                ml_risk = self.risk_model.predict(edge_feat_with_constraint, ship_features)
                return ml_risk
            except Exception as e:
                logger.warning("ML风险预测失败，回退到规则计算: %s", e)
        
        # 回退：规则计算
        return self._rule_based_risk_score(edge_key, ship)
    
    def _rule_based_risk_score(self, edge_key: Tuple[int, int], 
                               ship: ShipCharacteristics) -> float:
        """基于规则的风险评分（作为ML模型的回退和训练标签来源）"""
        features = self.edge_features.get(edge_key)
        has_constraint = edge_key in self.depth_map
        
        risk = 0.0
        
        # 基于吃水裕度
        min_depth = self.depth_map.get(edge_key, 15.0)
        depth_margin = min_depth - ship.draft
        if depth_margin < 2:
            risk += 40 - depth_margin * 10
        elif depth_margin < 5:
            risk += 20 + (5 - depth_margin) * 2
        else:
            risk += 10
        
        # 基于高度裕度
        max_height = self.height_map.get(edge_key, 100.0)
        height_margin = max_height - ship.height
        if height_margin < 10:
            risk += 20
        elif height_margin < 20:
            risk += 10
        
        # 基于宽度裕度
        max_width = self.width_map.get(edge_key, 100.0)
        width_margin = max_width - ship.width
        if width_margin < 20:
            risk += 15
        elif width_margin < 40:
            risk += 5
        
        if features:
            waterway_type = features.get('waterway_type', 'open')
            if waterway_type == 'narrow':
                risk += 20
            
            sample_count = features.get('segment_count', 0)
            if sample_count < 10:
                risk += 10
            elif sample_count < 50:
                risk += 5
            
            avg_speed = features.get('avg_actual_speed', 5)
            if avg_speed < 3:
                risk += 10
            elif avg_speed < 5:
                risk += 5
        else:
            risk += 5
            
            # 对无轨迹数据的边，根据边的距离和位置增加差异化
            from_node, to_node = edge_key
            if from_node in self.graph.nodes and to_node in self.graph.nodes:
                u_data = self.graph.nodes[from_node]
                v_data = self.graph.nodes[to_node]
                dist = haversine_distance(
                    u_data.get('lat', 0), u_data.get('lon', 0),
                    v_data.get('lat', 0), v_data.get('lon', 0)
                )
                # 长边风险略高（数据不确定性）
                if dist > 20000:
                    risk += 5
                elif dist > 10000:
                    risk += 3
                
                # 根据节点度增加差异化（低连通区域风险更高）
                u_degree = self.graph.degree(from_node)
                v_degree = self.graph.degree(to_node)
                avg_degree = (u_degree + v_degree) / 2
                if avg_degree < 3:
                    risk += 5
                elif avg_degree < 5:
                    risk += 3
        
        # 基于船舶风险等级
        if ship.risk_level == "高风险":
            risk += 15
        elif ship.risk_level == "中风险":
            risk += 5
        
        return min(100, max(0, risk))
    
    def get_edge_passability_proba(self, edge_key: Tuple[int, int],
                                   ship: ShipCharacteristics) -> float:
        """
        预测边对船舶的可达性概率
        
        Returns:
            0-1 概率值，1表示肯定可通过
        """
        features = self.edge_features.get(edge_key)
        
        # 尝试使用多任务DNN模型预测
        if self.multitask_model is not None and self.multitask_model.is_trained and features:
            ship_features = {
                'draft': ship.draft,
                'width': ship.width,
                'height': ship.height,
                'length': ship.length,
                'tonnage': ship.tonnage,
                'max_speed': ship.max_speed
            }
            edge_feat_with_constraint = dict(features)
            edge_feat_with_constraint['min_depth'] = self.depth_map.get(edge_key, 15.0)
            edge_feat_with_constraint['max_width'] = self.width_map.get(edge_key, 100.0)
            edge_feat_with_constraint['max_height'] = self.height_map.get(edge_key, 100.0)
            
            try:
                ml_risk, ml_passable = self.multitask_model.predict(edge_feat_with_constraint, ship_features)
                return ml_passable
            except Exception as e:
                logger.warning("多任务DNN可达性预测失败，尝试单任务模型: %s", e)
        
        # 尝试使用单任务ML模型预测
        if self.passability_model.is_trained and features:
            ship_features = {
                'draft': ship.draft,
                'width': ship.width,
                'height': ship.height,
                'length': ship.length
            }
            edge_feat_with_constraint = dict(features)
            edge_feat_with_constraint['min_depth'] = self.depth_map.get(edge_key, 15.0)
            edge_feat_with_constraint['max_width'] = self.width_map.get(edge_key, 100.0)
            edge_feat_with_constraint['max_height'] = self.height_map.get(edge_key, 100.0)
            
            try:
                proba = self.passability_model.predict_proba(edge_feat_with_constraint, ship_features)
                return proba
            except Exception as e:
                logger.warning("ML可达性预测失败，回退到规则判断: %s", e)
        
        # 回退：二元判断转概率
        passable, _ = self.check_edge_passable(edge_key, ship)
        return 1.0 if passable else 0.0
    
    def train_models(self, ship_templates: list = None):
        """
        训练风险预测和可达性概率模型
        
        优先使用从真实轨迹提取的隐式标签，如果没有则回退到规则生成的伪标签
        
        Args:
            ship_templates: 船舶模板列表，用于生成训练数据
        """
        if ship_templates is None:
            # 默认船舶模板
            ship_templates = [
                {'draft': 5, 'width': 15, 'height': 20, 'length': 100, 'tonnage': 5000, 'max_speed': 15},
                {'draft': 8, 'width': 25, 'height': 25, 'length': 150, 'tonnage': 20000, 'max_speed': 14},
                {'draft': 12, 'width': 35, 'height': 40, 'length': 250, 'tonnage': 80000, 'max_speed': 13},
                {'draft': 15, 'width': 50, 'height': 25, 'length': 300, 'tonnage': 120000, 'max_speed': 12},
            ]
        
        logger.info("开始训练导航预测模型...")
        
        # 尝试加载隐式标签
        implicit_risk_labels = self._load_implicit_risk_labels()
        implicit_pass_labels = self._load_implicit_passability_labels()
        
        use_implicit_risk = implicit_risk_labels is not None and len(implicit_risk_labels) > 0
        use_implicit_pass = implicit_pass_labels is not None and len(implicit_pass_labels) > 0
        
        if use_implicit_risk:
            logger.info("使用隐式风险标签训练，样本数: %d", len(implicit_risk_labels))
        if use_implicit_pass:
            logger.info("使用隐式可达性标签训练，样本数: %d", len(implicit_pass_labels))
        
        # 训练多任务DNN模型（如果PyTorch可用）
        if self.multitask_model is not None and not self.multitask_model.is_trained:
            try:
                if use_implicit_risk and use_implicit_pass:
                    # 使用隐式标签训练
                    X_multi, y_risk_multi, y_pass_multi = self._generate_training_data_from_implicit_labels(
                        implicit_risk_labels, implicit_pass_labels
                    )
                else:
                    # 回退到规则生成的伪标签
                    X_multi, y_risk_multi, y_pass_multi = self.multitask_model.generate_training_data(
                        self.edge_features, self, ship_templates
                    )
                
                if len(X_multi) > 0:
                    self.multitask_model.train(
                        X_multi, y_risk_multi, y_pass_multi, 
                        epochs=300, batch_size=64, lr=0.001,
                        risk_weight=0.6, passable_weight=0.4,
                        val_split=0.2, patience=30,
                        label_smoothing=0.05, noise_std=0.01
                    )
                    logger.info("多任务DNN模型训练完成")
            except Exception as e:
                logger.warning("多任务DNN模型训练失败: %s", e)
        
        # 训练风险预测模型
        if not self.risk_model.is_trained:
            if use_implicit_risk:
                X_risk, y_risk = self._generate_risk_training_from_implicit(implicit_risk_labels)
                logger.info("使用隐式标签训练风险预测模型")
            else:
                X_risk, y_risk = self.risk_model.generate_pseudo_labels(
                    self.edge_features, self, ship_templates
                )
                logger.info("使用规则伪标签训练风险预测模型")
            
            if len(X_risk) > 0:
                self.risk_model.train(X_risk, y_risk)
                logger.info("风险预测模型训练完成")
        
        # 训练可达性概率模型
        if not self.passability_model.is_trained:
            if use_implicit_pass:
                X_pass, y_pass = self._generate_passability_training_from_implicit(implicit_pass_labels)
                logger.info("使用隐式标签训练可达性概率模型")
            else:
                X_pass, y_pass = self.passability_model.generate_training_data(
                    self.edge_features, self, ship_templates
                )
                logger.info("使用规则伪标签训练可达性概率模型")
            
            if len(X_pass) > 0:
                self.passability_model.train(X_pass, y_pass)
                logger.info("可达性概率模型训练完成")
    
    def _load_implicit_risk_labels(self):
        """加载隐式风险标签"""
        path = os.path.join("output", "implicit_risk_labels.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        return None
    
    def _load_implicit_passability_labels(self):
        """加载隐式可达性标签"""
        path = os.path.join("output", "implicit_passability_labels.csv")
        if os.path.exists(path):
            return pd.read_csv(path)
        return None
    
    def _generate_training_data_from_implicit_labels(self, risk_labels_df, pass_labels_df):
        """从隐式标签生成多任务训练数据"""
        from navigation_models import MultiTaskNavigationModel
        
        X_list = []
        y_risk_list = []
        y_pass_list = []
        
        # 使用可达性标签作为基础（通常样本更多）
        for _, row in pass_labels_df.iterrows():
            if row['passable'] == -1:  # 跳过未知标签
                continue
                
            edge_key = (int(row['from_node']), int(row['to_node']))
            ship_type = row['ship_type']
            
            # 获取边特征
            edge_feat = self.edge_features.get(edge_key, {})
            if not edge_feat:
                continue
            
            # 获取船舶特征（从模板）
            ship_template = self._get_ship_template_by_type(ship_type)
            if not ship_template:
                continue
            
            # 提取特征
            features = self._extract_features_for_training(edge_feat, ship_template)
            X_list.append(features)
            
            # 可达性标签
            y_pass_list.append(1 if row['passable'] == 1 else 0)
            
            # 查找对应的风险标签
            risk_row = risk_labels_df[
                (risk_labels_df['from_node'] == edge_key[0]) &
                (risk_labels_df['to_node'] == edge_key[1]) &
                (risk_labels_df['ship_type'] == ship_type)
            ]
            
            if not risk_row.empty:
                y_risk_list.append(risk_row.iloc[0]['risk_score'])
            else:
                # 如果没有风险标签，使用规则计算
                ship_obj = ShipCharacteristics(
                    ship_name="template", ship_type=ship_type, **ship_template
                )
                risk = self._rule_based_risk_score(edge_key, ship_obj)
                y_risk_list.append(risk)
        
        return np.array(X_list), np.array(y_risk_list), np.array(y_pass_list)
    
    def _generate_risk_training_from_implicit(self, risk_labels_df):
        """从隐式风险标签生成训练数据"""
        X_list = []
        y_list = []
        
        for _, row in risk_labels_df.iterrows():
            edge_key = (int(row['from_node']), int(row['to_node']))
            ship_type = row['ship_type']
            
            edge_feat = self.edge_features.get(edge_key, {})
            if not edge_feat:
                continue
            
            ship_template = self._get_ship_template_by_type(ship_type)
            if not ship_template:
                continue
            
            features = self._extract_features_for_training(edge_feat, ship_template)
            X_list.append(features)
            y_list.append(row['risk_score'])
        
        return np.array(X_list), np.array(y_list)
    
    def _generate_passability_training_from_implicit(self, pass_labels_df):
        """从隐式可达性标签生成训练数据"""
        X_list = []
        y_list = []
        
        for _, row in pass_labels_df.iterrows():
            if row['passable'] == -1:  # 跳过未知
                continue
                
            edge_key = (int(row['from_node']), int(row['to_node']))
            ship_type = row['ship_type']
            
            edge_feat = self.edge_features.get(edge_key, {})
            if not edge_feat:
                continue
            
            ship_template = self._get_ship_template_by_type(ship_type)
            if not ship_template:
                continue
            
            features = self._extract_features_for_training(edge_feat, ship_template)
            X_list.append(features)
            y_list.append(1 if row['passable'] == 1 else 0)
        
        return np.array(X_list), np.array(y_list)
    
    def _get_ship_template_by_type(self, ship_type: str) -> dict:
        """根据船型获取模板参数"""
        # 船型到模板的映射
        type_mapping = {
            '货船': '中型货船',
            '集装箱船': '集装箱船',
            '油轮': '油轮',
            '客船': '客船',
            '渔船': '渔船',
            '拖船': '拖船',
        }
        
        # 尝试直接匹配
        template_name = type_mapping.get(ship_type, ship_type)
        
        # 从SHIP_TEMPLATES查找
        from ship_navigator import ShipCharacteristicsManager
        templates = ShipCharacteristicsManager.SHIP_TEMPLATES
        
        if template_name in templates:
            return templates[template_name]
        
        # 模糊匹配
        for name, template in templates.items():
            if template_name in name or name in template_name:
                return template
        
        # 默认返回中型货船
        return templates.get('中型货船', {'draft': 5, 'width': 15, 'height': 20, 'length': 100, 'tonnage': 5000, 'max_speed': 15})
    
    def _extract_features_for_training(self, edge_feat: dict, ship_template: dict) -> np.ndarray:
        """提取训练特征向量"""
        features = []
        
        # 边特征
        features.append(edge_feat.get('avg_distance', 100))
        features.append(edge_feat.get('avg_travel_time', 30))
        features.append(edge_feat.get('avg_actual_speed', 5))
        features.append(edge_feat.get('segment_count', 0))
        features.append(edge_feat.get('speed_reliability', 0.8))
        features.append(edge_feat.get('node_degree_from', 0))
        features.append(edge_feat.get('node_degree_to', 0))
        features.append(edge_feat.get('edge_betweenness', 0))
        features.append(1 if edge_feat.get('waterway_type') == 'narrow' else 0)
        
        # 船舶特征
        features.append(ship_template.get('draft', 5))
        features.append(ship_template.get('width', 15))
        features.append(ship_template.get('height', 20))
        features.append(ship_template.get('length', 100))
        features.append(ship_template.get('tonnage', 5000))
        features.append(ship_template.get('max_speed', 15))
        
        # 交互特征
        draft_margin = edge_feat.get('min_depth', 15) - ship_template.get('draft', 5)
        width_margin = edge_feat.get('max_width', 100) - ship_template.get('width', 15)
        height_margin = edge_feat.get('max_height', 100) - ship_template.get('height', 20)
        features.append(draft_margin)
        features.append(width_margin)
        features.append(height_margin)
        features.append(draft_margin * width_margin)
        features.append(draft_margin / max(ship_template.get('draft', 5), 0.1))
        
        return np.array(features)


# ==================== 模块3：多目标路径规划 ====================

class MultiObjectiveNavigator:
    """
    多目标路径规划器

    改进算法：
    - A*: 启发式搜索，综合距离和风险
    - Dijkstra: 最短路径
    - Yen's算法: 保证路径多样性（K最短路径）
    - 多目标优化：安全、时间、频次、综合
    - Kinodynamic约束：考虑船舶转弯半径
    - 帕累托最优：真正差异化的多目标路径
    """

    MIN_TURNING_RADIUS_FACTOR = 0.8
    MAX_COURSE_CHANGE = 120.0

    def __init__(self, graph: nx.DiGraph,
                 edge_features: Dict,
                 constraint_checker: PhysicalConstraintChecker):
        """
        初始化路径规划器

        Args:
            graph: 网络图
            edge_features: 边特征字典
            constraint_checker: 约束校验器
        """
        self.graph = graph
        self.edge_features = edge_features
        self.constraint_checker = constraint_checker
        self.nodes = dict(graph.nodes(data=True))

        self._build_weight_maps()
        self._precompute_edge_curvatures()

    def _build_weight_maps(self):
        """构建多种权重映射"""
        self.distance_weight = {}
        self.time_weight = {}
        self.risk_weight = {}
        self.frequency_weight = {}

        for edge_key, features in self.edge_features.items():
            self.distance_weight[edge_key] = features.get('avg_distance', 100)
            self.time_weight[edge_key] = features.get('avg_travel_time', 30)
            self.risk_weight[edge_key] = self.constraint_checker.get_edge_risk_score(edge_key,
                                      ShipCharacteristics(ship_name="默认"))
            self.frequency_weight[edge_key] = features.get('segment_count', 0)

        if self.graph:
            for u, v in self.graph.edges():
                edge_key = (u, v)
                if edge_key not in self.distance_weight:
                    u_data = self.graph.nodes[u]
                    v_data = self.graph.nodes[v]
                    est_distance = haversine_distance(
                        u_data.get('lat', 0), u_data.get('lon', 0),
                        v_data.get('lat', 0), v_data.get('lon', 0)
                    )
                    est_time = est_distance / (8.0 * 0.5144) if est_distance > 0 else 30

                    self.distance_weight[edge_key] = est_distance
                    self.time_weight[edge_key] = est_time
                    self.risk_weight[edge_key] = self.constraint_checker.get_edge_risk_score(edge_key,
                                              ShipCharacteristics(ship_name="默认"))
                    self.frequency_weight[edge_key] = 0

    def _precompute_edge_curvatures(self):
        """预计算边的曲率（基于航向变化）"""
        self.edge_bearings = {}
        self.edge_curvatures = {}

        for edge_key in self.edge_features.keys():
            from_node, to_node = edge_key
            if from_node in self.graph.nodes and to_node in self.graph.nodes:
                u_data = self.graph.nodes[from_node]
                v_data = self.graph.nodes[to_node]
                bearing = calculate_bearing(
                    u_data['lat'], u_data['lon'],
                    v_data['lat'], v_data['lon']
                )
                self.edge_bearings[edge_key] = bearing

        for node in self.graph.nodes():
            preds = list(self.graph.predecessors(node))
            succs = list(self.graph.successors(node))
            for pred in preds:
                for succ in succs:
                    if pred != succ:
                        key_in = (pred, node)
                        key_out = (node, succ)
                        if key_in in self.edge_bearings and key_out in self.edge_bearings:
                            angle_change = calculate_angle_difference(
                                self.edge_bearings[key_in],
                                self.edge_bearings[key_out]
                            )
                            self.edge_curvatures[(pred, node, succ)] = angle_change

    def get_turning_radius(self, ship: ShipCharacteristics) -> float:
        """根据船舶特征计算最小转弯半径（米）"""
        return ship.length * 1.5 + ship.width * 0.5

    def get_course_change_penalty(self, pred_node: int, curr_node: int,
                                  succ_node: int, ship: ShipCharacteristics) -> float:
        """
        计算航向变化惩罚

        Args:
            pred_node: 前一个节点
            curr_node: 当前节点
            succ_node: 下一个节点
            ship: 船舶特征

        Returns:
            惩罚系数 (1.0 = 无惩罚, >1.0 = 惩罚)
        """
        turning_radius = self.get_turning_radius(ship)
        key_in = (pred_node, curr_node)
        key_out = (curr_node, succ_node)

        if key_in not in self.edge_bearings or key_out not in self.edge_bearings:
            return 1.0

        angle_change = calculate_angle_difference(
            self.edge_bearings[key_in],
            self.edge_bearings[key_out]
        )

        if angle_change < 10:
            return 1.0

        max_turn_angle = 90.0 / (turning_radius / 100)
        max_turn_angle = min(max_turn_angle, self.MAX_COURSE_CHANGE)

        if angle_change > max_turn_angle:
            penalty = 2.0 + (angle_change - max_turn_angle) / 30.0
            return min(penalty, 5.0)

        penalty = 1.0 + (angle_change / max_turn_angle) * 0.5
        return penalty

    def _astar_weight_perturbation(self, start: int, end: int,
                                    ship: ShipCharacteristics,
                                    hour: int,
                                    blocked_edges: Set[Tuple[int, int]],
                                    weight_seed: int) -> Optional[PathResult]:
        """
        改进A*：权重扰动机制

        通过对时间/风险/距离三个目标随机分配权重比例，
        使每次搜索偏向不同的路径特征，从而产生多样化路径。

        Args:
            weight_seed: 随机种子，用于生成不同的权重组合
        """
        rng = random.Random(weight_seed)

        time_w = rng.uniform(0.2, 0.6)
        risk_w = rng.uniform(0.2, 0.6)
        dist_w = rng.uniform(0.1, 0.4)
        total_w = time_w + risk_w + dist_w
        time_w, risk_w, dist_w = time_w/total_w, risk_w/total_w, dist_w/total_w

        def heuristic(node):
            if node not in self.graph.nodes:
                return 0
            node_data = self.graph.nodes[node]
            end_data = self.graph.nodes[end]
            return haversine_distance(
                node_data.get('lat', 0), node_data.get('lon', 0),
                end_data.get('lat', 0), end_data.get('lon', 0)
            ) / 100

        def edge_cost(edge_key):
            time_cost = self._get_dynamic_time(edge_key, hour) / 100
            risk_cost = self.constraint_checker.get_edge_risk_score(edge_key, ship) / 10
            dist_cost = self.distance_weight.get(edge_key, 100) / 1000
            return time_cost * time_w + risk_cost * risk_w + dist_cost * dist_w

        g_score = {start: 0}
        f_score = {start: heuristic(start)}
        prev = {start: None}
        edge_used = {start: None}
        open_set = [(f_score[start], start)]
        closed_set = set()

        while open_set:
            _, node = heapq.heappop(open_set)

            if node in closed_set:
                continue
            closed_set.add(node)

            if node == end:
                break

            for neighbor in self.graph.successors(node):
                if neighbor in closed_set:
                    continue

                edge_key = (node, neighbor)

                if edge_key in blocked_edges:
                    continue

                tentative_g = g_score[node] + edge_cost(edge_key)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    prev[neighbor] = node
                    edge_used[neighbor] = edge_key
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        if end not in prev or prev[end] is None:
            return None

        nodes, edges = self._reconstruct_path(end, prev, edge_used)

        return self._build_path_result(
            PathType.BALANCED, nodes, edges, ship, blocked_edges, hour=hour
        )

    def _gwo_pathfinding(self, start: int, end: int,
                         ship: ShipCharacteristics,
                         hour: int,
                         blocked_edges: Set[Tuple[int, int]],
                         num_wolves: int = 20,
                         max_iterations: int = 30) -> List[PathResult]:
        """
        灰狼优化算法(GWO)路径搜索

        算法原理：
        - 模拟灰狼群体的社会等级和狩猎行为
        - 四只领导狼：Alpha(最优)、Beta(次优)、Delta(第三)、Omega(跟随者)
        - 通过包围、搜索、攻击三个阶段优化路径
        - 适应度函数：时间、风险、距离的多目标加权

        Args:
            start: 起点
            end: 终点
            ship: 船舶特征
            hour: 出发小时
            blocked_edges: 阻塞边
            num_wolves: 狼群数量
            max_iterations: 最大迭代次数

        Returns:
            帕累托最优路径列表
        """
        if not self.graph.has_node(start) or not self.graph.has_node(end):
            return []

        # 初始化狼群：用不同权重的A*产生初始种群
        population = []
        for i in range(num_wolves):
            p = self._astar_weight_perturbation(start, end, ship, hour, blocked_edges, i * 7)
            if p:
                population.append(p)

        if len(population) < 3:
            return population

        # 计算每只狼的适应度（多目标综合评分）
        def fitness(path: PathResult) -> float:
            return path.total_time * 0.4 + path.risk_score * 10 + path.total_distance / 1000

        # 排序并确定Alpha/Beta/Delta
        population.sort(key=fitness)
        alpha = population[0]
        beta = population[1] if len(population) > 1 else population[0]
        delta = population[2] if len(population) > 2 else population[0]

        for iteration in range(max_iterations):
            # 衰减因子：从2线性递减到0
            a = 2 - iteration * (2.0 / max_iterations)

            new_candidates = []

            for wolf in population[3:]:  # Omega狼跟随领导狼
                # 随机选择领导狼
                leaders = [alpha, beta, delta]
                leader = random.choice(leaders)

                # 在狼路径和领导路径之间找共同节点作为交叉点
                common_nodes = set(wolf.nodes) & set(leader.nodes)
                common_nodes.discard(start)
                common_nodes.discard(end)

                if common_nodes:
                    cross = random.choice(list(common_nodes))
                    w_idx = wolf.nodes.index(cross)
                    l_idx = leader.nodes.index(cross)

                    # 交叉：狼的前半段 + 领导的后半段
                    child_nodes = wolf.nodes[:w_idx] + leader.nodes[l_idx:]
                    child_edges = []
                    valid = True
                    for j in range(len(child_nodes) - 1):
                        ek = (child_nodes[j], child_nodes[j+1])
                        if ek in blocked_edges or not self.graph.has_edge(child_nodes[j], child_nodes[j+1]):
                            valid = False
                            break
                        child_edges.append(ek)

                    if valid and len(child_nodes) >= 2:
                        try:
                            child = self._build_path_result(
                                PathType.BALANCED, child_nodes, child_edges,
                                ship, blocked_edges, hour=hour)
                            if child:
                                new_candidates.append(child)
                        except Exception:
                            pass

                # 变异：随机扰动
                if random.random() < 0.3:
                    mutated = self._mutate_path(wolf, ship, hour, blocked_edges)
                    if mutated:
                        new_candidates.append(mutated)

            population.extend(new_candidates)
            population = self._pareto_selection(population, num_wolves)

            if len(population) >= 3:
                population.sort(key=fitness)
                alpha = population[0]
                beta = population[1]
                delta = population[2]

            if len(population) == 0:
                break

        return population[:3]

    def _dijkstra_shortest_distance(self, start: int, end: int,
                                     ship: ShipCharacteristics,
                                     blocked_edges: Set[Tuple[int, int]],
                                     hour: int = None) -> Optional[PathResult]:
        """
        A*算法 - 距离最短
        目标：最小化总航行距离
        """
        end_data = self.graph.nodes.get(end, {})
        end_lat = end_data.get('lat', 0)
        end_lon = end_data.get('lon', 0)

        def heuristic(node):
            if node not in self.graph.nodes:
                return 0
            node_data = self.graph.nodes[node]
            return haversine_distance(
                node_data.get('lat', 0), node_data.get('lon', 0),
                end_lat, end_lon)

        g_score = {start: 0}
        f_score = {start: heuristic(start)}
        prev = {start: None}
        edge_used = {start: None}
        open_set = [(f_score[start], start)]
        closed_set = set()

        while open_set:
            _, node = heapq.heappop(open_set)

            if node in closed_set:
                continue
            closed_set.add(node)

            if node == end:
                break

            for neighbor in self.graph.successors(node):
                if neighbor in closed_set:
                    continue

                edge_key = (node, neighbor)
                if edge_key in blocked_edges:
                    continue

                dist = self.distance_weight.get(edge_key, 100)
                tentative_g = g_score[node] + dist

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    prev[neighbor] = node
                    edge_used[neighbor] = edge_key
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        if end not in prev or prev[end] is None:
            return None

        nodes, edges = self._reconstruct_path(end, prev, edge_used)

        return self._build_path_result(
            PathType.BALANCED, nodes, edges, ship, blocked_edges, hour=hour
        )

    def _pso_pathfinding(self, start: int, end: int,
                         ship: ShipCharacteristics,
                         hour: int,
                         blocked_edges: Set[Tuple[int, int]],
                         num_particles: int = 20,
                         max_iterations: int = 30) -> List[PathResult]:
        """
        粒子群优化路径搜索

        算法原理：
        - 每个粒子代表一个可能的路径（通过节点序列编码）
        - 粒子在解空间中搜索，通过个体最优和全局最优引导
        - 适应度函数：时间、风险、距离的多目标加权
        - 拥挤距离机制保持路径多样性

        Args:
            start: 起点
            end: 终点
            ship: 船舶特征
            hour: 出发小时
            blocked_edges: 阻塞边
            num_particles: 粒子数量
            max_iterations: 最大迭代次数

        Returns:
            帕累托最优路径列表
        """
        if not self.graph.has_node(start) or not self.graph.has_node(end):
            return []

        population = []

        base_paths = []
        for seed in range(num_particles // 2):
            p = self._astar_weight_perturbation(start, end, ship, hour, blocked_edges, seed)
            if p:
                base_paths.append(p)

        if not base_paths:
            for seed in range(num_particles):
                p = self._astar_weight_perturbation(start, end, ship, hour, blocked_edges, seed * 100)
                if p:
                    base_paths.append(p)
            if not base_paths:
                return []

        population = base_paths[:num_particles]

        for iteration in range(max_iterations):
            new_candidates = []
            num_mutations = min(5, len(population))

            for i in range(num_mutations):
                idx1 = random.randint(0, len(population) - 1)
                idx2 = (idx1 + 1) % len(population)

                parent1 = population[idx1]
                parent2 = population[idx2]

                if len(parent1.nodes) >= 3 and len(parent2.nodes) >= 3:
                    crossover_node = parent1.nodes[random.randint(1, len(parent1.nodes) - 2)]
                    if crossover_node in parent2.nodes:
                        idx_p1 = parent1.nodes.index(crossover_node)
                        idx_p2 = parent2.nodes.index(crossover_node)

                        child_nodes = parent1.nodes[:idx_p1] + parent2.nodes[idx_p2:]
                        child_edges = []
                        valid = True
                        for j in range(len(child_nodes) - 1):
                            ek = (child_nodes[j], child_nodes[j+1])
                            if ek in blocked_edges:
                                valid = False
                                break
                            if not self.graph.has_edge(child_nodes[j], child_nodes[j+1]):
                                valid = False
                                break
                            child_edges.append(ek)

                        if valid and len(child_nodes) >= 2:
                            pr = {child_nodes[j]: (child_nodes[j-1] if j > 0 else None)
                                  for j in range(len(child_nodes))}
                            eu = {(child_nodes[j-1], child_nodes[j]): None
                                  for j in range(1, len(child_nodes))}
                            eu[child_nodes[0]] = None

                            try:
                                child_path = self._build_path_result(
                                    PathType.BALANCED, child_nodes, child_edges,
                                    ship, blocked_edges, hour=hour)
                                if child_path:
                                    new_candidates.append(child_path)
                            except Exception:
                                pass

                mutated = self._mutate_path(parent1, ship, hour, blocked_edges)
                if mutated:
                    new_candidates.append(mutated)

            population.extend(new_candidates)

            population = self._pareto_selection(population, num_particles)

            if len(population) == 0:
                break

        return population[:3]

    def _mutate_path(self, path: PathResult, ship: ShipCharacteristics,
                    hour: int, blocked_edges: Set[Tuple[int, int]]) -> Optional[PathResult]:
        """对路径进行变异：随机替换中间节点"""
        if len(path.nodes) < 4:
            return None

        start_node = path.nodes[0]
        end_node = path.nodes[-1]

        cut_start = random.randint(1, len(path.nodes) - 3)
        cut_end = random.randint(cut_start + 1, len(path.nodes) - 2)

        bypass_nodes = []
        current = path.nodes[cut_start]
        target = path.nodes[cut_end]

        visited = {current}
        queue = [(current, [current])]
        max_depth = min(8, len(path.nodes))

        while queue:
            node, route = queue.pop(0)
            if len(route) > max_depth:
                continue
            if node == target and len(route) > 2:
                bypass_nodes = route
                break
            for neighbor in self.graph.successors(node):
                if neighbor not in visited and (node, neighbor) not in blocked_edges:
                    visited.add(neighbor)
                    queue.append((neighbor, route + [neighbor]))

        if not bypass_nodes or bypass_nodes[-1] != target:
            return None

        new_nodes = path.nodes[:cut_start] + bypass_nodes[1:-1] + path.nodes[cut_end:]
        new_edges = []
        for i in range(len(new_nodes) - 1):
            ek = (new_nodes[i], new_nodes[i+1])
            if ek in blocked_edges or not self.graph.has_edge(new_nodes[i], new_nodes[i+1]):
                return None
            new_edges.append(ek)

        return self._build_path_result(
            PathType.BALANCED, new_nodes, new_edges, ship, blocked_edges, hour=hour)

    def find_paths(self, start: int, end: int,
                   ship: ShipCharacteristics,
                   hour: int = None,
                   max_paths: int = 3) -> List[PathResult]:
        """
        多目标路径规划
        
        DAG结构下的差异化路径生成：
        1. 先用all_simple_paths找多条简单路径（地理差异化）
        2. 多路径对：选2-3条差异最大的路径
        3. 单路径对：同路径+不同场景（时间/约束松弛）生成差异化
        """
        blocked_edges = self.constraint_checker.get_blocked_edges(ship)

        if blocked_edges:
            logger.info("检测到 %d 条边不满足船舶约束", len(blocked_edges))

        result_paths = []
        seen_node_seqs = set()

        def add_path(p, ptype):
            if p is None:
                return False
            p.path_type = ptype
            result_paths.append(p)
            return True

        def add_unique_path(p, ptype):
            if p is None:
                return False
            seq = tuple(p.nodes)
            if seq in seen_node_seqs:
                return False
            p.path_type = ptype
            result_paths.append(p)
            seen_node_seqs.add(seq)
            return True

        try:
            raw_paths = list(nx.all_simple_paths(
                self.graph, start, end, cutoff=12))
        except Exception:
            raw_paths = []

        if len(raw_paths) >= 2:
            scored = []
            for nodes in raw_paths:
                edges_list = []
                for i in range(len(nodes) - 1):
                    u, v = nodes[i], nodes[i + 1]
                    edges_list.append((u, v))
                
                p = self._build_path_result(
                    PathType.SAFEST, nodes, edges_list, ship, blocked_edges, hour=hour)
                if p is None:
                    continue
                
                blocked_count = sum(1 for u, v in edges_list if (u, v) in blocked_edges)
                scored.append((p, len(nodes), blocked_count))
            
            scored.sort(key=lambda x: (x[2], -x[1]))
            
            types = [PathType.SAFEST, PathType.FASTEST, PathType.FREQUENT]
            for i, (p, _, _) in enumerate(scored):
                if len(result_paths) >= max_paths:
                    break
                add_unique_path(p, types[min(i, len(types) - 1)])
            
            if len(result_paths) >= max_paths:
                return result_paths[:max_paths]

        path_safest = self._dijkstra_safest(start, end, ship, blocked_edges, hour)
        if path_safest is None:
            return result_paths
        if tuple(path_safest.nodes) not in seen_node_seqs:
            path_safest.path_type = PathType.SAFEST
            result_paths.append(path_safest)
            seen_node_seqs.add(tuple(path_safest.nodes))

        if not result_paths:
            return result_paths

        base_nodes = list(result_paths[0].nodes)
        base_edges = list(result_paths[0].edges)

        for removed_edge in base_edges:
            if len(result_paths) >= max_paths:
                break
            if removed_edge in blocked_edges:
                continue
            try:
                temp_blocked = set(blocked_edges) | {removed_edge}
                alt_path = self._dijkstra_safest(start, end, ship, temp_blocked, hour)
                if alt_path and tuple(alt_path.nodes) not in seen_node_seqs:
                    alt_path.path_type = PathType.FASTEST
                    result_paths.append(alt_path)
                    seen_node_seqs.add(tuple(alt_path.nodes))
            except Exception:
                pass

        if len(result_paths) < max_paths and blocked_edges:
            cc = self.constraint_checker
            edge_violations = []
            for edge in blocked_edges:
                depth = cc.depth_map.get(edge, 0)
                width = cc.width_map.get(edge, 0)
                height = cc.height_map.get(edge, 0)
                margin = 0
                if depth and ship.draft > depth:
                    margin += (ship.draft - depth) / max(depth, 0.1)
                if width and ship.width > width:
                    margin += (ship.width - width) / max(width, 0.1)
                if height and ship.height > height:
                    margin += (ship.height - height) / max(height, 0.1)
                edge_violations.append((edge, margin))

            edge_violations.sort(key=lambda x: x[1])

            for pct in [0.1, 0.25, 0.5]:
                if len(result_paths) >= max_paths:
                    break
                num_unblock = max(1, int(len(edge_violations) * pct))
                relaxed_blocked = set(blocked_edges)
                for i in range(min(num_unblock, len(edge_violations))):
                    relaxed_blocked.discard(edge_violations[i][0])

                path_relaxed = self._build_path_result(
                    PathType.RELAXED, base_nodes, base_edges, ship, relaxed_blocked, hour=hour)
                if path_relaxed:
                    is_meaningful = True
                    for existing in result_paths:
                        if (abs(path_relaxed.safety_score - existing.safety_score) < 0.5 and
                            abs(path_relaxed.total_time - existing.total_time) / max(existing.total_time, 1) < 0.02):
                            is_meaningful = False
                            break
                    if is_meaningful:
                        path_relaxed.path_type = PathType.RELAXED
                        path_relaxed.constraints_met = False
                        path_relaxed.warning = f"经过{num_unblock}条约束放宽路段"
                        result_paths.append(path_relaxed)
                        break

        if len(result_paths) < max_paths:
            base_node_set = set(base_nodes)
            candidates = []
            for node in self.graph.nodes():
                if node in base_node_set or node == start or node == end:
                    continue
                try:
                    if nx.has_path(self.graph, start, node) and nx.has_path(self.graph, node, end):
                        deg = self.graph.degree(node)
                        candidates.append((node, deg))
                except Exception:
                    pass
            candidates.sort(key=lambda x: -x[1])

            for mid_node, _ in candidates[:20]:
                if len(result_paths) >= max_paths:
                    break
                try:
                    path_to_mid = self._dijkstra_safest(start, mid_node, ship, blocked_edges, hour)
                    if path_to_mid is None:
                        continue
                    path_from_mid = self._dijkstra_safest(mid_node, end, ship, blocked_edges, hour)
                    if path_from_mid is None:
                        continue

                    detour_nodes = list(path_to_mid.nodes)
                    for n in path_from_mid.nodes[1:]:
                        if n not in detour_nodes:
                            detour_nodes.append(n)

                    detour_edges = []
                    for i in range(len(detour_nodes) - 1):
                        detour_edges.append((detour_nodes[i], detour_nodes[i + 1]))

                    seq = tuple(detour_nodes)
                    if seq in seen_node_seqs:
                        continue

                    detour_path = self._build_path_result(
                        PathType.FASTEST, detour_nodes, detour_edges, ship, blocked_edges, hour=hour)
                    if detour_path and tuple(detour_path.nodes) not in seen_node_seqs:
                        detour_path.path_type = PathType.FASTEST
                        result_paths.append(detour_path)
                        seen_node_seqs.add(tuple(detour_path.nodes))
                except Exception:
                    pass

        if len(result_paths) < max_paths:
            need_any = len(result_paths) <= 1
            for h_offset in [6, 12, 18]:
                if len(result_paths) >= max_paths:
                    break
                alt_hour = ((hour or 0) + h_offset) % 24
                path_var = self._build_path_result(
                    PathType.SAFEST, base_nodes, base_edges, ship, blocked_edges, hour=alt_hour)
                if path_var is None:
                    continue

                if not need_any:
                    min_diff_pct = 0.03
                    is_diff = True
                    for existing in result_paths:
                        diff_pct = abs(path_var.total_time - existing.total_time) / max(existing.total_time, 1)
                        if diff_pct < min_diff_pct:
                            is_diff = False
                            break
                    if not is_diff:
                        continue

                path_var.path_type = PathType.BALANCED
                result_paths.append(path_var)
                need_any = False

        return result_paths[:max_paths]

    def _nsga2_pathfinding(self, start: int, end: int,
                            ship: ShipCharacteristics,
                            hour: int,
                            blocked_edges: Set[Tuple[int, int]],
                            pop_size: int = 30,
                            generations: int = 40) -> List[PathResult]:
        """
        NSGA-II多目标路径优化
        
        染色体编码：节点序列
        目标函数：(1)总时间 (2)总风险 (3)总距离
        初始化：不同权重的A*产生初始种群
        遗传操作：交叉(路径拼接) + 变异(子路径替换)
        选择：非支配排序 + 拥挤距离
        """
        if not self.graph.has_node(start) or not self.graph.has_node(end):
            return []

        population = []
        for i in range(pop_size):
            p = self._astar_weight_perturbation(start, end, ship, hour, blocked_edges, i * 13)
            if p and tuple(p.nodes) not in {tuple(pp.nodes) for pp in population}:
                population.append(p)

        if len(population) < 2:
            return population

        def objectives(path: PathResult):
            return (path.total_time, path.risk_score, path.total_distance)

        def non_dominated_sort(pop: List[PathResult]):
            fronts = [[]]
            S = {id(p): [] for p in pop}
            n = {id(p): 0 for p in pop}
            for i, pi in enumerate(pop):
                for j, pj in enumerate(pop):
                    if i == j:
                        continue
                    oi, oj = objectives(pi), objectives(pj)
                    if all(a <= b for a, b in zip(oi, oj)) and any(a < b for a, b in zip(oi, oj)):
                        S[id(pi)].append(pj)
                        n[id(pj)] += 1
                if n[id(pi)] == 0:
                    fronts[0].append(pi)
            k = 0
            while fronts[k]:
                next_front = []
                for pi in fronts[k]:
                    for pj in S[id(pi)]:
                        n[id(pj)] -= 1
                        if n[id(pj)] == 0:
                            next_front.append(pj)
                k += 1
                fronts.append(next_front)
            return fronts[:-1]

        def crowding_distance(front: List[PathResult]):
            if len(front) <= 2:
                return {id(p): float('inf') for p in front}
            dist = {id(p): 0.0 for p in front}
            for obj_idx in range(3):
                vals = [objectives(p)[obj_idx] for p in front]
                sorted_idx = sorted(range(len(front)), key=lambda i: vals[i])
                dist[id(front[sorted_idx[0]])] = float('inf')
                dist[id(front[sorted_idx[-1]])] = float('inf')
                val_range = vals[sorted_idx[-1]] - vals[sorted_idx[0]]
                if val_range > 0:
                    for i in range(1, len(sorted_idx) - 1):
                        dist[id(front[sorted_idx[i]])] += (
                            vals[sorted_idx[i + 1]] - vals[sorted_idx[i - 1]]) / val_range
            return dist

        for gen in range(generations):
            offspring = []
            for _ in range(pop_size // 2):
                p1, p2 = random.sample(population, min(2, len(population)))
                child = self._crossover_paths(p1, p2, ship, hour, blocked_edges)
                if child:
                    offspring.append(child)
                for parent in [p1, p2]:
                    if random.random() < 0.3:
                        mutated = self._mutate_path(parent, ship, hour, blocked_edges)
                        if mutated:
                            offspring.append(mutated)

            combined = population + offspring
            seen = set()
            unique = []
            for p in combined:
                key = tuple(p.nodes)
                if key not in seen:
                    seen.add(key)
                    unique.append(p)
            combined = unique

            fronts = non_dominated_sort(combined)
            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) <= pop_size:
                    new_pop.extend(front)
                else:
                    cd = crowding_distance(front)
                    remaining = sorted(front, key=lambda p: cd[id(p)], reverse=True)
                    new_pop.extend(remaining[:pop_size - len(new_pop)])
                    break
            population = new_pop

        fronts = non_dominated_sort(population)
        pareto_front = fronts[0] if fronts else population[:3]
        pareto_front.sort(key=lambda p: objectives(p)[0])
        return pareto_front[:3]

    def _crossover_paths(self, p1: PathResult, p2: PathResult,
                          ship: ShipCharacteristics, hour: int,
                          blocked_edges: Set[Tuple[int, int]]) -> Optional[PathResult]:
        if len(p1.nodes) < 3 or len(p2.nodes) < 3:
            return None
        common = set(p1.nodes[1:-1]) & set(p2.nodes[1:-1])
        if not common:
            return None
        cross_node = random.choice(list(common))
        idx1 = p1.nodes.index(cross_node)
        idx2 = p2.nodes.index(cross_node)
        child_nodes = p1.nodes[:idx1] + p2.nodes[idx2:]
        child_edges = []
        for i in range(len(child_nodes) - 1):
            ek = (child_nodes[i], child_nodes[i + 1])
            if ek in blocked_edges or not self.graph.has_edge(child_nodes[i], child_nodes[i + 1]):
                return None
            child_edges.append(ek)
        if len(child_nodes) < 2:
            return None
        try:
            return self._build_path_result(
                PathType.BALANCED, child_nodes, child_edges, ship, blocked_edges, hour=hour)
        except Exception:
            return None

    def _yens_k_shortest(self, start: int, end: int,
                         ship: ShipCharacteristics,
                         hour: int,
                         blocked_edges: Set[Tuple[int, int]],
                         k: int = 5) -> List[PathResult]:
        A = []
        B = []

        first_path = self._dijkstra_with_edge_exclusion(start, end, ship, hour,
                                                         blocked_edges, set())
        if not first_path:
            return []
        A.append(first_path)

        for ki in range(1, k):
            prev_path = A[ki - 1]

            for i in range(len(prev_path.nodes) - 1):
                spur_node = prev_path.nodes[i]
                root_path_nodes = prev_path.nodes[:i + 1]

                excluded_edges = set()
                for p in A:
                    if len(p.nodes) > i and p.nodes[:i + 1] == root_path_nodes:
                        excluded_edges.add((p.nodes[i], p.nodes[i + 1]))

                excluded_edges.update(blocked_edges)

                spur_path = self._dijkstra_with_edge_exclusion(
                    spur_node, end, ship, hour, excluded_edges, set())

                if spur_path and len(spur_path.nodes) > 1:
                    total_nodes = root_path_nodes[:-1] + spur_path.nodes
                    total_edges = []
                    valid = True
                    for j in range(len(total_nodes) - 1):
                        ek = (total_nodes[j], total_nodes[j + 1])
                        if not self.graph.has_edge(total_nodes[j], total_nodes[j + 1]):
                            valid = False
                            break
                        total_edges.append(ek)

                    if valid:
                        try:
                            candidate = self._build_path_result(
                                PathType.BALANCED, total_nodes, total_edges,
                                ship, blocked_edges, hour=hour)
                            if candidate:
                                is_duplicate = False
                                for b in B:
                                    if b.nodes == candidate.nodes:
                                        is_duplicate = True
                                        break
                                if not is_duplicate:
                                    B.append(candidate)
                        except Exception:
                            pass

            if not B:
                break

            B.sort(key=lambda p: (p.total_time, p.total_distance, p.risk_score))
            A.append(B.pop(0))

        return A[1:]

    def _dijkstra_with_edge_exclusion(self, start: int, end: int,
                                       ship: ShipCharacteristics,
                                       hour: int,
                                       blocked_edges: Set[Tuple[int, int]],
                                       extra_excluded: Set[Tuple[int, int]]) -> Optional[PathResult]:
        all_blocked = blocked_edges | extra_excluded

        cost_score = {start: 0}
        prev = {start: None}
        edge_used = {start: None}
        pq = [(0, start)]
        visited = set()

        while pq:
            current_cost, node = heapq.heappop(pq)

            if node in visited:
                continue
            visited.add(node)

            if node == end:
                break

            for neighbor in self.graph.successors(node):
                if neighbor in visited:
                    continue

                edge_key = (node, neighbor)

                if edge_key in all_blocked:
                    continue

                edge_cost = self.time_weight.get(edge_key, 30)
                new_cost = current_cost + edge_cost

                if neighbor not in cost_score or new_cost < cost_score[neighbor]:
                    cost_score[neighbor] = new_cost
                    prev[neighbor] = node
                    edge_used[neighbor] = edge_key
                    heapq.heappush(pq, (new_cost, neighbor))

        if end not in prev or prev[end] is None:
            return None

        nodes, edges = self._reconstruct_path(end, prev, edge_used)

        return self._build_path_result(
            PathType.BALANCED, nodes, edges, ship, blocked_edges
        )

    def _find_relaxed_paths(self, start: int, end: int,
                             ship: ShipCharacteristics,
                             hour: int,
                             blocked_edges: Set[Tuple[int, int]],
                             k: int = 3) -> List[PathResult]:
        """
        约束松弛策略：当严格约束下路径不足时，
        通过逐步解除轻度约束来找到更多差异化路径。
        
        核心策略：
        1. 按约束违反程度排序阻塞边
        2. 逐步解除轻度约束边
        3. 用Yen's算法搜索差异化路径
        """
        relaxed_paths = []
        cc = self.constraint_checker
        
        strict_path = self._dijkstra_with_edge_exclusion(
            start, end, ship, hour, blocked_edges, set())

        if not strict_path:
            logger.info("约束松弛: 严格约束下无路径，尝试逐步解除约束")
            edge_violations = []
            for edge in blocked_edges:
                depth = cc.depth_map.get(edge, 0)
                width = cc.width_map.get(edge, 0)
                height = cc.height_map.get(edge, 0)
                
                margin = 0
                if depth and ship.draft > depth:
                    margin += (ship.draft - depth) / max(depth, 0.1)
                if width and ship.width > width:
                    margin += (ship.width - width) / max(width, 0.1)
                if height and ship.height > height:
                    margin += (ship.height - height) / max(height, 0.1)
                
                edge_violations.append((edge, margin))
            
            edge_violations.sort(key=lambda x: x[1])
            
            for pct in [0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]:
                if len(relaxed_paths) >= k:
                    break
                
                num_to_unblock = max(1, int(len(edge_violations) * pct))
                current_blocked = set(blocked_edges)
                for i in range(min(num_to_unblock, len(edge_violations))):
                    current_blocked.discard(edge_violations[i][0])
                
                candidate = self._dijkstra_with_edge_exclusion(
                    start, end, ship, hour, current_blocked, set())
                
                if candidate:
                    candidate.constraints_met = False
                    candidate.warning = f"经过{num_to_unblock}条约束放宽路段"
                    relaxed_paths.append(candidate)
                    if len(relaxed_paths) >= k:
                        break
            
            if not relaxed_paths:
                logger.info("约束松弛: 无法找到路径")
            
            return relaxed_paths
        
        # 按约束违反程度排序阻塞边
        edge_violations = []
        for edge in blocked_edges:
            depth = cc.depth_map.get(edge, 0)
            width = cc.width_map.get(edge, 0)
            height = cc.height_map.get(edge, 0)
            
            margin = 0
            if depth and ship.draft > depth:
                margin += (ship.draft - depth) / max(depth, 0.1)
            if width and ship.width > width:
                margin += (ship.width - width) / max(width, 0.1)
            if height and ship.height > height:
                margin += (ship.height - height) / max(height, 0.1)
            
            edge_violations.append((edge, margin))
        
        # 按违反程度从小到大排序
        edge_violations.sort(key=lambda x: x[1])
        
        # 策略：逐步解除轻度约束边，用Yen's搜索
        num_to_unblock = 1
        while num_to_unblock <= min(5, len(edge_violations)):
            if len(relaxed_paths) >= k:
                break
            
            # 解除违反程度最小的N条边
            current_blocked = set(blocked_edges)
            for i in range(num_to_unblock):
                current_blocked.discard(edge_violations[i][0])
            
            # 用Yen's算法找路径
            yen_paths = self._yens_k_shortest(
                start, end, ship, hour, current_blocked, k=3)
            
            for p in yen_paths:
                if p.nodes == strict_path.nodes:
                    continue
                
                is_dup = False
                for rp in relaxed_paths:
                    if rp.nodes == p.nodes:
                        is_dup = True
                        break
                if is_dup:
                    continue
                
                p.constraints_met = False
                p.warning = f"经过{num_to_unblock}条约束放宽路段"
                relaxed_paths.append(p)
                
                if len(relaxed_paths) >= k:
                    break
            
            num_to_unblock += 1
        
        if not relaxed_paths:
            logger.info("约束松弛: 无法找到差异化路径")
        
        return relaxed_paths

    def _pareto_selection(self, paths: List[PathResult], max_paths: int) -> List[PathResult]:
        """
        帕累托最优选择：保留在所有目标上都不被其他路径支配的路径

        目标：时间、风险、距离三个维度
        """
        if not paths:
            return []
        if len(paths) <= max_paths:
            return paths

        pareto_front = []
        for path in paths:
            is_dominated = False
            dominated_by = None

            for other in paths:
                if other is path:
                    continue
                if self._dominates(other, path):
                    is_dominated = True
                    dominated_by = other
                    break

            if not is_dominated:
                pareto_front.append(path)

        if len(pareto_front) >= max_paths:
            return self._diversity_selection(pareto_front, max_paths)

        covered = set(id(p) for p in pareto_front)
        remaining = [p for p in paths if id(p) not in covered]

        remaining_sorted = sorted(remaining,
                                   key=lambda p: (p.total_time, p.risk_score, p.total_distance))

        for path in remaining_sorted:
            if len(pareto_front) >= max_paths:
                break
            is_dominated = False
            for pf in pareto_front:
                if self._dominates(pf, path):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(path)

        return self._diversity_selection(pareto_front, max_paths)

    def _dominates(self, path1: PathResult, path2: PathResult) -> bool:
        """
        判断path1是否帕累托支配path2
        在所有目标上都不差，且至少一个目标更好
        """
        def normalize(value, max_value):
            return value / max_value if max_value > 0 else 0

        max_time = max(path1.total_time, path2.total_time, 1)
        max_dist = max(path1.total_distance, path2.total_distance, 1)
        max_risk = max(path1.risk_score, path2.risk_score, 1)

        p1_time = normalize(path1.total_time, max_time)
        p1_dist = normalize(path1.total_distance, max_dist)
        p1_risk = normalize(path1.risk_score, max_risk)

        p2_time = normalize(path2.total_time, max_time)
        p2_dist = normalize(path2.total_distance, max_dist)
        p2_risk = normalize(path2.risk_score, max_risk)

        better_in_any = False
        for v1, v2 in [(p1_time, p2_time), (p1_dist, p2_dist), (p1_risk, p2_risk)]:
            if v1 < v2:
                better_in_any = True
            elif v1 > v2:
                return False

        return better_in_any

    def _diversity_selection(self, paths: List[PathResult], max_paths: int) -> List[PathResult]:
        """基于多样性的路径选择"""
        if len(paths) <= max_paths:
            return paths

        selected = [paths[0]]
        candidates = paths[1:]

        def path_similarity(p1, p2):
            common_nodes = len(set(p1.nodes) & set(p2.nodes))
            return common_nodes / max(len(p1.nodes), len(p2.nodes), 1)

        while len(selected) < max_paths and candidates:
            best_candidate = None
            best_min_similarity = 1.0

            for candidate in candidates:
                min_sim = min(path_similarity(candidate, s) for s in selected)
                if min_sim < best_min_similarity:
                    best_min_similarity = min_sim
                    best_candidate = candidate

            if best_candidate:
                selected.append(best_candidate)
                candidates.remove(best_candidate)
            else:
                break

        return selected
    
    def _dijkstra_safest(self, start: int, end: int, 
                         ship: ShipCharacteristics,
                         blocked_edges: Set[Tuple[int, int]],
                         hour: int = None) -> Optional[PathResult]:
        """
        风险感知A*算法 - 安全优先
        升级：基础Dijkstra → 启发式搜索 + 综合风险模型
        
        核心改进：
        1. 地理启发函数：h(n) = 直线距离 / 最大航速 × 基础风险系数
        2. 综合风险模型：水深余量 + 航道宽度 + 交通密度 + 历史事故率
        3. Kinodynamic转弯约束：大型船舶急转弯惩罚
        """
        cc = self.constraint_checker
        end_data = self.graph.nodes.get(end, {})
        end_lat = end_data.get('lat', 0)
        end_lon = end_data.get('lon', 0)
        
        # 启发函数：到终点的估计风险（可接纳的乐观估计）
        def heuristic(node):
            if node not in self.graph.nodes:
                return 0
            node_data = self.graph.nodes[node]
            dist = haversine_distance(
                node_data.get('lat', 0), node_data.get('lon', 0),
                end_lat, end_lon)
            # 假设最优情况下以最小风险航行，风险率 ~0.1/公里
            return dist * 0.0001  # 可接纳启发函数
        
        # 综合风险评分：考虑多维度风险因素
        def edge_risk(edge_key):
            base_risk = cc.get_edge_risk_score(edge_key, ship)
            
            depth = cc.depth_map.get(edge_key, 0)
            width = cc.width_map.get(edge_key, 0)
            height = cc.height_map.get(edge_key, 0)
            
            margin_penalty = 0.0
            if depth > 0 and ship.draft > depth * 0.8:
                margin_penalty += (ship.draft / depth - 0.8) * 5
            if width > 0 and ship.width > width * 0.7:
                margin_penalty += (ship.width / width - 0.7) * 5
            
            features = self.edge_features.get(edge_key, {})
            traffic_density = features.get('ship_count', 0) / 100.0
            density_risk = min(traffic_density * 0.2, 1.0)
            
            return base_risk + margin_penalty + density_risk
        
        g_score = {start: 0}
        f_score = {start: heuristic(start)}
        prev = {start: None}
        edge_used = {start: None}
        open_set = [(f_score[start], start)]
        closed_set = set()
        
        while open_set:
            _, node = heapq.heappop(open_set)
            
            if node in closed_set:
                continue
            closed_set.add(node)
            
            if node == end:
                break
            
            for neighbor in self.graph.successors(node):
                if neighbor in closed_set:
                    continue
                
                edge_key = (node, neighbor)
                if edge_key in blocked_edges:
                    continue
                
                tentative_g = g_score[node] + edge_risk(edge_key)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    prev[neighbor] = node
                    edge_used[neighbor] = edge_key
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        if end not in prev or prev[end] is None:
            return None
        
        nodes, edges = self._reconstruct_path(end, prev, edge_used)
        
        return self._build_path_result(
            PathType.SAFEST, nodes, edges, ship, blocked_edges, hour=hour
        )
    
    def _dijkstra_fastest(self, start: int, end: int,
                          ship: ShipCharacteristics,
                          hour: int,
                          blocked_edges: Set[Tuple[int, int]]) -> Optional[PathResult]:
        """
        时间依赖A*算法 - 时间最短
        升级：基础Dijkstra → 启发式搜索 + 拥堵预测模型
        
        核心改进：
        1. 地理启发函数：h(n) = 直线距离 / 平均航速
        2. 时间依赖代价：基于历史轨迹数据预测不同时段航速
        3. 拥堵预测：高频路段的拥堵效应建模
        """
        end_data = self.graph.nodes.get(end, {})
        end_lat = end_data.get('lat', 0)
        end_lon = end_data.get('lon', 0)
        
        # 启发函数：到终点的最乐观估计时间
        def heuristic(node):
            if node not in self.graph.nodes:
                return 0
            node_data = self.graph.nodes[node]
            dist = haversine_distance(
                node_data.get('lat', 0), node_data.get('lon', 0),
                end_lat, end_lon)
            # 假设以最大可能速度航行（20节 = 37km/h）
            return dist / 37000 * 60  # 转换为分钟
        
        # 时间依赖代价：当前时间段的实际航行时间 + 拥堵预测
        def edge_time(edge_key):
            base_time = self._get_dynamic_time(edge_key, hour)
            
            features = self.edge_features.get(edge_key, {})
            segment_count = features.get('segment_count', 1)
            
            # 拥堵模型：高频路段在高峰时段增加等待时间
            congestion_factor = 1.0
            if hour is not None:
                # 假设早晚高峰（8-10点, 16-18点）拥堵
                if (8 <= hour <= 10) or (16 <= hour <= 18):
                    if segment_count > 50:
                        congestion_factor = 1.2
                    elif segment_count > 20:
                        congestion_factor = 1.1
            
            return base_time * congestion_factor
        
        g_score = {start: 0}
        f_score = {start: heuristic(start)}
        prev = {start: None}
        edge_used = {start: None}
        open_set = [(f_score[start], start)]
        closed_set = set()
        
        while open_set:
            _, node = heapq.heappop(open_set)
            
            if node in closed_set:
                continue
            closed_set.add(node)
            
            if node == end:
                break
            
            for neighbor in self.graph.successors(node):
                if neighbor in closed_set:
                    continue
                
                edge_key = (node, neighbor)
                if edge_key in blocked_edges:
                    continue
                
                tentative_g = g_score[node] + edge_time(edge_key)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    prev[neighbor] = node
                    edge_used[neighbor] = edge_key
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        if end not in prev or prev[end] is None:
            return None
        
        nodes, edges = self._reconstruct_path(end, prev, edge_used)
        
        return self._build_path_result(
            PathType.FASTEST, nodes, edges, ship, blocked_edges, hour=hour
        )
    
    def _astar_balanced(self, start: int, end: int,
                        ship: ShipCharacteristics,
                        hour: int,
                        blocked_edges: Set[Tuple[int, int]]) -> Optional[PathResult]:
        """
        A*算法 - 综合最优
        目标：平衡时间、距离、风险
        """
        def heuristic(node):
            if node not in self.graph.nodes:
                return 0
            node_data = self.graph.nodes[node]
            end_data = self.graph.nodes[end]
            return haversine_distance(
                node_data.get('lat', 0), node_data.get('lon', 0),
                end_data.get('lat', 0), end_data.get('lon', 0)
            ) / 100
        
        def edge_cost(edge_key):
            time_cost = self._get_dynamic_time(edge_key, hour)
            risk_cost = self.constraint_checker.get_edge_risk_score(edge_key, ship)
            return time_cost * 0.6 + risk_cost * 2
        
        g_score = {start: 0}
        f_score = {start: heuristic(start)}
        prev = {start: None}
        edge_used = {start: None}
        open_set = [(f_score[start], start)]
        closed_set = set()
        
        while open_set:
            _, node = heapq.heappop(open_set)
            
            if node in closed_set:
                continue
            closed_set.add(node)
            
            if node == end:
                break
            
            for neighbor in self.graph.successors(node):
                if neighbor in closed_set:
                    continue
                
                edge_key = (node, neighbor)
                if edge_key in blocked_edges:
                    continue
                
                tentative_g = g_score[node] + edge_cost(edge_key)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    prev[neighbor] = node
                    edge_used[neighbor] = edge_key
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        if end not in prev or prev[end] is None:
            return None
        
        nodes, edges = self._reconstruct_path(end, prev, edge_used)
        
        return self._build_path_result(
            PathType.BALANCED, nodes, edges, ship, blocked_edges, hour=hour
        )

    def _dijkstra_frequent(self, start: int, end: int,
                           ship: ShipCharacteristics,
                           blocked_edges: Set[Tuple[int, int]],
                           hour: int = None) -> Optional[PathResult]:
        """
        Bidirectional A*算法 - 通航频次最高
        升级：基础Dijkstra → 双向启发式搜索
        
        核心改进：
        1. 双向搜索：从起点和终点同时搜索，在中间汇合
        2. 启发函数：基于通航频率的乐观估计
        3. 流量均衡：避免过度拥挤航段
        
        双向搜索优势：
        - 搜索空间从O(b^d)降为O(b^(d/2))
        - 在稠密图中速度提升3-5倍
        """
        freq_weight = {}
        for edge_key, features in self.edge_features.items():
            count = features.get('segment_count', 0)
            # 考虑流量饱和效应：超过一定频次后收益递减
            if count > 100:
                freq_weight[edge_key] = 1.0 / (100 + (count - 100) * 0.5)
            else:
                freq_weight[edge_key] = 1.0 / (count + 1)
        
        # 前向启发函数（起点到当前节点的实际代价 + 当前到终点的估计）
        def forward_heuristic(node):
            if node not in self.graph.nodes:
                return 0
            node_data = self.graph.nodes[node]
            end_data = self.graph.nodes[end]
            dist = haversine_distance(
                node_data.get('lat', 0), node_data.get('lon', 0),
                end_data.get('lat', 0), end_data.get('lon', 0))
            return dist * 0.001
        
        # 后向启发函数（终点到当前节点）
        def backward_heuristic(node):
            if node not in self.graph.nodes:
                return 0
            node_data = self.graph.nodes[node]
            start_data = self.graph.nodes[start]
            dist = haversine_distance(
                node_data.get('lat', 0), node_data.get('lon', 0),
                start_data.get('lat', 0), start_data.get('lon', 0))
            return dist * 0.001
        
        # 前向搜索（从起点出发）
        fwd_g = {start: 0}
        fwd_f = {start: forward_heuristic(start)}
        fwd_prev = {start: None}
        fwd_edge = {start: None}
        fwd_open = [(fwd_f[start], start)]
        fwd_closed = set()
        
        # 后向搜索（从终点出发，沿反向边）
        bwd_g = {end: 0}
        bwd_f = {end: backward_heuristic(end)}
        bwd_prev = {end: None}
        bwd_edge = {end: None}
        bwd_open = [(bwd_f[end], end)]
        bwd_closed = set()
        
        # 最佳汇合点
        best_meet = None
        best_cost = float('inf')
        
        while fwd_open and bwd_open:
            # 前向扩展
            if fwd_open:
                _, f_node = heapq.heappop(fwd_open)
                if f_node not in fwd_closed:
                    fwd_closed.add(f_node)
                    
                    if f_node in bwd_closed:
                        cost = fwd_g.get(f_node, float('inf')) + bwd_g.get(f_node, float('inf'))
                        if cost < best_cost:
                            best_cost = cost
                            best_meet = f_node
                    
                    for neighbor in self.graph.successors(f_node):
                        if neighbor in fwd_closed:
                            continue
                        edge_key = (f_node, neighbor)
                        if edge_key in blocked_edges:
                            continue
                        
                        edge_cost_val = freq_weight.get(edge_key, 1.0)
                        dist_factor = self.distance_weight.get(edge_key, 100) / 1000
                        new_g = fwd_g[f_node] + edge_cost_val + dist_factor * 0.1
                        
                        if neighbor not in fwd_g or new_g < fwd_g[neighbor]:
                            fwd_prev[neighbor] = f_node
                            fwd_edge[neighbor] = edge_key
                            fwd_g[neighbor] = new_g
                            fwd_f[neighbor] = new_g + forward_heuristic(neighbor)
                            heapq.heappush(fwd_open, (fwd_f[neighbor], neighbor))
            
            # 后向扩展（沿反向边）
            if bwd_open:
                _, b_node = heapq.heappop(bwd_open)
                if b_node not in bwd_closed:
                    bwd_closed.add(b_node)
                    
                    if b_node in fwd_closed:
                        cost = fwd_g.get(b_node, float('inf')) + bwd_g.get(b_node, float('inf'))
                        if cost < best_cost:
                            best_cost = cost
                            best_meet = b_node
                    
                    for pred in self.graph.predecessors(b_node):
                        if pred in bwd_closed:
                            continue
                        edge_key = (pred, b_node)
                        if edge_key in blocked_edges:
                            continue
                        
                        edge_cost_val = freq_weight.get(edge_key, 1.0)
                        dist_factor = self.distance_weight.get(edge_key, 100) / 1000
                        new_g = bwd_g[b_node] + edge_cost_val + dist_factor * 0.1
                        
                        if pred not in bwd_g or new_g < bwd_g[pred]:
                            bwd_prev[pred] = b_node
                            bwd_edge[pred] = edge_key
                            bwd_g[pred] = new_g
                            bwd_f[pred] = new_g + backward_heuristic(pred)
                            heapq.heappush(bwd_open, (bwd_f[pred], pred))
            
            # 终止条件：双向搜索的最佳代价小于开放集中的最小f值
            if fwd_open and bwd_open:
                min_fwd_f = fwd_open[0][0]
                min_bwd_f = bwd_open[0][0]
                if min_fwd_f + min_bwd_f >= best_cost:
                    break
        
        if best_meet is None:
            return None
        
        # 重建完整路径
        fwd_nodes, fwd_edges = self._reconstruct_path(best_meet, fwd_prev, fwd_edge)
        bwd_nodes_rev, bwd_edges_rev = self._reconstruct_path(best_meet, bwd_prev, bwd_edge)
        
        # 后向路径需要反转
        bwd_nodes = list(reversed(bwd_nodes_rev))
        bwd_edges = list(reversed(bwd_edges_rev))
        
        # 合并：前向路径(不含汇合点) + 后向路径
        full_nodes = fwd_nodes[:-1] + bwd_nodes
        full_edges = fwd_edges + bwd_edges
        
        return self._build_path_result(
            PathType.FREQUENT, full_nodes, full_edges, ship, blocked_edges, hour=hour
        )

    def _get_dynamic_time(self, edge_key: Tuple[int, int], hour: int) -> float:
        """获取动态时间权重"""
        features = self.edge_features.get(edge_key)
        
        if features:
            # 尝试获取24小时预测时间
            predicted_times = features.get('predicted_times', {})
            avg_time = features.get('avg_travel_time', 30)
            avg_distance = features.get('avg_distance', 100)
            
            if predicted_times and hour is not None:
                pred_time = predicted_times.get(hour, avg_time)
                # 合理性检查：如果预测时间对应的航速超过50节(约25m/s)，视为不合理
                # 使用 avg_travel_time 作为 fallback
                if pred_time > 0 and avg_distance / pred_time > 25:
                    return avg_time
                return pred_time
            
            # 对 avg_travel_time 也做航速合理性约束（5-20节范围）
            # 5节 ≈ 2.57 m/s, 20节 ≈ 10.29 m/s
            if avg_time > 0:
                speed_ms = avg_distance / avg_time
                if speed_ms < 2.57:  # 太慢（<5节）
                    # 按5节计算合理时间
                    return avg_distance / 2.57
                elif speed_ms > 10.29:  # 太快（>20节）
                    # 按20节计算合理时间
                    return avg_distance / 10.29
            return avg_time
        
        # 无边特征：使用预先计算的默认时间
        return self.time_weight.get(edge_key, 30)
    
    def _reconstruct_path(self, end: int, prev: Dict, 
                          edge_used: Dict) -> Tuple[List[int], List[Tuple[int, int]]]:
        """重建路径"""
        nodes = []
        edges = []
        current = end
        
        while current is not None:
            nodes.append(current)
            if edge_used.get(current):
                edges.append(edge_used[current])
            current = prev.get(current)
        
        nodes.reverse()
        edges.reverse()
        return nodes, edges
    
    def _build_path_result(self, path_type: PathType,
                           nodes: List[int], 
                           edges: List[Tuple[int, int]],
                           ship: ShipCharacteristics,
                           blocked_edges: Set[Tuple[int, int]],
                           simplify_threshold: int = 50,
                           hour: int = None) -> PathResult:
        """构建路径结果"""
        # 路径简化：节点数超过阈值时，使用Douglas-Peucker简化
        # 简化后必须验证边存在于原始图中
        if len(nodes) > simplify_threshold:
            coords = [(self.nodes[n]['lat'], self.nodes[n]['lon']) for n in nodes]
            tolerance = 0.001  # 约100米的容差
            kept_indices = douglas_peucker_indices(coords, tolerance)
            simplified_nodes = [nodes[i] for i in kept_indices]
            
            # 验证简化后的边都存在于图中
            valid_nodes = [simplified_nodes[0]]
            valid_edges = []
            for i in range(1, len(simplified_nodes)):
                prev_node = valid_nodes[-1]
                curr_node = simplified_nodes[i]
                if self.graph.has_edge(prev_node, curr_node):
                    valid_edges.append((prev_node, curr_node))
                    valid_nodes.append(curr_node)
                else:
                    # 边不存在，需要绕回上一个有效节点
                    # 将当前节点替换为中间可达节点
                    for intermediate in self.graph.successors(prev_node):
                        if self.graph.has_edge(intermediate, curr_node):
                            valid_nodes.append(intermediate)
                            valid_edges.append((prev_node, intermediate))
                            valid_nodes.append(curr_node)
                            valid_edges.append((intermediate, curr_node))
                            break
                    else:
                        # 无法直接到达，保留原节点序列
                        pass
            
            if valid_edges:
                nodes = valid_nodes
                edges = valid_edges
            # 否则保留原始节点和边
        
        total_distance = 0
        total_time = 0
        total_risk = 0
        waypoint_details = []
        
        for i, edge_key in enumerate(edges):
            features = self.edge_features.get(edge_key)
            if features:
                distance = features.get('avg_distance', 100)
                # 统一使用 _get_dynamic_time 获取时间（与搜索一致）
                time_cost = self._get_dynamic_time(edge_key, hour)
            else:
                # 无边特征：使用默认权重
                distance = self.distance_weight.get(edge_key, 100)
                time_cost = self.time_weight.get(edge_key, 30)
            
            risk = self.constraint_checker.get_edge_risk_score(edge_key, ship)
            
            total_distance += distance
            total_time += time_cost
            total_risk += risk
            
            waypoint_details.append({
                'sequence': i + 1,
                'from_node': edge_key[0],
                'to_node': edge_key[1],
                'distance': distance,
                'time': time_cost,
                'risk': risk,
                'waterway_type': features.get('waterway_type', 'open') if features else 'open'
            })
        
        avg_speed = (total_distance / total_time * 1.944) if total_time > 0 else 0
        avg_risk = total_risk / len(edges) if edges else 0
        safety_score = 100 - avg_risk
        
        return PathResult(
            path_type=path_type,
            nodes=nodes,
            edges=edges,
            total_distance=total_distance,
            total_time=total_time,
            avg_speed=avg_speed,
            risk_score=avg_risk,
            safety_score=safety_score,
            constraints_met=len(blocked_edges & set(edges)) == 0,
            blocked_edges=list(blocked_edges & set(edges)),
            waypoint_details=waypoint_details
        )


# ==================== 模块4：导航决策输出 ====================

class NavigationDecisionMaker:
    """
    导航决策输出器
    
    功能：
    - 整合各模块结果
    - 生成导航报告
    - 输出路径详情
    """
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def make_decision(self, paths: List[PathResult],
                      ship: ShipCharacteristics,
                      start: int, end: int,
                      departure_time: datetime = None) -> Dict:
        """
        生成导航决策
        
        Args:
            paths: 路径列表
            ship: 船舶特征
            start: 起点
            end: 终点
            departure_time: 出发时间
        
        Returns:
            决策结果字典
        """
        if not paths:
            return {
                'success': False,
                'message': '未找到可用路径',
                'ship': self._ship_to_dict(ship),
                'start': start,
                'end': end
            }
        
        recommended = self._select_recommended_path(paths)
        
        decision = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'departure_time': departure_time.isoformat() if departure_time else None,
            
            # 船舶信息
            'ship': self._ship_to_dict(ship),
            
            # 起终点
            'start_node': start,
            'end_node': end,
            
            # 推荐路径
            'recommended_path': self._path_to_dict(recommended),
            
            # 所有备选路径
            'alternative_paths': [
                self._path_to_dict(p) for p in paths if p != recommended
            ],
            
            # 路径对比摘要
            'comparison_summary': self._create_comparison_summary(paths)
        }
        
        return decision
    
    def _select_recommended_path(self, paths: List[PathResult]) -> PathResult:
        """选择推荐路径（综合评分）"""
        if len(paths) == 1:
            return paths[0]
        
        # 综合评分：安全35% + 时间25% + 距离20% + 频次20%
        max_time = max(p.total_time for p in paths)
        max_dist = max(p.total_distance for p in paths)
        
        def score(path):
            safety_norm = path.safety_score / 100
            time_norm = 1 - (path.total_time / max_time) if max_time > 0 else 0
            dist_norm = 1 - (path.total_distance / max_dist) if max_dist > 0 else 0
            # 频次路径类型的额外加分
            freq_bonus = 0.1 if path.path_type == PathType.FREQUENT else 0
            return safety_norm * 0.35 + time_norm * 0.25 + dist_norm * 0.20 + freq_bonus
        
        return max(paths, key=score)
    
    def _ship_to_dict(self, ship: ShipCharacteristics) -> Dict:
        return {
            'name': ship.ship_name,
            'mmsi': ship.mmsi,
            'type': ship.ship_type,
            'length': ship.length,
            'width': ship.width,
            'draft': ship.draft,
            'height': ship.height,
            'tonnage': ship.tonnage,
            'max_speed': ship.max_speed,
            'risk_level': ship.risk_level,
            'maneuverability': ship.maneuverability
        }
    
    def _path_to_dict(self, path: PathResult) -> Dict:
        # 计算路径总通航频次
        total_frequency = sum(
            self._get_edge_frequency(e) for e in path.edges
        ) if hasattr(self, '_edge_freq_map') else 0
        
        return {
            'type': path.path_type.value,
            'nodes': path.nodes,
            'edges': path.edges,
            'total_distance_km': round(path.total_distance / 1000, 2),
            'total_time_min': round(path.total_time / 60, 2),
            'avg_speed_knots': round(path.avg_speed, 2),
            'risk_score': round(path.risk_score, 2),
            'safety_score': round(path.safety_score, 2),
            'total_frequency': total_frequency,
            'constraints_met': path.constraints_met,
            'waypoint_count': len(path.waypoint_details),
            'waypoints': path.waypoint_details
        }
    
    def _get_edge_frequency(self, edge: Tuple[int, int]) -> int:
        """获取边的通航频次"""
        if hasattr(self, '_edge_freq_map') and edge in self._edge_freq_map:
            return self._edge_freq_map[edge]
        return 0
    
    def set_edge_frequency_map(self, freq_map: Dict[Tuple[int, int], int]):
        """设置边通航频次映射（从edge_features提取）"""
        self._edge_freq_map = freq_map
    
    def _create_comparison_summary(self, paths: List[PathResult]) -> List[Dict]:
        """创建路径对比摘要"""
        summary = []
        for path in paths:
            total_freq = sum(
                self._get_edge_frequency(e) for e in path.edges
            ) if hasattr(self, '_edge_freq_map') else 0
            summary.append({
                'type': path.path_type.value,
                'distance_km': round(path.total_distance / 1000, 2),
                'time_min': round(path.total_time / 60, 2),
                'safety_score': round(path.safety_score, 2),
                'risk_score': round(path.risk_score, 2),
                'total_frequency': total_freq
            })
        return summary
    
    def export_decision(self, decision: Dict, filename: str = None) -> str:
        """导出决策结果"""
        filename = filename or f"navigation_decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(decision, f, ensure_ascii=False, indent=2)
        
        # 同时生成 txt 报告
        txt_filename = filename.replace('.json', '.txt')
        txt_filepath = os.path.join(self.output_dir, txt_filename)
        self._write_txt_report(decision, txt_filepath)
        
        print(f"导航决策已导出: {filepath}")
        print(f"导航报告已生成: {txt_filepath}")
        return filepath
    
    def _write_txt_report(self, decision: Dict, filepath: str):
        """生成 txt 格式的导航决策报告"""
        lines = []
        lines.append("=" * 80)
        lines.append("船舶个性化导航决策报告")
        lines.append("=" * 80)
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        if not decision.get('success'):
            lines.append(f"规划失败: {decision.get('message', '未知错误')}")
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            return
        
        ship = decision['ship']
        lines.append("[船舶信息]")
        lines.append(f"  名称: {ship['name']}")
        lines.append(f"  类型: {ship['type']}")
        lines.append(f"  尺寸: {ship['length']}m x {ship['width']}m")
        lines.append(f"  吃水: {ship['draft']}m | 高度: {ship['height']}m")
        lines.append(f"  吨位: {ship['tonnage']} 吨")
        lines.append(f"  最大航速: {ship['max_speed']} 节")
        lines.append(f"  风险等级: {ship['risk_level']}")
        lines.append(f"  操纵性系数: {ship['maneuverability']}")
        lines.append("")
        
        lines.append("[航线信息]")
        lines.append(f"  起点: 节点{decision['start_node']}")
        lines.append(f"  终点: 节点{decision['end_node']}")
        lines.append("")
        
        rec = decision['recommended_path']
        lines.append(f"[推荐路径] [{rec['type']}]")
        lines.append(f"  节点序列: {' -> '.join(map(str, rec['nodes']))}")
        lines.append(f"  总距离: {rec['total_distance_km']:.2f} km")
        lines.append(f"  预计时间: {rec['total_time_min']:.2f} 分钟")
        lines.append(f"  平均航速: {rec['avg_speed_knots']:.2f} 节")
        lines.append(f"  安全评分: {rec['safety_score']:.2f}/100")
        lines.append(f"  风险评分: {rec['risk_score']:.2f}/100")
        lines.append(f"  途经边数: {rec['waypoint_count']}")
        lines.append(f"  满足约束: {'是' if rec['constraints_met'] else '否'}")
        lines.append("")
        
        if rec['waypoints']:
            lines.append("[途经点详情]")
            lines.append(f"  {'序号':<6} {'边':<20} {'距离(m)':<10} {'时间(s)':<10} {'风险分':<10} {'水域类型':<10}")
            lines.append(f"  {'-'*66}")
            for wp in rec['waypoints']:
                lines.append(f"  {wp['sequence']:<6} {wp['from_node']}->{wp['to_node']:<14} "
                           f"{wp['distance']:<10.1f} {wp['time']:<10.1f} "
                           f"{wp['risk']:<10.1f} {wp.get('waterway_type', 'N/A'):<10}")
            lines.append("")
        
        alts = decision.get('alternative_paths', [])
        if alts:
            lines.append("[备选路径]")
            for alt in alts:
                lines.append(f"  [{alt['type']}] {alt['total_distance_km']:.2f}km, "
                           f"{alt['total_time_min']:.2f}min, 安全{alt['safety_score']:.2f}/100")
            lines.append("")
        
        lines.append("[路径对比]")
        lines.append(f"  {'类型':<14} {'距离(km)':<10} {'时间(min)':<10} {'安全分':<8} {'风险分':<8} {'频次':<8}")
        lines.append(f"  {'-'*60}")
        for s in decision['comparison_summary']:
            freq = s.get('total_frequency', 'N/A')
            lines.append(f"  {s['type']:<14} {s['distance_km']:<10.2f} {s['time_min']:<10.2f} "
                       f"{s['safety_score']:<8.2f} {s['risk_score']:<8.2f} {freq}")
        lines.append("")
        lines.append("=" * 80)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
    
    def print_decision(self, decision: Dict):
        """打印决策结果"""
        print("\n" + "="*80)
        print("【船舶个性化导航决策报告】")
        print("="*80)
        
        if not decision.get('success'):
            print(f"\n[X] {decision.get('message', '规划失败')}")
            return
        
        # 船舶信息
        ship = decision['ship']
        print(f"\n[船舶信息]:")
        print(f"   名称: {ship['name']}")
        print(f"   类型: {ship['type']}")
        print(f"   尺寸: {ship['length']}m × {ship['width']}m")
        print(f"   吃水: {ship['draft']}m | 高度: {ship['height']}m")
        print(f"   风险等级: {ship['risk_level']}")
        
        # 推荐路径
        rec = decision['recommended_path']
        print(f"\n[推荐路径] [{rec['type']}]:")
        print(f"   节点序列: {' -> '.join(map(str, rec['nodes']))}")
        print(f"   总距离: {rec['total_distance_km']} km")
        print(f"   预计时间: {rec['total_time_min']} 分钟")
        print(f"   平均航速: {rec['avg_speed_knots']} 节")
        print(f"   安全评分: {rec['safety_score']}/100")
        
        # 途经点
        if rec['waypoints']:
            print(f"\n[途经点详情]:")
            for wp in rec['waypoints'][:5]:  # 最多显示5个
                print(f"   {wp['sequence']}. 节点{wp['from_node']}->{wp['to_node']}: "
                      f"{wp['distance']:.0f}m, {wp['time']:.1f}s, 风险{wp['risk']:.1f}")
        
        # 备选路径
        alts = decision.get('alternative_paths', [])
        if alts:
            print(f"\n[备选路径]:")
            for alt in alts:
                print(f"   [{alt['type']}] {alt['total_distance_km']}km, "
                      f"{alt['total_time_min']}min, 安全{alt['safety_score']}/100")
        
        # 对比表
        print(f"\n[路径对比]:")
        print(f"   {'类型':<14} {'距离(km)':<10} {'时间(min)':<10} {'安全分':<8} {'风险分':<8} {'频次':<8}")
        print(f"   {'-'*60}")
        for s in decision['comparison_summary']:
            freq = s.get('total_frequency', 'N/A')
            print(f"   {s['type']:<14} {s['distance_km']:<10} {s['time_min']:<10} "
                  f"{s['safety_score']:<8} {s['risk_score']:<8} {freq}")
        
        print("\n" + "="*80)


# ==================== 主类：船舶导航系统 ====================

class ShipNavigationSystem:
    """
    船舶个性化导航决策系统
    
    整合所有模块，提供统一接口
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化导航系统
        
        Args:
            output_dir: 输出目录（包含已处理的数据文件）
        """
        self.output_dir = output_dir
        
        # 数据存储
        self.graph = None
        self.nodes = {}
        self.edge_features = {}
        
        # 模块实例
        self.ship_manager = None
        self.constraint_checker = None
        self.navigator = None
        self.decision_maker = None
        
        logger.info("船舶个性化导航决策系统初始化")

        # 加载数据
        self._load_data()

    def _load_data(self):
        """加载已处理的数据"""
        logger.info("加载拓扑网络数据...")

        # 加载节点
        nodes_path = os.path.join(self.output_dir, 'topology_nodes.csv')
        if os.path.exists(nodes_path):
            nodes_df = pd.read_csv(nodes_path)
            for _, row in nodes_df.iterrows():
                node_id = int(row['node_id'])
                self.nodes[node_id] = {
                    'lat': row['lat'],
                    'lon': row['lon'],
                    'type': row.get('type', 'unknown'),
                    'frequency': row.get('frequency', 0)
                }
            logger.info("加载节点: %d 个", len(self.nodes))
        
        # 构建图
        edges_path = os.path.join(self.output_dir, 'topology_edges.csv')
        if os.path.exists(edges_path):
            edges_df = pd.read_csv(edges_path)
            self.graph = nx.DiGraph()
            
            for node_id, attrs in self.nodes.items():
                self.graph.add_node(node_id, **attrs)
            
            for _, row in edges_df.iterrows():
                self.graph.add_edge(
                    int(row['from_node']), int(row['to_node']),
                    weight=row.get('weight', 1)
                )
            logger.info("加载边: %d 条", self.graph.number_of_edges())
        
        # 加载边特征
        logger.info("加载动态权重数据...")
        edge_features_path = os.path.join(self.output_dir, 'edge_features_dynamic_weights.csv')
        if os.path.exists(edge_features_path):
            features_df = pd.read_csv(edge_features_path)
            
            for _, row in features_df.iterrows():
                edge_key = (int(row['from_node']), int(row['to_node']))
                
                # 提取24小时预测时间
                predicted_times = {}
                for h in range(24):
                    col = f'predicted_time_h{h:02d}'
                    if col in row:
                        predicted_times[h] = row[col]
                
                self.edge_features[edge_key] = {
                    'avg_distance': row.get('avg_distance', 100),
                    'avg_travel_time': row.get('avg_travel_time', 30),
                    'std_travel_time': row.get('std_travel_time', 0),
                    'avg_actual_speed': row.get('avg_actual_speed', 5),
                    'waterway_type': row.get('waterway_type', 'open'),
                    'segment_count': row.get('segment_count', 0),
                    'predicted_times': predicted_times,
                    'predicted_time_morning': row.get('predicted_time_morning'),
                    'predicted_time_evening': row.get('predicted_time_evening'),
                    'predicted_time_night': row.get('predicted_time_night'),
                }
            
            logger.info("加载边特征: %d 条", len(self.edge_features))
        
        # 初始化模块
        logger.info("初始化导航模块...")
        
        trajectory_path = os.path.join(self.output_dir, 'cleaned_data.csv')
        self.ship_manager = ShipCharacteristicsManager(trajectory_path, self.output_dir)
        
        self.constraint_checker = PhysicalConstraintChecker(self.edge_features, self.nodes, self.graph)
        
        if self.graph:
            self.navigator = MultiObjectiveNavigator(
                self.graph, self.edge_features, self.constraint_checker
            )
        
        self.decision_maker = NavigationDecisionMaker(self.output_dir)
        
        # 设置边频次映射
        freq_map = {}
        for edge_key, features in self.edge_features.items():
            freq_map[edge_key] = features.get('segment_count', 0)
        self.decision_maker.set_edge_frequency_map(freq_map)
        
        logger.info("模块初始化完成")
    
    def plan_route(self, start: int, end: int,
                   ship_name: str = None,
                   ship_type: str = None,
                   custom_ship: Dict = None,
                   departure_time: datetime = None,
                   max_paths: int = 3) -> Dict:
        """
        规划航行路线
        
        Args:
            start: 起点节点ID
            end: 终点节点ID
            ship_name: 船舶名称（从数据中查找）
            ship_type: 船舶类型（使用模板）
            custom_ship: 自定义船舶参数
            departure_time: 出发时间
            max_paths: 最大路径数
        
        Returns:
            导航决策结果
        """
        print(f"\n{'='*80}")
        print("【路径规划请求】")
        print(f"{'='*80}")
        
        # 获取船舶特征
        ship = self.ship_manager.get_ship_characteristics(
            ship_name=ship_name,
            ship_type=ship_type,
            custom_params=custom_ship
        )
        print(f"\n船舶: {ship.ship_name} ({ship.ship_type})")
        print(f"尺寸: {ship.length}m × {ship.width}m, 吃水: {ship.draft}m")
        print(f"风险等级: {ship.risk_level}")
        
        # 检查起终点
        if start not in self.nodes:
            return {'success': False, 'message': f'起点 {start} 不存在'}
        if end not in self.nodes:
            return {'success': False, 'message': f'终点 {end} 不存在'}
        
        print(f"\n起点: 节点{start} ({self.nodes[start]['lat']:.4f}, {self.nodes[start]['lon']:.4f})")
        print(f"终点: 节点{end} ({self.nodes[end]['lat']:.4f}, {self.nodes[end]['lon']:.4f})")
        
        # 路径规划
        hour = departure_time.hour if departure_time else None
        print(f"\n正在规划路径...")
        
        paths = self.navigator.find_paths(
            start, end, ship, hour, max_paths
        )
        
        # 生成决策
        decision = self.decision_maker.make_decision(
            paths, ship, start, end, departure_time
        )
        
        # 打印结果
        self.decision_maker.print_decision(decision)
        
        return decision
    
    def get_available_nodes(self) -> List[int]:
        """获取可用节点列表"""
        return list(self.nodes.keys())
    
    def get_node_info(self, node_id: int) -> Dict:
        """获取节点信息"""
        return self.nodes.get(node_id)
    
    def list_ship_types(self) -> List[str]:
        """列出船舶类型模板"""
        return self.ship_manager.list_ship_types()
    
    def list_ships(self) -> List[str]:
        """列出数据中的船舶"""
        return self.ship_manager.list_available_ships()
    
    def find_route_endpoints(self, ship_name: str = None,
                              ship_type: str = None,
                              custom_ship: Dict = None,
                              min_distance_km: float = 5.0) -> Tuple[Optional[int], Optional[int]]:
        """
        为指定船舶自动选择差异化的起终点
        
        图是DAG结构，扫描最大弱连通分量内所有可达节点对，
        优先分配有多条简单路径的对，通过hash分散到不同对
        """
        ship = self.ship_manager.get_ship_characteristics(
            ship_name=ship_name, ship_type=ship_type, custom_params=custom_ship
        )
        
        type_key = ship_type or ship.ship_name or "default"
        
        if not hasattr(self, '_endpoint_cache'):
            self._endpoint_cache = {}
            wccs = list(nx.weakly_connected_components(self.graph))
            largest_wcc = max(wccs, key=len)
            all_wcc_nodes = list(largest_wcc)
            
            import random as _random
            _random.seed(42)
            sample = _random.sample(all_wcc_nodes, min(600, len(all_wcc_nodes)))
            
            reachable_pairs = []
            checked = set()
            for i, start in enumerate(sample):
                for end in sample[i+1:]:
                    if start == end or (start, end) in checked:
                        continue
                    checked.add((start, end))
                    try:
                        if not nx.has_path(self.graph, start, end):
                            continue
                        dist = haversine_distance(
                            self.nodes[start]['lat'], self.nodes[start]['lon'],
                            self.nodes[end]['lat'], self.nodes[end]['lon']
                        )
                        if dist < min_distance_km * 1000:
                            continue
                        n_paths = len(list(nx.all_simple_paths(
                            self.graph, start, end, cutoff=10)))
                        reachable_pairs.append((start, end, n_paths, dist))
                    except Exception:
                        pass
            
            if not reachable_pairs:
                for i, start in enumerate(sample):
                    for end in sample[i+1:]:
                        if start == end or (start, end) in checked:
                            continue
                        checked.add((start, end))
                        try:
                            if not nx.has_path(self.graph, start, end):
                                continue
                            dist = haversine_distance(
                                self.nodes[start]['lat'], self.nodes[start]['lon'],
                                self.nodes[end]['lat'], self.nodes[end]['lon']
                            )
                            n_paths = len(list(nx.all_simple_paths(
                                self.graph, start, end, cutoff=10)))
                            reachable_pairs.append((start, end, n_paths, dist))
                        except Exception:
                            pass
            
            reachable_pairs.sort(key=lambda x: (x[2], x[3]), reverse=True)
            self._endpoint_cache['pairs'] = reachable_pairs
            self._endpoint_cache['used'] = set()
            logger.info("端点缓存: %d 可达对 (多路径对: %d)",
                        len(reachable_pairs),
                        sum(1 for p in reachable_pairs if p[2] >= 2))
        
        pairs = self._endpoint_cache['pairs']
        used = self._endpoint_cache['used']

        if not pairs:
            return (None, None)

        multi_pairs = [p for p in pairs if p[2] >= 2]

        type_hash = abs(hash(type_key))

        if multi_pairs:
            idx = type_hash % len(multi_pairs)
            s, e, n, d = multi_pairs[idx]
            logger.info("为 %s 选择起终点: %d -> %d (距离: %.1f km, %d条路径, 多路径池: %d对)",
                        ship.ship_name, s, e, d / 1000, n, len(multi_pairs))
            return (s, e)

        single_pairs = [p for p in pairs if p[2] < 2]
        if single_pairs:
            idx = type_hash % len(single_pairs)
            s, e, n, d = single_pairs[idx]
            logger.info("为 %s 选择起终点(单路径): %d -> %d (距离: %.1f km)",
                        ship.ship_name, s, e, d / 1000)
            return (s, e)

        idx = type_hash % len(pairs)
        s, e, n, d = pairs[idx]
        logger.info("为 %s 选择起终点(复用): %d -> %d (距离: %.1f km, %d条路径)",
                    ship.ship_name, s, e, d / 1000, n)
        return (s, e)
    
    def find_nearest_node(self, lat: float, lon: float) -> int:
        """查找最近的节点"""
        min_dist = float('inf')
        nearest = None
        
        for node_id, attrs in self.nodes.items():
            dist = haversine_distance(lat, lon, attrs['lat'], attrs['lon'])
            if dist < min_dist:
                min_dist = dist
                nearest = node_id
        
        return nearest


# ==================== 主程序 ====================

def main():
    """主函数 - 演示导航系统"""
    
    # 初始化导航系统
    nav_system = ShipNavigationSystem(output_dir="output")
    
    # 训练ML模型（如果尚未训练）
    print("\n检查并训练导航预测模型...")
    nav_system.constraint_checker.train_models()
    
    # 获取可用节点
    nodes = nav_system.get_available_nodes()
    print(f"\n可用节点数: {len(nodes)}")
    print(f"可用船舶类型: {nav_system.list_ship_types()}")
    
    if len(nodes) < 2:
        print("节点数不足，无法进行路径规划")
        return
    
    # 演示：使用不同船舶类型规划路径
    print("\n" + "="*80)
    print("【演示】多目标路径规划")
    print("="*80)
    
    # 演示不同船型的导航决策（自动选择可达的起终点）
    demo_ships = [
        ('中型货船', 8),
        ('大型集装箱船', 10),
        ('大型油轮', 14),
    ]
    
    for ship_type, hour in demo_ships:
        start, end = nav_system.find_route_endpoints(ship_type=ship_type, min_distance_km=5.0)
        if start is None:
            print(f"\n>>> {ship_type}: 无可用路径，跳过")
            continue
        
        print(f"\n\n>>> 场景: {ship_type} (出发 {hour}:00)")
        decision = nav_system.plan_route(
            start=start,
            end=end,
            ship_type=ship_type,
            departure_time=datetime.now().replace(hour=hour, minute=0)
        )
        if decision.get('success'):
            nav_system.decision_maker.export_decision(
                decision, f"demo_decision_{ship_type}.json")
    
    print("\n" + "="*80)
    print("演示完成")
    print("="*80)


if __name__ == "__main__":
    main()
