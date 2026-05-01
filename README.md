# 航道拓扑节点网络提取系统

基于海量AIS轨迹数据的航道拓扑网络构建与船舶智能导航路径规划系统。

## 功能概述

- **数据预处理**：清洗异常数据、卡尔曼/EMA轨迹平滑
- **节点提取**：识别拐点、分岔点、汇合点（方向变化检测+Douglas-Peucker简化）
- **节点聚类**：DBSCAN聚类识别高频节点
- **拓扑构建**：构建航道有向拓扑网络
- **权重建模**：多算法对比的动态路段耗时预测（RF/XGBoost/LightGBM/MLP/GNN）
- **可视化**：轨迹、节点、网络、统计图表可视化
- **导航决策**：船舶个性化导航系统（特征检索+物理约束+多目标路径规划）

## 项目结构

```
Data_Element/
├── config.py                 # 全局配置参数
├── main.py                   # 主入口（任务调度）
├── data_preprocessor.py      # 数据预处理（清洗、平滑）
├── node_extractor.py         # 节点提取
├── node_cluster.py           # 节点聚类
├── topology_builder.py       # 拓扑网络构建
├── advanced_weight_model.py  # 动态权重建模（多算法对比）
├── ship_navigator.py         # 船舶导航决策系统
├── visualize.py              # 可视化
├── utils.py                  # 工具函数（距离、方位角等）
├── Data/                     # 原始数据（.xlsx）
└── output/                   # 输出结果
    ├── cleaned_data.csv                # 清洗后轨迹数据
    ├── extracted_nodes.csv             # 提取的原始节点
    ├── clustered_nodes.csv             # 聚类后节点
    ├── topology_nodes.csv              # 拓扑网络节点
    ├── topology_edges.csv              # 拓扑网络边
    ├── waterway_topology.json          # 完整拓扑结构
    ├── edge_features_dynamic_weights.csv # 动态权重特征
    ├── dynamic_time_matrix.csv         # 动态耗时矩阵
    ├── ship_characteristics_db.csv     # 船舶特征数据库
    ├── navigation_*.json               # 导航决策结果
    ├── model_report.txt                # 模型评估报告
    ├── feature_importance.csv          # 特征重要性
    ├── summary_report.txt              # 汇总报告
    └── img/                            # 可视化图片
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> GNN模型需要额外安装 PyTorch Geometric，如不需要可跳过。

### 2. 运行完整流程

```bash
python main.py
```

### 3. 运行指定任务

```bash
# 只运行任务1-4
python main.py --task "1,2,3,4"

# 跳过任务7（高级模型对比）
python main.py --skip "7"

# 强制重新计算（忽略缓存）
python main.py --force
```

## 任务说明

| 任务 | 名称 | 描述 | 依赖 |
|------|------|------|------|
| 1 | 数据预处理 | 异常值过滤、卡尔曼/EMA平滑 | - |
| 2 | 节点提取 | 方向变化检测、Douglas-Peucker简化 | Task1 |
| 3 | 节点聚类 | DBSCAN聚类识别关键节点 | Task2 |
| 4 | 拓扑构建 | 构建有向加权图 | Task1,3 |
| 5 | 权重建模 | 多模型对比（XGBoost/LightGBM/RF/GNN）预测路段耗时 | Task1,4 |
| 6 | 可视化 | 生成轨迹/节点/网络/统计图 | Task1,3,4 |
| 7 | 导航决策 | 船舶个性化导航（特征检索+物理约束+多目标路径规划） | Task1-5 |

## 导航系统

### 架构

系统由四个核心模块组成：

1. **ShipCharacteristicsManager** - 船舶特征属性检索（船长、船宽、吃水、吨位等）
2. **PhysicalConstraintChecker** - 物理约束校验（吃水深度、限高、航道宽度）
3. **MultiObjectiveNavigator** - 多目标路径规划（改进A*/Dijkstra）
4. **NavigationDecisionMaker** - 导航决策输出

### 路径类型

- **安全优先** - 规避高风险航段，优先宽阔航道
- **时间最短** - 最快到达，接受一定风险
- **综合最优** - 安全与效率平衡
- **通航频次最高** - 选择高频通行航段
- **约束放宽路径** - 在严格约束无解时放宽条件

### 使用示例

```python
from ship_navigator import ShipNavigationSystem
from datetime import datetime

# 初始化
nav = ShipNavigationSystem(output_dir="output")

# 查看可用节点和船舶类型
nodes = nav.get_available_nodes()
ship_types = nav.list_ship_types()

# 路径规划
result = nav.plan_route(
    start=1,
    end=20,
    ship_type='中型货船',
    departure_time=datetime(2025, 1, 1, 8, 0)
)

# 根据坐标找最近节点
nearest_node = nav.find_nearest_node(lat=31.2, lon=121.5)

# 自动选择可达的远距离起终点
start, end = nav.find_route_endpoints(ship_type='大型油轮')
```

### 支持的船舶类型

- 小型货船 / 中型货船 / 大型货船
- 集装箱船 / 大型集装箱船
- 油轮 / 大型油轮
- 客船 / 渔船 / 拖船

## 配置参数

主要配置位于 `config.py`：

```python
# 数据清洗
CLEANING_CONFIG = {
    'max_speed': 30.0,        # 最大航速(节)
    'min_speed': 0.1,         # 最小航速
    'max_acceleration': 5.0,  # 最大加速度
    'max_time_gap': 3600,     # 最大时间间隔(秒)
    'max_distance_jump': 500, # 最大距离跳变(米)
}

# 节点聚类
CLUSTERING_CONFIG = {
    'eps': 100.0,             # DBSCAN邻域半径(米)
    'min_samples': 5,         # 最小样本数
}

# 拓扑构建
TOPOLOGY_CONFIG = {
    'edge_connection_distance': 200.0,  # 边连接距离(米)
    'min_edge_weight': 3,               # 最小边权重
}
```

## 可视化输出

位于 `output/img/`：

| 文件 | 说明 |
|------|------|
| `trajectory_sample.png` | 轨迹样本图 |
| `node_distribution.png` | 节点分布统计 |
| `topology_network.png` | 拓扑网络图 |
| `network_statistics.png` | 网络统计图 |

## 依赖库

| 库 | 用途 | 必需 |
|----|------|------|
| numpy, pandas | 数据处理 | 是 |
| openpyxl | Excel数据读取 | 是 |
| networkx | 图结构 | 是 |
| scikit-learn | 机器学习（RF等） | 是 |
| matplotlib | 可视化 | 是 |
| xgboost | 梯度提升树 | 可选 |
| lightgbm | 轻量级梯度提升 | 可选 |
| torch | 深度学习（MLP） | 可选 |
| torch-geometric | 图神经网络（GNN） | 可选 |

## 许可证

MIT License
