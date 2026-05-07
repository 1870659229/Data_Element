# 水上航道智能路径规划系统

> 基于海量AIS轨迹数据的航道拓扑网络构建与船舶个性化智能导航

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目简介

本系统从海量原始AIS船舶轨迹数据出发，通过数据清洗、节点提取聚类、拓扑网络构建、动态权重建模，最终实现面向不同船舶类型的**个性化多目标智能导航决策**。系统完整覆盖了从原始GPS数据到可视化导航结果的端到端流水线。

### 核心创新

- **数据驱动拓扑构建**：从真实轨迹中无监督聚类获取高频航行节点，不依赖人工标注航道
- **GNN动态耗时预测**：图神经网络融合空间拓扑与时序特征，路段耗时预测 R² 达 0.56
- **物理约束感知导航**：结合船舶吃水/限高/宽度等多维物理约束的差异化路径规划
- **多目标路径输出**：同时输出安全优先、时间最短、通航频次最高等多条差异化路径

## 实际运行结果

| 处理阶段 | 产出 |
|----------|------|
| 数据预处理 | 110.5 万条清洗记录（原始 AIS 轨迹 → 异常过滤 → 卡尔曼平滑） |
| 节点提取 | 36,651 个候选节点（8,872 拐点 + 27,324 途经点 + 455 停泊点） |
| 节点聚类 | 2,438 个聚类节点（DBSCAN + 航向感知特征） |
| 拓扑网络 | **1,718 节点 / 3,388 条有向边**，含 HMM 地图匹配 |
| 权重建模 | 6,192 条边动态耗时预测，**GNN 最优** |
| 导航决策 | **10 种船型全部成功规划**，2-3 条差异化路径/船型 |

## 模型性能

| 模型 | MAE(s) | RMSE(s) | R² | MAPE(%) |
|------|--------|---------|-----|---------|
| XGBoost | 11.76 | 25.27 | 0.316 | 52.02 |
| LightGBM | 11.67 | 24.33 | 0.366 | 50.79 |
| Random Forest | 11.73 | 24.98 | 0.332 | 51.65 |
| **GNN ★** | **8.21** | **24.81** | **0.557** | **17.72** |

GNN 相比传统方法 MAE 降低 30%，R² 提升 52%，MAPE 从 50%+ 降至 17.7%。

## 系统架构

```
原始AIS数据 (.xlsx)
       │
       ▼
┌──────────────────┐
│  Task1  数据预处理 │  异常过滤 → IsolationForest 漂移检测 → 卡尔曼平滑
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Task2  节点提取   │  方向变化检测 → Douglas-Peucker 简化 → DBSCAN 航向聚类
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Task3  节点聚类   │  DBSCAN/HDBSCAN + 航向感知特征聚类
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Task4  拓扑构建   │  有向图 DiGraph → HMM/Viterbi 地图匹配 → 双向边
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Task5  权重建模   │  22维特征工程 → XGBoost/LightGBM/RF/GNN 对比
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Task7  导航决策   │  特征检索 → 物理约束校验 → 多目标路径规划
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Web API 服务     │  Flask 后端 → RESTful API → Leaflet.js 前端
└────────┬─────────┘
         │
         ▼
  导航结果可视化 (地图展示 + 路径对比)
```

### 导航系统模块

| 模块 | 功能 |
|------|------|
| **ShipCharacteristicsManager** | 船舶特征检索（船长/宽/高/吃水/吨位），10 种模板 + 航速推断 |
| **PhysicalConstraintChecker** | 吃水深度校验、桥梁限高、航道宽度 + ML风险预测 |
| **MultiObjectiveNavigator** | A* / Dijkstra / Yen's K最短路径 + Kinodynamic转弯约束 |
| **NavigationDecisionMaker** | 多路径评分 → 推荐路径 + 备选路径 + 对比摘要 |

### 路径类型

| 类型 | 策略 |
|------|------|
| **安全优先** | 规避高风险航段，优先宽阔开阔水道 |
| **时间最短** | 24h动态耗时最小，接受可控风险 |
| **通航频次最高** | 选择高频通行航段，群体智慧路径 |
| **综合最优** | 安全(35%) + 时间(25%) + 距离(20%) + 频次(20%) 加权评分 |

## 项目结构

```
Data_Element/
├── main.py                       # 主入口，模块化任务调度
├── app.py                        # Web API 服务（Flask + Leaflet.js 前端）
├── config.py                     # 全局配置参数
├── data_preprocessor.py          # 数据预处理（清洗、平滑、轨迹分割）
├── node_extractor.py             # 节点提取（拐点/分岔点/汇合点识别）
├── node_cluster.py               # 节点聚类（DBSCAN/HDBSCAN）
├── topology_builder.py           # 拓扑网络构建（有向图 + HMM匹配）
├── advanced_weight_model.py      # 动态权重建模（多模型对比 + 特征工程）
├── navigation_models.py          # 导航模型（MLP/GNN + 多任务学习）
├── ship_navigator.py             # 船舶导航系统（特征检索 + 约束 + 规划）
├── visualize.py                  # 可视化（轨迹/节点/网络/统计图）
├── utils.py                      # 工具函数（Haversine距离、方位角等）
├── requirements.txt              # Python 依赖
├── README.md                     # 本文档
├── 需求.md                       # 需求规格说明
├── Data/                         # 原始AIS轨迹数据
│   └── *.xlsx (2个)
├── templates/                    # Web 前端页面
│   └── index.html                #   地图可视化页面（Leaflet.js）
└── output/                       # 输出结果
    ├── cleaned_data.csv                    # 清洗后轨迹 (110.5万条)
    ├── extracted_nodes.csv                 # 提取节点 (36,651个)
    ├── clustered_nodes.csv                 # 聚类节点 (2,438个)
    ├── topology_nodes.csv / topology_edges.csv  # 拓扑网络 (1,718节点/3,388边)
    ├── waterway_topology.json              # 完整拓扑结构 (JSON)
    ├── edge_features_dynamic_weights.csv   # 动态权重特征 (6,192条边×24h)
    ├── dynamic_time_matrix.csv             # 动态耗时矩阵
    ├── ship_characteristics_db.csv         # 船舶特征数据库
    ├── feature_importance.csv              # 特征重要性排名
    ├── model_metadata.json                 # 模型元数据（含对比结果）
    ├── model_report.txt                    # 模型评估报告
    ├── summary_report.txt                  # 项目汇总报告
    ├── weight_model_gnn.pkl                # GNN最优模型
    ├── navigation_*.json / .txt            # 10种船型导航决策结果
    └── img/                                # 可视化图片
        ├── trajectory_sample.png
        ├── node_distribution.png
        ├── topology_network.png
        └── network_statistics.png
```

## 快速开始

### 环境要求

- Python 3.10+
- 推荐使用虚拟环境

### 安装依赖

```bash
pip install -r requirements.txt
```

> GNN 模型需要 PyTorch + PyTorch Geometric。纯传统方法（XGBoost/LightGBM/RF）无需 GPU。

### 运行完整流程

```bash
python main.py
```

首次运行将从原始 xlsx 数据开始，依次执行 7 个任务，最终生成导航决策结果。中间结果会缓存至 `output/`，后续运行自动跳过已完成步骤。

### 选择性运行

```bash
# 仅运行数据预处理和拓扑构建
python main.py --task "1,2,3,4"

# 跳过权重建模和可视化，仅导航
python main.py --skip "5,6"

# 强制重新计算所有步骤
python main.py --force
```

### API 调用示例

```python
from ship_navigator import ShipNavigationSystem
from datetime import datetime

nav = ShipNavigationSystem(output_dir="output")

# 列出可用节点和船型
nodes = nav.get_available_nodes()
print(nav.list_ship_types())

# 路径规划
result = nav.plan_route(
    start=1,
    end=20,
    ship_type='中型货船',
    departure_time=datetime(2025, 1, 1, 8, 0)
)

# 按坐标查找最近节点
nearest = nav.find_nearest_node(lat=31.2, lon=121.5)

# 自动为指定船型选择合适的起终点
start, end = nav.find_route_endpoints(ship_type='大型油轮')
```

## Web 可视化前端

系统提供基于 Flask + Leaflet.js 的 Web 可视化界面，支持地图交互式路径规划。

### 启动方式

```bash
python app.py
```

启动后访问 http://localhost:5000 即可打开导航页面。

### 功能特性

- **地图选点**：点击地图选择起点/终点，也支持手动输入 GPS 坐标
- **拓扑节点参考**：地图上显示航道拓扑节点作为参考点
- **多船型支持**：下拉选择 10 种船舶类型
- **多路径展示**：同时展示综合最优、时间最短、距离最短等多条路径
- **路径详情**：每条路径显示距离、耗时、平均航速、风险评分等指标
- **节点列表**：展示路径途经的航道节点序列

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/api/ship_types` | GET | 获取支持的船舶类型列表 |
| `/api/topology_nodes` | GET | 获取所有拓扑节点坐标 |
| `/api/plan` | POST | 执行路径规划 |

#### /api/plan 请求示例

```json
{
  "start_lat": 22.9442,
  "start_lon": 113.5442,
  "end_lat": 23.0582,
  "end_lon": 113.4953,
  "ship_type": "大型货船"
}
```

#### /api/plan 响应示例

```json
{
  "success": true,
  "paths": [
    {
      "type": "综合最优",
      "total_distance_km": 18.52,
      "total_time_min": 45.3,
      "avg_speed_knots": 12.5,
      "risk_score": 35.2,
      "safety_score": 64.8,
      "nodes": [165, 182, 210, ...],
      "waypoints": [...]
    }
  ],
  "start_node": 165,
  "end_node": 1991
}
```

## 支持的船舶类型

| 类型 | 船长(m) | 船宽(m) | 吃水(m) | 限高(m) | 吨位 |
|------|---------|---------|---------|---------|------|
| 小型货船 | 80 | 12 | 4.5 | 15 | 3,000 |
| 中型货船 | 150 | 22 | 7.5 | 25 | 15,000 |
| 大型货船 | 250 | 32 | 11.0 | 30 | 50,000 |
| 集装箱船 | 200 | 30 | 10.0 | 40 | 35,000 |
| 大型集装箱船 | 350 | 45 | 14.0 | 50 | 100,000 |
| 油轮 | 180 | 28 | 9.0 | 20 | 25,000 |
| 大型油轮 | 300 | 50 | 15.0 | 25 | 120,000 |
| 客船 | 100 | 18 | 5.0 | 30 | 8,000 |
| 渔船 | 30 | 6 | 2.5 | 8 | 200 |
| 拖船 | 25 | 8 | 3.0 | 10 | 300 |

## 核心配置

主要参数位于 `config.py`，可按需调整：

```python
CLEANING_CONFIG = {
    'max_speed': 30.0,         # 最大航速(节)
    'min_speed': 0.1,          # 最小航速
    'max_acceleration': 5.0,   # 最大加速度(m/s²)
    'max_distance_jump': 500,  # 最大距离跳变(m)
}

CLUSTERING_CONFIG = {
    'eps': 100.0,              # DBSCAN 邻域半径(m)
    'min_samples': 5,          # 最小样本数
}

TOPOLOGY_CONFIG = {
    'edge_connection_distance': 200.0,  # 边连接距离(m)
}
```

## 依赖说明

| 库 | 用途 | 必需 |
|----|------|:--:|
| numpy, pandas | 数据处理与数值计算 | ✅ |
| openpyxl | Excel 数据读取 | ✅ |
| networkx | 图结构存储与路径算法 | ✅ |
| scikit-learn | IsolationForest / RF / 特征工程 | ✅ |
| matplotlib | 可视化图表 | ✅ |
| flask | Web API 服务 | ✅ |
| xgboost | 梯度提升树 | 可选 |
| lightgbm | 轻量级梯度提升 | 可选 |
| torch | 深度学习框架 | 可选 |
| torch-geometric | 图神经网络 | 可选 |

## 许可证

MIT License
