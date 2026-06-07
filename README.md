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
| 数据预处理 | 110.6 万条清洗记录（原始 AIS 轨迹 → 异常过滤 → 卡尔曼平滑） |
| 节点提取 | 53,581 个候选节点（763 拐点 + 52,537 途经点 + 281 停泊点） |
| 节点聚类 | 1,244 个聚类节点（HDBSCAN + 航向感知特征 + KDE 聚类中心） |
| 拓扑网络 | **416 节点 / 506 条有向边**，含 HMM/Viterbi 地图匹配 |
| 权重建模 | 506 条有向边 → 动态耗时权重，**PNA ★ R²=0.9030**（7 模型对比） |
| 导航决策 | **10 种船型全部成功规划**，2-3 条差异化路径/船型 |

## 模型性能

| 模型 | MAE(s) | RMSE(s) | R² | MAPE(%) |
|------|--------|---------|-----|---------|
| xgboost | 9.39 | 27.76 | 0.7325 | 14.64 |
| lightgbm | 9.50 | 25.43 | 0.7754 | 15.95 |
| lightgbm_tweedie | 9.92 | 25.38 | 0.7764 | 17.12 |
| random_forest | 8.97 | 23.76 | 0.8040 | 14.48 |
| ngboost | 7.86 | 20.67 | 0.8517 | 13.24 |
| gnn | 8.57 | 22.33 | 0.8268 | 14.78 |
| **pna ★** | **7.61** | **16.72** | **0.9030** | **13.79** |

7 种模型统一对比（数据源 `output/model_metadata.json`），**PNA（Principal Neighbourhood Aggregation）综合最优**，R² 达 0.9030 显著高于其他模型。

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
| **通航频次最高** | 双向A*(频次倒数权重) + 张策2025奖惩函数，优先通行高密度航段 |
| **综合最优** | 安全(35%) + 时间(25%) + 距离(20%) + 频次(20%) 加权评分 |
| **约束放宽** | 物理约束全部阻塞时的兜底路径，标记风险提示 |

## 项目结构

```
Data_Element/
├── main.py                       # 主入口，模块化任务调度
├── app.py                        # Web API 服务（Flask + Leaflet.js 前端）
├── config.py                     # 全局配置参数
├── data_preprocessor.py          # 数据预处理（清洗、平滑、轨迹分割）
├── node_extractor.py             # 节点提取（拐点/分岔点/汇合点识别）
├── node_cluster.py               # 节点聚类（HDBSCAN/DBSCAN + 航向感知特征）
├── topology_builder.py           # 拓扑网络构建（有向图 + HMM/Viterbi匹配）
├── advanced_weight_model.py      # 动态权重建模（7模型对比 + 22维特征工程）
├── navigation_models.py          # 导航模型（ML风险预测 + 通过概率 + 多任务DNN）
├── adra_star_planner.py          # ADR-A* 自适应方向限制路径规划器
├── ship_navigator.py             # 船舶导航决策系统（特征检索 + 物理约束 + 多目标规划）
├── tests/                        # 单元测试与回归测试
│   └── test_frequent_path_bug.py #   通航频次最高路径 bug 修复回归测试（4 个用例）
├── visualize.py                  # 可视化（轨迹/节点/网络/统计图）
├── utils.py                      # 工具函数（Haversine距离、方位角等）
├── preprocess_edge_waypoints.py  # 边途经点预处理
├── fetch_waterways.py            # OSM 水道数据获取
├── requirements.txt              # Python 依赖
├── README.md                     # 本文档
├── 需求.md                       # 需求规格说明
├── 技术报告.md                    # 详细技术报告（含算法原理与评估）
├── 方法评估报告.md                 # 方法评估与对比
├── 初始需求.md                    # 初始需求文档
├── Data/                         # 原始AIS轨迹数据
│   └── *.xlsx (2个)
├── templates/                    # Web 前端页面
│   └── index.html                #   地图可视化页面（Leaflet.js + 暗色主题）
├── scripts/                      # 可视化脚本
│   ├── viz_model_comparison.py   #   模型对比图
│   ├── viz_feature_importance.py #   特征重要性图
│   ├── viz_prediction_scatter.py #   预测散点图
│   ├── viz_path_radar.py         #   路径雷达图
│   ├── viz_adra_benchmark.py     #   ADR-A* 基准测试
│   ├── viz_ablation_study.py     #   消融实验
│   ├── viz_module_timing.py      #   模块耗时分析
│   └── run_all_viz.py            #   批量运行所有可视化
├── data_osm/                     # OSM 水道地理数据
│   └── waterways.geojson
├── output/                       # 输出结果
│   ├── cleaned_data.csv                    # 清洗后轨迹 (110.6万条)
│   ├── extracted_nodes.csv                 # 提取节点 (53,581个)
│   ├── clustered_nodes.csv                 # 聚类节点 (1,244个)
│   ├── topology_nodes.csv / topology_edges.csv  # 拓扑网络 (416节点/506边)
│   ├── waterway_topology.json              # 完整拓扑结构 (JSON)
│   ├── edge_features_dynamic_weights.csv   # 动态权重特征 (4,103条边×24h)
│   ├── dynamic_time_matrix.csv             # 动态耗时矩阵
│   ├── ship_characteristics_db.csv         # 船舶特征数据库
│   ├── feature_importance.csv              # 特征重要性排名
│   ├── model_metadata.json                 # 模型元数据（含7模型对比结果）
│   ├── model_report.txt                    # 模型评估报告
│   ├── summary_report.txt                  # 项目汇总报告
│   ├── weight_model_gnn.pkl                # GNN模型
│   ├── weight_model_gnn_gnn_state.pt       # GNN state_dict
│   ├── weight_model_pna.pkl                # PNA最优模型
│   ├── weight_model_pna_gnn_state.pt       # PNA state_dict
│   ├── passability_model.pkl               # 通过性预测模型
│   ├── risk_prediction_model.pkl           # 风险预测模型
│   ├── multitask_navigation_model.pt       # 多任务导航DNN
│   ├── multitask_navigation_scaler.pkl     # 多任务导航归一化器
│   ├── navigation_random_sample.json      # 随机采样测试结果(由 tests/test_navigation_random.py 生成)
│   └── img/                                # 可视化图片
│       ├── trajectory_sample.png
│       ├── node_distribution.png
│       ├── topology_network.png
│       └── network_statistics.png
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
- **多路径展示**：同时展示安全优先、时间最短、综合最优、通航频次最高 4 种路径，颜色区分 + 图例标注
- **路径详情**：每条路径显示距离、耗时、平均航速、风险评分等指标
- **节点列表**：展示路径途经的航道节点序列

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/api/ship_types` | GET | 获取支持的船舶类型列表 |
| `/api/topology_nodes` | GET | 获取所有拓扑节点坐标 |
| `/api/plan` | POST | 执行路径规划，返回 4 种类型路径 |
| `/api/trajectory_sample` | GET | 获取采样轨迹数据用于回放动画 |

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
  "data": {
    "routes": [
      {
        "path_name": "安全优先",
        "statistics": {
          "total_distance_km": 18.52,
          "total_time_min": 45.3
        }
      },
      {
        "path_name": "时间最短",
        ...
      },
      {
        "path_name": "综合最优",
        ...
      },
      {
        "path_name": "通航频次最高",
        ...
      }
    ]
  }
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

## 最近更新

### 2026-06-05：导航报告精简 + 多 OD 测试修复

- **Task 7 不再生成 20 个 `navigation_*.json/txt` 文件**：原 `_task7_navigation` 改为内存缓存（`self.task7_results`），`summary_report` 从内存读取结果，避免 10 船型 × 2 格式 = 20 个冗余文件
- **`tests/test_navigation_random.py` 替代固定模板报告**：随机采样 30 组 (ship_type, OD, hour) 组合验证 `plan_route` 鲁棒性，结果汇总到单一 `navigation_random_sample.json`
- **大型油轮 NO_PATH 修复**：`_dijkstra_safest` 在 `blocked_edges` 存在但主路径返回 None 时，RELAXED 兜底逻辑未执行，已在 `return` 前补一次无约束调用
- **hour=0 与 hour=8 路径结果一致修复**：`edge_features_dynamic_weights.csv` 缺少 `predicted_time_h00..h23` 列导致动态时间回退到均值，改为时段级预测
- **短距离高跳数风险评分修复**：原 `total_risk = Σrisk` 简单累加，改为距离加权平均 `Σ(risk × distance) / Σdistance`
- **清理 20 个冗余 navigation_*.json/txt 和 8 个 `_xxx.py` 诊断脚本**

### 2026-06-03：频次路径可视化 + 物理约束松弛策略

- **Web 前端支持 4 种路径**：前端地图新增"通航频次最高"路径（粉紫色 `#e84393`）的渲染和图例，4 种路径颜色清晰区分
- **app.py 接入多目标导航器**：后端 `plan_paths` 重写为 `MultiObjectiveNavigator.find_paths()`，输出安全优先/时间最短/综合最优/通航频次最高 4 种路径
- **物理约束松弛兜底**：当某型船舶因吃水/船宽/限高全部被阻塞时，`find_paths` 自动放宽约束生成 `RELAXED` 次优路径，附带风险提示

### 2026-05-31：通航频次最高路径 bug 修复

修复了 `find_paths` 中导致"通航频次最高"路径无法正确生成的 3 个 bug：

1. `PathType('distance')` 抛 ValueError 被静默吞掉 — 'distance'/'time' 路径全部丢失
2. label 循环漏掉 `'frequent'` — `_bidirectional_a_star_frequent` 成为死代码
3. fallback 分支随意贴 FREQUENT 标签

**修复方案：** 用 `label_plan` 字典解耦"标签→算法+PathType"映射，'frequent' 走 双向A*(频次倒数) + 张策2025奖惩函数。

**验证：**
- 4 个新回归测试全部通过（`pytest tests/test_frequent_path_bug.py`）
- 真实拓扑 612→1403 路径总频次 = **178**（修复前为 0）
- 17/17 现有测试零回归

## 许可证

MIT License
