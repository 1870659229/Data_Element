# 水上航道智能路径规划系统

> 基于海量AIS轨迹数据的航道拓扑网络构建与船舶个性化智能导航

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 项目简介

本系统从海量原始AIS船舶轨迹数据出发，通过数据清洗、节点提取聚类、拓扑网络构建、动态权重建模，最终实现面向不同船舶类型的**个性化多目标智能导航决策**。系统完整覆盖了从原始GPS数据到可视化导航结果的端到端流水线。

### 核心创新

- **数据驱动拓扑构建**：从真实轨迹中无监督聚类获取高频航行节点，不依赖人工标注航道
- **GNN动态耗时预测**：图神经网络融合空间拓扑与时序特征，路段耗时预测 R² 达 0.8879
- **物理约束感知导航**：结合船舶吃水/限高/宽度等多维物理约束的差异化路径规划
- **多目标路径输出**：同时输出安全优先、时间最短、频次优先等多条差异化路径

## 实际运行结果

| 处理阶段 | 产出 |
|----------|------|
| 数据预处理 | 110.6 万条清洗记录（原始 AIS 轨迹 → 异常过滤 → 卡尔曼平滑） |
| 节点提取 | 53,581 个候选节点（763 拐点 + 52,537 途经点 + 281 停泊点） |
| 节点聚类 | 1,244 个聚类节点（HDBSCAN + 航向感知特征 + KDE 聚类中心） |
| 拓扑网络 | **416 节点 / 506 条有向边**，含 HMM/Viterbi 地图匹配 |
| 权重建模 | 506 条有向边 → 动态耗时权重，**PNA ★ R²=0.8879**（7 模型对比） |
| 导航决策 | **10 种船型全部成功规划**，2-3 条差异化路径/船型 |

## 模型性能

| 模型 | MAE(s) | RMSE(s) | R² | MAPE(%) |
|------|--------|---------|-----|---------|
| **pna ★** | **7.89** | **17.97** | **0.8879** | **13.61** |
| pna_5seed_stability_mean | 8.29 | 20.89 | 0.8485 | 13.45 |
| lightgbm_tweedie | 9.92 | 25.38 | 0.7764 | 17.12 |
| ngboost | 8.32 | 22.17 | 0.8294 | 13.82 |
| gnn (GAT) | 8.74 | 23.93 | 0.8012 | 14.69 |
| xgboost | 9.39 | 27.76 | 0.7325 | 14.64 |
| random_forest | 8.97 | 23.76 | 0.8040 | 14.48 |
| lightgbm | 9.50 | 25.43 | 0.7754 | 15.95 |

7 种模型统一对比 + PNA 5-seed 稳定性均值（7 种模型数据源 `output/model_metadata.json`；PNA 5-seed 数据源 `output/weight_model_pna_stability_5seed.pkl`），**PNA（Principal Neighbourhood Aggregation）综合最优**，单次训练 R² 达 0.8879 显著高于次优 NGBoost(0.8294)；NGBoost 调参对种子敏感（种 42 R²=0.8594 vs 种 7 R²=0.8294），实际使用建议多种子平均。

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
│  Task5  权重建模   │  28维特征工程 → XGBoost/LightGBM/RF/GNN/PNA 7模型对比
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

### 文件级数据流

上图是概念流程，实际运行时各 Task 之间的数据传递通过 `output/` 目录下的文件完成：

```mermaid
graph TD
    XLSX[Data/*.xlsx<br/>原始AIS轨迹]
    T1[Task1 数据预处理]
    T2[Task2 节点提取]
    T3[Task3 节点聚类]
    T4[Task4 拓扑构建]
    T5[Task5 权重建模]
    T55[Task5.5 中心性]
    T6[Task6 可视化]
    T7[Task7 导航决策]

    C1[output/cleaned_data.csv]
    C2[output/extracted_nodes.csv]
    C3[output/clustered_nodes.csv]
    C4[output/topology_nodes.csv<br/>topology_edges.csv<br/>waterway_topology.json]
    C5[output/edge_features_dynamic_weights.csv<br/>weight_model_*.pkl<br/>model_metadata.json]
    C55[output/node_centrality.csv]
    C6[output/img/*.png]
    C7[内存 task7_results<br/>+ risk_prediction_model.pkl<br/>+ passability_model.pkl]

    XLSX --> T1 --> C1
    C1 --> T2 --> C2
    C2 --> T3 --> C3
    C3 --> T4
    C1 -.-> T4
    T4 --> C4
    C4 --> T5
    C1 -.-> T5
    T5 --> C5
    C4 -.-> T55
    T55 --> C55
    C1 --> T6
    C3 --> T6
    C4 --> T6
    T6 --> C6
    C4 --> T7
    C5 --> T7
    T7 --> C7
```

各任务输入输出对照：

| Task | 输入 | 输出 | 说明 |
|------|------|------|------|
| 1 | `Data/*.xlsx` | `cleaned_data.csv` | 异常过滤 + IsolationForest + Kalman平滑 |
| 2 | `cleaned_data.csv` | `extracted_nodes.csv` | 拐点/分岔点/停泊点提取 |
| 3 | `extracted_nodes.csv` | `clustered_nodes.csv` | HDBSCAN + 航向感知聚类 |
| 4 | `clustered_nodes.csv` + `cleaned_data.csv` | `topology_nodes.csv` `topology_edges.csv` `waterway_topology.json` | HMM/Viterbi 地图匹配 → 有向图 |
| 5 | 拓扑文件 + `cleaned_data.csv` | `edge_features_dynamic_weights.csv` `weight_model_*.pkl` `model_metadata.json` | 7模型对比 + PNA集成 |
| 5.5 | 拓扑文件 | `node_centrality.csv` | 4种中心性 + PCA自动赋权 |
| 6 | `cleaned_data.csv` + `clustered_nodes.csv` + 拓扑文件 | `img/*.png` | 4张核心可视化图 |
| 7 | 拓扑文件 + `edge_features_dynamic_weights.csv` | 内存（`task7_results`）+ `risk_prediction_model.pkl` `passability_model.pkl` `multitask_navigation_scaler.pkl` | 10种船型导航决策不落盘；但 `train_models()` 训练的导航ML模型会保存为 pkl |

### 导航系统模块

| 模块 | 功能 |
|------|------|
| **ShipCharacteristicsManager** | 船舶特征检索（船长/宽/高/吃水/吨位），309 艘真实数据 + 8 种模板兜底 |
| **PhysicalConstraintChecker** | 吃水深度校验、桥梁限高、航道宽度 + ML风险预测 |
| **MultiObjectiveNavigator** | 改进A*(风险感知/时间依赖) + 双向A*(频次) + 船型感知权重扰动 |
| **NavigationDecisionMaker** | 多路径评分 → 推荐路径 + 备选路径 + 对比摘要 |

### 路径类型

| 类型 | 策略 |
|------|------|
| **安全优先** | 规避高风险航段，优先宽阔开阔水道 |
| **时间最短** | 24h动态耗时最小，接受可控风险 |
| **距离最短** | 物理距离最短路径，A*搜索优化 |
| **频次优先** | 双向A*(频次倒数权重) + 流量饱和衰减，优先通行高密度航段 |
| **综合最优** | 安全(35%) + 时间(25%) + 距离(20%) + 频次(20%) 加权评分 |
| **约束放宽** | 物理约束全部阻塞时的兜底路径，标记风险提示 |

## 项目结构

```
Data_Element/
├── main.py                       # 主入口，模块化任务调度（Task1-7）
├── app.py                        # Web API 服务（Flask + Leaflet.js 前端）
├── config.py                     # 全局配置参数
├── data_preprocessor.py          # 数据预处理（Kalman平滑 + IsolationForest）
├── node_extractor.py             # 节点提取（自适应阈值 + Douglas-Peucker）
├── node_cluster.py               # 节点聚类（HDBSCAN + 航向感知特征）
├── topology_builder.py           # 拓扑网络构建（HMM/Viterbi + 多进程）
├── advanced_weight_model.py      # 动态权重建模（7模型对比 + PNA 5-seed集成）
├── navigation_models.py          # 导航ML模型（风险预测 + 可达性概率 + 多任务DNN）
├── ship_navigator.py             # 船舶导航决策系统（改进A* + 物理约束 + 6种路径类型）
├── node_centrality.py            # 节点中心性分析（4种指标 + PCA自动赋权）
├── visualize.py                  # 可视化（12 种图表方法：轨迹/拓扑/网络统计/模型对比/路径对比等）
├── utils.py                      # 工具函数（Haversine距离、方位角、Douglas-Peucker）
├── requirements.txt              # Python 依赖
├── .gitignore                    # Git 忽略规则
├── README.md                     # 本文档
├── 技术报告.md                    # 详细技术报告
├── 初始需求.md                    # 赛题初始需求
│
├── templates/                    # Web 前端
│   └── index.html                #   Leaflet.js 暗色主题地图界面
│
├── tests/                        # 测试（9 个文件：8 个 test_*.py + conftest.py）
│   ├── conftest.py               #   pytest 配置（UTF-8 模式）
│   ├── test_node_centrality.py   #   节点中心性单元测试
│   ├── test_navigation_random.py #   导航随机采样鲁棒性测试（30组OD）
│   ├── test_relaxed_path_fallback.py # 约束放宽路径回退测试
│   ├── test_snap_distance_warning.py # 起终点吸附距离告警测试
│   ├── test_frequent_path_bug.py #   频次优先路径 bug 回归测试
│   ├── test_route_geojson_simplify.py # GeoJSON 路径简化测试
│   ├── test_api_simple.py        #   Flask API 端点测试
│   └── test_app_browser.py       #   Playwright 浏览器自动化测试
│
├── scripts/                      # 分析与出图脚本（19 个文件，独立于流水线）
│   ├── algorithm_benchmark.py    #   Dijkstra vs A* vs 改进A* 基准对比
│   ├── benchmark_efficiency.py   #   运行效率基准（时间+内存）
│   ├── benchmark_robustness.py   #   异常输入鲁棒性测试
│   ├── gen_eda_figures.py        #   EDA 探索性分析图
│   ├── gen_eda_quality_report.py #   EDA 数据质量报告
│   ├── gen_p0p1_figures.py       #   参赛报告 P0+P1 图
│   ├── gen_connectivity_analysis.py # 网络连通性 + 可扩展性分析
│   ├── gen_data_potential_r2.py  #   数据潜能 R² 上限诊断
│   ├── gen_simulation_experiment.py # 仿真实验（燃油消耗对比）
│   ├── gen_diff_routes.py        #   差异化路径对比图
│   ├── gen_data_fusion_diagram.py #  数据融合示意图
│   ├── gen_validity_report.py    #   有效性报告
│   ├── generate_extra_figures.py #   补充图表
│   ├── generate_flowcharts.py    #   技术路线 + 改进A* 流程图
│   ├── extract_ref_figures.py    #   参考文献图表提取
│   ├── compare_empirical_vs_pna.py # 经验模型 vs PNA 对比
│   ├── analyze_tonnage_by_type.py # 载重吨按船型回归分析
│   ├── convert_to_docx.py        #   pandoc 文档转换辅助
│   └── fetch_sol_zc_batch.py     #   Playwright 批量采集航运在线船舶数据
│
├── docs/                         # 技术文档（5 个文件，另有 3 个论文参考文档被 .gitignore 忽略）
│   ├── 6段vs2段对比分析.md         #   时段划分决策记录
│   ├── 吨位估算改进方案.md         #   DWT 公式推导过程
│   ├── 船讯网数据采集技术报告.md    #   shipxy API 数据采集方案
│   ├── 2026-06-17-ship-selector-fix.md # 船舶选择器 Bug 修复记录
│   └── references.md             #   参考文献清单
│
├── Data/                         # 原始AIS轨迹数据（.gitignore 排除）
│   └── *.xlsx (2个)
│
└── output/                       # 输出结果
    ├── topology_nodes.csv / topology_edges.csv  # 拓扑网络（416节点/506边）
    ├── waterway_topology.json                   # 完整拓扑结构
    ├── ship_characteristics_db.csv              # 309艘船舶特征数据库
    ├── feature_importance.csv                   # 特征重要性排名
    ├── model_metadata.json                      # 模型元数据（7模型对比）
    ├── model_report.txt                         # 模型评估报告
    ├── summary_report.txt                       # 运行汇总报告
    ├── algorithm_benchmark.json                 # 算法基准对比数据
    ├── connectivity_analysis.json               # 网络连通性分析
    ├── simulation_experiment.json               # 仿真实验数据
    ├── eda_summary.json                         # EDA 汇总
    ├── wilcoxon/                                # 统计检验数据
    └── img/                                     # 可视化图片（49 张）
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

任务编号：1 数据预处理 / 2 节点提取 / 3 节点聚类 / 4 拓扑构建 / 5 权重建模 / 6 可视化 / 7 导航决策。

> `--task` 只接受整数 1-7。**5.5 中心性分析**是 `run_all` 在 Task5 完成后自动触发的子任务，无法通过 `--task` 单独指定；如需重跑，删除 `output/node_centrality.csv` 后运行 `python main.py`（会从 Task1 开始检查缓存，到 Task5 后自动重跑 5.5）。

### 缓存与增量运行

流水线采用**文件存在即跳过**策略：

- 每个 Task 开始时检查输出文件是否存在，存在则直接加载，不重算
- `--force` 参数强制忽略缓存，重新计算所有步骤
- Task5 优先加载已有 `weight_model_*.pkl`，找不到才训练；若 pkl 损坏（best_model 为 None）自动回退训练

**只重跑某一步**的方法：删除对应的输出文件，再 `python main.py --task N`。例如：

```bash
# 只重跑权重建模
del output\edge_features_dynamic_weights.csv
python main.py --task 5

# 只重跑导航决策
python main.py --task 7
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

启动后访问 http://localhost:5000（host=0.0.0.0，port=5000）即可打开导航页面。

> **前置条件**：`output/` 目录下必须已有 Task1-5 的产出文件，以及 Task7 训练保存的导航 ML 模型 pkl。Web 服务**不会重跑流水线**，只读取已有结果。

**app.py 启动时读取的文件**：

- `output/topology_nodes.csv` `output/topology_edges.csv` —— 拓扑结构
- `output/edge_features_dynamic_weights.csv` —— 24h动态耗时权重
- `output/ship_characteristics_db.csv` —— 309 艘船舶特征数据库
- `output/risk_prediction_model.pkl` —— 风险预测模型（由 `PhysicalConstraintChecker` 初始化时自动加载；pkl 不存在则回退规则计算）
- `output/passability_model.pkl` —— 可达性概率模型（同上）

> 这两个 pkl 是 `main.py` Task7 中 `train_models()` 训练后保存的。`app.py` 直接加载使用，不会重训。`weight_model_*.pkl`（Task5 的耗时权重模型）是另一套模型，`app.py` **不加载**它（耗时权重已写入 `edge_features_dynamic_weights.csv`）。

### 功能特性

- **地图选点**：点击地图选择起点/终点，也支持手动输入 GPS 坐标
- **拓扑节点参考**：地图上显示航道拓扑节点作为参考点
- **船名搜索选择**：船型过滤 + 船名关键词搜索，选中后自动使用真实参数（309 艘 shipxy/CCS 数据）
- **多路径展示**：同时展示最多 4 条差异化路径（`max_paths=4`），颜色区分 + 图例标注
- **路径详情**：每条路径显示距离、耗时、平均航速、风险评分等指标
- **节点列表**：展示路径途经的航道节点序列

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/api/ship_types` | GET | 获取支持的船舶类型列表 |
| `/api/ships` | GET | 获取船舶列表，支持 `?ship_type=油轮&keyword=东运` 过滤搜索 |
| `/api/topology_nodes` | GET | 获取所有拓扑节点坐标 |
| `/api/plan` | POST | 执行路径规划，支持 `ship_name`（真实参数）或 `ship_type`（模板兜底） |
| `/api/trajectory_sample` | GET | 获取采样轨迹数据用于回放动画 |

#### /api/plan 请求示例

**按船名查询真实参数（推荐）：**

```json
{
  "start_lat": 22.9442,
  "start_lon": 113.5442,
  "end_lat": 23.0582,
  "end_lon": 113.4953,
  "ship_name": "东运628"
}
```

**按船型模板兜底（兼容旧接口）：**

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
        "path_name": "频次优先",
        ...
      }
    ]
  }
}
```

> `/api/plan` 请求 `max_paths=4`，返回最多 4 条差异化路径。`path_name` 取值随路径规划结果动态变化，可能为：`安全优先` / `时间最短` / `距离最短` / `频次优先` / `综合最优` / `约束放宽路径`（兜底场景，附带 `warning` 字段）。

## 支持的船舶类型

> 每种模板对应一艘数据库中的真实船舶（2026-06-10 更新），按船型分位数选取代表性船只。"大型"模板取对应船型各字段 max 值（2026-06-18 修正）。数据来源: shipxy 309 艘真实船舶特征数据库。

| 类型 | 代表船舶 | 船长(m) | 船宽(m) | 吃水(m) | 限高(m) | 吨位 | 最大航速(节) |
|------|---------|:---:|:---:|:---:|:---:|:---:|:---:|
| 小型货船 | 粤诚辉168 | 53 | 11 | 2.6 | 15 | 3,000 | 8 |
| 中型货船 | 锦江2003 | 63 | 13 | 3.2 | 15 | 994 | 8 |
| 大型货船 | 顺利2338 | 77 | 16 | 3.7 | 15 | 3,000 | 8 |
| 集装箱船 | 泰航5088 | 49 | 13 | 2.4 | 15 | 3,000 | 8 |
| 大型集装箱船 | max(5艘) | 70 | 18 | 3.8 | 15 | 3,000 | 9.2 |
| 油轮 | 运达油13 | 63 | 13 | 3.31 | 15 | 1,187 | 8 |
| 大型油轮 | max(7艘)+CCS | 96 | 16 | 5.894 | 15 | 3,572 | 11 |
| 客船 | 广发证券号 | 44 | 9 | 1.85 | 15 | 488 | 8 |
| 渔船 | 粤穗渔11132 | 17 | 4 | 1.5 | 8 | 200 | 6 |
| 拖船 | 穗救拖16 | 31 | 10 | 2.2 | 10 | 300 | 8 |

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
    'eps': 150.0,              # DBSCAN 邻域半径(m)
    'min_samples': 3,          # 最小样本数
}

TOPOLOGY_CONFIG = {
    'edge_connection_distance': 200.0,  # 边连接距离(m)
}
```

## 依赖说明

| 库 | 用途 | 必需 |
|----|------|:--:|
| numpy, pandas, scipy | 数据处理与数值计算 | ✅ |
| openpyxl | Excel 数据读取 | ✅ |
| networkx | 图结构存储与路径算法 | ✅ |
| shapely | 几何计算（水域/跨陆地检测） | ✅ |
| scikit-learn | IsolationForest / RF / 特征工程 | ✅ |
| hdbscan | HDBSCAN 层次密度聚类（Task3） | ✅ |
| matplotlib | 可视化图表 | ✅ |
| flask | Web API 服务 | ✅ |
| pytest | 测试框架 | ✅ |
| xgboost | 梯度提升树（Task5 模型对比） | 可选 |
| lightgbm | 轻量级梯度提升（Task5 模型对比） | 可选 |
| ngboost | 自然梯度提升（Task5 模型对比） | 可选 |
| optuna | 超参数优化（Task5） | 可选 |
| torch | 深度学习框架（GNN/PNA） | 可选 |
| torch-geometric | 图神经网络（GAT/PNA） | 可选 |

## 最近更新

### 2026-06-16：V4前端渲染修复与路径增强

- **图例颜色与路径选项框颜色不匹配修复**：将颜色绑定从索引绑定改为语义绑定（PATH_COLOR_MAP），确保图例、路径选项框、路径折线、船舶动画颜色完全一致
- **路径折线折点平滑修复**：新增 Pass 8 锚点拐角平滑v3，迭代删除大于140度转折（最多5轮），残留锚点尖角做中点切角
- **新增距离最短路径类型**：PathType.SHORTEST 实现物理距离最短路径规划，使用 _dijkstra_shortest_distance 算法
- **路线跨陆地检测**：新增基于水域面 GeoJSON 的跨陆地检测功能，自动标记穿越陆地的路径
- **验证通过**：22 个测试通过，1 个预存失败（非回归）

### 2026-06-14：核心可视化图视觉修复

- **`pna_scatter.png` 重生成**：在 `scripts/gen_p0p1_figures.py` 中新增 `gen_pna_scatter()`，**严格复用**项目已有 `_load_predictions_from_csv()`（从 `edge_features_dynamic_weights.csv` 读取 `avg_travel_time` 与 `predicted_time_h*` 均值）+ `load_metadata()`（从 `model_metadata.json` 读取 R²/MAE/RMSE/MAPE 官方指标）。R² 修正为 **0.8879**（集成 PNA），新增 ±10% 误差带、KDE 密度等高线、IQR 长尾菱形标记
- **`path_comparison.png` 视觉升级**：改进 `visualize.py` 中 `plot_path_comparison()` 的**样式**层（海图底色、经纬度格式、指北针、比例尺、起终点配色、方向箭头、4 路径阴影描边）。**选路逻辑完全沿用** `scripts/generate_extra_figures.py` 第 113-127 行 4 组分位节点 (start/q1/q2/q3/end) 方式，未改动。4 条路径（频次优先 / 安全优先 / 时间最短 / 综合最优）在空间上真正可分
- **论文同步更新**：`paper_rewriting_output/final_paper/参赛报告.docx` 用最新图片重生成

### 2026-06-10：真实船舶参数替换模板预设值

- **SHIP_TEMPLATES 基于真实数据更新**：模板值从偏大 2-3 倍的猜测值更新为 shipxy/CCS 309 艘真实数据按船型中位数聚合值（如中型货船 length 150→63, draft 7.5→3.2）
- **CCS 数据合并**：14 艘 CCS 真实数据（含真实 tonnage/draft/depth）合并到 `ship_characteristics_db.csv`
- **新增 `/api/ships` 接口**：返回 309 艘船舶列表，支持按船型过滤 + 关键词搜索
- **`/api/plan` 支持按船名查真实参数**：传 `ship_name` 优先从数据库查真实参数，不传则模板兜底，完全向后兼容
- **前端船名搜索选择器**：船型过滤下拉 + 船名搜索框，选中后显示真实参数（长度/宽度/吃水/数据来源）
- **核心改进**：物理约束校验不再过严，小船不再被误判为"无法通过"

### 2026-06-05：导航报告精简 + 多 OD 测试修复

- **Task 7 不再生成 20 个 `navigation_*.json/txt` 文件**：原 `_task7_navigation` 改为内存缓存（`self.task7_results`），`summary_report` 从内存读取结果，避免 10 船型 × 2 格式 = 20 个冗余文件
- **`tests/test_navigation_random.py` 替代固定模板报告**：随机采样 30 组 (ship_type, OD, hour) 组合验证 `plan_route` 鲁棒性，结果汇总到单一 `navigation_random_sample.json`
- **大型油轮 NO_PATH 修复**：`_dijkstra_safest` 在 `blocked_edges` 存在但主路径返回 None 时，RELAXED 兜底逻辑未执行，已在 `return` 前补一次无约束调用
- **hour=0 与 hour=8 路径结果一致修复**：`edge_features_dynamic_weights.csv` 缺少 `predicted_time_h00..h23` 列导致动态时间回退到均值，改为时段级预测
- **短距离高跳数风险评分修复**：原 `total_risk = Σrisk` 简单累加，改为距离加权平均 `Σ(risk × distance) / Σdistance`
- **清理 20 个冗余 navigation_*.json/txt 和 8 个 `_xxx.py` 诊断脚本**

### 2026-06-03：频次路径可视化 + 物理约束松弛策略

- **Web 前端支持 4 种路径**：前端地图新增"频次优先"路径（粉紫色 `#e84393`）的渲染和图例，4 种路径颜色清晰区分
- **app.py 接入多目标导航器**：后端 `plan_paths` 重写为 `MultiObjectiveNavigator.find_paths()`，输出安全优先/时间最短/综合最优/频次优先 4 种路径
- **物理约束松弛兜底**：当某型船舶因吃水/船宽/限高全部被阻塞时，`find_paths` 自动放宽约束生成 `RELAXED` 次优路径，附带风险提示

### 2026-05-31：频次优先路径 bug 修复

修复了 `find_paths` 中导致"频次优先"路径无法正确生成的 3 个 bug：

1. `PathType('distance')` 抛 ValueError 被静默吞掉 — 'distance'/'time' 路径全部丢失
2. label 循环漏掉 `'frequent'` — `_bidirectional_a_star_frequent` 成为死代码
3. fallback 分支随意贴 FREQUENT 标签

**修复方案：** 用顺序调用重构解耦"路径类型→算法+PathType"映射，'frequent' 走 双向A*(频次倒数) + 流量饱和衰减。

**验证：**
- 4 个新回归测试全部通过（`pytest tests/test_frequent_path_bug.py`）
- 真实拓扑 612→1403 路径总频次 = **178**（修复前为 0）
- 17/17 现有测试零回归

## 许可证

MIT License
