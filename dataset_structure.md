# 数据集结构说明

> 本文档总结系统各阶段产出的 CSV 数据集结构与统计特征，按数据流顺序排列。
> 生成时间：2026-07-04
> 数据状态：基于 PNA 推断边 ratio clip(0.5, 3.0) + 燃油公式用等效速度 + speed_reliability 已修复后的最新版本。

## 数据流总览

```
Task1: cleaned_data.csv (110万 AIS 点)
  ↓
Task2: extracted_nodes.csv (5.3万 候选节点)
  ↓
Task3: clustered_nodes.csv (1244 聚类节点)
  ↓
Task4: topology_nodes.csv (416) + topology_edges.csv (506) + edge_waypoints.csv (1.6万)
  ↓
Task5: edge_features_dynamic_weights.csv (966 边 × 48 列) ← 核心数据
       feature_matrix.csv (829 训练样本) + feature_importance.csv (14 特征)
  ↓
Task5.5: node_centrality.csv (416 节点)
  ↓
Task7: ship_characteristics_db.csv (309 船) → 路径规划
  ↓
仿真: simulation_experiment.json (30 组 OD)
```

---

## Task1 - cleaned_data.csv

**阶段：** 数据预处理
**说明：** 清洗平滑后的 AIS 轨迹数据。

| 指标 | 值 |
|---|---|
| 行数 | 1,105,879 |
| 列数 | 9 |
| 文件大小 | 115 MB |
| 时间范围 | 2026-03-17 ~ 2026-03-24 |
| 经度范围 | 108.27 ~ 120.40°E |
| 纬度范围 | 21.88 ~ 31.97°N |
| 船舶数 | 309 艘 |
| 航速中位 | 4.5 节 |

### 列结构

| 类型 | 列名 |
|---|---|
| 数值列（6） | 序号、航向、航速、纬度、经度、trajectory_segment |
| 分类列（3） | 船舶名称、船舶英文名称、时间 |

### 关键统计

- 船舶名称：309 类，top3：海巡09210（29526 条）、海顺63（13613 条）、扬帆2378（11225 条）
- 时间：283727 个唯一时间戳
- 无缺失值

---

## Task2 - extracted_nodes.csv

**阶段：** 节点提取
**说明：** 从轨迹数据提取的候选节点（拐点、分岔点、汇合点、航点）。

| 指标 | 值 |
|---|---|
| 行数 | 53,581 |
| 列数 | 10 |
| 文件大小 | 5.7 MB |
| frequency 中位 | 1.0 |
| ship_count 中位 | 1.0 |
| 经纬度范围 | 与原始数据一致 |

### 节点类型分布

| 类型 | 数量 | 占比 |
|---|---|---|
| waypoint | 52,537 | 98.0% |
| turn_point | 763 | 1.4% |
| stop_point | 281 | 0.5% |

### 列结构

| 类型 | 列名 |
|---|---|
| 数值列（7） | node_id、lat、lon、frequency、ship_count、heading、heading_concentration |
| 分类列（3） | type、type_distribution、detailed_type |

### detailed_type 分布

- turn_point: 29,431
- waypoint: 23,365
- merge_point: 662

---

## Task3 - clustered_nodes.csv

**阶段：** 节点聚类
**说明：** 高频节点聚类合并后的节点集。

| 指标 | 值 |
|---|---|
| 行数 | 1,244 |
| 列数 | 14 |
| 文件大小 | 187 KB |
| cluster_id 范围 | -1（噪声）~ 359 |
| frequency 中位 | 24.5 |
| ship_count 中位 | 22 |
| node_count 中位 | 1.0（max 597）|
| 经度范围 | 108.33 ~ 117.85°E |
| 纬度范围 | 21.89 ~ 30.50°N |

### final_type 分布

| 类型 | 数量 |
|---|---|
| turn_point | 1,126 |
| waypoint | 90 |
| merge_point | 27 |
| (其他噪声/未分类) | 1 |

### 列结构

| 类型 | 列名 |
|---|---|
| 数值列（9） | node_id、cluster_id、lat、lon、frequency、ship_count、node_count、heading、heading_concentration |
| 分类列（4） | type、type_distribution、detailed_type、final_type |
| 布尔列（1） | is_noise |

---

## Task4 - topology_nodes.csv

**阶段：** 拓扑构建
**说明：** 最终拓扑网络的节点表。

| 指标 | 值 |
|---|---|
| 行数 | 416 |
| 列数 | 6 |
| 文件大小 | 24 KB |
| 经度范围 | 110.91 ~ 114.12°E（珠三角）|
| 纬度范围 | 22.15 ~ 24.26°N |
| frequency 中位 | 109 |
| ship_count 中位 | 57.5 |

### type 分布

| 类型 | 数量 | 占比 |
|---|---|---|
| turn_point | 348 | 83.7% |
| waypoint | 57 | 13.7% |
| merge_point | 11 | 2.6% |

### 列结构

node_id、lat、lon、type、frequency、ship_count

---

## Task4 - topology_edges.csv

**阶段：** 拓扑构建
**说明：** 拓扑网络的边表。

| 指标 | 值 |
|---|---|
| 行数 | 506 |
| 列数 | 8 |
| 文件大小 | 34 KB |
| avg_speed 中位 | 5.28 节（合理）|
| avg_distance 中位 | 1092 m |
| avg_time 中位 | 675 s |
| is_bidirectional | 460 条双向 / 46 条单向 |

### 列结构

| 列名 | 说明 |
|---|---|
| from_node | 起点节点 ID |
| to_node | 终点节点 ID |
| weight | 边权重 |
| ship_count | 经过船舶数 |
| avg_speed | 平均航速（节）|
| avg_distance | 平均距离（米）|
| avg_time | 平均耗时（秒）|
| is_bidirectional | 是否双向 |

---

## Task4 - edge_waypoints.csv

**阶段：** 拓扑构建
**说明：** 每条边的航路点序列（用于路径可视化）。

| 指标 | 值 |
|---|---|
| 行数 | 16,250 |
| 列数 | 5 |
| 文件大小 | 767 KB |
| sequence 中位 | 28 |
| 经度范围 | 110.94 ~ 114.05°E |
| 纬度范围 | 22.27 ~ 24.12°N |

### 列结构

from_node、to_node、sequence、lat、lon

---

## Task5 - edge_features_dynamic_weights.csv（核心数据）

**阶段：** 权重建模
**说明：** 边的动态耗时权重表，含 PNA 推断的 24 小时预测。**核心数据，下游导航与仿真均依赖此文件。**

| 指标 | 值 |
|---|---|
| 行数 | 966 |
| 列数 | 48 |
| 文件大小 | 598 KB |
| model_used 分布 | pna 531（55%）、empirical 435（45%）|
| segment_count 中位 | 0（531 条 pna 推断边无数据）|
| speed_reliability | 中位 0.80，范围 [0, 1]（已修复）|

### 列结构（48 列）

| 类别 | 列名 | 数量 |
|---|---|---|
| 基础 | from_node、to_node、model_used、segment_count | 4 |
| 距离耗时 | avg_distance、avg_travel_time、std_travel_time、min_travel_time、max_travel_time、median_travel_time、theoretical_time | 7 |
| 速度 | avg_actual_speed、std_actual_speed、avg_reported_speed、speed_reliability | 4 |
| 拓扑 | waterway_type、waterway_type_code、node_degree_from、node_degree_to、edge_betweenness | 5 |
| 方向 | avg_bearing、std_bearing、avg_course_change、is_bidirectional | 4 |
| 24小时预测 | predicted_time_h00 ~ predicted_time_h23 | 24 |

### model_used 说明

- **empirical（435 条）：** 有 AIS 实测数据的边，segment_count ≥ 2
- **pna（531 条）：** 无实测数据的长距离边，由 PNA 模型 transductive 推断

### 关键统计

| 指标 | empirical 边 | pna 推断边 |
|---|---|---|
| avg_distance 中位 | 94 m | 3088 m |
| avg_travel_time 中位 | 35.4 s | 2870.9 s |
| avg_actual_speed 中位 | 5.35 节 | 5.0 节 |
| speed_reliability 中位 | 0.22 | 0.80 |

---

## Task5 - feature_matrix.csv

**阶段：** 权重建模
**说明：** GNN 模型训练用的特征矩阵（仅 empirical 边）。

| 指标 | 值 |
|---|---|
| 行数 | 829 |
| 列数 | 33 |
| 文件大小 | 328 KB |
| time_ratio 中位 | 1.03（max 3.63）|
| period 分布 | day 424、night 405 |

### 列结构（33 列）

| 类别 | 列名 | 数量 |
|---|---|---|
| 标识 | from_node、to_node、period | 3 |
| 时段统计 | avg_travel_time、theoretical_time、time_ratio、sample_count | 4 |
| 速度 | avg_reported_speed、std_reported_speed、speed_cv、edge_speed_median、edge_speed_iqr | 5 |
| 方向 | bearing、bearing_sin、bearing_cos、avg_course_change、std_course_change、course_change_x_narrow | 6 |
| 拓扑 | waterway_type、node_degree_from、node_degree_to、edge_betweenness | 4 |
| 邻居 | neighbor_count、neighbor_speed_median、log_sample_count | 3 |
| 时段编码 | period_morning、period_midday、period_afternoon、period_night、distance | 5 |
| 时间编码 | hour_sin、hour_cos、speed_decay | 3 |

---

## Task5 - feature_importance.csv

**阶段：** 权重建模
**说明：** 模型特征重要性排名（Permutation Importance）。

| 指标 | 值 |
|---|---|
| 行数 | 14 |
| 列数 | 2 |
| 文件大小 | 0.4 KB |

### 列结构

feature、importance

### 重要性排名

| 排名 | 特征 | 重要性 |
|---|---|---|
| 1 | speed_decay | 73.09% |
| 2 | avg_course_change | 10.35% |
| 3 | speed_iqr | 9.33% |
| 4 | edge_betweenness | 4.52% |
| 5 ~ 14 | 其他 10 个特征 | 共 2.71%（其中 8 个为 0）|

**说明：** 特征贡献集中度高，speed_decay 单特征占 73%。8 个特征重要性为 0（distance、theoretical_time、bearing_cos、waterway_type、node_degree_from、node_degree_to、hour_sin、hour_cos）。

---

## Task5.5 - node_centrality.csv

**阶段：** 中心性分析
**说明：** 节点的综合中心性指标（基于 PCA 加权）。

| 指标 | 值 |
|---|---|
| 行数 | 416（与 topology_nodes 一致）|
| 列数 | 9 |
| 文件大小 | 229 KB |
| composite 中位 | 0.0735 |
| betweenness 中位 | 0.0424 |
| betweenness max | 0.6220 |

### PCA 自动权重

| 中心性指标 | 权重 |
|---|---|
| betweenness | 94.66% |
| eigenvector | 3.91% |
| closeness | 1.20% |
| degree | 0.23% |

**说明：** 介数中心性主导，符合航运网络关键节点识别逻辑。

### 列结构

node_id、degree、betweenness、closeness、eigenvector、composite、lat、lon、frequency

---

## Task7 - ship_characteristics_db.csv

**阶段：** 导航
**说明：** 船型特征库，用于约束检查与路径规划。

| 指标 | 值 |
|---|---|
| 行数 | 309 |
| 列数 | 23 |
| 文件大小 | 42 KB |
| max_speed 中位 | 8.0 节 |
| length 中位 | 61 m |
| tonnage 中位 | 3000 吨 |
| draft 缺失 | 104 条（33.7%）|

### 字段缺失情况

| 字段 | 缺失数 | 缺失率 |
|---|---|---|
| draft | 104 | 33.7% |
| net_tonnage、depth、shipyard、build_year、me_power_kw、class_notation | 295 | 95.5% |
| imo | 11 | 3.6% |
| deadweight | 4 | 1.3% |

### ship_type 分布

| 类型 | 数量 |
|---|---|
| 货船 | 267 |
| 执法船 | 9 |
| 客船 | 8 |
| 其他 | 25 |

### inferred_type 分布

| 类型 | 数量 |
|---|---|
| 中型货船 | 95 |
| 小型货船 | 88 |
| 渔船 | 82 |
| 其他 | 44 |

### 列结构（23 列）

| 类别 | 列名 | 数量 |
|---|---|---|
| 标识 | ship_name、ship_type、mmsi、imo、data_source | 5 |
| 速度 | max_speed、avg_speed | 2 |
| 尺寸 | length、width、draft、height、tonnage | 5 |
| 吨位 | net_tonnage、deadweight、dwt_source | 3 |
| 建造 | build_year、shipyard、class_notation、ccs_status | 4 |
| 动力 | me_power_kw | 1 |
| 分类 | inferred_type | 1 |
| 统计 | record_count | 1 |

---

## 仿真实验 - simulation_experiment.json

**阶段：** P1-3 模拟航行对比实验
**说明：** 30 组 OD 对的 Dijkstra 基线 vs PNA 动态权重改进 A* 对比。

| 指标 | 值 |
|---|---|
| OD 对数 | 30 |
| 路径不同数 | 24/30（80%）|
| 燃油节约均值 | 6.24% |
| 燃油节约中位 | 2.16% |
| 燃油节约范围 | [0.00%, 47.34%] |
| 时间节约均值 | 6.48% |
| smart 路径等效速度中位 | 1.82 节 |

### 燃油模型参数

- IMO GHG Study 2020：`fuel_rate = a * v^3 + b`（a=0.012, b=5.0 吨/天）
- 燃油计算用等效速度（distance/time 反推），保证物理一致性
- VLSFO 价格：1050 USD/吨
- USD/CNY：7.25

---

## 数据集健康状态总结

| CSV | 状态 | 说明 |
|---|---|---|
| cleaned_data.csv | ✅ 正常 | 无缺失，范围合理 |
| extracted_nodes.csv | ✅ 正常 | 类型分布合理 |
| clustered_nodes.csv | ✅ 正常 | 聚类后节点数减少符合预期 |
| topology_nodes.csv | ⚠️ 范围偏小 | 仅覆盖珠三角，与原始数据范围不一致 |
| topology_edges.csv | ⚠️ avg_time max 偏高 | 28.6h，存在停留点污染 |
| edge_waypoints.csv | ✅ 正常 | - |
| edge_features_dynamic_weights.csv | ✅ 已修复 | speed_reliability 已修，clip 0.5-3.0 |
| feature_matrix.csv | ✅ 正常 | time_ratio 中位 1.03 合理 |
| feature_importance.csv | ⚠️ 过拟合 | speed_decay 占 73%，8 特征为 0 |
| node_centrality.csv | ✅ 已修复 | 416 节点，与拓扑一致 |
| ship_characteristics_db.csv | ⚠️ 字段缺失 | draft 缺 33.7%，6 字段缺 95.5% |

---

## 已知问题与修复记录

| 问题 | 修复状态 | 修复方式 |
|---|---|---|
| speed_reliability 全 0 | ✅ 已修复 | advanced_weight_model.py:2715 改用 `1 - std/mean` 计算 |
| node_centrality 节点数不匹配 | ✅ 已修复 | 重跑 Task5.5，从 1589 → 416 |
| PNA ratio clip 过宽 | ✅ 已修复 | clip(0.1, 20.0) → clip(0.5, 3.0) |
| 燃油公式速度-时间不一致 | ✅ 已修复 | gen_simulation_experiment.py 改用等效速度 |
| PNA 在长距离边 OOD 偏高 | ⚠️ 未修 | 训练数据全是短距离，OOD 区域 ratio 偏高 2.8 倍 |
| feature_importance 过拟合 | ⚠️ 未修 | speed_decay 占 73%，需在论文中说明 |
| ship_characteristics draft 缺失 | ⚠️ 未修 | 33.7% 船舶跳过吃水约束 |
