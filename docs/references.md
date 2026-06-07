# 参考文献清单 (References)

> 本文档汇总项目"边权预测 / GATv2Conv 精度提升"主题下需要引用的所有参考文献，包括原始英文论文与中文 CNKI 检索结果。
> 后续撰写报告、学位论文、期刊论文时直接复用本清单即可。

最后更新：2026-06-02

---

## A. 英文原始论文（必引 · Foundational）

### [1] Veličković P, Cucurull G, Casanova A, et al.
**Graph Attention Networks.**
// ICLR 2018 // 2017-10-12 (arXiv preprint)
- arXiv: 1710.10903
- 用途：项目基础 GAT 原理引用，Plan Task 5 报告"引言/背景"段
- BibTeX 关键字段：`author={Veličković, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Liò, Pietro and Bengio, Yoshua}, title={Graph Attention Networks}, booktitle={ICLR 2018}, year={2018}`

### [2] Brody S, Alon U, Yahav E.
**How Attentive are Graph Attention Networks?**
// ICLR 2022 // 2021-12-02 (arXiv v3)
- arXiv: 2105.14491
- 用途：项目核心 GATv2Conv 原理引用，Plan Task 5 报告"为什么选 GATv2"段
- 关键论点：证明 GAT 注意力是"静态"的（排序与查询节点无关），提出 GATv2 通过调整操作顺序实现"动态"注意力
- BibTeX 关键字段：`author={Brody, Shaked and Alon, Uri and Yahav, Eran}, title={How Attentive are Graph Attention Networks?}, booktitle={ICLR 2022}, year={2022}`

### [3] Feng W, Zhang J, Dong Y, et al.
**Graph Random Neural Network for Semi-Supervised Learning on Graphs.**
// NeurIPS 2020 // 2020-05-22 (arXiv)
- arXiv: 2005.13579
- 用途：DropEdge 原始论文，Plan Task 5 报告"方法 4 DropEdge"段
- 关键论点：随机删边作为数据增强，提升 GNN 鲁棒性
- BibTeX 关键字段：`author={Feng, Wenzheng and Zhang, Jie and Dong, Yuxiao and Han, Yin and Luan, Huanbo and Xu, Qian and Yang, Qiang and Kharlamov, Evgeny and Tang, Jie}, title={Graph Random Neural Network for Semi-Supervised Learning on Graphs}, booktitle={NeurIPS 2020}, year={2020}`

---

## B. 中文文献（CNKI 检索结果 · 按 Related Work 章节组织）

### B1. 图注意力机制基础（背景支撑）

#### [4] 罗小元, 耿艺帆, 吴莉艳, 王新宇.
**基于GATv2模型的虚假数据注入攻击检测方法.**
电气工程学报, 2024, 19(3): 354-365.
- 主题：GATv2 在电力系统 FDIA 检测中的应用
- 评估：IEEE 14/118 节点系统上检测准确率 >98%
- 用途：Related Work "图注意力机制应用现状"段
- 检索来源：CNKI 主题 = "GATv2"，命中 91 篇

### B2. GATv2 架构改进（对应方法 2：调宽+调深架构）

#### [5] 曹炳尧, 姜莲卿.
**基于GATv2与DQN增强的图特征融合代码克隆检测技术.**
计算机科学, 2026-05-15.
- 主题：分层残差 GATv2 + Transformer 特征融合，用于代码克隆检测
- 关键方法：分层残差 GATv2 网络强化 PDG 结构特征
- 用途：Related Work "GATv2 后续改进"段；为 Plan Task 2（3 层+残差）提供工业证据
- 检索来源：CNKI 主题 = "GATv2"

#### [6] 邓淼磊, 周鑫, 樊少珺, 孙川川, 李远博.
**MAGAT-IDS：一种少数类感知的图注意力入侵检测模型.**
计算机应用研究, 2026-03-24.
- 主题：GATv2 + 少数类敏感编码 + 边特征非线性残差映射
- 评估：NF-BoT-IoT/NF-ToN-IoT/NF-UNSW-NB15-v2 上 F1 99.88%-99.99%
- 用途：Related Work "GATv2 + 边特征融合"段
- 注意：仅借鉴"边特征非线性残差"写法，不借鉴"少数类感知"（你无此问题）
- 检索来源：CNKI 主题 = "GATv2"

#### [7] 袁军, 周丽君, 张俊然.
**一种高效的对称注意力图神经网络模型.**
工程科学与技术, 2025-11-25.
- 主题：对称注意力 + ReLU 替代 softmax，用于节点分类与链路预测
- 关键贡献：在 13 个基准上与 GAT、GATv2、GTN 直接对比
- 用途：Related Work "GATv2 替代方案"段；可作为精度对比基线引用
- 检索来源：CNKI 主题 = "GATv2"

#### [8] 张明明, 王乐萱, 王小元, 陈聪, 袁峰, 李晓晖, 丁静.
**基于GATv2的三维成矿预测方法研究.**
地学前缘, 2026-04-08.
- 主题：GATv2 对体素图节点进行动态注意力建模
- 用途：Related Work "GATv2 在空间离散图上的应用"段（凑字数）
- 检索来源：CNKI 主题 = "GATv2"

### B3. 数据增强 + 模型集成（对应方法 3、4：5-seed 集成 + DropEdge）

#### [9] 杨炳新, 郭艳蓉, 郝世杰, 洪日昌.
**基于数据增广和模型集成策略的图神经网络在抑郁症识别上的应用.**
计算机科学, 2022-07-15.
- 主题：EEG 脑电 → 皮尔逊相关构图 → GNN → 多数投票集成
- 评估：MODMA 数据集分类准确率 77%
- 用途：Related Work "GNN 数据增广+模型集成"段
- 检索来源：CNKI 主题 = "图神经网络 数据增广 模型集成"

#### [10] 耿梦娇.
**面向图神经网络的图数据增强方法研究[硕士论文].**
南京: 东南大学, 2024-06-13.
- 主题：GNN 数据增强系统方法学（边 drop / 节点 drop / 特征 mask / 图混合）
- 用途：Related Work "GNN 数据增强方法"段；为 Plan Task 4（DropEdge p=0.1）提供参数选择依据
- 检索来源：CNKI 主题 = "图神经网络 数据增强"

### B4. 链路/边权预测（应用场景同源）

#### [11] 梅鹏.
**基于图表示学习的动态网络链路预测方法研究[硕士论文].**
包头: 内蒙古科技大学, 2025-06-05.
- 主题：GCN + RNN + 多头注意力的离散/连续时间动态链路预测
- 用途：Related Work "图学习在链路预测中的应用"段
- 注意：你的图是静态的，仅引用其问题背景，不引用其动态建模方法
- 检索来源：CNKI 主题 = "图表示学习 链路预测"

---

## C. 推荐补充引用（Optional · 写作时视情况加）

### [12] Hamilton W, Ying Z, Leskovec J.
**Inductive Representation Learning on Large Graphs.**
// NeurIPS 2017 // 2017-06-07
- arXiv: 1706.02216
- 用途：邻居采样与消息传递的经典方法（GraphSAGE），可作为 Plan 报告"方法背景"补充

### [13] Xu K, Hu W, Leskovec J, et al.
**How Powerful are Graph Neural Networks?**
// ICLR 2019 // 2018-10-01
- arXiv: 1810.00826
- 用途：GIN 论文，GNN 表达能力分析（与 GAT/PNA 对比时可引）

---

## D. 推荐 **不**引用的文献（已筛掉）

| 论文 | 排除理由 |
|---|---|
| 罗小元 2024 (FDIA) | 若 B1[4] 已引则不重复 |
| 张明明 2026 (成矿) | 任务差异大，价值低于 B2[5-7] |
| 邓淼磊 2026 (少数类) | 你的问题无数据不平衡 |
| 梅鹏 2025 (动态链路) | 仅引一次即可，不重复 |

---

## E. GB/T 7714-2015 标准引用格式（中文文献模板）

如需按国内标准格式整理报告参考文献，可参考以下模板：

```
[4] 罗小元, 耿艺帆, 吴莉艳, 王新宇. 基于GATv2模型的虚假数据注入攻击检测方法[J]. 电气工程学报, 2024, 19(3): 354-365.
[5] 曹炳尧, 姜莲卿. 基于GATv2与DQN增强的图特征融合代码克隆检测技术[J]. 计算机科学, 2026-05-15. (预出版/网络首发)
[6] 邓淼磊, 周鑫, 樊少珺, 等. MAGAT-IDS: 一种少数类感知的图注意力入侵检测模型[J]. 计算机应用研究, 2026-03-24. (预出版/网络首发)
[7] 袁军, 周丽君, 张俊然. 一种高效的对称注意力图神经网络模型[J]. 工程科学与技术, 2025-11-25. (预出版/网络首发)
[8] 张明明, 王乐萱, 王小元, 等. 基于GATv2的三维成矿预测方法研究[J]. 地学前缘, 2026-04-08. (预出版/网络首发)
[9] 杨炳新, 郭艳蓉, 郝世杰, 等. 基于数据增广和模型集成策略的图神经网络在抑郁症识别上的应用[J]. 计算机科学, 2022-07-15.
[10] 耿梦娇. 面向图神经网络的图数据增强方法研究[D]. 南京: 东南大学, 2024.
[11] 梅鹏. 基于图表示学习的动态网络链路预测方法研究[D]. 包头: 内蒙古科技大学, 2025.
```

注：[5][6][7][8] 为网络首发文献，正式卷期号待确认。

---

## F. 使用建议

- **Plan Task 5 报告 Related Work 段**：直接引 [1][2][3][5][6][9][10]，共 7 篇
- **学位论文 References 段**：引 A 部分全部 + B 部分全部 = 11 篇
- **期刊投稿（中文核心）**：引 [1][2] + B 部分 [4][5][6][7][9][10] = 8 篇
- **会议/技术报告**：引 [1][2] + [5] + [9] + [10] = 4 篇（精简版）

---

## G. BibTeX 批量导出（待生成）

如需 BibTeX 格式文件 `docs/references.bib`，可调用以下命令自动生成（需要 GB/T 7714 → BibTeX 转换工具，或人工核对）：

```bash
# 暂未提供，待后续生成
```

---

**总计**：A 3 篇 + B 8 篇 + C 2 篇 = **13 篇可引用文献**，覆盖项目所有方法的引用需求。
