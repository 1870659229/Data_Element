# 魔搭 ModelScope PAI-DSW AMD GPU 训练 GATv2Conv 教程 (v2: Optuna + 5-seed 集成)

> **适用场景**：本地 Windows + 无独立显卡 / 想要免费白嫖 AMD MI 加速 GNN 调参
> **目标**：在 [PAI-DSW](https://www.modelscope.cn) 的 AMD GPU 镜像上**用 Optuna 贝叶斯调参 + 5-seed 集成** 训练 `advanced_weight_model.py` 中的 GATv2Conv, 把训练好的模型下载回本地项目使用
> **预计耗时**：首次 30 分钟 (含环境启动), 后续每次调参 40-60 分钟
> **更新**: 2026-06-03 重构为「Optuna 贝叶斯搜索 + 5-seed 集成」方案 (与方法论统一, 报告好解释), 推荐使用新版

---

## 0. 背景：我们为什么在 AMD 上跑 GATv2Conv？

我们的项目 [advanced_weight_model.py](../../advanced_weight_model.py) 中已经横向对比了 7 个模型 (gnn/xgboost/pna/random_forest/lightgbm/ngboost/lightgbm_tweedie), GATv2Conv 以 **R²=0.8005 / MAE=6.27s** 拿下第一.

但 1.4% 的 R² 领先优势**不够稳** —— xgboost 是亚军 (0.7893), PNA 是季军 (0.7839). 要让 GATv2Conv "稳赢", 需要做:

1. **超参贝叶斯搜索** (Optuna, hidden_dim/num_layers/dropout/lr) —— GPU 比 CPU 快 10-50 倍
2. **多 seed 集成** (5 个不同随机种子训练后取平均预测) —— GPU 必备
3. **加入 edge_attr** (边特征进 attention) —— 代码改动 + 重新训练
4. **更深的网络** (4-5 层 GAT + skip connection) —— 慢

**AMD MI250X 的 192G 显存可以同时把 batch_size 拉到 512+**, 调参效率远超本地 CPU.

> **附录：为什么没选 GAT-LSTM?** 我们做过 A/B 实验 (4 时段粒度下 GAT-LSTM R²=0.9008 vs Baseline R²=0.9391), LSTM 在 3 步短序列上没增益, 反而引入过拟合. 所以边权预测任务**保持 GATv2Conv 不动**.

---

## ✨ v2 更新 (2026-06-03): Optuna 贝叶斯搜索 + 5-seed 集成

之前的方案是用手写 Grid Search + 5-seed 集成 (见 [历史归档](#附录-c-v1-历史方案归档)), 新版改用 Optuna 贝叶斯搜索, 优势:

| 对比项 | v1 Grid Search | **v2 Optuna (当前推荐)** |
|--------|---------------|--------------------------|
| 搜索空间 | 离散笛卡尔积 (54 组) | TPE 贝叶斯 (8 trial) |
| 耗时 | ~45 分钟 | ~20 分钟 (trial 数更少, 但每 trial 用 5-fold CV 评估) |
| R² 提升预期 | +1~3% | **+2~4%** (贝叶斯在低预算下效率更高) |
| 方法论一致性 | 与 XGBoost 调参不同 | **与 XGBoost/LightGBM 调参统一** ✅ |
| 报告可解释性 | 一般 | 强 (Optuna 搜索历史可直接画图) |

**v2 全部代码已打包到 [cloud_gnn_package/](../../cloud_gnn_package/), 一键上传到魔搭即可, 不需要任何源码改动.**

---

## 1. 准备工作

### 1.1 注册魔搭账号
1. 打开 https://www.modelscope.cn 注册/登录 (支持支付宝/微信扫码)
2. 完成实名认证 (**必须**, 否则无法创建 Notebook)

### 1.2 本地准备: cloud_gnn_package 目录
**v2 推荐使用 `cloud_gnn_package/` 一键包** (替代 v1 的逐文件上传):

```
d:\py_project\Data_Element\cloud_gnn_package\
├── README.md                       # 详细运行说明
├── standalone_train.py             # Optuna + 5-seed 集成主训练脚本
├── advanced_weight_model.py        # 原项目主模块 (只读, 不修改)
├── config.py                       # 配置
├── utils.py                        # 通用工具
├── topology_builder.py             # 图构建工具
├── upload_these.txt                # 待上传文件清单
└── output_subset/                  # task 1-4 产物子集
    ├── cleaned_data.csv            # task 1 (118 MB)
    ├── waterway_topology.json      # task 4 (8.6 MB)
    ├── topology_nodes.csv          # task 4 (备用)
    └── topology_edges.csv          # task 4 (备用)
```

> **数据来源**: `output_subset/` 里就是项目 `output/` 目录的 task 1 (cleaned_data.csv) + task 4 (waterway_topology.json + topology_nodes/edges.csv) 产物. 你只需要在本地确认 `output/` 里有这 4 个文件, 上传时整目录打包即可.

---

## 2. 启动 AMD GPU 镜像

### 2.1 进入 PAI-DSW
1. 登录后访问 https://www.modelscope.cn/my/myNotebook
2. 点击左侧导航 **我的 Notebook**
3. 点击右上角 **创建 Notebook**

### 2.2 选择 AMD 方式三
1. 在「环境」下拉中选 **方式三 AMD GPU 环境**
2. 镜像默认就是 `ubuntu22.04-rocm7.2.1-py312-torch2.9.1-1.36.3`，**不用改**
3. 资源选最大（24h 持续使用 + 100h 总额度）
4. 点击 **启动**

> 启动需要 2-5 分钟，等待状态变为「运行中」。

---

## 3. 上传代码和数据 (v2: 整目录一键上传)

### 3.1 方法 A: 整目录上传 (v2 推荐)
1. 进入 Notebook 后默认是 JupyterLab 界面
2. 左侧文件浏览器, 进入 `/mnt/data/` (推荐, 避开系统目录)
3. 创建子目录 `cloud_gnn/`
4. **本地打包**: 在 PowerShell 跑
   ```powershell
   cd d:\py_project\Data_Element
   Compress-Archive -Path cloud_gnn_package\* -DestinationPath cloud_gnn_package.zip -Force
   ```
5. 在 JupyterLab 左侧右键 `/mnt/data/cloud_gnn/` → **Upload Files**, 上传 `cloud_gnn_package.zip`
6. **云端解压**:
   ```bash
   cd /mnt/data/cloud_gnn
   unzip ../cloud_gnn_package.zip
   ls -lh
   ```
   应看到 `standalone_train.py`, `advanced_weight_model.py`, `output_subset/` 等.

### 3.2 方法 B: 直接拖拽文件夹
JupyterLab 支持直接拖拽文件夹到文件浏览器:
1. 把 `d:\py_project\Data_Element\cloud_gnn_package\` 整个目录拖到 JupyterLab 左侧文件树
2. 等待上传完成 (大文件约需 5-10 分钟)

### 3.3 验证数据
在云端 Terminal 跑:
```bash
ls -lh /mnt/data/cloud_gnn/
ls -lh /mnt/data/cloud_gnn/output_subset/
head -2 /mnt/data/cloud_gnn/output_subset/cleaned_data.csv
```
应看到 4 个数据文件大小与本地一致.

---

## 4. 安装依赖 (v2 精简)

```bash
cd /mnt/data/cloud_gnn
pip install optuna          # AMD 镜像可能没预装
pip install torch_geometric # AMD ROCm 镜像通常已预装, 不行再装
```

---

## 5. 启动 v2 训练 (Optuna + 5-seed 集成)

### 5.1 验证 GPU 可用
```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```
应输出 `Device: AMD Instinct MI210` 或 `AMD Instinct MI250X`.

### 5.2 后台启动训练
```bash
cd /mnt/data/cloud_gnn
nohup python standalone_train.py > train.log 2>&1 &
echo "PID=$!"
```

### 5.3 实时查看日志
```bash
tail -f train.log
```

### 5.4 训练流程 (约 40-60 分钟)
`standalone_train.py` 内部会自动跑完 4 个阶段:

| 阶段 | 动作 | 预计耗时 |
|------|------|----------|
| [1/4] | 加载 `cleaned_data.csv` + `waterway_topology.json` | 1-2 分钟 |
| [2/4] | 特征构建 (`build_weights_with_comparison`, 跳过所有模型) | 2-3 分钟 |
| [3/4] | **Optuna 贝叶斯搜索**: 8 trial × 5-fold CV × 80 epoch | 15-25 分钟 |
| [4/4] | **5-seed 集成训练**: 5 seed × 300 epoch | 20-30 分钟 |

> **源码改动**: 0 行. `standalone_train.py` 通过 `inspect.getsource()` + `re.sub()` + `exec()` 动态替换 `_train_gnn` 函数体中的硬编码超参, 原 `advanced_weight_model.py` 保持原样.

---

## 6. 下载训练结果 (v2)

### 6.1 训练完成的产物
```
/mnt/data/cloud_gnn/
├── train.log                          # 完整日志
├── output_subset/
│   ├── weight_model_gnn_optuna.pkl           # ⭐ 主模型 (兼容 main.py 加载)
│   ├── weight_model_gnn_optuna_gnn_state.pt  # ⭐ GNN 权重
│   ├── best_gnn_hparams.json                 # 最佳超参 (报告里要引用)
│   ├── optuna_study.csv                      # Optuna 搜索历史 (画图用)
│   ├── gnn_ensemble_r2.json                  # 集成 R² / MAE / 各 seed 结果
│   └── gnn_ensemble_y_pred.npy               # 测试集预测 (备用)
```

### 6.2 打包下载
```bash
cd /mnt/data/cloud_gnn
zip -r cloud_gnn_results.zip \
    output_subset/weight_model_gnn_optuna.pkl \
    output_subset/weight_model_gnn_optuna_gnn_state.pt \
    output_subset/best_gnn_hparams.json \
    output_subset/optuna_study.csv \
    output_subset/gnn_ensemble_r2.json \
    train.log
```
在 JupyterLab 左侧文件浏览器右键 `cloud_gnn_results.zip` → **Download**.

### 6.3 集成到本地项目
把 zip 下载到 `d:\py_project\Data_Element\output\` 目录下解压:
```powershell
# 假设下载到 d:\Downloads\
Copy-Item d:\Downloads\cloud_gnn_results.zip d:\py_project\Data_Element\output\ -Force
Expand-Archive d:\py_project\Data_Element\output\cloud_gnn_results.zip -DestinationPath d:\py_project\Data_Element\output\ -Force
```

### 6.4 本地 main.py 自动加载
`main.py` 启动时会自动扫描 `output/weight_model_gnn_*.pkl`, 按 R² 排序选最优. 云端下载的 `weight_model_gnn_optuna.pkl` 文件名包含 `gnn_`, 会被自动识别.

---

## 9. 常见问题 FAQ

### Q1. 启动 AMD 镜像报 "资源不足"
**A**: AMD 实例库存有限, 每天 9:00-10:00 释放资源, 建议错峰启动. 实在抢不到先用方式二 NVIDIA 顶一下.

### Q2. `torch.cuda.is_available()` 返回 False
**A**:
- 确认你选的是「方式三 AMD」而不是「方式一 CPU」
- 在 Terminal 跑 `rocminfo`, 应能看到 GPU 设备列表
- 如果都没有, 重启 Notebook

### Q3. OOM (Out of Memory)
**A**:
- 降低 `batch_size` (如 32)
- 降低 `hidden_dim` (如 32)
- AMD 有 192G 显存, 正常参数不会 OOM

### Q4. Optuna 搜索太慢 / 想减少 trial 数
**A**:
- 修改 `standalone_train.py` 末尾 `optuna_search(builder, n_trials=8, ...)` → `n_trials=4` 或更少
- trial 数越少, 总耗时越短, 但搜索质量也越低

### Q5. 报 "未找到硬编码的 best_hp 行"
**A**: `advanced_weight_model.py` 源码被改动, 跟 `inspect.getsource` 拿到的字符串不匹配. 请检查:
```python
# 应该在 AdvancedWeightModel._train_gnn 函数体中 (大约 1610-1613 行)
if gnn_arch == 'pna':
    best_hp = {'hidden_dim': 64, 'num_layers': 3, 'lr': 0.002, 'dropout': 0.2}
else:
    best_hp = {'hidden_dim': 96, 'num_layers': 3, 'lr': 0.002, 'dropout': 0.2}
```

### Q6. 训练结果跟本地不一致
**A**:
- 检查 `output_subset/cleaned_data.csv` 跟本地完全一致 (字节数 / md5)
- 切分函数 `standard_split` 已硬编码 `seed=42`, 应与本地一致
- AMD 浮点精度可能跟 CPU 微小差异 (<0.1% R²), 正常

### Q7. 怎么把云端训练好的模型集成到本地 ship_navigator?
**A**: 下载 zip 解压到 `output/` 后, `main.py` 启动时会自动扫描 `weight_model_gnn_*.pkl`, R² 最高的会成为 `best_model`. 集成预测是 `np.mean` 5 个 seed 的预测, 不需要改 `ship_navigator.py`.

### Q8. 想跳过 Optuna 直接用本地 96/3/0.002/0.2
**A**: 修改 `standalone_train.py` 末尾:
```python
# 注释掉 optuna_search
# best_hp = optuna_search(builder, n_trials=8, n_epochs=80, patience=15)
best_hp = {'hidden_dim': 96, 'num_layers': 3, 'dropout': 0.2, 'lr': 0.002}
```

### Q9. 报告里怎么解释 "为什么 GNN 用 Optuna 而树模型早就在用"?
**A**: 这正是 v2 改版的核心动机. 旧版 GNN 硬编码超参 (96/3/0.002/0.2), 树模型 (XGBoost/LightGBM) 用 Optuna 贝叶斯搜索, **方法论不一致**. v2 让 GNN 也用 Optuna 搜索, 这样:
- 报告里可以统一写 "本项目所有非集成模型均采用 Optuna TPE 贝叶斯搜索 (8-12 trial × 5-fold CV)"
- 评委不会质疑 "为什么树模型调了, GNN 不调"

---

## 10. 复现 checklist (v2)

- [ ] 注册魔搭 + 实名认证
- [ ] 创建 AMD GPU 方式三 Notebook
- [ ] 本地打包: `Compress-Archive cloud_gnn_package\* cloud_gnn_package.zip`
- [ ] 上传 zip 到 `/mnt/data/cloud_gnn/`, 解压
- [ ] `pip install optuna` (torch_geometric 通常已预装)
- [ ] 验证 GPU 可用
- [ ] 后台启动 `python standalone_train.py`
- [ ] 等待 40-60 分钟, 实时看 `tail -f train.log`
- [ ] 训练完成后打包结果 zip
- [ ] 下载 zip, 解压到本地 `output/`
- [ ] 运行 `python main.py` task 5 验证 R² 提升

---

## 附录 A: 完整命令速查 (v2)

### 本地 PowerShell (打包)
```powershell
cd d:\py_project\Data_Element
Compress-Archive -Path cloud_gnn_package\* -DestinationPath cloud_gnn_package.zip -Force
```

### 云端 Terminal (一气呵成)
```bash
# 1. 上传 cloud_gnn_package.zip 到 /mnt/data/, 解压
mkdir -p /mnt/data/cloud_gnn
cd /mnt/data/cloud_gnn
unzip /mnt/data/cloud_gnn_package.zip
# 2. 安装 optuna
pip install optuna
# 3. 后台启动
nohup python standalone_train.py > train.log 2>&1 &
echo "PID=$!"
# 4. 实时看日志
tail -f train.log
# 5. 训练完成后打包
zip -r cloud_gnn_results.zip \
    output_subset/weight_model_gnn_optuna.pkl \
    output_subset/weight_model_gnn_optuna_gnn_state.pt \
    output_subset/best_gnn_hparams.json \
    output_subset/optuna_study.csv \
    output_subset/gnn_ensemble_r2.json \
    train.log
```

### 本地 PowerShell (验证)
```powershell
cd d:\py_project\Data_Element
# 解压下载的 zip 到 output/
Expand-Archive .\Downloads\cloud_gnn_results.zip -DestinationPath .\output\ -Force
# 验证
python -c "import main; main.run_task5()" 2>&1 | Tee-Object output\verify_cloud_gnn.log
```

## 附录 B: 相关文件

- [cloud_gnn_package/](../../cloud_gnn_package/) — **v2 一键训练包** (Optuna + 5-seed 集成)
- [cloud_gnn_package/README.md](../../cloud_gnn_package/README.md) — 详细运行说明
- [cloud_gnn_package/standalone_train.py](../../cloud_gnn_package/standalone_train.py) — 主训练脚本
- [advanced_weight_model.py](../../advanced_weight_model.py) — GATv2Conv 主程序
- [config.py](../../config.py) — 配置文件
- [output/model_report.txt](../../output/model_report.txt) — 当前 7 模型对比报告

## 附录 C: v1 历史方案归档 (Grid Search + 5-seed 集成)

> **状态**: v1 已被 v2 (Optuna) 取代, 仅供对比参考. 旧方法也可在魔搭跑, 但耗时更长且方法论不一致.

v1 的方案是手写 Grid Search 笛卡尔积遍历超参, 跟本项目 XGBoost/LightGBM 用的 Optuna 贝叶斯搜索不一致, 报告里不好解释. v2 改用 Optuna 后方法论统一.

### v1 网格搜索脚本 (deprecated)

```python
# tune_gat_v2.py (v1, 不推荐)
import os, json, itertools
import torch
import numpy as np
from advanced_weight_model import train_and_eval_gnn, prepare_graph_data

OUT = '/mnt/data/cloud_gnn/output_subset'
data = prepare_graph_data(OUT)

grid = {
    'gat_heads': [2, 4, 8],
    'hidden_dim': [32, 64, 128],
    'n_layers': [2, 3, 4],
    'dropout': [0.1, 0.2, 0.3],
}

results = []
for heads, hd, layers, drop in itertools.product(*grid.values()):
    cfg = dict(gat_heads=heads, hidden_dim=hd, n_layers=layers, dropout=drop)
    print(f"\n=== Config: {cfg} ===")
    r = train_and_eval_gnn(data, **cfg, n_epochs=50)
    results.append({**cfg, **r})

with open('/mnt/data/cloud_gnn/tune_results.json', 'w') as f:
    json.dump(results, f, indent=2)

results.sort(key=lambda x: -x['r2'])
print("\n=== Top 5 ===")
for r in results[:5]:
    print(r)
```

**v1 缺点**:
- 3×3×3×2 = 54 组, 每组 50 epoch, 约 45 分钟 (CPU 8 小时)
- 不分 train/val, 全部用 test 评估, R² 偏乐观
- 跟项目 XGBoost/LightGBM 的 Optuna 调参策略不一致

### v1 集成脚本 (deprecated)

```python
# ensemble_gat_v2.py (v1, 不推荐)
import numpy as np
import torch
from advanced_weight_model import train_and_eval_gnn, prepare_graph_data

OUT = '/mnt/data/cloud_gnn/output_subset'
data = prepare_graph_data(OUT)

best_cfg = dict(gat_heads=4, hidden_dim=64, n_layers=3, dropout=0.2)

preds_list = []
for seed in [42, 123, 456, 789, 2024]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    r = train_and_eval_gnn(data, **best_cfg, n_epochs=100, return_predictions=True)
    preds_list.append(r['predictions'])

ensemble_pred = np.mean(preds_list, axis=0)
from sklearn.metrics import r2_score, mean_absolute_error
y_true = r['y_true']
print(f"Ensemble R2: {r2_score(y_true, ensemble_pred):.4f}")
print(f"Ensemble MAE: {mean_absolute_error(y_true, ensemble_pred):.2f} sec")
```

**v1 → v2 迁移**: v2 把网格搜索和集成训练合并到 `standalone_train.py` 一个脚本, 通过 `inspect.getsource` 注入超参, 0 源码改动, 自动保存兼容 `main.py` 加载的 `.pkl + .pt` 文件.
