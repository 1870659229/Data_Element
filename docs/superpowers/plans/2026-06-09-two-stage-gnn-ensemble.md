# 两阶段 GNN 集成 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `advanced_weight_model.py` 入口加一个"找最优 + 稳定性验证"步骤:从 `build_weights_with_comparison` 跑出的 7 个模型里选 R² 最高的,如果最优是 GNN(GAT 或 PNA)就再跑 5 个 seed 看 mean ± std,其他模型(XGBoost/LightGBM/RF/NGBoost)单次 R² 即代表稳定性。

**Architecture:** 单文件改造,改动局限在 `if __name__ == "__main__":` 入口、原 GNN 集成块的位置。先 `max(_model_results, key=R²)` 选最优,然后 `if best ∈ {gnn, pna} and HAS_PYG:` 跑 5 seed 稳定性,else 跳过。

**简化迭代(2026-06-09 晚):** 最初设计是"两阶段"——Stage 1 选 GAT/PNA 赢家,Stage 2 对赢家做 5-seed 集成。这个设计对 GAT vs PNA 的 arch 选择是公平的,但有两个问题:(1) 选 arch 时只看 GAT vs PNA,不看 XGBoost/LightGBM 等,可能选错;(2) 跑 5 seed 是为了"集成提分",但用户实际想要的是"验证稳定性"。简化版直接选 7 个模型里的最优,只对 GNN 做 5-seed 验证(其他模型波动小,单次即代表)。

**Tech Stack:** Python 3, PyTorch, PyTorch Geometric, NumPy, sklearn (R²/MAE/RMSE), 现有 `AdvancedWeightModel._train_gnn` + 顶层函数 `train_gnn_with_seed`

---

## 文件结构

| 文件 | 类型 | 职责 |
|---|---|---|
| `advanced_weight_model.py` | 修改 | 替换原 2853-2900 行的 GNN 集成块为两阶段逻辑 |
| `output/model_metadata.json` | 重新生成 | 重跑后自动更新,包含 `gat_ensemble_5seed` 或 `pna_ensemble_5seed` |
| `output/img/model_comparison.png` | 重新生成 | 由 `visualize.py` 读 metadata 出图 |
| `output/model_report.txt` | 重新生成 | 控制台输出,文档化结果 |

**不修改**:
- `visualize.py` — 数据驱动,新 ensemble 名字自动出现
- `train_gnn_with_seed` — 已经是通用接口,接受 `gnn_arch` 参数
- `AdvancedWeightModel` 类 — 无需改

---

## Task 1: 在 1 个 seed 上对比 GAT 和 PNA,选赢家

**Files:**
- Modify: `advanced_weight_model.py:2853-2900`(原 GNN 集成块)

- [ ] **Step 1: 定位原集成块的起点**

打开 `advanced_weight_model.py`,跳到 2853 行附近,确认是这段:

```python
# ===== GNN 多 seed 集成 =====
if HAS_PYG:
    SEEDS = [42, 123, 456, 789, 1011]
    print(f"\n{'='*60}")
    print(f"GNN 多 seed 集成 (n={len(SEEDS)})")
    print(f"{'='*60}")
```

- [ ] **Step 2: 替换为 Stage 1 + Stage 2 两阶段块**

把从 `# ===== GNN 多 seed 集成 =====` 开始,到 `model.best_model = gnn_seed_results[-1].model` 结束的全部内容,**整段替换**为下面这段:

```python
    # ===== GNN 两阶段集成 =====
    if HAS_PYG:
        SEEDS = [42, 123, 456, 789, 1011]
        GNN_ARCHES = ['gat', 'pna']

        # ---------- Stage 1: 单 seed 选 arch ----------
        print(f"\n{'='*60}")
        print(f"GNN Stage 1: 各 arch 单 seed 选赢家 (seed=42)")
        print(f"{'='*60}")

        stage1_r2 = {}
        stage1_results = {}
        for arch in GNN_ARCHES:
            r = train_gnn_with_seed(model, 42, gnn_arch=arch)
            stage1_results[arch] = r
            stage1_r2[arch] = r.r2
            print(f"  {arch} (seed=42): R2={r.r2:.4f}  MAE={r.mae:.2f}s")

        winner_arch = max(stage1_r2, key=stage1_r2.get)
        print(f"\n  -> Stage 1 赢家: {winner_arch} (R2={stage1_r2[winner_arch]:.4f})")

        # ---------- Stage 2: 赢家 arch 跑 5 seed 集成 ----------
        print(f"\n{'='*60}")
        print(f"GNN Stage 2: {winner_arch} 5 seed 集成 (n={len(SEEDS)})")
        print(f"{'='*60}")

        gnn_seed_results = []
        gnn_all_preds = []
        for seed in SEEDS:
            r = train_gnn_with_seed(model, seed, gnn_arch=winner_arch)
            print(f"  {winner_arch} seed={seed}: R2={r.r2:.4f} MAE={r.mae:.2f}s")
            gnn_seed_results.append(r)
            if r.predictions is not None:
                gnn_all_preds.append(r.predictions)

        if gnn_all_preds:
            avg_pred = np.mean(gnn_all_preds, axis=0)
            y_true = gnn_seed_results[0].y_test

            ens_mae = mean_absolute_error(y_true, avg_pred)
            ens_rmse = np.sqrt(mean_squared_error(y_true, avg_pred))
            ens_r2 = r2_score(y_true, avg_pred)
            mask = y_true != 0
            ens_mape = np.mean(np.abs((y_true[mask] - avg_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0

            ens_avg_r2 = float(np.mean([r.r2 for r in gnn_seed_results]))
            ens_std_r2 = float(np.std([r.r2 for r in gnn_seed_results]))

            print(f"\n  {winner_arch} 集成 ({len(SEEDS)} seed): R2={ens_r2:.4f}  MAE={ens_mae:.2f}s  RMSE={ens_rmse:.2f}s  MAPE={ens_mape:.2f}%")
            print(f"  {winner_arch} 单 seed 平均: R2={ens_avg_r2:.4f} ± {ens_std_r2:.4f}")

            ens_name = f'{winner_arch}_ensemble_{len(SEEDS)}seed'
            ens_result = ModelResult(
                model_name=ens_name,
                train_time=sum(r.train_time for r in gnn_seed_results),
                mae=ens_mae,
                rmse=ens_rmse,
                r2=ens_r2,
                mape=ens_mape,
                model=gnn_seed_results[-1].model,
                predictions=avg_pred,
                use_log_transform=False,
                y_test=y_true
            )
            model._model_results[ens_name] = ens_result

            best_r2 = model._model_results[model.best_model_name].r2 if model.best_model_name else -1
            if ens_r2 > best_r2:
                print(f"  -> 集成模型 (R2={ens_r2:.4f}) 优于当前最优 {model.best_model_name} (R2={best_r2:.4f}), 设为最佳模型")
                model.best_model_name = ens_name
                model.best_model = gnn_seed_results[-1].model
```

- [ ] **Step 3: 静态检查 - Python 语法验证**

在 `d:\py_project\Data_Element` 目录执行:

```bash
python -c "import ast; ast.parse(open('advanced_weight_model.py', encoding='utf-8').read()); print('SYNTAX OK')"
```

预期输出:`SYNTAX OK`

如报错,根据错误信息回到 Step 2 修正缩进或符号。

- [ ] **Step 4: Commit**

```bash
cd d:\py_project\Data_Element
git add advanced_weight_model.py
git commit -m "refactor(model): GNN ensemble 改两阶段,先选 arch 再 5-seed 集成"
```

---

## Task 2: 运行训练,验证赢家选出

**Files:**
- 重生成: `output/model_metadata.json`, `output/model_report.txt`

- [ ] **Step 1: 运行主训练脚本**

```bash
cd d:\py_project\Data_Element
python main.py 2>&1 | tee _run_main.log
```

或在 IDE 调试器里运行 `main.py`,但建议用命令行,日志方便 grep。

- [ ] **Step 2: 验证 Stage 1 输出**

在日志中 grep:

```bash
grep -E "Stage 1|gat \(seed=42\)|pna \(seed=42\)|Stage 1 赢家" _run_main.log
```

预期看到(顺序固定):

```
GNN Stage 1: 各 arch 单 seed 选赢家 (seed=42)
  gat (seed=42): R2=0.xxxx  MAE=x.xxs
  pna (seed=42): R2=0.xxxx  MAE=x.xxs
  -> Stage 1 赢家: pna (R2=0.xxxx)        ← 或 gat,取决于数据
```

- [ ] **Step 3: 验证 Stage 2 输出**

```bash
grep -E "Stage 2|集成 \(5 seed\)|优于当前最优" _run_main.log
```

预期看到:

```
GNN Stage 2: pna 5 seed 集成 (n=5)        ← 或 gat
  pna seed=42: R2=0.xxxx ...
  ...
  pna 集成 (5 seed): R2=0.xxxx ...
  -> 集成模型 (R2=0.xxxx) 优于当前最优 xgboost (R2=0.xxxx), 设为最佳模型
```

如果**没出现 "优于当前最优"**,说明集成 R² 不如 XGBoost,这是正常情况,后续对比图会显示所有 9 个模型。

- [ ] **Step 4: 验证 metadata 含新 ensemble 名字**

```bash
python -c "
import json
m = json.load(open('output/model_metadata.json', encoding='utf-8'))
keys = list(m.get('model_comparison', {}).keys())
print('models in metadata:', keys)
ens = [k for k in keys if 'ensemble' in k]
print('ensembles:', ens)
assert len(ens) == 1, f'Expected 1 ensemble, got {len(ens)}: {ens}'
assert ens[0] in ('gat_ensemble_5seed', 'pna_ensemble_5seed'), f'Unexpected: {ens[0]}'
print('OK')
"
```

预期输出:

```
models in metadata: ['xgboost', 'lightgbm', 'lightgbm_tweedie', 'random_forest', 'ngboost', 'gnn', 'pna', 'pna_ensemble_5seed']
ensembles: ['pna_ensemble_5seed']
OK
```

(具体模型名字顺序可能略有不同,关键是有 1 个 `*_ensemble_5seed`)

- [ ] **Step 5: 删除临时日志**

```bash
rm _run_main.log
```

---

## Task 3: 重新出图,验证对比图

**Files:**
- 重生成: `output/img/model_comparison.png`

- [ ] **Step 1: 运行可视化脚本**

```bash
cd d:\py_project\Data_Element
python visualize.py 2>&1 | tee _run_viz.log
```

- [ ] **Step 2: 验证 model_comparison.png 已更新**

```bash
ls -la output/img/model_comparison.png
file output/img/model_comparison.png
```

预期:文件存在,大于 50KB(经验值),PNG 格式。

- [ ] **Step 3: 打开图片确认 8 个模型柱状图(应该有新集成模型柱)**

在 IDE 资源管理器里双击打开 `output/img/model_comparison.png`,确认:
- R²/MAE/RMSE 三个子图
- 8 个柱(原来 7 个 + 新增的 `*_ensemble_5seed` 柱)
- 新柱条应该是 5-seed 集成的赢家,标星 ★

- [ ] **Step 4: 删除临时日志**

```bash
rm _run_viz.log
```

---

## Task 4: 验证最终输出,确认无回归

**Files:**
- 检查: `output/model_report.txt`, `output/model_metadata.json`

- [ ] **Step 1: 验证最佳模型是集成版本**

```bash
python -c "
import json
m = json.load(open('output/model_metadata.json', encoding='utf-8'))
best = m.get('best_model')
print('best_model:', best)
assert 'ensemble' in best, f'Best model should be ensemble, got: {best}'
print('OK')
"
```

预期:`best_model: pna_ensemble_5seed` 或 `best_model: gat_ensemble_5seed`

- [ ] **Step 2: 验证 R² 合理(0.7-0.9)**

```bash
python -c "
import json
m = json.load(open('output/model_metadata.json', encoding='utf-8'))
best = m.get('best_model')
r2 = m['model_comparison'][best]['r2']
print(f'best={best}, R2={r2:.4f}')
assert 0.70 <= r2 <= 0.92, f'R2 out of range: {r2}'
print('OK')
"
```

预期:`best=pna_ensemble_5seed, R2=0.xxxx` 且 `OK`

- [ ] **Step 3: 查看 model_report.txt 头部**

```bash
head -20 output/model_report.txt
```

预期看到表格里有 `pna_ensemble_5seed ★` 标记为最优模型。

- [ ] **Step 4: 更新技术报告引用(可选,非必须)**

如果 `技术报告.md` 里引用了旧的 `ngboost` 作为最佳模型,搜索并更新为新的 `*_ensemble_5seed`:

```bash
grep -n "ngboost" 技术报告.md
```

如果搜到,在 IDE 里手工更新。

- [ ] **Step 5: Commit 最终状态**

```bash
cd d:\py_project\Data_Element
git add output/model_metadata.json output/model_report.txt output/feature_importance.csv output/img/model_comparison.png
git status
git commit -m "chore(output): 2 段 + 两阶段 GNN 集成 后重新跑训练与出图"
```

---

## 自查清单(写完后回头看)

- [x] **Spec 覆盖**:
  - 修硬编码 'gat' → Task 1 Step 2 ✅
  - 两阶段策略(Stage 1 + Stage 2)→ Task 1 Step 2 ✅
  - 1.1× 时间成本估算 → 写在 Goal 里 ✅
  - 模型对比图含新集成 → Task 3 ✅
  - 最佳模型是集成版本 → Task 4 ✅

- [x] **占位符扫描**:全文没有"TBD"/"TODO"/"implement later"。

- [x] **类型一致**:
  - `train_gnn_with_seed(model, seed, gnn_arch=arch)` 与现有定义一致
  - `ModelResult` 字段名(model_name, mae, rmse, r2, mape, predictions, y_test)与 230-242 行定义一致
  - `model._model_results` 与 2408-2420 行的 `_model_results` 字典访问一致
  - `ens_name = f'{winner_arch}_ensemble_{len(SEEDS)}seed'` 与 Task 4 Step 1 的检查一致

---

## 预计产出

| 项 | 改动前 | 改动后 |
|---|---|---|
| 集成 arch 数量 | 1(GAT) | 1(赢家: GAT 或 PNA) |
| 集成 seed 数量 | 5 | 5 |
| 总训练时间 | 1.0× | 1.1×(多 1 个 seed) |
| `model_comparison.png` 模型数 | 7 | 8 |
| 最佳模型 | gnn_ensemble (硬编码) | 赢家 ens (GAT/PNA 自动选) |
| 公平性 | ❌ PNA 受歧视 | ✅ PNA 有机会 |

**Plan 完成时间预估**:Tasks 1-4 总计约 30-60 分钟(含重跑训练 20-30 分钟)。
