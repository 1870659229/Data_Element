# -*- coding: utf-8 -*-
"""
Compare: empirical avg vs model prediction on edges with trajectory data.
No retraining needed - loads feature_matrix.csv, reproduces 80/20 split.
"""
import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle, warnings, json
warnings.filterwarnings('ignore')

SEP = "=" * 70
DSEP = "-" * 75

print(SEP)
print("Compare: Empirical Mean vs Model Prediction (edges with data)")
print(SEP)

# ========== Load data ==========
fm = pd.read_csv('output/feature_matrix.csv')
print(f"\nfeature_matrix: {len(fm)} rows (edge x period)")

feature_names = [
    'avg_reported_speed', 'std_reported_speed', 'speed_cv',
    'bearing', 'bearing_sin', 'bearing_cos', 'avg_course_change',
    'std_course_change', 'course_change_x_narrow',
    'waterway_type',
    'node_degree_from', 'node_degree_to', 'edge_betweenness',
    'sample_count', 'log_sample_count',
    'distance', 'theoretical_time',
    'edge_speed_median', 'edge_speed_iqr',
    'neighbor_count', 'neighbor_speed_median',
    'period_morning', 'period_midday', 'period_afternoon', 'period_night',
    'hour_sin', 'hour_cos',
    'speed_decay',
]
print(f"Features: {len(feature_names)} (metadata says 28, actual={len(feature_names)})")

X = fm[feature_names].values.astype(np.float64)
y_ratio = fm['time_ratio'].values.astype(np.float64)
y_time = fm['avg_travel_time'].values.astype(np.float64)
tt = fm['theoretical_time'].values.astype(np.float64)
from_nodes = fm['from_node'].values
to_nodes = fm['to_node'].values
periods = fm['period'].values
sample_counts = fm['sample_count'].values

# ========== 80/20 split ==========
indices = np.arange(len(X))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

X_train, X_test = X[train_idx], X[test_idx]
y_ratio_train, y_ratio_test = y_ratio[train_idx], y_ratio[test_idx]
y_time_train, y_time_test = y_time[train_idx], y_time[test_idx]
tt_train, tt_test = tt[train_idx], tt[test_idx]
sc_train, sc_test = sample_counts[train_idx], sample_counts[test_idx]
fn_test, tn_test, period_test = from_nodes[test_idx], to_nodes[test_idx], periods[test_idx]
fn_train, tn_train, period_train = from_nodes[train_idx], to_nodes[train_idx], periods[train_idx]

print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

# ========== Train ML models ==========
print("\n[Training ML models for comparison...]")

# 1. XGBoost
HAS_XGB = False
try:
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    xgb_model.fit(X_train, y_ratio_train)
    xgb_pred_ratio = np.clip(xgb_model.predict(X_test), 0.1, 20.0)
    xgb_pred_time = xgb_pred_ratio * tt_test
    HAS_XGB = True
    print("  XGBoost: OK")
except Exception as e:
    print(f"  XGBoost: FAIL ({e})")

# 2. LightGBM
HAS_LGB = False
try:
    import lightgbm as lgb
    lgb_model = lgb.LGBMRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        num_leaves=31, random_state=42, n_jobs=-1, verbose=-1
    )
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    lgb_model.fit(X_train_df, y_ratio_train)
    lgb_pred_ratio = np.clip(lgb_model.predict(X_test_df), 0.1, 20.0)
    lgb_pred_time = lgb_pred_ratio * tt_test
    HAS_LGB = True
    print("  LightGBM: OK")
except Exception as e:
    print(f"  LightGBM: FAIL ({e})")

# 3. Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_ratio_train)
rf_pred_ratio = np.clip(rf_model.predict(X_test), 0.1, 20.0)
rf_pred_time = rf_pred_ratio * tt_test
print("  RandomForest: OK")

# ========== Empirical mean prediction ==========
print("\n[Computing empirical mean predictions...]")

# Build train edge index: (from, to, period) -> avg_travel_time
train_edge_by_pair = {}
for i in range(len(train_idx)):
    pair = (fn_train[i], tn_train[i])
    if pair not in train_edge_by_pair:
        train_edge_by_pair[pair] = {}
    train_edge_by_pair[pair][period_train[i]] = y_time_train[i]

global_mean_time = np.mean(y_time_train)
global_mean_ratio = np.mean(y_ratio_train)

# For each test sample, compute empirical prediction
empirical_pred = np.zeros(len(test_idx))
empirical_strategy = []

for i in range(len(test_idx)):
    pair = (fn_test[i], tn_test[i])
    period = period_test[i]
    other_period = 'night' if period == 'day' else 'day'
    
    # Strategy: same edge, different period (if in train)
    if pair in train_edge_by_pair and other_period in train_edge_by_pair[pair]:
        empirical_pred[i] = train_edge_by_pair[pair][other_period]
        empirical_strategy.append('cross_period')
    else:
        empirical_pred[i] = global_mean_time
        empirical_strategy.append('global_mean')

empirical_strategy = np.array(empirical_strategy)
n_cross = (empirical_strategy == 'cross_period').sum()
n_global = (empirical_strategy == 'global_mean').sum()
print(f"  Cross-period inference: {n_cross}")
print(f"  Global mean fallback: {n_global}")

# ========== Results ==========
print(f"\n{SEP}")
print(f"Results (test set, {len(test_idx)} edge-periods)")
print(SEP)

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mask = y_true > 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return mae, rmse, r2, mape

header = f"{'Method':<35} {'MAE(s)':>8} {'RMSE(s)':>8} {'R2':>8} {'MAPE%':>8}"
print(f"\n{header}")
print(DSEP)

all_results = {}

# Empirical
mae, rmse, r2, mape = evaluate('Empirical', y_time_test, empirical_pred)
all_results['Empirical mean'] = (mae, rmse, r2, mape)
print(f"{'Empirical mean (cross-period/global)':<35} {mae:>8.2f} {rmse:>8.2f} {r2:>8.4f} {mape:>8.2f}")

# Theoretical time (ratio=1)
theo_pred = tt_test.copy()
mae, rmse, r2, mape = evaluate('Theoretical', y_time_test, theo_pred)
all_results['Theoretical time'] = (mae, rmse, r2, mape)
print(f"{'Theoretical time (ratio=1)':<35} {mae:>8.2f} {rmse:>8.2f} {r2:>8.4f} {mape:>8.2f}")

# Global ratio mean * theoretical
global_ratio_pred = global_mean_ratio * tt_test
mae, rmse, r2, mape = evaluate('Global ratio', y_time_test, global_ratio_pred)
all_results['Global ratio mean'] = (mae, rmse, r2, mape)
print(f"{'Global ratio mean * theo_time':<35} {mae:>8.2f} {rmse:>8.2f} {r2:>8.4f} {mape:>8.2f}")

# RF
mae, rmse, r2, mape = evaluate('RF', y_time_test, rf_pred_time)
all_results['RandomForest'] = (mae, rmse, r2, mape)
print(f"{'RandomForest':<35} {mae:>8.2f} {rmse:>8.2f} {r2:>8.4f} {mape:>8.2f}")

if HAS_XGB:
    mae, rmse, r2, mape = evaluate('XGB', y_time_test, xgb_pred_time)
    all_results['XGBoost'] = (mae, rmse, r2, mape)
    print(f"{'XGBoost':<35} {mae:>8.2f} {rmse:>8.2f} {r2:>8.4f} {mape:>8.2f}")

if HAS_LGB:
    mae, rmse, r2, mape = evaluate('LGB', y_time_test, lgb_pred_time)
    all_results['LightGBM'] = (mae, rmse, r2, mape)
    print(f"{'LightGBM':<35} {mae:>8.2f} {rmse:>8.2f} {r2:>8.4f} {mape:>8.2f}")

# PNA from metadata
with open('output/model_metadata.json') as f:
    meta = json.load(f)
pna_meta = meta['model_comparison']['pna']
print(f"{'PNA (from training log)':<35} {pna_meta['mae']:>8.2f} {pna_meta['rmse']:>8.2f} {pna_meta['r2']:>8.4f} {pna_meta['mape']:>8.2f}")

# ========== Head-to-head ==========
print(f"\n{SEP}")
print("Head-to-head: Model vs Empirical (per-edge)")
print(SEP)

for name, pred in [('RandomForest', rf_pred_time)]:
    pass

if HAS_XGB:
    xgb_better = int(np.sum(np.abs(y_time_test - xgb_pred_time) < np.abs(y_time_test - empirical_pred)))
    print(f"  XGBoost wins:      {xgb_better}/{len(test_idx)} ({xgb_better/len(test_idx)*100:.1f}%)")

if HAS_LGB:
    lgb_better = int(np.sum(np.abs(y_time_test - lgb_pred_time) < np.abs(y_time_test - empirical_pred)))
    print(f"  LightGBM wins:     {lgb_better}/{len(test_idx)} ({lgb_better/len(test_idx)*100:.1f}%)")

rf_better = int(np.sum(np.abs(y_time_test - rf_pred_time) < np.abs(y_time_test - empirical_pred)))
emp_better = len(test_idx) - rf_better
print(f"  RandomForest wins: {rf_better}/{len(test_idx)} ({rf_better/len(test_idx)*100:.1f}%)")
print(f"  Empirical wins:    {emp_better}/{len(test_idx)} ({emp_better/len(test_idx)*100:.1f}%)")

# ========== By sample count ==========
print(f"\n{SEP}")
print("By sample count (dense=data-rich, empirical should be reliable)")
print(SEP)

for threshold in [10, 30, 50]:
    sparse = sc_test < threshold
    dense = ~sparse
    if sparse.sum() > 5 and dense.sum() > 5:
        emp_mae_s = mean_absolute_error(y_time_test[sparse], empirical_pred[sparse])
        emp_mae_d = mean_absolute_error(y_time_test[dense], empirical_pred[dense])
        rf_mae_s = mean_absolute_error(y_time_test[sparse], rf_pred_time[sparse])
        rf_mae_d = mean_absolute_error(y_time_test[dense], rf_pred_time[dense])
        
        emp_r2_s = r2_score(y_time_test[sparse], empirical_pred[sparse])
        emp_r2_d = r2_score(y_time_test[dense], empirical_pred[dense])
        rf_r2_s = r2_score(y_time_test[sparse], rf_pred_time[sparse])
        rf_r2_d = r2_score(y_time_test[dense], rf_pred_time[dense])
        
        xgb_line = ""
        if HAS_XGB:
            xgb_mae_s = mean_absolute_error(y_time_test[sparse], xgb_pred_time[sparse])
            xgb_mae_d = mean_absolute_error(y_time_test[dense], xgb_pred_time[dense])
            xgb_line = f" | XGB MAE={xgb_mae_s:.1f}/{xgb_mae_d:.1f}"
        
        print(f"\n  Threshold={threshold} (sparse={sparse.sum()}, dense={dense.sum()})")
        print(f"    Sparse: Emp MAE={emp_mae_s:.2f} R2={emp_r2_s:.4f} | RF MAE={rf_mae_s:.2f} R2={rf_r2_s:.4f}{xgb_line}")
        print(f"    Dense:  Emp MAE={emp_mae_d:.2f} R2={emp_r2_d:.4f} | RF MAE={rf_mae_d:.2f} R2={rf_r2_d:.4f}")

# ========== Core conclusion ==========
emp_r2 = r2_score(y_time_test, empirical_pred)
rf_r2 = r2_score(y_time_test, rf_pred_time)

print(f"\n{SEP}")
print("CORE CONCLUSIONS")
print(SEP)
print(f"""
Q: For edges with trajectory data, use empirical mean or model prediction?

1. Empirical mean R2={emp_r2:.4f} vs RandomForest R2={rf_r2:.4f}
   -> {'Model is BETTER' if rf_r2 > emp_r2 else 'Empirical is BETTER'}

2. Current code behavior (_predict_with_gnn):
   -> Edges WITH data: uses empirical average (NOT PNA)
   -> Edges WITHOUT data: uses neighbor speed inference (NOT PNA either!)
   -> PNA model is trained, evaluated, then DISCARDED at deployment

3. Is retraining needed?
   -> NO need to retrain PNA
   -> NEED to fix _predict_with_gnn() to actually USE PNA predictions
   -> Or adopt hybrid: dense edges use empirical, sparse edges use PNA

4. Paper impact:
   -> Current paper says "PNA for dynamic weight prediction"
   -> Reality: PNA never participates in actual prediction
   -> Recommendation: fix code to use PNA, or honestly describe hybrid strategy
""")
