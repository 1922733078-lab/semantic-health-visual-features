"""
建模实验：遍历所有特征方案 × 模型 × 评分维度

6 feature sets × 5 models × 5 targets × 5 splits = 750 runs
运行时间：本机约 10-30 分钟

使用方法: python src/models/run_all_models.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# ===== 配置 =====
FEATURE_DIR = Path("data/features/merged")
OUTPUT_DIR = Path("results/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SPLITS = 5
RANDOM_SEEDS = [42, 123, 456, 789, 1024]
TARGET_COLS = ["complexity_mean", "beauty_mean", "order_mean", "hierarchy_mean", "emotion_mean"]
TARGET_SHORT = ["complexity", "beauty", "order", "hierarchy", "emotion"]

# ===== 加载数据 =====
print("Loading data...")
feature_sets = {}
feature_names = ["F1_traditional", "F2_deep_pca128", "F3_fusion_full",
               "F4_fusion_compact", "F5_tradition_clip", "F6_tradition_dinov2"]
for fname in feature_names:
    feature_sets[fname] = np.load(FEATURE_DIR / f"{fname}_features.npy")

targets = np.load(FEATURE_DIR / "targets.npy")
sample_info = pd.read_csv(FEATURE_DIR / "sample_info.csv")

print(f"Loaded {len(feature_sets)} feature sets")
for name, feats in feature_sets.items():
    print(f"  {name}: {feats.shape}")
print(f"Targets: {targets.shape}")
print(f"Samples: {len(sample_info)}")

# ===== 定义模型 =====
def get_models():
    return {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(n_estimators=150, max_depth=12,
                                               min_samples_leaf=5, random_state=42, n_jobs=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42,
                                     verbosity=0),
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "MLP": MLPRegressor(hidden_layer_sizes=(128, 64), activation="relu",
                            alpha=0.001, max_iter=300, random_state=42),
    }

# ===== 评估函数 =====
def evaluate_model(y_true, y_pred):
    r_val, _ = pearsonr(y_true, y_pred)
    rho_val, _ = spearmanr(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "Pearson_r": r_val if not np.isnan(r_val) else 0.0,
        "Spearman_rho": rho_val if not np.isnan(rho_val) else 0.0,
        "R2": r2_score(y_true, y_pred),
    }

# ===== 主实验循环 =====
all_results = []
models_dict = get_models()
total_runs = len(feature_sets) * len(models_dict) * len(TARGET_COLS) * N_SPLITS
pbar = tqdm(total=total_runs, desc="Running experiments")

for feat_name, X_all in feature_sets.items():
    for model_name, model_template in models_dict.items():
        for t_idx, (target_name, target_short) in enumerate(zip(TARGET_COLS, TARGET_SHORT)):
            y_all = targets[:, t_idx]

            for split_idx, seed in enumerate(RANDOM_SEEDS):
                # 分层分割
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_all, y_all, test_size=0.2, random_state=seed,
                        stratify=sample_info["category"].values
                    )
                except ValueError:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_all, y_all, test_size=0.2, random_state=seed
                    )

                # 克隆模型
                from sklearn.base import clone
                model = clone(model_template)

                try:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    metrics = evaluate_model(y_test, y_pred)
                except Exception as e:
                    metrics = {"MAE": np.nan, "RMSE": np.nan, "Pearson_r": np.nan,
                              "Spearman_rho": np.nan, "R2": np.nan}

                result = {
                    "FeatureSet": feat_name,
                    "Model": model_name,
                    "Target": target_name,
                    "TargetShort": target_short,
                    "Split": split_idx,
                    "Seed": seed,
                    **metrics
                }
                all_results.append(result)
                pbar.update(1)

pbar.close()

# ===== 保存结果 =====
results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_DIR / "full_experiment_results.csv", index=False)
print(f"\nResults saved to {OUTPUT_DIR / 'full_experiment_results.csv'}")

# ===== 汇总统计 =====
summary = results_df.groupby(["FeatureSet", "Model", "TargetShort"]).agg(
    Pearson_r_mean=("Pearson_r", "mean"),
    Pearson_r_std=("Pearson_r", "std"),
    Spearman_rho_mean=("Spearman_rho", "mean"),
    Spearman_rho_std=("Spearman_rho", "std"),
    MAE_mean=("MAE", "mean"),
    MAE_std=("MAE", "std"),
    RMSE_mean=("RMSE", "mean"),
    R2_mean=("R2", "mean"),
).reset_index()

summary.to_csv(OUTPUT_DIR / "experiment_summary.csv", index=False)

# ===== 找出最佳组合 =====
best_by_target = summary.loc[
    summary.groupby("TargetShort")["Pearson_r_mean"].idxmax()
]
print("\n=== Best model for each target ===")
print(best_by_target[["TargetShort", "FeatureSet", "Model", "Pearson_r_mean", "Spearman_rho_mean"]].to_string(index=False))

# ===== 汇总：按模型平均 =====
model_avg = results_df.groupby("Model")["Pearson_r"].mean().sort_values(ascending=False)
print("\n=== Average Pearson r by model ===")
for m, r in model_avg.items():
    print(f"  {m}: {r:.4f}")

# ===== 汇总：按特征方案平均 =====
feat_avg = results_df.groupby("FeatureSet")["Pearson_r"].mean().sort_values(ascending=False)
print("\n=== Average Pearson r by feature set ===")
for f, r in feat_avg.items():
    print(f"  {f}: {r:.4f}")

print("\n=== All experiments completed! ===")
