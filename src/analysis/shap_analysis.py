"""
SHAP 可解释性分析
对最佳模型 (F3 + XGBoost) 的 5 个评分维度分别做 SHAP 分析。

使用方法: python src/analysis/shap_analysis.py
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from pathlib import Path
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

# ===== 配置 =====
FEATURE_DIR = Path("data/features/merged")
OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = Path("results/tables")

TARGET_COLS = ["complexity_mean", "beauty_mean", "order_mean", "hierarchy_mean", "emotion_mean"]
TARGET_SHORT = ["complexity", "beauty", "order", "hierarchy", "emotion"]

# ===== 加载数据 =====
X = np.load(FEATURE_DIR / "F3_fusion_full_features.npy")
targets = np.load(FEATURE_DIR / "targets.npy")
sample_info = pd.read_csv(FEATURE_DIR / "sample_info.csv")

# 特征名
trad_df = pd.read_csv("data/features/traditional_features.csv")
meta_cols = ["image_id", "category"]
trad_feat_cols = [c for c in trad_df.columns if c not in meta_cols]
deep_cols = [f"deep_pca_{i}" for i in range(X.shape[1] - len(trad_feat_cols))]
feature_names = trad_feat_cols + deep_cols
feature_names = feature_names[:X.shape[1]]

print(f"Feature matrix: {X.shape}")
print(f"Feature names: {len(feature_names)}")

shap_importance_all = {}

for t_idx, (target_name, short_name) in enumerate(zip(TARGET_COLS, TARGET_SHORT)):
    print(f"\n{'='*60}")
    print(f"SHAP analysis for: {short_name}")
    print(f"{'='*60}")

    y = targets[:, t_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=sample_info["category"].values
    )

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r_val, _ = pearsonr(y_test, y_pred)
    print(f"  Test Pearson r: {r_val:.4f}")

    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(model)
    # 使用 200 个测试样本
    n_test = min(200, len(X_test))
    X_test_sub = X_test[:n_test]
    shap_values = explainer.shap_values(X_test_sub)

    # 全局特征重要性
    shap_importance = np.abs(shap_values).mean(axis=0)

    # 保存 SHAP 重要性
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "importance_mean": np.abs(shap_values).mean(axis=0),
        "importance_std": np.abs(shap_values).std(axis=0),
    })
    shap_df = shap_df.sort_values("importance_mean", ascending=False)
    shap_df.to_csv(TABLES_DIR / f"shap_importance_{short_name}.csv", index=False)
    shap_importance_all[short_name] = shap_df

    # 图 1：SHAP Summary Bar
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test_sub,
        feature_names=feature_names,
        plot_type="bar", max_display=15, show=False
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_bar_{short_name}.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"shap_bar_{short_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: shap_bar_{short_name}")

    # 图 2：SHAP Summary Beeswarm
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X_test_sub,
        feature_names=feature_names,
        max_display=15, show=False
    )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_beeswarm_{short_name}.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"shap_beeswarm_{short_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: shap_beeswarm_{short_name}")

    # 图 3：SHAP Dependence Plot (top 4 特征)
    top_indices = np.argsort(shap_importance)[-4:]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for idx, feat_idx in enumerate(reversed(top_indices)):
        ax = axes[idx // 2, idx % 2]
        shap.dependence_plot(
            feat_idx, shap_values, X_test_sub,
            feature_names=feature_names,
            ax=ax, show=False
        )
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_dependence_{short_name}.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"shap_dependence_{short_name}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: shap_dependence_{short_name}")

    # 打印 top-10 特征
    print(f"\n  Top-10 features for {short_name}:")
    for i, row in shap_df.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance_mean']:.4f}")

# 汇总 SHAP 重要性到一张表
shap_summary = pd.DataFrame()
for short_name, shap_df in shap_importance_all.items():
    if shap_summary.empty:
        shap_summary = shap_df[["feature", "importance_mean"]].rename(
            columns={"importance_mean": short_name})
    else:
        shap_summary = shap_summary.merge(
            shap_df[["feature", "importance_mean"]].rename(
                columns={"importance_mean": short_name}),
            on="feature", how="outer"
        )

shap_summary = shap_summary.fillna(0)
shap_summary["mean_importance"] = shap_summary[TARGET_SHORT].mean(axis=1)
shap_summary = shap_summary.sort_values("mean_importance", ascending=False)
shap_summary.to_csv(TABLES_DIR / "shap_importance_summary.csv", index=False)

print(f"\n{'='*60}")
print("All SHAP analyses complete!")
print(f"{'='*60}")
