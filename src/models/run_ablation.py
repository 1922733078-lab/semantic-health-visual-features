"""
消融实验：逐一移除特征组，评估性能变化

使用方法: python src/models/run_ablation.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import xgboost as xgb
from scipy.stats import pearsonr
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

FEATURE_DIR = Path("data/features/merged")
OUTPUT_DIR = Path("results/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLS = ["complexity_mean", "beauty_mean", "order_mean", "hierarchy_mean", "emotion_mean"]
TARGET_SHORT = ["complexity", "beauty", "order", "hierarchy", "emotion"]

# 加载传统特征的原始列
trad_df = pd.read_csv("data/features/traditional_features.csv")
meta_cols = ["image_id", "category"]
trad_feat_cols = [c for c in trad_df.columns if c not in meta_cols]

COLOR_FEATS = [c for c in trad_feat_cols if any(k in c for k in
    ["color", "saturation", "value_", "lightness", "hue", "warm", "harmony", "dominant"])]
TEXTURE_FEATS = [c for c in trad_feat_cols if any(k in c for k in
    ["edge", "glcm", "gray_entropy"])]
COMPOSITION_FEATS = [c for c in trad_feat_cols if any(k in c for k in
    ["symmetry", "rule_of_thirds", "center_offset", "whitespace", "fg_bg", "diagonal"])]
TYPOGRAPHY_FEATS = [c for c in trad_feat_cols if any(k in c for k in
    ["text_", "has_text", "font_size"])]
SALIENCY_FEATS = [c for c in trad_feat_cols if any(k in c for k in
    ["saliency", "salient"])]

feature_groups = {
    "All_traditional": trad_feat_cols,
    "No_Color": [c for c in trad_feat_cols if c not in COLOR_FEATS],
    "No_Texture": [c for c in trad_feat_cols if c not in TEXTURE_FEATS],
    "No_Composition": [c for c in trad_feat_cols if c not in COMPOSITION_FEATS],
    "No_Typography": [c for c in trad_feat_cols if c not in TYPOGRAPHY_FEATS],
    "No_Saliency": [c for c in trad_feat_cols if c not in SALIENCY_FEATS],
    "Color_only": COLOR_FEATS,
    "Texture_only": TEXTURE_FEATS,
    "Composition_only": COMPOSITION_FEATS,
    "Saliency_only": SALIENCY_FEATS,
}

# 加载 F3 全融合特征
X_full = np.load(FEATURE_DIR / "F3_fusion_full_features.npy")
targets = np.load(FEATURE_DIR / "targets.npy")
sample_info = pd.read_csv(FEATURE_DIR / "sample_info.csv")

# 获取列索引映射
trad_col_idx = {col: i for i, col in enumerate(trad_feat_cols)}

ablation_results = []

for group_name, selected_cols in tqdm(feature_groups.items(), desc="Ablation"):
    trad_indices = [trad_col_idx[c] for c in selected_cols if c in trad_col_idx]
    n_trad = len(trad_feat_cols)
    deep_indices = list(range(n_trad, X_full.shape[1]))
    all_indices = trad_indices + deep_indices

    X_ablated = X_full[:, all_indices]

    for t_idx, (target_name, target_short) in enumerate(zip(TARGET_COLS, TARGET_SHORT)):
        y = targets[:, t_idx]

        scores = []
        for seed in [42, 123, 456]:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_ablated, y, test_size=0.2, random_state=seed,
                    stratify=sample_info["category"].values
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_ablated, y, test_size=0.2, random_state=seed
                )

            model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                     random_state=42, verbosity=0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r_val, _ = pearsonr(y_test, y_pred)
            scores.append(r_val if not np.isnan(r_val) else 0.0)

        ablation_results.append({
            "Ablation": group_name,
            "Target": target_short,
            "Pearson_r_mean": np.mean(scores),
            "Pearson_r_std": np.std(scores),
            "n_features": len(all_indices),
        })

ablation_df = pd.DataFrame(ablation_results)
ablation_df.to_csv(OUTPUT_DIR / "ablation_results.csv", index=False)

# 打印摘要
print("\n=== Ablation Summary ===")
baseline = ablation_df[ablation_df["Ablation"] == "All_traditional"]
for target in TARGET_SHORT:
    baseline_r = baseline[baseline["Target"] == target]["Pearson_r_mean"].values[0]
    others = ablation_df[(ablation_df["Target"] == target) & (ablation_df["Ablation"] != "All_traditional")]
    print(f"\n{target} (baseline r={baseline_r:.4f}):")
    for _, row in others.iterrows():
        delta = row["Pearson_r_mean"] - baseline_r
        print(f"  {row['Ablation']}: r={row['Pearson_r_mean']:.4f}, n={row['n_features']}, Δ={delta:+.4f}")

print("\nAblation study complete!")
