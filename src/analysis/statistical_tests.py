"""
统计检验：类别间差异检验 + 模型间性能差异检验

使用方法: python src/analysis/statistical_tests.py
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, kruskal, shapiro, levene
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("results/tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 1. 加载评分数据 =====
ratings = pd.read_csv("data/ratings/aggregated_ratings.csv")
metadata = pd.read_csv("data/processed/metadata.csv")
df = ratings.merge(metadata[["image_id", "category"]], on="image_id")
print(f"Data shape: {df.shape}")
print(f"Categories: {df['category'].value_counts().to_dict()}")

dim_cols = ["complexity_mean", "beauty_mean", "order_mean", "hierarchy_mean", "emotion_mean"]
short_names = ["Complexity", "Beauty", "Order", "Hierarchy", "Emotion"]

# ===== 2. 正态性与方差齐性检验 =====
normality = []
for dim in dim_cols:
    for cat in df["category"].unique():
        cat_data = df[df["category"] == cat][dim]
        if len(cat_data) >= 3:
            stat, p = shapiro(cat_data)
            normality.append({"Dimension": dim, "Category": cat, "Shapiro_W": stat, "p_value": p})

norm_df = pd.DataFrame(normality)
norm_df.to_csv(OUTPUT_DIR / "normality_tests.csv", index=False)

# Levene 检验
levene_results = []
for dim in dim_cols:
    groups = [df[df["category"] == cat][dim].dropna().values for cat in df["category"].unique() if len(df[df["category"] == cat]) > 0]
    if len(groups) >= 2:
        stat, p = levene(*groups)
        levene_results.append({"Dimension": dim, "Levene_W": stat, "p_value": p})

levene_df = pd.DataFrame(levene_results)
levene_df.to_csv(OUTPUT_DIR / "levene_tests.csv", index=False)

# ===== 3. 类别间差异检验 =====
comparison_results = []

for dim, short in zip(dim_cols, short_names):
    groups = {cat: df[df["category"] == cat][dim].dropna().values for cat in df["category"].unique()}

    normal_ok = all(
        shapiro(v)[1] > 0.05 for v in groups.values() if len(v) >= 3
    )
    levene_ok = levene(*[v for v in groups.values() if len(v) > 0])[1] > 0.05

    if normal_ok and levene_ok and len(groups) >= 3:
        f_stat, p_val = f_oneway(*groups.values())
        test_used = "ANOVA"
        grand_mean = np.concatenate(list(groups.values())).mean()
        ss_between = sum(len(v) * (v.mean() - grand_mean) ** 2 for v in groups.values())
        ss_total = sum(((v - grand_mean) ** 2).sum() for v in groups.values())
        eta_sq = ss_between / ss_total if ss_total > 0 else 0
    else:
        h_stat, p_val = kruskal(*groups.values())
        test_used = "Kruskal-Wallis"
        eta_sq = np.nan

    comparison_results.append({
        "Dimension": short,
        "Test": test_used,
        "Statistic": f_stat if test_used == "ANOVA" else h_stat,
        "p_value": p_val,
        "Effect_Size": eta_sq,
        "Significant": "Yes" if p_val < 0.05 else "No",
    })

    print(f"{short}: {test_used}, p={p_val:.6f}, {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

comparison_df = pd.DataFrame(comparison_results)
comparison_df.to_csv(OUTPUT_DIR / "category_differences.csv", index=False)

# ===== 4. 模型间性能差异检验 =====
experiment_results = pd.read_csv(OUTPUT_DIR / "full_experiment_results.csv")

model_comparison = []
for target in experiment_results["TargetShort"].unique():
    for feat in experiment_results["FeatureSet"].unique():
        subset = experiment_results[(experiment_results["TargetShort"] == target) &
                                     (experiment_results["FeatureSet"] == feat)]
        models = subset["Model"].unique()

        model_scores = {}
        for m in models:
            model_scores[m] = subset[subset["Model"] == m]["Pearson_r"].values

        if len(models) >= 3:
            scores_matrix = np.array([model_scores[m][:5] for m in models])
            if scores_matrix.shape[1] >= 3:
                friedman_stat, friedman_p = stats.friedmanchisquare(*scores_matrix)
                model_comparison.append({
                    "Target": target,
                    "FeatureSet": feat,
                    "Friedman_stat": friedman_stat,
                    "Friedman_p": friedman_p,
                    "Significant": "Yes" if friedman_p < 0.05 else "No",
                })

model_comp_df = pd.DataFrame(model_comparison)
model_comp_df.to_csv(OUTPUT_DIR / "model_comparisons.csv", index=False)

print("\n=== Model Comparisons (significant cases) ===")
sig_cases = model_comp_df[model_comp_df["Significant"] == "Yes"]
if len(sig_cases) > 0:
    print(sig_cases.head(10).to_string(index=False))
else:
    print("  No significant model differences found.")

# ===== 5. 维度间相关性 =====
from scipy.stats import pearsonr as pcorr
dim_corr = []
for i, d1 in enumerate(dim_cols):
    for j, d2 in enumerate(dim_cols):
        if i < j:
            r_val, p_val = pcorr(df[d1], df[d2])
            dim_corr.append({
                "Dim1": short_names[i], "Dim2": short_names[j],
                "Pearson_r": r_val, "p_value": p_val
            })

dim_corr_df = pd.DataFrame(dim_corr)
dim_corr_df.to_csv(OUTPUT_DIR / "dimension_correlations.csv", index=False)
print("\n=== Dimension Correlations ===")
print(dim_corr_df.to_string(index=False))

print("\nAll statistical tests complete!")
