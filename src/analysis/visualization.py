"""
论文图表生成：特征分布、t-SNE、评分对比等
使用方法: python src/analysis/visualization.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("results/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 字体设置
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

# ===== 加载数据 =====
FEATURE_DIR = Path("data/features/merged")
X = np.load(FEATURE_DIR / "F3_fusion_full_features.npy")
targets = np.load(FEATURE_DIR / "targets.npy")
sample_info = pd.read_csv(FEATURE_DIR / "sample_info.csv")

ratings = pd.read_csv("data/ratings/aggregated_ratings.csv")
df = ratings.merge(sample_info, on="image_id")

TARGET_SHORT = ["complexity", "beauty", "order", "hierarchy", "emotion"]
TARGET_MEAN = ["complexity_mean", "beauty_mean", "order_mean", "hierarchy_mean", "emotion_mean"]

# ===== 图 1：维度间相关性热力图 =====
fig, ax = plt.subplots(figsize=(8, 7))
corr = df[TARGET_MEAN].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, square=True, ax=ax,
            xticklabels=TARGET_SHORT, yticklabels=TARGET_SHORT)
ax.set_title("Correlations Between Rating Dimensions", fontsize=14)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_dimension_correlations.pdf", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "fig_dimension_correlations.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_dimension_correlations")

# ===== 图 2：各类别评分均值条形图 =====
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, (dim_mean, dim_short) in enumerate(zip(TARGET_MEAN, TARGET_SHORT)):
    cat_means = df.groupby("category")[dim_mean].agg(["mean", "std"])
    axes[i].bar(cat_means.index, cat_means["mean"], yerr=cat_means["std"],
                capsize=5, color=sns.color_palette("Set2", len(cat_means)))
    axes[i].set_title(dim_short.capitalize(), fontsize=12)
    axes[i].set_xticklabels(cat_means.index, rotation=45, ha="right", fontsize=7)
    axes[i].set_ylim(1, 7)
    axes[i].set_ylabel("Rating")
plt.suptitle("Rating Distribution by Image Category", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_category_ratings.pdf", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "fig_category_ratings.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_category_ratings")

# ===== 图 3：主要实验结果对比 =====
summary = pd.read_csv("results/tables/experiment_summary.csv")

for target in TARGET_SHORT:
    target_df = summary[summary["TargetShort"] == target]
    top = target_df.nlargest(10, "Pearson_r_mean")

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(
        [f"{r['Model']} + {r['FeatureSet']}" for _, r in top.iterrows()],
        top["Pearson_r_mean"].values,
        xerr=top["Pearson_r_std"].values,
        capsize=3
    )
    ax.set_xlabel("Pearson r")
    ax.set_title(f"Top 10 Model-Feature Combinations for {target.capitalize()}")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"fig_top_models_{target}.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / f"fig_top_models_{target}.png", dpi=150, bbox_inches="tight")
    plt.close()
print("Saved: fig_top_models_*.pdf")

# ===== 图 4：消融实验结果 =====
ablation = pd.read_csv("results/tables/ablation_results.csv")

fig, axes = plt.subplots(1, 5, figsize=(24, 5))
for i, target in enumerate(TARGET_SHORT):
    ax = axes[i]
    target_ablation = ablation[ablation["Target"] == target].copy()
    target_ablation = target_ablation.sort_values("Pearson_r_mean", ascending=True)
    colors = ["red" if x == "All_traditional" else "steelblue" for x in target_ablation["Ablation"]]
    ax.barh(target_ablation["Ablation"], target_ablation["Pearson_r_mean"], color=colors)
    ax.set_xlabel("Pearson r")
    ax.set_title(target.capitalize(), fontsize=11)
    ax.set_xlim(0, 1)
plt.suptitle("Ablation Study: Feature Group Importance", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_ablation.pdf", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "fig_ablation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_ablation")

# ===== 图 5：t-SNE 特征空间可视化 =====
np.random.seed(42)
X_subset = X[:500]
cat_subset = sample_info["category"].values[:500]

pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(X_subset)

tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_pca)

fig, ax = plt.subplots(figsize=(10, 8))
for cat in np.unique(cat_subset):
    mask = cat_subset == cat
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=cat, alpha=0.7, s=30)
ax.legend(fontsize=10)
ax.set_title("t-SNE Visualization of Feature Space by Category")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_tsne.pdf", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "fig_tsne.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_tsne")

# ===== 图 6：箱线图 - 各类别评分分布 =====
fig, ax = plt.subplots(figsize=(12, 6))
df_melted = df.melt(id_vars=["image_id", "category"], value_vars=TARGET_MEAN,
                     var_name="Dimension", value_name="Rating")
df_melted["Dimension"] = df_melted["Dimension"].map(dict(zip(TARGET_MEAN, TARGET_SHORT)))
sns.boxplot(data=df_melted, x="Dimension", y="Rating", hue="category", ax=ax, palette="Set2")
ax.set_title("Rating Distributions by Dimension and Category")
ax.legend(title="Category", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_rating_boxplots.pdf", dpi=150, bbox_inches="tight")
plt.savefig(OUTPUT_DIR / "fig_rating_boxplots.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: fig_rating_boxplots")

print("\nAll visualizations complete!")
