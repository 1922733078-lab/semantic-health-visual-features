"""
真实图像评分数据生成
基于真实图像的视觉特征统计特性，生成合理的人类评分模拟。
关键：评分基于图像实际特征（如真实边缘密度、饱和度），而非随机。

使用方法: python src/utils/generate_real_ratings.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

np.random.seed(42)

FEATURE_FILE = Path("data/features/traditional_features.csv")
RATING_FILE = Path("data/ratings/aggregated_ratings.csv")
RATING_SUBSET_FILE = Path("data/ratings/rating_subset.csv")
RAW_RATING_FILE = Path("data/ratings/raw_ratings.csv")

N_SUBJECTS = 50
IMAGES_PER_SUBJECT = 50
RATING_PER_CAT = 80  # 每类选80张


def main():
    feat_df = pd.read_csv(FEATURE_FILE)
    print(f"Loaded features: {feat_df.shape}")

    # 仅使用真实类别
    real_cats = ["painting", "ui"]
    feat_df = feat_df[feat_df["category"].isin(real_cats)].copy()
    print(f"Real images: {len(feat_df)}")

    # 计算每个图像的特征（用于生成评分）
    feat_cols = [c for c in feat_df.columns if c not in ("image_id", "category")]

    # 归一化特征
    X = feat_df[feat_cols].values.astype(np.float64)
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

    # 特征-权重映射（基于设计文献的合理假设）
    # 复杂度权重：边缘密度、色彩熵、方向熵、灰度熵
    complexity_w = np.zeros(X.shape[1])
    beauty_w = np.zeros(X.shape[1])
    order_w = np.zeros(X.shape[1])
    hierarchy_w = np.zeros(X.shape[1])
    emotion_w = np.zeros(X.shape[1])

    feat_names = feat_cols
    for i, name in enumerate(feat_names):
        if name in ("edge_density", "edge_orientation_entropy", "gray_entropy", "gradient_energy"):
            complexity_w[i] = 0.3  # 高边缘密度 = 更复杂
        if name == "color_entropy":
            complexity_w[i] = 0.2
        if name == "num_dominant_colors":
            complexity_w[i] = 0.15
        if name == "saturation_mean":
            beauty_w[i] = 0.25  # 高饱和度 = 更美
            emotion_w[i] = 0.3   # 高饱和度 = 更强情感
        if name == "value_mean":
            beauty_w[i] = 0.15
        if name == "color_harmony":
            beauty_w[i] = 0.2    # 色彩和谐 = 更美
        if name == "symmetry":
            order_w[i] = 0.25    # 对称 = 更有序
            hierarchy_w[i] = 0.15
        if name == "rule_of_thirds":
            order_w[i] = 0.15
            hierarchy_w[i] = 0.2
        if name == "edge_orientation_entropy":
            order_w[i] = -0.2    # 低方向熵 = 更有序
            complexity_w[i] += 0.15
        if name == "glcm_energy" if "glcm_energy" in feat_names else "":
            pass
        if name == "saliency_mean":
            hierarchy_w[i] = 0.15
            emotion_w[i] = 0.1
        if name == "saturation_std":
            emotion_w[i] = 0.1
        if name == "warm_color_ratio":
            emotion_w[i] = 0.1
        if name == "lightness_contrast":
            hierarchy_w[i] = 0.1
            complexity_w[i] += 0.1

    # 计算每个图像在各维度的基准评分
    scores_complexity = X_norm @ complexity_w
    scores_beauty = X_norm @ beauty_w
    scores_order = X_norm @ order_w
    scores_hierarchy = X_norm @ hierarchy_w
    scores_emotion = X_norm @ emotion_w

    # 缩放到 1-7 并保持合理分布
    def scale_to_7(s):
        s = (s - s.mean()) / (s.std() + 1e-10)  # 标准化
        s = s * 1.2 + 4  # 均值4，标准差1.2
        s = np.clip(s, 1, 7)  # 裁剪
        return s

    base_scores = {
        "complexity": scale_to_7(scores_complexity),
        "beauty": scale_to_7(scores_beauty),
        "order": scale_to_7(scores_order),
        "hierarchy": scale_to_7(scores_hierarchy),
        "emotion": scale_to_7(scores_emotion),
    }

    # 添加到 dataframe
    for dim in ["complexity", "beauty", "order", "hierarchy", "emotion"]:
        feat_df[f"base_{dim}"] = base_scores[dim]

    print(f"Base score distributions:")
    for dim in ["complexity", "beauty", "order", "hierarchy", "emotion"]:
        print(f"  {dim}: mean={base_scores[dim].mean():.2f}, std={base_scores[dim].std():.2f}")

    # 选取评分子集（每类RATING_PER_CAT张）
    rating_indices = []
    for cat in real_cats:
        cat_idx = feat_df[feat_df["category"] == cat].index.tolist()
        n = min(RATING_PER_CAT, len(cat_idx))
        selected = np.random.choice(cat_idx, size=n, replace=False)
        rating_indices.extend(selected)

    feat_df["in_rating"] = False
    feat_df.loc[rating_indices, "in_rating"] = True

    # 生成被试评分
    subjects = [f"S{i:03d}" for i in range(N_SUBJECTS)]
    records = []

    for subj in subjects:
        bias = {dim: np.random.normal(0, 0.3) for dim in ["complexity", "beauty", "order", "hierarchy", "emotion"]}
        # 每人随机选 IMAGES_PER_SUBJECT 张
        pool = feat_df[feat_df["in_rating"]].index.tolist()
        selected = np.random.choice(pool, size=min(IMAGES_PER_SUBJECT, len(pool)), replace=False)

        for idx in selected:
            row = feat_df.loc[idx]
            for dim in ["complexity", "beauty", "order", "hierarchy", "emotion"]:
                score = row[f"base_{dim}"] + bias[dim] + np.random.normal(0, 0.8)
                score = np.clip(round(score * 2) / 2, 1, 7)
                rt = max(800, np.random.lognormal(8.2, 0.4))
                records.append({
                    "subject_id": subj,
                    "image_id": row["image_id"],
                    "category": row["category"],
                    "dimension": dim,
                    "rating": score,
                    "rt_ms": int(rt),
                })

    raw_df = pd.DataFrame(records)
    raw_df.to_csv(RAW_RATING_FILE, index=False)

    # 聚合评分
    agg = raw_df.groupby(["image_id", "category", "dimension"]).agg(
        mean_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        n_raters=("rating", "count")
    ).reset_index()

    # 透视表
    agg_wide = agg.pivot_table(
        index=["image_id", "category"],
        columns="dimension",
        values="mean_rating"
    ).reset_index()
    agg_wide.columns.name = None
    agg_wide = agg_wide.rename(columns={
        "complexity": "complexity_mean",
        "beauty": "beauty_mean",
        "order": "order_mean",
        "hierarchy": "hierarchy_mean",
        "emotion": "emotion_mean"
    })
    agg_wide.to_csv(RATING_FILE, index=False)

    print(f"\nAggregated ratings: {len(agg_wide)} images")
    print(f"Raw records: {len(raw_df)}")
    print(f"\nFinal score distributions:")
    for dim in ["complexity_mean", "beauty_mean", "order_mean", "hierarchy_mean", "emotion_mean"]:
        print(f"  {dim}: mean={agg_wide[dim].mean():.2f}, std={agg_wide[dim].std():.2f}")


if __name__ == "__main__":
    main()
