"""
主观评分实验模拟

模拟人类被试对图像的 5 维度评分：
  complexity, beauty, order, hierarchy, emotion

流程：
1. 从每类选取 ~150 张进入评分子集
2. 模拟 50 名被试，每人随机评 50 张（每张被评 3-5 次）
3. 生成 raw_ratings.csv, 清洗后生成 aggregated_ratings.csv

评分基准基于图像类别特征 + 个体差异噪声 + 随机误差。

使用方法: python src/utils/simulate_ratings.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import random

random.seed(42)
np.random.seed(42)

# ===== 配置 =====
RATINGS_DIR = Path("data/ratings")
RATINGS_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = Path("data/processed")
METADATA_CSV = PROCESSED_DIR / "metadata.csv"

N_SUBJECTS = 50
IMAGES_PER_SUBJECT = 50
RATING_SUBSET_PER_CAT = 150

# 每个类别在各维度上的基准均值（基于设计文献的先验）
CATEGORY_PROFILES = {
    "painting": {
        "complexity": (4.5, 1.2), "beauty": (4.2, 1.1), "order": (3.8, 1.0),
        "hierarchy": (3.9, 1.0), "emotion": (4.5, 1.2),
        "color_entropy": 0.7, "edge_density": 0.6, "symmetry": 0.3,
    },
    "poster": {
        "complexity": (4.0, 1.0), "beauty": (4.8, 0.9), "order": (5.0, 0.8),
        "hierarchy": (5.2, 0.8), "emotion": (4.8, 1.0),
        "color_entropy": 0.8, "edge_density": 0.4, "symmetry": 0.6,
    },
    "ui": {
        "complexity": (3.2, 0.9), "beauty": (4.0, 1.0), "order": (5.5, 0.7),
        "hierarchy": (5.5, 0.7), "emotion": (3.0, 0.9),
        "color_entropy": 0.3, "edge_density": 0.2, "symmetry": 0.75,
    },
    "packaging": {
        "complexity": (3.8, 1.0), "beauty": (4.0, 1.0), "order": (4.2, 0.9),
        "hierarchy": (4.0, 0.9), "emotion": (3.8, 1.0),
        "color_entropy": 0.6, "edge_density": 0.45, "symmetry": 0.5,
    },
    "banner": {
        "complexity": (3.5, 1.0), "beauty": (4.5, 1.0), "order": (4.8, 0.8),
        "hierarchy": (5.0, 0.8), "emotion": (4.2, 1.0),
        "color_entropy": 0.7, "edge_density": 0.3, "symmetry": 0.65,
    },
}


def select_rating_subset(metadata_df):
    """从每类中随机选取 RATING_SUBSET_PER_CAT 张"""
    subsets = []
    for cat in metadata_df["category"].unique():
        cat_df = metadata_df[metadata_df["category"] == cat]
        n_sample = min(RATING_SUBSET_PER_CAT, len(cat_df))
        sampled = cat_df.sample(n=n_sample, random_state=42)
        subsets.append(sampled)

    subset_df = pd.concat(subsets)
    metadata_df["in_human_subset"] = metadata_df["image_id"].isin(subset_df["image_id"])
    metadata_df.to_csv(METADATA_CSV, index=False)
    subset_df[["image_id", "category", "standardized_path"]].to_csv(
        RATINGS_DIR / "rating_subset.csv", index=False
    )
    print(f"Rating subset: {len(subset_df)} images")
    return subset_df


def generate_individual_bias():
    """生成被试个体偏差（每个被试对每个维度有轻微不同的评分倾向）"""
    biases = {}
    for dim in ["complexity", "beauty", "order", "hierarchy", "emotion"]:
        # 个体偏差 ~ N(0, 0.3)
        biases[dim] = np.random.normal(0, 0.3)
    return biases


def generate_raw_ratings(subset_df):
    """生成模拟评分数据"""
    raw_rows = []
    image_ids = subset_df["image_id"].tolist()

    for subject_idx in range(N_SUBJECTS):
        subject_id = f"subject_{subject_idx:03d}"
        bias = generate_individual_bias()

        # 每名被试随机选取 IMAGES_PER_SUBJECT 张
        selected = random.sample(image_ids, min(IMAGES_PER_SUBJECT, len(image_ids)))

        for img_id in selected:
            cat = subset_df[subset_df["image_id"] == img_id]["category"].values[0]
            profile = CATEGORY_PROFILES[cat]
            noise_seed = hash(f"{subject_idx}_{img_id}") % 10000
            np.random.seed(noise_seed)

            ratings = {}
            for dim in ["complexity", "beauty", "order", "hierarchy", "emotion"]:
                mean, std = profile[dim]
                # 基准 + 个体偏差 + 随机误差
                rating = mean + bias[dim] + np.random.normal(0, std * 0.5)
                rating = np.clip(round(rating * 2) / 2, 1, 7)  # 0.5 步长，范围 1-7
                ratings[dim] = rating

            # 反应时长（模拟），正常范围 1500-8000ms
            rt = max(800, np.random.lognormal(8.2, 0.4))

            raw_rows.append({
                "subject_id": subject_id,
                "image_id": img_id,
                "category": cat,
                "complexity": ratings["complexity"],
                "beauty": ratings["beauty"],
                "order": ratings["order"],
                "hierarchy": ratings["hierarchy"],
                "emotion": ratings["emotion"],
                "reaction_time": int(rt),
            })

        if (subject_idx + 1) % 10 == 0:
            print(f"  Generated ratings for {subject_idx + 1}/{N_SUBJECTS} subjects")

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(RATINGS_DIR / "raw_ratings.csv", index=False)
    print(f"Raw ratings: {len(raw_df)} rows, {raw_df['subject_id'].nunique()} subjects")
    return raw_df


def clean_and_aggregate(raw_df):
    """数据清洗与聚合"""
    print(f"\nData cleaning:")
    print(f"  Raw data: {len(raw_df)} rows")

    # 排除反应时 < 1000ms 的被试
    subject_rt = raw_df.groupby("subject_id")["reaction_time"].mean()
    fast_subjects = subject_rt[subject_rt < 1000].index
    print(f"  Excluded fast responders: {len(fast_subjects)}")

    # 排除评分方差极低的被试
    subject_std = raw_df.groupby("subject_id")["complexity"].std()
    low_var = subject_std[subject_std < 0.3].index
    print(f"  Excluded low-variance subjects: {len(low_var)}")

    excluded = set(fast_subjects) | set(low_var)
    clean_df = raw_df[~raw_df["subject_id"].isin(excluded)]
    print(f"  Clean data: {len(clean_df)} rows, {clean_df['subject_id'].nunique()} subjects")

    # 聚合评分
    agg = clean_df.groupby("image_id").agg(
        complexity_mean=("complexity", "mean"),
        complexity_std=("complexity", "std"),
        beauty_mean=("beauty", "mean"),
        beauty_std=("beauty", "std"),
        order_mean=("order", "mean"),
        order_std=("order", "std"),
        hierarchy_mean=("hierarchy", "mean"),
        hierarchy_std=("hierarchy", "std"),
        emotion_mean=("emotion", "mean"),
        emotion_std=("emotion", "std"),
        n_raters=("subject_id", "count"),
    ).reset_index()

    agg.to_csv(RATINGS_DIR / "aggregated_ratings.csv", index=False)
    print(f"\nAggregated ratings: {len(agg)} images")
    print(f"Saved to data/ratings/aggregated_ratings.csv")

    # 打印各维度统计
    for dim in ["complexity_mean", "beauty_mean", "order_mean", "hierarchy_mean", "emotion_mean"]:
        print(f"  {dim}: mean={agg[dim].mean():.2f}, std={agg[dim].std():.2f}")

    return agg


def main():
    print("Loading metadata...")
    metadata_df = pd.read_csv(METADATA_CSV)

    print("\nStep 1: Select rating subset...")
    subset_df = select_rating_subset(metadata_df)

    print("\nStep 2: Generate simulated ratings...")
    raw_df = generate_raw_ratings(subset_df)

    print("\nStep 3: Clean and aggregate ratings...")
    agg_df = clean_and_aggregate(raw_df)

    print("\nAll rating data generated successfully!")


if __name__ == "__main__":
    main()
