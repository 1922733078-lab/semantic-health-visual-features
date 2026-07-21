"""
重建元数据 + 构建人评分子集
将所有已处理图像（真实+合成）统一生成新的 metadata.csv，
并选取人评分子集（rating_subset.csv）。

使用方法: python src/utils/rebuild_metadata_and_rating_set.py
"""

import random
import pandas as pd
from pathlib import Path

random.seed(42)

PROCESSED_DIR = Path("data/processed")
RATINGS_DIR = Path("data/ratings")
RATINGS_DIR.mkdir(parents=True, exist_ok=True)

PER_CATEGORY_RATING = 120  # 每类选120张用于评分


def main():
    # ===== 1. 扫描所有已处理图像 =====
    categories = ["painting", "ui"]  # 真实数据类别
    # 也保留其他合成类别（用于数据增强/对比）
    all_categories = []
    for d in PROCESSED_DIR.iterdir():
        if d.is_dir() and d.name not in ("metadata.csv",):
            images = list(d.glob("*.jpg")) + list(d.glob("*.png"))
            if images:
                all_categories.append((d.name, images))

    print(f"Found {len(all_categories)} categories with images")
    for cat_name, imgs in all_categories:
        print(f"  {cat_name}: {len(imgs)} images")

    # ===== 2. 构建完整元数据 =====
    metadata = []
    for cat_name, images in all_categories:
        for img_path in sorted(images):
            image_id = img_path.stem
            metadata.append({
                "image_id": image_id,
                "category": cat_name,
                "standardized_path": str(img_path),
                "data_source": "real" if cat_name in ("painting", "ui") else "synthetic",
            })

    df = pd.DataFrame(metadata)
    df.to_csv(PROCESSED_DIR / "metadata.csv", index=False)
    print(f"\nTotal images: {len(df)}")
    print(f"By category:\n{df['category'].value_counts()}")

    # ===== 3. 选取人评分子集 =====
    rating_samples = []
    for cat_name, images in all_categories:
        cat_df = df[df["category"] == cat_name]
        n_sample = min(PER_CATEGORY_RATING, len(cat_df))
        sampled = cat_df.sample(n=n_sample, random_state=42)
        rating_samples.append(sampled)

    rating_df = pd.concat(rating_samples)
    rating_df["in_human_subset"] = True
    df["in_human_subset"] = df["image_id"].isin(rating_df["image_id"])
    df.to_csv(PROCESSED_DIR / "metadata.csv", index=False)

    # 保存评分子集
    rating_df[["image_id", "category", "standardized_path"]].to_csv(
        RATINGS_DIR / "rating_subset.csv", index=False
    )
    print(f"\nRating subset: {len(rating_df)} images")
    print(f"  {rating_df['category'].value_counts().to_dict()}")

    return df, rating_df


if __name__ == "__main__":
    main()
