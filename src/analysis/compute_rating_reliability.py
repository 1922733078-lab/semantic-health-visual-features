#!/usr/bin/env python3
"""
Step 9 — Compute Human-Rating Reliability (ICC)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "redesign" / "tables"
FIGURES = ROOT / "results" / "redesign" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

DIMENSION_MAP = {
    "视觉复杂度": "complexity",
    "美感吸引力": "beauty",
    "秩序感": "order",
    "视觉层级清晰度": "hierarchy",
    "情感强度": "emotion",
}
DIMS = ["complexity", "beauty", "order", "hierarchy", "emotion"]


def compute_icc(df_long, dimension):
    sub = df_long[df_long["dimension_en"] == dimension].copy()
    sub["rater_num"] = sub["rater_id"].astype("category").cat.codes
    sub["image_num"] = sub["image_id"].astype("category").cat.codes
    icc = pg.intraclass_corr(data=sub, targets="image_num", raters="rater_num", ratings="rating")
    return icc


def main():
    ratings = pd.read_csv(ROOT / "盲评问卷" / "ratings" / "real_human_ratings.csv")
    ratings["dimension_en"] = ratings["dimension"].map(DIMENSION_MAP)

    rows_dim = []
    for dim in DIMS:
        icc = compute_icc(ratings, dim)
        # ICC(2,1) absolute agreement ~ ICC(A,1) in pingouin
        # ICC(2,k) absolute agreement ~ ICC(A,k) in pingouin
        icc_21 = icc[icc["Type"] == "ICC(A,1)"].iloc[0]
        icc_2k = icc[icc["Type"] == "ICC(A,k)"].iloc[0]
        vals = ratings[ratings["dimension_en"] == dim]["rating"].values
        rows_dim.append({
            "dimension": dim,
            "icc2_1": icc_21["ICC"],
            "icc2_1_ci_low": icc_21["CI95"][0],
            "icc2_1_ci_high": icc_21["CI95"][1],
            "icc2_1_f": icc_21["F"],
            "icc2_1_df1": icc_21["df1"],
            "icc2_1_df2": icc_21["df2"],
            "icc2_1_p": icc_21["pval"],
            "icc2_k": icc_2k["ICC"],
            "icc2_k_ci_low": icc_2k["CI95"][0],
            "icc2_k_ci_high": icc_2k["CI95"][1],
            "mean_rating": np.mean(vals),
            "std_rating": np.std(vals),
            "min_rating": np.min(vals),
            "max_rating": np.max(vals),
            "n_images": ratings[ratings["dimension_en"] == dim]["image_id"].nunique(),
            "n_raters": ratings[ratings["dimension_en"] == dim]["rater_id"].nunique(),
        })

    icc_dim_df = pd.DataFrame(rows_dim)
    icc_dim_df.to_csv(RESULTS / "icc_by_dimension.csv", index=False)
    print(f"Wrote {RESULTS / 'icc_by_dimension.csv'}")
    print(icc_dim_df[["dimension", "icc2_1", "icc2_k", "mean_rating", "std_rating"]].to_string(index=False))

    # Per-category ICC(2,k)
    rows_cat = []
    for dim in DIMS:
        for cat in ratings["category"].unique():
            sub = ratings[(ratings["dimension_en"] == dim) & (ratings["category"] == cat)].copy()
            if sub["image_id"].nunique() < 2 or sub["rater_id"].nunique() < 2:
                continue
            sub["rater_num"] = sub["rater_id"].astype("category").cat.codes
            sub["image_num"] = sub["image_id"].astype("category").cat.codes
            try:
                icc = pg.intraclass_corr(data=sub, targets="image_num", raters="rater_num", ratings="rating")
                icc_2k = icc[icc["Type"] == "ICC(A,k)"].iloc[0]["ICC"]
            except Exception:
                icc_2k = np.nan
            rows_cat.append({"dimension": dim, "category": cat, "icc2_k": icc_2k})

    icc_cat_df = pd.DataFrame(rows_cat)
    icc_cat_df.to_csv(RESULTS / "icc_by_category.csv", index=False)
    print(f"Wrote {RESULTS / 'icc_by_category.csv'}")

    # Rating distribution figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, dim in zip(axes, DIMS):
        vals = ratings[ratings["dimension_en"] == dim]["rating"].values
        ax.hist(vals, bins=np.arange(0.5, 8.5, 1), edgecolor="black")
        ax.set_title(dim)
        ax.set_xlabel("rating")
        ax.set_ylabel("count")
    plt.tight_layout()
    fig.savefig(FIGURES / "rating_distribution_by_dimension.png", dpi=150)
    fig.savefig(FIGURES / "rating_distribution_by_dimension.pdf")
    print(f"Wrote {FIGURES / 'rating_distribution_by_dimension.png'}")

    print("Step 9 complete.")


if __name__ == "__main__":
    main()
