#!/usr/bin/env python3
"""
Build deterministic theory-derived proxy targets for the full D-proxy set.
The scoring formula is taken from run_ava_experiment.py with a fixed random seed
so the targets are reproducible. These are proxy labels, not human ground truth.
"""
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "ratings" / "proxy_targets_full.csv"


def score(row):
    s = {}
    s["overall"] = np.clip(
        4 + 0.15 * (row.get("saturation_mean", 50) / 50 - 0.5) * 5
        + 0.2 * (row.get("color_harmony", 0.5) - 0.5) * 5
        + 0.15 * (row.get("symmetry", 0.5) - 0.5) * 5
        + np.random.normal(0, 0.3),
        1, 7,
    )
    s["beauty"] = np.clip(
        4 + 0.15 * (row.get("warm_color_ratio", 0.5) - 0.5) * 5
        + 0.15 * (row.get("color_entropy", 2) / 4 - 0.5) * 5
        + 0.1 * (row.get("saturation_std", 30) / 60 - 0.5) * 5
        + np.random.normal(0, 0.4),
        1, 7,
    )
    s["complexity"] = np.clip(
        4 + 0.2 * (row.get("edge_density", 0.1) - 0.05) * 10
        + 0.15 * (row.get("num_dominant_colors", 3) / 5 - 0.5) * 5
        - 0.1 * (row.get("symmetry", 0.5) - 0.5) * 5
        + np.random.normal(0, 0.5),
        1, 7,
    )
    s["order"] = np.clip(
        4 - 0.15 * (row.get("edge_density", 0.1) - 0.05) * 10
        + 0.15 * (row.get("symmetry", 0.5) - 0.5) * 5
        + 0.1 * (1 - row.get("whitespace_ratio", 0.1)) * 5
        + np.random.normal(0, 0.4),
        1, 7,
    )
    s["emotion"] = np.clip(
        4 + 0.15 * (row.get("saturation_mean", 50) / 50 - 0.5) * 5
        + 0.15 * (row.get("warm_color_ratio", 0.5) - 0.5) * 5
        + 0.1 * (row.get("lightness_contrast", 30) / 60 - 0.5) * 5
        + np.random.normal(0, 0.4),
        1, 7,
    )
    return s


def main():
    feat_df = pd.read_csv(ROOT / "data" / "features" / "traditional_features.csv")
    meta = pd.read_csv(ROOT / "data" / "processed" / "metadata.csv")
    combo = feat_df.merge(meta[["image_id", "category"]], on="image_id", how="inner", suffixes=("", "_meta"))
    print(f"Building proxy targets for {len(combo)} images across {combo['category'].nunique()} categories")

    scores = []
    for _, row in combo.iterrows():
        s = score(row)
        s["image_id"] = row["image_id"]
        s["category"] = row["category"]
        scores.append(s)

    scores_df = pd.DataFrame(scores)
    scores_df = scores_df[["image_id", "category", "overall", "beauty", "complexity", "order", "emotion"]]
    scores_df.to_csv(OUT, index=False)
    print(f"Saved {OUT}")

    for dim in ["overall", "beauty", "complexity", "order", "emotion"]:
        vals = scores_df[dim]
        print(f"  {dim:12s}: mean={vals.mean():.2f}, std={vals.std():.2f}")


if __name__ == "__main__":
    main()
