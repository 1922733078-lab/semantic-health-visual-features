#!/usr/bin/env python3
"""
Step 12 — Run the Rated-Image Saturation Experiment
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
N_VALUES = [20, 40, 60, 80, 100]
N_BOOT = 1000
SEED = 42


def main():
    rng = np.random.RandomState(SEED)

    ratings = pd.read_csv(ROOT / "盲评问卷" / "ratings" / "real_human_ratings.csv")
    ratings["dimension_en"] = ratings["dimension"].map(DIMENSION_MAP)

    ref_means = {}
    categories = {}
    for dim in DIMS:
        sub = ratings[ratings["dimension_en"] == dim]
        means = sub.groupby("image_id").agg({"rating": "mean", "category": "first"})
        ref_means[dim] = means["rating"].values
        categories[dim] = means["category"].values
        if dim == DIMS[0]:
            image_ids = means.index.values

    # Frozen predictions
    with open(ROOT / "results" / "redesign" / "models" / "frozen_primary_model.pkl", "rb") as f:
        frozen = pickle.load(f)
    models = frozen["models"]
    feature_cols = frozen["feature_cols"]
    feats = pd.read_csv(ROOT / "data" / "features" / "traditional_features.csv")
    df = pd.DataFrame({"image_id": image_ids}).merge(feats, on="image_id", how="inner")
    X = df[feature_cols].values.astype(np.float64)
    frozen_preds = {dim: models[dim].predict(X) if dim in models else None for dim in DIMS}

    rows = []
    for dim in DIMS:
        ref = ref_means[dim]
        cats = categories[dim]
        pred = frozen_preds[dim]
        if pred is None:
            continue
        full_rho = stats.spearmanr(pred, ref)[0]
        for n in N_VALUES:
            rho_boot = []
            for _ in range(N_BOOT):
                # stratified sample
                selected = []
                for cat in np.unique(cats):
                    cat_idx = np.where(cats == cat)[0]
                    n_per_cat = n // len(np.unique(cats))
                    selected.extend(rng.choice(cat_idx, size=n_per_cat, replace=False))
                selected = np.array(selected)
                rho_boot.append(stats.spearmanr(pred[selected], ref[selected])[0])
            rho_boot = np.array(rho_boot)
            rows.append({
                "dimension": dim,
                "n": n,
                "full_rho": full_rho,
                "rho_median": np.median(rho_boot),
                "rho_bias": np.median(rho_boot) - full_rho,
                "rho_std": np.std(rho_boot),
                "rho_p2.5": np.percentile(rho_boot, 2.5),
                "rho_p97.5": np.percentile(rho_boot, 97.5),
                "ci_width": np.percentile(rho_boot, 97.5) - np.percentile(rho_boot, 2.5),
            })

    sat_df = pd.DataFrame(rows)
    sat_df.to_csv(RESULTS / "image_saturation.csv", index=False)
    print(f"Wrote {RESULTS / 'image_saturation.csv'}")
    print(sat_df[["dimension", "n", "rho_median", "rho_bias", "ci_width"]].to_string(index=False))

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    plot_dims = [d for d in DIMS if d in sat_df["dimension"].unique()]
    for j, dim in enumerate(plot_dims):
        ax = axes[j]
        sub = sat_df[sat_df["dimension"] == dim]
        ax.plot(sub["n"], sub["rho_median"], marker="o")
        ax.fill_between(sub["n"], sub["rho_p2.5"], sub["rho_p97.5"], alpha=0.3)
        ax.axhline(sub["full_rho"].iloc[0], color="r", linestyle="--", label="full 100-image")
        ax.set_title(dim)
        ax.set_xlabel("number of images")
        ax.set_ylabel("Spearman rho")
        ax.legend()
    axes[-1].axis("off")
    plt.tight_layout()
    fig.savefig(FIGURES / "image_saturation_by_dimension.png", dpi=150)
    fig.savefig(FIGURES / "image_saturation_by_dimension.pdf")
    print(f"Wrote {FIGURES / 'image_saturation_by_dimension.png'}")

    print("Step 12 complete.")


if __name__ == "__main__":
    main()
