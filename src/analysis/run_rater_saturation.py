#!/usr/bin/env python3
"""
Step 11 — Run the Rater-Saturation Experiment
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
M_VALUES = [1, 3, 5, 10, 15, 20, 30]
N_BOOT = 1000
SEED = 42
SUFF_CRITERIA = {
    "rho_ref": 0.95,
    "rho_lower": 0.90,
    "model_rho_tol": 0.05,
}


def main():
    rng = np.random.RandomState(SEED)

    ratings = pd.read_csv(ROOT / "盲评问卷" / "ratings" / "real_human_ratings.csv")
    ratings["dimension_en"] = ratings["dimension"].map(DIMENSION_MAP)

    # Pivot to image x rater matrices per dimension
    matrices = {}
    ref_means = {}
    for dim in DIMS:
        sub = ratings[ratings["dimension_en"] == dim]
        mat = sub.pivot_table(index="image_id", columns="rater_id", values="rating").values
        matrices[dim] = mat
        ref_means[dim] = mat.mean(axis=1)

    # Frozen model predictions for 100 images
    with open(ROOT / "results" / "redesign" / "models" / "frozen_primary_model.pkl", "rb") as f:
        frozen = pickle.load(f)
    models = frozen["models"]
    feature_cols = frozen["feature_cols"]

    feats = pd.read_csv(ROOT / "data" / "features" / "traditional_features.csv")
    # Map matrix rows to image_ids preserving pivot order
    image_ids = ratings[ratings["dimension_en"] == DIMS[0]].pivot_table(
        index="image_id", columns="rater_id", values="rating"
    ).index.values
    df = pd.DataFrame({"image_id": image_ids}).merge(feats, on="image_id", how="inner")
    X = df[feature_cols].values.astype(np.float64)
    frozen_preds = {dim: models[dim].predict(X) if dim in models else None for dim in DIMS}

    rows = []
    for dim in DIMS:
        mat = matrices[dim]
        n_images, n_raters = mat.shape
        ref = ref_means[dim]
        for m in M_VALUES:
            rho_to_ref = []
            mad_to_ref = []
            top_bottom_stability = []
            model_rho = []
            for _ in range(N_BOOT):
                selected = rng.choice(n_raters, size=m, replace=False)
                sub_mean = mat[:, selected].mean(axis=1)
                rho_to_ref.append(stats.spearmanr(sub_mean, ref)[0])
                mad_to_ref.append(np.mean(np.abs(sub_mean - ref)))
                # Rank stability of top/bottom 10%
                k = max(1, n_images // 10)
                top_ref = set(np.argsort(ref)[:k]).union(set(np.argsort(ref)[-k:]))
                top_sub = set(np.argsort(sub_mean)[:k]).union(set(np.argsort(sub_mean)[-k:]))
                top_bottom_stability.append(len(top_ref & top_sub) / len(top_ref))
                if frozen_preds[dim] is not None:
                    model_rho.append(stats.spearmanr(frozen_preds[dim], sub_mean)[0])

            def pct(arr, p):
                return np.percentile(arr, p)

            row = {
                "dimension": dim,
                "m": m,
                "rho_to_ref_median": pct(rho_to_ref, 50),
                "rho_to_ref_p2.5": pct(rho_to_ref, 2.5),
                "rho_to_ref_p97.5": pct(rho_to_ref, 97.5),
                "mad_to_ref_median": pct(mad_to_ref, 50),
                "top_bottom_stability_median": pct(top_bottom_stability, 50),
            }
            if model_rho:
                row["model_rho_median"] = pct(model_rho, 50)
                row["model_rho_p2.5"] = pct(model_rho, 2.5)
                row["model_rho_p97.5"] = pct(model_rho, 97.5)
            rows.append(row)

    sat_df = pd.DataFrame(rows)
    sat_df.to_csv(RESULTS / "rater_saturation.csv", index=False)
    print(f"Wrote {RESULTS / 'rater_saturation.csv'}")

    # Sufficiency summary
    suff_rows = []
    for dim in DIMS:
        sub = sat_df[sat_df["dimension"] == dim]
        ok = sub[
            (sub["rho_to_ref_median"] >= SUFF_CRITERIA["rho_ref"])
            & (sub["rho_to_ref_p2.5"] >= SUFF_CRITERIA["rho_lower"])
        ]
        min_m = ok["m"].min() if len(ok) > 0 else np.nan
        suff_rows.append({"dimension": dim, "minimum_m": min_m})
    suff_df = pd.DataFrame(suff_rows)
    suff_df.to_csv(RESULTS / "rater_saturation_sufficiency.csv", index=False)
    print("Sufficiency (median rho>=0.95 & lower 95% CI>=0.90):")
    print(suff_df.to_string(index=False))

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for j, dim in enumerate(DIMS):
        ax = axes[j]
        sub = sat_df[sat_df["dimension"] == dim]
        ax.plot(sub["m"], sub["rho_to_ref_median"], marker="o", label="rho to 30-rater ref")
        ax.fill_between(sub["m"], sub["rho_to_ref_p2.5"], sub["rho_to_ref_p97.5"], alpha=0.3)
        ax.axhline(SUFF_CRITERIA["rho_ref"], color="r", linestyle="--", label="rho=0.95")
        ax.set_title(dim)
        ax.set_xlabel("number of raters")
        ax.set_ylabel("Spearman rho")
        ax.set_ylim(0, 1)
        ax.legend()
    axes[-1].axis("off")
    plt.tight_layout()
    fig.savefig(FIGURES / "rater_saturation_by_dimension.png", dpi=150)
    fig.savefig(FIGURES / "rater_saturation_by_dimension.pdf")
    print(f"Wrote {FIGURES / 'rater_saturation_by_dimension.png'}")

    print("Step 11 complete.")


if __name__ == "__main__":
    main()
