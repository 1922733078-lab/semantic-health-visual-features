#!/usr/bin/env python3
"""
Step 13 — Optional Small-Sample Calibration
Linear mapping from frozen proxy predictions to human mean ratings.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge

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
CAL_SIZES = [5, 10, 20, 40, 60]
SEED = 42
N_REPS = 100


def main():
    rng = np.random.RandomState(SEED)

    ratings = pd.read_csv(ROOT / "盲评问卷" / "ratings" / "real_human_ratings.csv")
    ratings["dimension_en"] = ratings["dimension"].map(DIMENSION_MAP)
    human_means = ratings.groupby(["image_id", "category", "dimension_en"])["rating"].mean().reset_index()
    human_wide = human_means.pivot_table(index=["image_id", "category"], columns="dimension_en", values="rating").reset_index()
    human_wide.columns.name = None

    with open(ROOT / "results" / "redesign" / "models" / "frozen_primary_model.pkl", "rb") as f:
        frozen = pickle.load(f)
    models = frozen["models"]
    feature_cols = frozen["feature_cols"]

    feats = pd.read_csv(ROOT / "data" / "features" / "traditional_features.csv")
    df = human_wide.merge(feats, on="image_id", how="inner", suffixes=("", "_feat"))
    df = df.drop(columns=[c for c in df.columns if c.endswith("_feat")])
    X = df[feature_cols].values.astype(np.float64)
    preds = {dim: models[dim].predict(X) if dim in models else None for dim in DIMS}

    rows = []
    for dim in DIMS:
        if preds[dim] is None:
            continue
        y = df[dim].values.astype(np.float64)
        pred = preds[dim]
        full_rho = stats.spearmanr(pred, y)[0]

        for cal_size in CAL_SIZES:
            test_rhos = []
            for _ in range(N_REPS):
                # stratified sample
                selected = []
                for cat in df["category"].unique():
                    cat_idx = df[df["category"] == cat].index.values
                    n_per_cat = cal_size // len(df["category"].unique())
                    selected.extend(rng.choice(cat_idx, size=n_per_cat, replace=False))
                selected = np.array(selected)
                test_idx = np.setdiff1d(np.arange(len(df)), selected)

                # linear calibration
                cal = Ridge(alpha=1.0)
                cal.fit(pred[selected].reshape(-1, 1), y[selected])
                pred_cal = cal.predict(pred[test_idx].reshape(-1, 1))
                test_rhos.append(stats.spearmanr(pred_cal, y[test_idx])[0])

            rows.append({
                "dimension": dim,
                "cal_size": cal_size,
                "frozen_rho": full_rho,
                "calibrated_rho_median": np.median(test_rhos),
                "calibrated_rho_std": np.std(test_rhos),
                "gain": np.median(test_rhos) - full_rho,
            })

    cal_df = pd.DataFrame(rows)
    cal_df.to_csv(RESULTS / "calibration_label_efficiency.csv", index=False)
    print(f"Wrote {RESULTS / 'calibration_label_efficiency.csv'}")
    print(cal_df.to_string(index=False))

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    plot_dims = [d for d in DIMS if d in cal_df["dimension"].unique()]
    for j, dim in enumerate(plot_dims):
        ax = axes[j]
        sub = cal_df[cal_df["dimension"] == dim]
        ax.axhline(sub["frozen_rho"].iloc[0], color="k", linestyle="--", label="frozen")
        ax.plot(sub["cal_size"], sub["calibrated_rho_median"], marker="o", label="calibrated")
        ax.set_title(dim)
        ax.set_xlabel("calibration images")
        ax.set_ylabel("Spearman rho")
        ax.legend()
    axes[-1].axis("off")
    plt.tight_layout()
    fig.savefig(FIGURES / "calibration_curve.png", dpi=150)
    fig.savefig(FIGURES / "calibration_curve.pdf")
    print(f"Wrote {FIGURES / 'calibration_curve.png'}")

    print("Step 13 complete.")


if __name__ == "__main__":
    main()
