#!/usr/bin/env python3
"""
Step 10 — Run Frozen External Validation on D-Human
Apply the pre-fitted primary model to the 100 human-rated images.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

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
HUMAN_DIMS = ["complexity", "beauty", "order", "hierarchy", "emotion"]
FROZEN_DIMS = ["beauty", "complexity", "order", "emotion"]  # overlapping proxy targets


def bootstrap_corr(x, y, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(x)
    rho_boot = []
    tau_boot = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        rho_boot.append(stats.spearmanr(x[idx], y[idx])[0])
        tau_boot.append(stats.kendalltau(x[idx], y[idx])[0])
    rho_boot = np.array(rho_boot)
    tau_boot = np.array(tau_boot)
    return (
        np.percentile(rho_boot, [2.5, 97.5]),
        np.percentile(tau_boot, [2.5, 97.5]),
    )


def main():
    # Load frozen model
    with open(ROOT / "results" / "redesign" / "models" / "frozen_primary_model.pkl", "rb") as f:
        frozen = pd.read_pickle(f)
    models = frozen["models"]
    feature_cols = frozen["feature_cols"]
    print(f"Loaded frozen model trained on {frozen['config']['d_proxy_clean_size']} clean proxy images")

    # Load human ratings
    ratings = pd.read_csv(ROOT / "盲评问卷" / "ratings" / "real_human_ratings.csv")
    ratings["dimension_en"] = ratings["dimension"].map(DIMENSION_MAP)
    human_means = ratings.groupby(["image_id", "category", "dimension_en"])["rating"].mean().reset_index()
    human_wide = human_means.pivot_table(index=["image_id", "category"], columns="dimension_en", values="rating").reset_index()
    human_wide.columns.name = None

    # Load features
    feats = pd.read_csv(ROOT / "data" / "features" / "traditional_features.csv")
    df = human_wide.merge(feats, on="image_id", how="inner")
    print(f"Frozen validation images: {len(df)}")

    X = df[feature_cols].values.astype(np.float64)

    # Predict with frozen model for overlapping proxy targets
    preds = np.zeros((len(df), len(FROZEN_DIMS)))
    for j, dim in enumerate(FROZEN_DIMS):
        preds[:, j] = models[dim].predict(X)

    # Evaluate
    rows = []
    pvals = []
    for j, dim in enumerate(FROZEN_DIMS):
        y_true = df[dim].values.astype(np.float64)
        y_pred = preds[:, j]
        rho, rho_p = stats.spearmanr(y_true, y_pred)
        tau, tau_p = stats.kendalltau(y_true, y_pred)
        rho_ci, tau_ci = bootstrap_corr(y_true, y_pred)
        rows.append({
            "dimension": dim,
            "n_images": len(df),
            "n_raters": 30,
            "spearman_rho": rho,
            "spearman_p": rho_p,
            "rho_ci_low": rho_ci[0],
            "rho_ci_high": rho_ci[1],
            "kendall_tau": tau,
            "tau_ci_low": tau_ci[0],
            "tau_ci_high": tau_ci[1],
        })
        pvals.append(rho_p)

    # Multiple testing correction (Benjamini-Hochberg)
    _, pvals_fdr, _, _ = multipletests(pvals, method="fdr_bh")
    for row, pfdr in zip(rows, pvals_fdr):
        row["spearman_p_fdr"] = pfdr

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS / "frozen_human_validation.csv", index=False)
    print(f"Wrote {RESULTS / 'frozen_human_validation.csv'}")
    print(results_df[["dimension", "spearman_rho", "rho_ci_low", "rho_ci_high", "spearman_p", "spearman_p_fdr"]].to_string(index=False))

    # Prediction-vs-human scatter plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for j, dim in enumerate(FROZEN_DIMS):
        ax = axes[j]
        y_true = df[dim].values
        y_pred = preds[:, j]
        ax.scatter(y_true, y_pred, alpha=0.6)
        ax.plot([1, 7], [1, 7], "r--", lw=1)
        ax.set_xlabel("human mean rating")
        ax.set_ylabel("frozen proxy prediction")
        ax.set_title(f"{dim}: rho={rows[j]['spearman_rho']:.3f}")
    axes[-1].axis("off")
    plt.tight_layout()
    fig.savefig(FIGURES / "prediction_vs_human_mean.png", dpi=150)
    fig.savefig(FIGURES / "prediction_vs_human_mean.pdf")
    print(f"Wrote {FIGURES / 'prediction_vs_human_mean.png'}")

    # Save human means for later steps
    human_wide.to_csv(RESULTS / "human_mean_ratings.csv", index=False)
    print(f"Wrote {RESULTS / 'human_mean_ratings.csv'}")

    print("Step 10 complete.")


if __name__ == "__main__":
    main()
