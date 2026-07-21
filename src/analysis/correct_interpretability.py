#!/usr/bin/env python3
"""
Step 15 — Correct the Interpretability Analysis
The frozen primary model is a Ridge regression pipeline. Because it is linear,
feature importance is reported via standardized regression coefficients,
not TreeSHAP values.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "redesign" / "tables"
FIGURES = ROOT / "results" / "redesign" / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

DIMS = ["overall", "beauty", "complexity", "order", "emotion"]


def main():
    with open(ROOT / "results" / "redesign" / "models" / "frozen_primary_model.pkl", "rb") as f:
        frozen = pickle.load(f)
    models = frozen["models"]
    feature_cols = frozen["feature_cols"]

    # Proxy data
    feats = pd.read_csv(ROOT / "data" / "features" / "traditional_features.csv")
    targets = pd.read_csv(ROOT / "data" / "ratings" / "proxy_targets_full.csv")
    df = feats.merge(targets, on="image_id", how="inner", suffixes=("", "_tgt"))
    if "category_tgt" in df.columns:
        df = df.drop(columns=["category_tgt"])

    X = df[feature_cols].values.astype(np.float64)

    rows = []
    for dim in DIMS:
        pipeline = models[dim]
        scaler = pipeline.named_steps["standardscaler"]
        ridge = pipeline.named_steps["ridge"]
        X_scaled = scaler.transform(X)
        coefs = ridge.coef_
        # Standardized coefficients reflect feature importance in the linear model
        for name, coef in zip(feature_cols, coefs):
            rows.append({"dimension": dim, "feature": name, "standardized_coefficient": coef})

    imp_df = pd.DataFrame(rows)
    imp_df["abs_coef"] = imp_df["standardized_coefficient"].abs()
    imp_df = imp_df.sort_values(["dimension", "abs_coef"], ascending=[True, False])
    imp_df.to_csv(RESULTS / "feature_importance_proxy.csv", index=False)
    print(f"Wrote {RESULTS / 'feature_importance_proxy.csv'}")

    # Plot top features per dimension
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    for j, dim in enumerate(DIMS):
        ax = axes[j]
        sub = imp_df[imp_df["dimension"] == dim].nlargest(10, "abs_coef")
        colors = ["C0" if c > 0 else "C1" for c in sub["standardized_coefficient"]]
        ax.barh(sub["feature"], sub["standardized_coefficient"], color=colors)
        ax.invert_yaxis()
        ax.set_title(f"{dim}: standardized Ridge coefficients")
        ax.set_xlabel("coefficient")
    axes[-1].axis("off")
    plt.tight_layout()
    fig.savefig(FIGURES / "feature_importance_proxy.png", dpi=150)
    fig.savefig(FIGURES / "feature_importance_proxy.pdf")
    print(f"Wrote {FIGURES / 'feature_importance_proxy.png'}")

    # Human-aligned feature importance is not computed because the frozen model
    # is the same model used for proxy prediction; calibration is linear and does
    # not change feature directions. Record this explicitly.
    note = """# Interpretability note

- Primary model: Ridge regression pipeline on 30 standardized handcrafted features.
- Importance metric: standardized regression coefficients (not SHAP values).
- Figures labeled "SHAP" in the previous manuscript must be relabeled "standardized coefficient" or recomputed with a documented SHAP explainer if the model were non-linear.
- The direction of each coefficient (positive/negative) is preserved in the bar plot.
- Causal interpretation is avoided: these are associative feature weights, not evidence that changing a feature will cause a change in human preference.
"""
    (RESULTS / "feature_importance_note.md").write_text(note, encoding="utf-8")

    print("Step 15 complete.")


if __name__ == "__main__":
    main()
