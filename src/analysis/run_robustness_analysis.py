#!/usr/bin/env python3
"""
Step 16 — Add Robustness and Boundary Analyses
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict

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


def main():
    with open(ROOT / "results" / "redesign" / "models" / "frozen_primary_model.pkl", "rb") as f:
        frozen = pickle.load(f)
    models = frozen["models"]
    feature_cols = frozen["feature_cols"]

    ratings = pd.read_csv(ROOT / "盲评问卷" / "ratings" / "real_human_ratings.csv")
    ratings["dimension_en"] = ratings["dimension"].map(DIMENSION_MAP)
    human_means = ratings.groupby(["image_id", "category", "dimension_en"])["rating"].mean().reset_index()
    human_wide = human_means.pivot_table(index=["image_id", "category"], columns="dimension_en", values="rating").reset_index()
    human_wide.columns.name = None

    feats = pd.read_csv(ROOT / "data" / "features" / "traditional_features.csv")
    df = human_wide.merge(feats, on="image_id", how="inner", suffixes=("", "_feat"))
    df = df.drop(columns=[c for c in df.columns if c.endswith("_feat")])

    X = df[feature_cols].values.astype(np.float64)
    preds = {dim: models[dim].predict(X) if dim in models else None for dim in HUMAN_DIMS}

    # Per-category frozen validation
    rows = []
    for dim in ["complexity", "beauty", "order", "emotion"]:
        for cat in df["category"].unique():
            mask = df["category"] == cat
            if mask.sum() < 3:
                continue
            rho, p = stats.spearmanr(preds[dim][mask], df[dim][mask])
            rows.append({"dimension": dim, "category": cat, "n": mask.sum(), "spearman_rho": rho, "p": p})
    per_cat_df = pd.DataFrame(rows)
    per_cat_df.to_csv(RESULTS / "robustness_per_category.csv", index=False)
    print(f"Wrote {RESULTS / 'robustness_per_category.csv'}")
    print(per_cat_df.to_string(index=False))

    # Ablation: without OCR/layout features
    ocr_cols = ["text_coverage", "text_block_count", "font_size_cv", "has_text"]
    non_ocr_cols = [c for c in feature_cols if c not in ocr_cols]
    rows = []
    for dim in ["complexity", "beauty", "order", "emotion"]:
        # Refit Ridge on clean proxy data without OCR (frozen model refit for ablation only)
        proxy_df = feats.merge(pd.read_csv(ROOT / "data" / "ratings" / "proxy_targets_full.csv"), on="image_id", how="inner", suffixes=("", "_tgt"))
        if "category_tgt" in proxy_df.columns:
            proxy_df = proxy_df.drop(columns=["category_tgt"])
        X_proxy = proxy_df[non_ocr_cols].values.astype(np.float64)
        y_proxy = proxy_df[dim].values.astype(np.float64)
        model_no_ocr = Ridge(alpha=1.0)
        model_no_ocr.fit(X_proxy, y_proxy)
        pred_no_ocr = model_no_ocr.predict(df[non_ocr_cols].values.astype(np.float64))
        rho, _ = stats.spearmanr(pred_no_ocr, df[dim])
        rows.append({"dimension": dim, "ablation": "without_ocr_layout", "spearman_rho": rho})
        # With all features (frozen)
        rows.append({"dimension": dim, "ablation": "with_all_30", "spearman_rho": stats.spearmanr(preds[dim], df[dim])[0]})
    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(RESULTS / "robustness_ablation.csv", index=False)
    print(f"Wrote {RESULTS / 'robustness_ablation.csv'}")
    print(ablation_df.to_string(index=False))

    # Failure cases for hierarchy (no proxy target) and order
    rows = []
    for dim in ["order"]:
        residuals = np.abs(preds[dim] - df[dim].values)
        worst_idx = np.argsort(residuals)[-10:]
        for idx in worst_idx:
            rows.append({
                "dimension": dim,
                "image_id": df.iloc[idx]["image_id"],
                "category": df.iloc[idx]["category"],
                "human_mean": df.iloc[idx][dim],
                "frozen_pred": preds[dim][idx],
                "abs_error": residuals[idx],
            })
    failure_df = pd.DataFrame(rows)
    failure_df.to_csv(RESULTS / "failure_cases.csv", index=False)
    print(f"Wrote {RESULTS / 'failure_cases.csv'}")

    # Efficiency: inference time and model size
    import time
    n_rep = 100
    start = time.time()
    for _ in range(n_rep):
        _ = models["complexity"].predict(X[:1])
    elapsed = (time.time() - start) / n_rep

    model_size = (ROOT / "results" / "redesign" / "models" / "frozen_primary_model.pkl").stat().st_size

    eff = pd.DataFrame([{
        "model": "frozen_primary_ridge",
        "inference_time_ms_per_image": elapsed * 1000,
        "model_size_bytes": model_size,
        "n_features": len(feature_cols),
    }])
    eff.to_csv(RESULTS / "efficiency_comparison.csv", index=False)
    print(f"Wrote {RESULTS / 'efficiency_comparison.csv'}")
    print(eff.to_string(index=False))

    # Robustness summary
    summary_rows = []
    for dim in ["complexity", "beauty", "order", "emotion"]:
        full_rho = stats.spearmanr(preds[dim], df[dim])[0]
        no_ocr_rho = ablation_df[(ablation_df["dimension"] == dim) & (ablation_df["ablation"] == "without_ocr_layout")]["spearman_rho"].values[0]
        summary_rows.append({
            "dimension": dim,
            "frozen_rho": full_rho,
            "without_ocr_rho": no_ocr_rho,
            "drop_due_to_ocr_removal": full_rho - no_ocr_rho,
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS / "robustness_summary.csv", index=False)
    print(f"Wrote {RESULTS / 'robustness_summary.csv'}")
    print(summary_df.to_string(index=False))

    print("Step 16 complete.")


if __name__ == "__main__":
    main()
