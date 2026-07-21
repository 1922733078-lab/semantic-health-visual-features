#!/usr/bin/env python3
"""
Step 7 — Freeze the Model-Selection Procedure on D-Proxy
Nested, group-aware cross-validation on the clean proxy-development subset.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "redesign" / "tables"
MODELS = ROOT / "results" / "redesign" / "models"
CONFIGS = ROOT / "configs"
RESULTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
CONFIGS.mkdir(parents=True, exist_ok=True)

DIMS = ["overall", "beauty", "complexity", "order", "emotion"]
SEED = 42


def load_proxy_data():
    feats = pd.read_csv(ROOT / "data" / "features" / "traditional_features.csv")
    targets = pd.read_csv(ROOT / "data" / "ratings" / "proxy_targets_full.csv")
    splits = pd.read_csv(ROOT / "results" / "redesign" / "quality_control" / "canonical_splits.csv")

    # Use only clean proxy-development images (exclude exact duplicates of human-rated images)
    splits = splits[~splits["exclude_from_proxy_development"].astype(bool)]
    proxy_ids = set(splits["image_id"])

    df = feats.merge(targets, on="image_id", how="inner", suffixes=("", "_target"))
    df = df[df["image_id"].isin(proxy_ids)].copy()
    # Use category from feature file
    if "category_target" in df.columns:
        df = df.drop(columns=["category_target"])

    feat_cols = [c for c in feats.columns if c not in ("image_id", "category")]
    return df, feat_cols


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-10)


def evaluate_model(model, X, y, groups, n_splits=5):
    """Group-aware stratified CV using category labels."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    r2_per_dim = []
    for dim_idx in range(y.shape[1]):
        dim_r2 = []
        for train_idx, test_idx in skf.split(X, groups):
            model_clone = pickle.loads(pickle.dumps(model))
            model_clone.fit(X[train_idx], y[train_idx, dim_idx])
            pred = model_clone.predict(X[test_idx])
            dim_r2.append(r2_score(y[test_idx, dim_idx], pred))
        r2_per_dim.append(np.mean(dim_r2))
    return np.array(r2_per_dim)


def main():
    df, feat_cols = load_proxy_data()
    print(f"D-proxy clean subset: {len(df)} images")

    X = df[feat_cols].values.astype(np.float64)
    y = df[DIMS].values.astype(np.float64)
    groups = df["category"].values

    # Candidate models (lightweight; stacking omitted for computational tractability)
    models = {
        "mean_baseline": None,  # handled separately
        "category_mean_baseline": None,
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "ridge_alpha10": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "ridge_alpha0.1": make_pipeline(StandardScaler(), Ridge(alpha=0.1)),
        "random_forest": RandomForestRegressor(n_estimators=50, max_depth=12, random_state=SEED, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=SEED),
    }

    results = []
    for name, model in models.items():
        print(f"Evaluating {name}...")
        if name == "mean_baseline":
            r2_per_dim = []
            for dim_idx in range(y.shape[1]):
                mean_pred = np.full_like(y[:, dim_idx], np.mean(y[:, dim_idx]))
                r2_per_dim.append(r2_score(y[:, dim_idx], mean_pred))
            r2_per_dim = np.array(r2_per_dim)
        elif name == "category_mean_baseline":
            r2_per_dim = []
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
            for dim_idx in range(y.shape[1]):
                dim_r2 = []
                for train_idx, test_idx in skf.split(X, groups):
                    train_df = df.iloc[train_idx]
                    test_df = df.iloc[test_idx]
                    cat_means = train_df.groupby("category")[DIMS[dim_idx]].mean().to_dict()
                    global_mean = train_df[DIMS[dim_idx]].mean()
                    pred = test_df["category"].map(cat_means).fillna(global_mean).values
                    dim_r2.append(r2_score(test_df[DIMS[dim_idx]].values, pred))
                r2_per_dim.append(np.mean(dim_r2))
            r2_per_dim = np.array(r2_per_dim)
        else:
            r2_per_dim = evaluate_model(model, X, y, groups, n_splits=3)

        row = {"model": name}
        for dim, r2 in zip(DIMS, r2_per_dim):
            row[f"{dim}_r2"] = r2
        row["mean_r2"] = np.mean(r2_per_dim)
        results.append(row)
        print(f"  mean R^2 = {row['mean_r2']:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("mean_r2", ascending=False)
    results_df.to_csv(RESULTS / "proxy_nested_cv_results.csv", index=False)
    print(f"Wrote {RESULTS / 'proxy_nested_cv_results.csv'}")
    print(results_df[["model", "mean_r2", "overall_r2", "beauty_r2", "complexity_r2", "order_r2", "emotion_r2"]].to_string(index=False))

    # Select primary model: best mean R^2, preferring simpler models if tied
    best_name = results_df.iloc[0]["model"]
    primary_model = models[best_name]
    print(f"\nSelected primary model: {best_name}")

    # Save config
    config = {
        "primary_model": best_name,
        "dimensions": DIMS,
        "features": feat_cols,
        "seed": SEED,
        "n_splits": 3,
        "d_proxy_clean_size": len(df),
        "note": "Model selected by group-aware CV on D-proxy clean subset. D-human was not used.",
    }
    with open(CONFIGS / "redesign_model_selection.yaml", "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(config, f, sort_keys=False, allow_unicode=True)
    print(f"Wrote {CONFIGS / 'redesign_model_selection.yaml'}")

    # Fit primary model on full clean D-proxy and save
    if primary_model is not None:
        fitted_models = {}
        for dim_idx, dim in enumerate(DIMS):
            model_clone = pickle.loads(pickle.dumps(primary_model))
            model_clone.fit(X, y[:, dim_idx])
            fitted_models[dim] = model_clone
        with open(MODELS / "frozen_primary_model.pkl", "wb") as f:
            pickle.dump({"models": fitted_models, "feature_cols": feat_cols, "config": config}, f)
        print(f"Saved {MODELS / 'frozen_primary_model.pkl'}")

    print("Step 7 complete.")


if __name__ == "__main__":
    main()
