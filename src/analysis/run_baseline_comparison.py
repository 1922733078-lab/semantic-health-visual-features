#!/usr/bin/env python3
"""
Step 8 — Establish Fair Deep and Classical Baselines
Evaluate all baselines on the same 1,500-image subset that has deep features.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "redesign" / "tables"
RESULTS.mkdir(parents=True, exist_ok=True)

DIMS = ["overall", "beauty", "complexity", "order", "emotion"]
SEED = 42


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-10)


def theory_score(row):
    """Original theory-derived proxy formula from run_ava_experiment.py (deterministic part)."""
    s = {}
    s["overall"] = np.clip(
        4 + 0.15 * (row.get("saturation_mean", 50) / 50 - 0.5) * 5
        + 0.2 * (row.get("color_harmony", 0.5) - 0.5) * 5
        + 0.15 * (row.get("symmetry", 0.5) - 0.5) * 5,
        1, 7,
    )
    s["beauty"] = np.clip(
        4 + 0.15 * (row.get("warm_color_ratio", 0.5) - 0.5) * 5
        + 0.15 * (row.get("color_entropy", 2) / 4 - 0.5) * 5
        + 0.1 * (row.get("saturation_std", 30) / 60 - 0.5) * 5,
        1, 7,
    )
    s["complexity"] = np.clip(
        4 + 0.2 * (row.get("edge_density", 0.1) - 0.05) * 10
        + 0.15 * (row.get("num_dominant_colors", 3) / 5 - 0.5) * 5
        - 0.1 * (row.get("symmetry", 0.5) - 0.5) * 5,
        1, 7,
    )
    s["order"] = np.clip(
        4 - 0.15 * (row.get("edge_density", 0.1) - 0.05) * 10
        + 0.15 * (row.get("symmetry", 0.5) - 0.5) * 5
        + 0.1 * (1 - row.get("whitespace_ratio", 0.1)) * 5,
        1, 7,
    )
    s["emotion"] = np.clip(
        4 + 0.15 * (row.get("saturation_mean", 50) / 50 - 0.5) * 5
        + 0.15 * (row.get("warm_color_ratio", 0.5) - 0.5) * 5
        + 0.1 * (row.get("lightness_contrast", 30) / 60 - 0.5) * 5,
        1, 7,
    )
    return s


def cv_ridge(X, y, groups, n_splits=5, pca_dim=None):
    """Group-aware CV with optional PCA fitted inside each training fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    r2_per_dim = []
    for dim_idx in range(y.shape[1]):
        dim_r2 = []
        for train_idx, test_idx in skf.split(X, groups):
            X_tr, X_te = X[train_idx], X[test_idx]
            if pca_dim is not None:
                pca = PCA(n_components=pca_dim, random_state=SEED)
                X_tr = pca.fit_transform(X_tr)
                X_te = pca.transform(X_te)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            model = Ridge(alpha=1.0)
            model.fit(X_tr_s, y[train_idx, dim_idx])
            pred = model.predict(X_te_s)
            dim_r2.append(r2_score(y[test_idx, dim_idx], pred))
        r2_per_dim.append(np.mean(dim_r2))
    return np.array(r2_per_dim)


def main():
    # Load deep feature image list
    deep_ids = pd.read_csv(ROOT / "data" / "features" / "deep" / "image_ids.csv")
    target_ids = set(deep_ids["image_id"])

    # Load features and targets, restrict to common subset
    trad_df = pd.read_csv(ROOT / "data" / "features" / "traditional_features.csv")
    enh_df = pd.read_csv(ROOT / "data" / "features" / "enhanced_features.csv")
    targets = pd.read_csv(ROOT / "data" / "ratings" / "proxy_targets_full.csv")

    df = deep_ids.merge(trad_df, on="image_id", how="inner")
    df = df.merge(targets, on="image_id", how="inner", suffixes=("", "_tgt"))
    # Drop duplicated target category column if present
    if "category_tgt" in df.columns:
        df = df.drop(columns=["category_tgt"])
    if "category" not in df.columns:
        df["category"] = df["image_id"].apply(lambda x: x.split("_")[0])

    print(f"Common evaluation subset: {len(df)} images")

    trad_cols = [c for c in trad_df.columns if c not in ("image_id", "category")]

    X_trad = df[trad_cols].values.astype(np.float64)
    y = df[DIMS].values.astype(np.float64)
    groups = df["category"].values

    # Load deep features and align to the common subset by image_id
    deep_features_raw = {}
    for name in ["resnet50_features", "clip_features", "dinov2_features", "efficientnet_b0_features"]:
        arr = np.load(ROOT / "data" / "features" / "deep" / f"{name}.npy", allow_pickle=True)
        deep_features_raw[name.replace("_features", "")] = arr.astype(np.float64)

    # deep_ids rows align with deep feature arrays; reorder/filter to match df
    id_to_idx = {img_id: idx for idx, img_id in enumerate(deep_ids["image_id"])}
    deep_features = {}
    for name, arr in deep_features_raw.items():
        idxs = [id_to_idx[iid] for iid in df["image_id"] if iid in id_to_idx]
        deep_features[name] = arr[idxs]

    # Theory baseline predictions
    theory_preds = np.array([[theory_score(row)[dim] for dim in DIMS] for _, row in df.iterrows()])

    results = []

    # Naive baselines
    for name, preds in [("global_mean", np.tile(y.mean(axis=0), (len(df), 1))),
                        ("category_mean", None)]:
        if name == "category_mean":
            preds = np.zeros_like(y)
            for cat in np.unique(groups):
                mask = groups == cat
                preds[mask] = y[mask].mean(axis=0)
        r2_per_dim = np.array([r2_score(y[:, j], preds[:, j]) for j in range(y.shape[1])])
        row = {"model": name, "mean_r2": r2_per_dim.mean()}
        for dim, r2 in zip(DIMS, r2_per_dim):
            row[f"{dim}_r2"] = r2
        results.append(row)
        print(f"{name}: mean R^2 = {row['mean_r2']:.4f}")

    # Theory baseline
    r2_theory = np.array([r2_score(y[:, j], theory_preds[:, j]) for j in range(y.shape[1])])
    row = {"model": "theory_formula", "mean_r2": r2_theory.mean()}
    for dim, r2 in zip(DIMS, r2_theory):
        row[f"{dim}_r2"] = r2
    results.append(row)
    print(f"theory_formula: mean R^2 = {row['mean_r2']:.4f}")

    # Classical features
    for name, X in [("trad_30_ridge", X_trad)]:
        r2_per_dim = cv_ridge(X, y, groups, n_splits=3, pca_dim=None)
        row = {"model": name, "mean_r2": r2_per_dim.mean()}
        for dim, r2 in zip(DIMS, r2_per_dim):
            row[f"{dim}_r2"] = r2
        results.append(row)
        print(f"{name}: mean R^2 = {row['mean_r2']:.4f}")

    # Deep features
    for feat_name, X_deep in deep_features.items():
        # Raw deep + Ridge
        r2_per_dim = cv_ridge(X_deep, y, groups, n_splits=3, pca_dim=None)
        row = {"model": f"{feat_name}_ridge", "mean_r2": r2_per_dim.mean()}
        for dim, r2 in zip(DIMS, r2_per_dim):
            row[f"{dim}_r2"] = r2
        results.append(row)
        print(f"{feat_name}_ridge: mean R^2 = {row['mean_r2']:.4f}")

        # PCA128 + Ridge
        r2_per_dim = cv_ridge(X_deep, y, groups, n_splits=3, pca_dim=min(128, X_deep.shape[1]))
        row = {"model": f"{feat_name}_pca128_ridge", "mean_r2": r2_per_dim.mean()}
        for dim, r2 in zip(DIMS, r2_per_dim):
            row[f"{dim}_r2"] = r2
        results.append(row)
        print(f"{feat_name}_pca128_ridge: mean R^2 = {row['mean_r2']:.4f}")

    results_df = pd.DataFrame(results).sort_values("mean_r2", ascending=False)
    results_df.to_csv(RESULTS / "baseline_comparison_proxy.csv", index=False)
    print(f"Wrote {RESULTS / 'baseline_comparison_proxy.csv'}")
    print(results_df[["model", "mean_r2"]].to_string(index=False))

    print("Step 8 (proxy baseline comparison) complete.")


if __name__ == "__main__":
    main()
