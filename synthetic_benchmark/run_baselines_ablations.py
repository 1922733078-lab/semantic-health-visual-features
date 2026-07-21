#!/usr/bin/env python3
"""
Run baseline comparison, feature-group ablation, and weight-robustness analyses
on the synthetic benchmark.

Outputs are written to results/no_human/runs/<run-id>/baselines/.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, kendalltau

import sys
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.no_human.baselines import compute_baselines, BASELINE_TARGET_MAP, best_baseline_for_target

METRICS = {
    "visual_complexity_proxy": {
        "features": ["num_dominant_colors", "color_entropy", "edge_density",
                     "edge_orientation_entropy", "gradient_energy",
                     "text_block_count", "whitespace_ratio", "symmetry"],
        "weights": [0.20, 0.15, 0.25, 0.15, 0.10, 0.05, -0.05, -0.05],
        "groups": ["colour", "colour", "edge_texture", "edge_texture", "edge_texture",
                   "projection_legacy", "composition", "composition"],
    },
    "layout_order_proxy": {
        "features": ["symmetry", "whitespace_ratio", "rule_of_thirds", "edge_density",
                     "text_block_count", "lightness_contrast"],
        "weights": [0.25, 0.20, 0.15, -0.25, -0.10, -0.05],
        "groups": ["composition", "composition", "composition", "edge_texture",
                   "projection_legacy", "colour"],
    },
    "colour_harmony_proxy": {
        "features": ["color_harmony", "warm_color_ratio", "saturation_std",
                     "hue_contrast", "num_dominant_colors"],
        "weights": [0.30, 0.25, 0.20, -0.15, -0.10],
        "groups": ["colour", "colour", "colour", "colour", "colour"],
    },
    "visual_intensity_proxy": {
        "features": ["saturation_mean", "lightness_contrast", "gradient_energy",
                     "warm_color_ratio", "saturation_std"],
        "weights": [0.35, 0.25, 0.20, 0.10, 0.10],
        "groups": ["colour", "colour", "edge_texture", "colour", "colour"],
    },
    "layout_hierarchy_proxy": {
        "features": ["edge_density", "saliency_std", "fg_bg_ratio",
                     "text_block_count", "has_text", "saliency_mean", "whitespace_ratio"],
        "weights": [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, -0.05],
        "groups": ["edge_texture", "saliency", "composition", "projection_legacy",
                   "projection_legacy", "saliency", "composition"],
    },
}

TARGET_MAP = {
    "visual_complexity_proxy": "structural_clutter_index",
    "layout_order_proxy": "geometric_order_index",
    "colour_harmony_proxy": "palette_coherence_index",
    "visual_intensity_proxy": "visual_salience_index",
    "layout_hierarchy_proxy": "focal_hierarchy_index",
}

TARGET_COLS = list(TARGET_MAP.values())
ID_COLS = {"image_id", "category"}


def _numeric_feature_cols(df):
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric if c not in ID_COLS and c not in TARGET_COLS]


def load_frozen_stats():
    preproc_path = PROJECT_ROOT / "configs" / "no_human_preprocessing.yaml"
    preproc = yaml.safe_load(preproc_path.read_text(encoding="utf-8"))
    return preproc["normalization_constants"], preproc["percentile_bounds"]


def standardize_features(df, feature_cols, stats):
    z = pd.DataFrame(index=df.index, columns=feature_cols, dtype=float)
    for col in feature_cols:
        mean = stats["mean"].get(col, df[col].mean())
        std = stats["std"].get(col, df[col].std())
        if std == 0:
            std = 1.0
        z[col] = ((df[col] - mean) / std).clip(-4, 4).fillna(0)
    return z


def compute_proxy(z_df, metric, percentile_bounds):
    spec = METRICS[metric]
    raw = np.zeros(len(z_df))
    for feat, w in zip(spec["features"], spec["weights"]):
        if feat in z_df.columns:
            raw += z_df[feat].values * w
    pb = percentile_bounds[metric]
    score = 100 * (raw - pb["p0.5"]) / (pb["p99.5"] - pb["p0.5"] + 1e-10)
    return np.clip(score, 0, 100)


def ablate_group(z_df, metric, group_to_remove):
    spec = METRICS[metric]
    raw = np.zeros(len(z_df))
    for feat, w, grp in zip(spec["features"], spec["weights"], spec["groups"]):
        if grp == group_to_remove:
            continue
        if feat in z_df.columns:
            raw += z_df[feat].values * w
    return raw


def bootstrap_delta_ci(proposed, baseline, y, n_boot=1000, alpha=0.05, seed=42):
    """Paired bootstrap CI for proposed_rho - baseline_rho on the same data."""
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pr, _ = spearmanr(proposed[idx], y[idx])
        br, _ = spearmanr(baseline[idx], y[idx])
        if np.isfinite(pr) and np.isfinite(br):
            deltas.append(pr - br)
    deltas = np.array(deltas)
    lo = np.percentile(deltas, 100 * alpha / 2)
    hi = np.percentile(deltas, 100 * (1 - alpha / 2))
    return lo, hi


def run_baselines(df, baselines_df, run_dir, split_name="G-test"):
    rows = []
    for proxy, target in TARGET_MAP.items():
        y = df[target].values
        proposed = compute_proxy_from_features(df, proxy)
        prop_rho, _ = spearmanr(proposed, y)

        candidates = BASELINE_TARGET_MAP[proxy]
        best_name, best_rho, best_baseline_values = None, 0.0, None
        for col, direction in candidates:
            try:
                vals = baselines_df[col].values
                rho, _ = spearmanr(vals, y)
                if np.isnan(rho):
                    rho = 0.0
                rho = rho * direction
            except Exception:
                rho = 0.0
                vals = np.zeros(len(y))
            if abs(rho) > abs(best_rho):
                best_rho = rho
                best_name = col
                best_baseline_values = vals * direction
        if best_name is None and candidates:
            best_name = candidates[0][0]
            best_baseline_values = baselines_df[candidates[0][0]].values * candidates[0][1]

        delta_lo, delta_hi = bootstrap_delta_ci(proposed, best_baseline_values, y)

        rows.append({
            "proxy": proxy,
            "target": target,
            "evaluation_split": split_name,
            "proposed_rho": prop_rho,
            "best_baseline": best_name,
            "best_baseline_rho": best_rho,
            "delta_rho": prop_rho - best_rho,
            "delta_ci_lower": delta_lo,
            "delta_ci_upper": delta_hi,
            "baseline_superior": (prop_rho - best_rho) < 0,
        })
    return pd.DataFrame(rows)


def compute_proxy_from_features(df, proxy):
    feature_cols = _numeric_feature_cols(df)
    stats, pb = load_frozen_stats()
    z_df = standardize_features(df, feature_cols, stats)
    return compute_proxy(z_df, proxy, pb)


def run_ablations(df, run_dir):
    rows = []
    feature_cols = _numeric_feature_cols(df)
    stats, pb = load_frozen_stats()
    z_df = standardize_features(df, feature_cols, stats)

    for proxy, target in TARGET_MAP.items():
        full_score = compute_proxy(z_df, proxy, pb)
        full_rho, _ = spearmanr(full_score, df[target].values)
        full_rho = 0.0 if np.isnan(full_rho) else full_rho
        spec = METRICS[proxy]
        groups = set(spec["groups"])
        for grp in ["full"] + sorted(groups):
            if grp == "full":
                score = full_score
            else:
                raw = ablate_group(z_df, proxy, grp)
                # rescale using full metric bounds for comparability
                score = 100 * (raw - pb[proxy]["p0.5"]) / (pb[proxy]["p99.5"] - pb[proxy]["p0.5"] + 1e-10)
                score = np.clip(score, 0, 100)
            rho, _ = spearmanr(score, df[target].values)
            rho = 0.0 if np.isnan(rho) else rho
            rows.append({
                "proxy": proxy,
                "target": target,
                "ablated_group": grp,
                "spearman_rho": rho,
                "delta_from_full": rho - full_rho,
            })
    return pd.DataFrame(rows)


def run_weight_robustness(df, run_dir, n_samples=1000):
    rows = []
    feature_cols = _numeric_feature_cols(df)
    stats, pb = load_frozen_stats()
    z_df = standardize_features(df, feature_cols, stats)
    rng = np.random.default_rng(42)

    for proxy, target in TARGET_MAP.items():
        spec = METRICS[proxy]
        weights = np.array(spec["weights"], dtype=float)
        full_score = compute_proxy(z_df, proxy, pb)
        full_rho, _ = spearmanr(full_score, df[target].values)

        sample_rhos = []
        for _ in range(n_samples):
            perturb = rng.uniform(0.9, 1.1, size=len(weights))
            new_weights = weights * perturb
            # preserve sign and rescale sum of absolute values
            new_weights = np.sign(new_weights) * (np.abs(new_weights) / np.abs(new_weights).sum() * np.abs(weights).sum())
            raw = np.zeros(len(z_df))
            for feat, w in zip(spec["features"], new_weights):
                if feat in z_df.columns:
                    raw += z_df[feat].values * w
            score = 100 * (raw - pb[proxy]["p0.5"]) / (pb[proxy]["p99.5"] - pb[proxy]["p0.5"] + 1e-10)
            score = np.clip(score, 0, 100)
            rho, _ = spearmanr(score, df[target].values)
            if np.isfinite(rho):
                sample_rhos.append(rho)

        sample_rhos = np.array(sample_rhos)
        if len(sample_rhos) == 0:
            sample_rhos = np.array([0.0])
        rows.append({
            "proxy": proxy,
            "target": target,
            "full_rho": full_rho,
            "perturbed_mean_rho": sample_rhos.mean(),
            "perturbed_std_rho": sample_rhos.std(),
            "perturbed_min_rho": sample_rhos.min(),
            "perturbed_max_rho": sample_rhos.max(),
            "proportion_below_full": np.mean(sample_rhos < full_rho),
        })
    return pd.DataFrame(rows)


def plot_baseline_forest(baseline_df, run_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(baseline_df))
    ax.barh(y_pos, baseline_df["proposed_rho"], color="steelblue", label="Proposed")
    ax.barh(y_pos, baseline_df["best_baseline_rho"], color="coral", label="Best baseline", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(baseline_df["proxy"])
    ax.set_xlabel("Spearman rho")
    ax.legend()
    ax.set_title("Proposed proxy vs best simple baseline")
    fig.tight_layout()
    fig.savefig(run_dir / "baselines" / "figures" / "baseline_forest_plot.pdf")
    plt.close(fig)


def plot_weight_robustness(weight_df, run_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    proxies = weight_df["proxy"].values
    x = np.arange(len(proxies))
    ax.bar(x, weight_df["full_rho"], color="steelblue", label="Frozen weights")
    ax.errorbar(x, weight_df["perturbed_mean_rho"], yerr=weight_df["perturbed_std_rho"],
                fmt="o", color="coral", label="Mean ± SD (±10% perturbations)")
    ax.set_xticks(x)
    ax.set_xticklabels(proxies, rotation=45, ha="right")
    ax.set_ylabel("Spearman rho")
    ax.legend()
    ax.set_title("Weight robustness")
    fig.tight_layout()
    fig.savefig(run_dir / "baselines" / "figures" / "weight_robustness.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--evaluation-split", default="G-test",
                        choices=["G-dev", "G-validation", "G-test", "interaction-test", "pooled"],
                        help="Evaluation split required by protocol.")
    parser.add_argument("--features", default="synthetic_benchmark/metadata/synthetic_traditional_features.csv")
    parser.add_argument("--targets", default="synthetic_benchmark/metadata/target_metadata.csv")
    args = parser.parse_args()

    run_dir = PROJECT_ROOT / "results" / "no_human" / "runs" / args.run_id
    for sub in ["baselines/tables", "baselines/figures", "baselines/logs"]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "baselines" / "logs" / "run.log"
    log = open(log_path, "w", encoding="utf-8")
    log.write(f"Baseline/ablation/robustness started at {pd.Timestamp.now().isoformat()}\n")
    log.write(f"Evaluation split: {args.evaluation_split}\n")

    features = pd.read_csv(args.features)
    targets = pd.read_csv(args.targets)
    df = features.merge(targets, on="image_id", how="inner")
    if args.evaluation_split != "pooled":
        df = df[df["split"] == args.evaluation_split].copy()
    log.write(f"Loaded {len(df)} images after split filter.\n")

    baselines_df = compute_baselines(features)
    baselines_df = baselines_df[baselines_df["image_id"].isin(df["image_id"])].sort_values("image_id").reset_index(drop=True)
    df = df.sort_values("image_id").reset_index(drop=True)
    baseline_df = run_baselines(df, baselines_df, run_dir, split_name=args.evaluation_split)
    baseline_df.to_csv(run_dir / "baselines" / "tables" / "baseline_comparison.csv", index=False)
    log.write(f"Baseline comparison complete.\n")

    ablation_df = run_ablations(df, run_dir)
    ablation_df.to_csv(run_dir / "baselines" / "tables" / "ablation_results.csv", index=False)
    log.write(f"Ablation complete.\n")

    weight_df = run_weight_robustness(df, run_dir)
    weight_df.to_csv(run_dir / "baselines" / "tables" / "weight_robustness.csv", index=False)
    log.write(f"Weight robustness complete.\n")

    plot_baseline_forest(baseline_df, run_dir)
    plot_weight_robustness(weight_df, run_dir)
    log.write("Figures saved.\n")
    log.close()

    print("Baseline comparison:")
    print(baseline_df[["proxy", "proposed_rho", "best_baseline_rho", "delta_rho", "baseline_superior"]].to_string(index=False))


if __name__ == "__main__":
    main()
