#!/usr/bin/env python3
"""
Steps 11 and 12 — Structural discriminability and cross-category generalization.

Uses the per-image metric trace to:
  1. Compare score distributions across structural groups (category, text-heavy vs text-light).
  2. Summarize metric behaviour per category and in leave-one-category-out fashion.

Outputs:
    results/no_human/tables/structural_group_comparison.csv
    results/no_human/figures/score_distributions_by_structural_group.pdf
    results/no_human/tables/category_generalization.csv
    results/no_human/tables/source_held_out_results.csv
    results/no_human/figures/category_metric_profiles.pdf
"""

import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.environ.get("NO_HUMAN_OUTPUT_ROOT", ROOT / "results" / "no_human"))
TRACE_PATH = OUTPUT_ROOT / "data" / "per_image_metric_trace.csv"
SPLITS_PATH = OUTPUT_ROOT / "quality_control" / "canonical_splits.csv"

CONFIG_PATH = Path(os.environ.get("NO_HUMAN_CONFIG_PATH", ROOT / "configs" / "no_human_design.yaml"))


def get_traditional_features_path():
    """Resolve the traditional-features path from the active config."""
    default = ROOT / "data" / "features" / "traditional_features.csv"
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        rel = cfg.get("paths", {}).get("traditional_features")
        if rel:
            candidate = ROOT / rel
            if candidate.exists():
                return candidate
    except Exception:
        pass
    return default if default.exists() else None

METRICS = [
    "visual_complexity_proxy",
    "layout_order_proxy",
    "colour_harmony_proxy",
    "visual_intensity_proxy",
    "layout_hierarchy_proxy",
]


def load_trace():
    trace = pd.read_csv(TRACE_PATH)
    splits = pd.read_csv(SPLITS_PATH)[["image_id", "role"]]
    trace = trace.merge(splits, on="image_id", how="left")
    # Bring in raw features for structural grouping
    feat_path = get_traditional_features_path()
    if feat_path is not None:
        feats = pd.read_csv(feat_path)
        feat_cols = [c for c in feats.columns if c not in ["category"]]
        trace = trace.merge(feats[feat_cols], on="image_id", how="left")
    else:
        print("Warning: traditional feature file not found; structural grouping will be limited.")
    return trace


def define_structural_groups(trace):
    trace = trace.copy()
    if "edge_density" in trace.columns:
        trace["edge_group"] = np.where(trace["edge_density"] > trace["edge_density"].median(),
                                       "edge_dense", "edge_sparse")
    else:
        trace["edge_group"] = "unknown"
    trace["structural_group"] = trace["category"].astype(str) + "_" + trace["edge_group"]
    return trace


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return float((x.mean() - y.mean()) / pooled_std)


def bootstrap_ci(x, y, n_boot=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        diffs.append(xb.mean() - yb.mean())
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def structural_group_comparison(trace):
    rows = []
    # Category-level comparisons: painting vs UI
    cat_pairs = [("painting", "ui"), ("ui", "packaging"), ("packaging", "poster")]
    for metric in METRICS:
        for g1, g2 in cat_pairs:
            x = trace[trace["category"] == g1][metric]
            y = trace[trace["category"] == g2][metric]
            if len(x) < 10 or len(y) < 10:
                continue
            d = cohens_d(x, y)
            ci_low, ci_high = bootstrap_ci(x, y)
            rows.append({
                "comparison_type": "category",
                "group_1": g1,
                "group_2": g2,
                "metric": metric,
                "n_1": len(x),
                "n_2": len(y),
                "mean_1": float(x.mean()),
                "mean_2": float(y.mean()),
                "cohens_d": d,
                "mean_diff_ci_low": ci_low,
                "mean_diff_ci_high": ci_high,
            })

    # Text-heavy vs text-light within each category
    for metric in METRICS:
        for cat in trace["category"].unique():
            sub = trace[trace["category"] == cat]
            x = sub[sub["text_group"] == "text_heavy"][metric]
            y = sub[sub["text_group"] == "text_light"][metric]
            if len(x) < 10 or len(y) < 10:
                continue
            d = cohens_d(x, y)
            ci_low, ci_high = bootstrap_ci(x, y)
            rows.append({
                "comparison_type": "text_group_within_category",
                "group_1": f"{cat}_text_heavy",
                "group_2": f"{cat}_text_light",
                "metric": metric,
                "n_1": len(x),
                "n_2": len(y),
                "mean_1": float(x.mean()),
                "mean_2": float(y.mean()),
                "cohens_d": d,
                "mean_diff_ci_low": ci_low,
                "mean_diff_ci_high": ci_high,
            })
    return pd.DataFrame(rows)


def plot_score_distributions(trace):
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(10, 16))
    for ax, metric in zip(axes, METRICS):
        sns.violinplot(data=trace, x="category", y=metric, ax=ax, palette="muted", inner="quart")
        ax.set_title(metric)
        ax.set_ylim(-5, 105)
    plt.tight_layout()
    out_path = OUTPUT_ROOT / "figures" / "score_distributions_by_structural_group.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def category_generalization(trace):
    rows = []
    for cat in sorted(trace["category"].unique()):
        sub = trace[trace["category"] == cat]
        row = {"category": cat, "n": len(sub)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std())
            row[f"{metric}_median"] = float(sub[metric].median())
        rows.append(row)

    # Add leave-one-category-out: stats computed on all categories except one
    for held_out in sorted(trace["category"].unique()):
        sub = trace[trace["category"] != held_out]
        row = {"category": f"exclude_{held_out}", "n": len(sub)}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std())
            row[f"{metric}_median"] = float(sub[metric].median())
        rows.append(row)
    return pd.DataFrame(rows)


def source_held_out_results(trace):
    # Metadata lacks source column; document limitation and provide category-level proxy.
    rows = []
    rows.append({
        "source_held_out": "not_available",
        "reason": "metadata has no source column",
        "proxy_used": "leave-one-category-out summaries in category_generalization.csv",
        "n_categories": trace["category"].nunique(),
    })
    return pd.DataFrame(rows)


def plot_category_profiles(trace):
    cat_summary = category_generalization(trace)
    cat_only = cat_summary[~cat_summary["category"].str.startswith("exclude_")].copy()
    metrics_mean = [f"{m}_mean" for m in METRICS]
    data = cat_only.set_index("category")[metrics_mean]
    data.columns = METRICS

    fig, ax = plt.subplots(figsize=(10, 6))
    data.T.plot(kind="bar", ax=ax, colormap="viridis")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mean proxy score")
    ax.set_title("Metric profiles by category")
    ax.legend(title="Category", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out_path = OUTPUT_ROOT / "figures" / "category_metric_profiles.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    trace = load_trace()
    trace = define_structural_groups(trace)

    print("Computing structural group comparisons...")
    struct_comp = structural_group_comparison(trace)
    struct_comp.to_csv(OUTPUT_ROOT / "tables" / "structural_group_comparison.csv", index=False)
    print(f"Wrote {len(struct_comp)} comparisons")

    print("Plotting score distributions...")
    plot_score_distributions(trace)

    print("Computing category generalization...")
    cat_gen = category_generalization(trace)
    cat_gen.to_csv(OUTPUT_ROOT / "tables" / "category_generalization.csv", index=False)

    print("Documenting source-held-out limitation...")
    source_out = source_held_out_results(trace)
    source_out.to_csv(OUTPUT_ROOT / "tables" / "source_held_out_results.csv", index=False)

    print("Plotting category profiles...")
    plot_category_profiles(trace)

    print("Done.")


if __name__ == "__main__":
    main()
