#!/usr/bin/env python3
"""
Step 13 — Independent computational-agreement analyses.

Correlates the primary direct metrics with independent computational references:
  - Fixed pretrained deep embeddings (ResNet50, EfficientNet-B0, CLIP, DINOv2)
  - Segmentation-geometry descriptors from the enhanced feature set

Outputs:
    results/no_human/tables/computational_convergence.csv
    results/no_human/figures/metric_vs_independent_references.pdf
"""

import os
import warnings
from pathlib import Path

import matplotlib
import yaml
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from fdr_bh import fdr_control

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(os.environ.get("NO_HUMAN_CONFIG_PATH", ROOT / "configs" / "no_human_design.yaml"))
OUTPUT_ROOT = Path(os.environ.get("NO_HUMAN_OUTPUT_ROOT", ROOT / "results" / "no_human"))
TRACE_PATH = OUTPUT_ROOT / "data" / "per_image_metric_trace.csv"
DEEP_DIR = ROOT / "data" / "features" / "deep"
ENHANCED_PATH = ROOT / "data" / "features" / "enhanced_features.csv"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Synthetic-objective fallback descriptors for the public demonstration benchmark.
SYNTHETIC_OBJECTIVE_MAP = {
    "visual_complexity_proxy": "structural_clutter_index",
    "layout_order_proxy": "geometric_order_index",
    "colour_harmony_proxy": "palette_coherence_index",
    "visual_intensity_proxy": "visual_salience_index",
    "layout_hierarchy_proxy": "focal_hierarchy_index",
}

METRICS = [
    "visual_complexity_proxy",
    "layout_order_proxy",
    "colour_harmony_proxy",
    "visual_intensity_proxy",
    "layout_hierarchy_proxy",
]

# Map primary metrics to theoretically aligned reference descriptors
REFERENCE_MAP = {
    "visual_complexity_proxy": [
        ("deep_resnet50_norm", "deep"),
        ("deep_clip_norm", "deep"),
        ("region_dispersion", "segmentation"),
    ],
    "layout_order_proxy": [
        ("grid_alignment", "segmentation"),
        ("quadrant_balance", "segmentation"),
        ("deep_dinov2_norm", "deep"),
    ],
    "colour_harmony_proxy": [
        ("warm_cool_balance", "segmentation"),
        ("deep_clip_norm", "deep"),
    ],
    "visual_intensity_proxy": [
        ("deep_efficientnet_norm", "deep"),
        ("colourfulness", "segmentation"),
    ],
    "layout_hierarchy_proxy": [
        ("figure_ground", "segmentation"),
        ("size_concentration", "segmentation"),
    ],
}


def bootstrap_correlation(x, y, n_boot=1000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    spearman_rhos = []
    pearson_rs = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        xb, yb = x[idx], y[idx]
        if np.unique(xb).size < 2 or np.unique(yb).size < 2:
            continue
        spearman_rhos.append(spearmanr(xb, yb)[0])
        pearson_rs.append(pearsonr(xb, yb)[0])
    if not spearman_rhos:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(spearman_rhos)),
        float(np.percentile(spearman_rhos, 2.5)),
        float(np.percentile(spearman_rhos, 97.5)),
        float(np.mean(pearson_rs)),
        float(np.percentile(pearson_rs, 2.5)),
        float(np.percentile(pearson_rs, 97.5)),
    )


def load_deep_references():
    if not (DEEP_DIR / "image_ids.csv").exists():
        return pd.DataFrame()
    ids_df = pd.read_csv(DEEP_DIR / "image_ids.csv")
    refs = {}
    for model in ["resnet50", "efficientnet_b0", "clip", "dinov2"]:
        feat_path = DEEP_DIR / f"{model}_features.npy"
        if not feat_path.exists():
            continue
        feats = np.load(feat_path)
        # Center and compute norm as a simple univariate reference
        centered = feats - feats.mean(axis=0)
        norm = np.linalg.norm(centered, axis=1)
        refs[f"deep_{model.replace('_b0', '')}_norm"] = norm
    if not refs:
        return pd.DataFrame()
    refs["image_id"] = ids_df["image_id"].values
    return pd.DataFrame(refs)


def load_segmentation_references():
    if not ENHANCED_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(ENHANCED_PATH)
    # Build theoretically aligned descriptors from enhanced features
    refs = pd.DataFrame({"image_id": df["image_id"]})
    refs["region_dispersion"] = df["fractal_dim"] * (1 - df["repetition_fft"])
    refs["grid_alignment"] = df["grid_alignment"]
    refs["quadrant_balance"] = df["quadrant_balance"]
    refs["warm_cool_balance"] = df["warm_cool_balance"]
    refs["colourfulness"] = df["color_harmony_adv"] * df["warm_cool_balance"]
    refs["figure_ground"] = df["figure_ground"]
    refs["size_concentration"] = 1.0 - df["proximity_cv"]
    return refs


def load_synthetic_objective_references():
    """Use the synthetic benchmark's objective indices as independent references."""
    cfg = load_config()
    metadata_path = cfg.get("paths", {}).get("metadata")
    if not metadata_path:
        return pd.DataFrame()
    path = ROOT / metadata_path
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = ["image_id"] + list(SYNTHETIC_OBJECTIVE_MAP.values())
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
    return df[["image_id"] + list(SYNTHETIC_OBJECTIVE_MAP.values())].copy()


def main():
    trace = pd.read_csv(TRACE_PATH)[["image_id"] + METRICS]
    deep_refs = load_deep_references()
    seg_refs = load_segmentation_references()
    synth_refs = load_synthetic_objective_references()

    rows = []
    pvals_spearman = []
    pvals_pearson = []
    plot_data = []

    reference_map = dict(REFERENCE_MAP)
    # Add synthetic-objective references when the public demo benchmark is used.
    if not synth_refs.empty:
        for metric, col in SYNTHETIC_OBJECTIVE_MAP.items():
            reference_map.setdefault(metric, []).append((col, "synthetic_objective"))

    for metric, ref_list in reference_map.items():
        for ref_name, ref_family in ref_list:
            if ref_family == "deep":
                if deep_refs.empty or ref_name not in deep_refs.columns:
                    continue
                merged = trace.merge(deep_refs[["image_id", ref_name]], on="image_id", how="inner")
            elif ref_family == "segmentation":
                if seg_refs.empty or ref_name not in seg_refs.columns:
                    continue
                merged = trace.merge(seg_refs[["image_id", ref_name]], on="image_id", how="inner")
            elif ref_family == "synthetic_objective":
                if synth_refs.empty or ref_name not in synth_refs.columns:
                    continue
                merged = trace.merge(synth_refs[["image_id", ref_name]], on="image_id", how="inner")
            else:
                continue
            if len(merged) < 30:
                continue
            x = merged[metric].values
            y = merged[ref_name].values
            rho, p_spearman = spearmanr(x, y)
            r, p_pearson = pearsonr(x, y)
            rho_mean, rho_lo, rho_hi, r_mean, r_lo, r_hi = bootstrap_correlation(x, y)
            rows.append({
                "metric": metric,
                "reference_family": ref_family,
                "reference_descriptor": ref_name,
                "n": len(merged),
                "spearman_rho": float(rho),
                "spearman_p": float(p_spearman),
                "spearman_ci_low": rho_lo,
                "spearman_ci_high": rho_hi,
                "pearson_r": float(r),
                "pearson_p": float(p_pearson),
                "pearson_ci_low": r_lo,
                "pearson_ci_high": r_hi,
            })
            pvals_spearman.append(p_spearman)
            pvals_pearson.append(p_pearson)
            plot_data.append((metric, ref_name, ref_family, merged))

    results_df = pd.DataFrame(rows)
    if len(results_df) > 0:
        results_df["spearman_q"] = fdr_control(pvals_spearman)
        results_df["pearson_q"] = fdr_control(pvals_pearson)

    (OUTPUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_ROOT / "tables" / "computational_convergence.csv", index=False)
    print(f"Wrote {len(results_df)} convergence comparisons")
    if not results_df.empty:
        print(results_df[["metric", "reference_descriptor", "n", "spearman_rho", "spearman_ci_low", "spearman_ci_high"]].to_string(index=False))

    # Scatter plots
    n_plots = len(plot_data)
    if n_plots > 0:
        cols = 3
        rows_needed = int(np.ceil(n_plots / cols))
        fig, axes = plt.subplots(rows_needed, cols, figsize=(15, 4 * rows_needed))
        axes = np.array(axes).flatten()
        for ax, (metric, ref_name, ref_family, merged) in zip(axes, plot_data):
            ax.scatter(merged[metric], merged[ref_name], alpha=0.3, s=5)
            ax.set_xlabel(metric)
            ax.set_ylabel(ref_name)
            ax.set_title(f"{ref_family}: {ref_name}")
        for ax in axes[n_plots:]:
            ax.axis("off")
        plt.tight_layout()
        (OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
        plt.savefig(OUTPUT_ROOT / "figures" / "metric_vs_independent_references.pdf", dpi=300)
        plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
