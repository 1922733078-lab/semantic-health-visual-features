#!/usr/bin/env python3
"""
Mandatory non-learned baselines for the no-human proxy comparison.

Each baseline is a deterministic, independently implemented metric computed from
standard image descriptors. They are not optimized against human scores.
"""

import numpy as np
import pandas as pd


def compute_baselines(features_df):
    """Compute baseline scores from a traditional-features DataFrame."""
    df = features_df.copy()
    baselines = pd.DataFrame({"image_id": df["image_id"]})

    # Visual complexity baselines
    baselines["baseline_complexity_gray_entropy"] = df["gray_entropy"]
    baselines["baseline_complexity_edge_density"] = df["edge_density"]
    # Contour density proxy: edge_density * (1 - whitespace_ratio)
    baselines["baseline_complexity_contour_density"] = df["edge_density"] * (1 - df["whitespace_ratio"])

    # Layout order baselines
    # Alignment dispersion proxy: 1 - symmetry
    baselines["baseline_order_alignment"] = 1 - df["symmetry"]
    # Spacing CV proxy: font_size_cv (higher = less regular)
    baselines["baseline_order_spacing_cv"] = df["font_size_cv"]
    # Symmetry score (higher = more ordered)
    baselines["baseline_order_symmetry"] = df["symmetry"]

    # Colour harmony baselines
    baselines["baseline_harmony_hue_entropy"] = df["color_entropy"]
    # Palette size proxy: num_dominant_colors (more = less harmonious)
    baselines["baseline_harmony_palette_size"] = df["num_dominant_colors"]
    # Colourfulness proxy
    baselines["baseline_harmony_colourfulness"] = np.sqrt(
        df["saturation_std"] ** 2 + df["saturation_mean"] ** 2
    )

    # Visual intensity baselines
    baselines["baseline_intensity_rms_contrast"] = df["lightness_contrast"]
    baselines["baseline_intensity_luminance_range"] = df["value_std"]
    baselines["baseline_intensity_saturation"] = df["saturation_mean"]

    # Layout hierarchy baselines
    # Saliency concentration: std of saliency map
    baselines["baseline_hierarchy_saliency_std"] = df["saliency_std"]
    # Area concentration: fg_bg_ratio
    baselines["baseline_hierarchy_area_concentration"] = df["fg_bg_ratio"]
    # Edge coverage is an elementary pixel baseline, not a text detector.
    baselines["baseline_hierarchy_edge_coverage"] = df["edge_density"]

    return baselines


BASELINE_TARGET_MAP = {
    "visual_complexity_proxy": [
        ("baseline_complexity_gray_entropy", 1),
        ("baseline_complexity_edge_density", 1),
        ("baseline_complexity_contour_density", 1),
    ],
    "layout_order_proxy": [
        ("baseline_order_alignment", -1),
        ("baseline_order_spacing_cv", -1),
        ("baseline_order_symmetry", 1),
    ],
    "colour_harmony_proxy": [
        ("baseline_harmony_hue_entropy", -1),
        ("baseline_harmony_palette_size", -1),
        ("baseline_harmony_colourfulness", -1),
    ],
    "visual_intensity_proxy": [
        ("baseline_intensity_rms_contrast", 1),
        ("baseline_intensity_luminance_range", 1),
        ("baseline_intensity_saturation", 1),
    ],
    "layout_hierarchy_proxy": [
        ("baseline_hierarchy_saliency_std", 1),
        ("baseline_hierarchy_area_concentration", 1),
        ("baseline_hierarchy_edge_coverage", 1),
    ],
}


def best_baseline_for_target(baselines_df, target_col, candidates, direction=1):
    """Return the baseline name with highest absolute Spearman correlation to target."""
    from scipy.stats import spearmanr
    best_name = None
    best_rho = np.nan
    best_abs_rho = -np.inf
    for col in candidates:
        rho, _ = spearmanr(baselines_df[col].values, target_col)
        rho = rho * direction
        if np.isfinite(rho) and abs(rho) > best_abs_rho:
            best_abs_rho = abs(rho)
            best_rho = rho
            best_name = col
    if best_name is None:
        raise ValueError("no baseline candidate produced a finite Spearman correlation")
    return best_name, best_rho
