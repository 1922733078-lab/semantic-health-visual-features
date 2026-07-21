#!/usr/bin/env python3
"""
Compute objective target variables from synthetic benchmark metadata.

Target formulas are defined in synthetic_benchmark/PROTOCOL.md and must not use
pixel features extracted by the proposed proxy.

Usage:
    python synthetic_benchmark/generator/compute_objective_targets.py
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
METADATA_DIR = ROOT / "synthetic_benchmark" / "metadata"
TARGET_CSV = METADATA_DIR / "target_metadata.csv"


def standard_normal_cdf(z):
    """Map z-score to [0, 1]."""
    return norm.cdf(z)


def z_score(series):
    """Standardize a pandas Series to zero mean, unit std."""
    std = float(series.std())
    if std == 0:
        std = 1.0
    return (series - series.mean()) / std


def compute_targets(metadata_df):
    df = metadata_df.copy()

    # Standardize raw parameters over the full generated set
    z_n = z_score(df["n_primitives"])
    z_overlap = z_score(df["overlap_ratio_param"])
    z_edge = z_score(df["edge_density_factor"])
    z_spacing = z_score(df["spacing_irregularity"])
    z_whitespace = z_score(df["actual_whitespace"])

    z_jitter = z_score(df["alignment_jitter"])
    z_grid = z_score(df["grid_deviation"])
    z_spacing_var = z_score(df["spacing_variance"])

    z_hue = z_score(df["hue_dispersion"])
    z_chroma = z_score(df["chroma_dispersion"])
    z_palette = z_score(df["palette_incoherence"])

    z_contrast = z_score(df["contrast_ratio"])
    z_lum = z_score(df["luminance_contrast"])
    z_edge_contrast = z_score(df["edge_contrast"])
    z_focal_contrast = z_score(df["focal_contrast"])

    z_focal_size = z_score(df["focal_size_ratio"])
    z_secondary = z_score(df["secondary_count"])

    # Structural clutter index (visual complexity target)
    clutter = z_n + 2 * z_overlap + z_edge + z_spacing - z_whitespace
    df["structural_clutter_index"] = 100 * standard_normal_cdf(clutter / 4)

    # Geometric order index (layout order target)
    order = -z_jitter - z_grid - z_spacing_var - z_overlap
    df["geometric_order_index"] = 100 * standard_normal_cdf(order / 4)

    # Palette coherence index (colour harmony target)
    coherence = -z_hue - z_chroma - z_palette
    df["palette_coherence_index"] = 100 * standard_normal_cdf(coherence / 3)

    # Visual salience index (visual intensity target)
    salience = z_contrast + z_lum + z_edge_contrast + z_focal_contrast
    df["visual_salience_index"] = 100 * standard_normal_cdf(salience / 4)

    # Focal hierarchy index (layout hierarchy target)
    hierarchy = z_focal_size + z_focal_contrast - z_secondary - z_whitespace
    df["focal_hierarchy_index"] = 100 * standard_normal_cdf(hierarchy / 4)

    # Target specification hash
    formula = {
        "structural_clutter_index": "(z_n + 2*z_overlap + z_edge + z_spacing - z_whitespace) / 4",
        "geometric_order_index": "(-z_jitter - z_grid - z_spacing_var - z_overlap) / 4",
        "palette_coherence_index": "(-z_hue - z_chroma - z_palette) / 3",
        "visual_salience_index": "(z_contrast + z_lum + z_edge_contrast + z_focal_contrast) / 4",
        "focal_hierarchy_index": "(z_focal_size + z_focal_contrast - z_secondary - z_whitespace) / 4",
    }
    df["target_formula_version"] = hashlib.sha256(
        json.dumps(formula, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=METADATA_DIR / "image_metadata.csv")
    parser.add_argument("--output", type=Path, default=TARGET_CSV)
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    print(f"Loaded {len(df)} metadata rows.")

    df = compute_targets(df)
    df.to_csv(args.output, index=False)
    print(f"Wrote target metadata to {args.output}")

    # Print summary by construct and level
    cols = [
        "structural_clutter_index",
        "geometric_order_index",
        "palette_coherence_index",
        "visual_salience_index",
        "focal_hierarchy_index",
    ]
    print("\nMean target values by construct/level:")
    summary = df.groupby(["construct", "level"])[cols].mean()
    print(summary.round(2))


if __name__ == "__main__":
    main()
