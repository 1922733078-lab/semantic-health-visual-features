#!/usr/bin/env python3
"""Protocol v5: semantic-health repair and leakage-controlled reanalysis.

The five composites are treated as legacy software artefacts under audit, not
as externally validated perceptual scales.  All target and feature scaling is
fit on G-dev; baseline choice is made on G-dev and locked before G-test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import skimage
from scipy.special import ndtr
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.no_human.semantic_health import (
    assert_weighted_features_healthy,
    rank_equivalent_pairs,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/no_human/runs/run_20260720_semantic_health_v5"
POST_FEATURES = ROOT / "synthetic_benchmark/metadata/synthetic_traditional_features.csv"
PRE_FEATURES = OUT / "pre_fix_synthetic_traditional_features.csv"
TARGET_METADATA = ROOT / "synthetic_benchmark/metadata/target_metadata.csv"
N_BOOT = 2000
RNG_SEED = 260720

TARGET_MAP = {
    "visual_complexity_proxy": "structural_clutter_index",
    "layout_order_proxy": "geometric_order_index",
    "colour_harmony_proxy": "palette_coherence_index",
    "visual_intensity_proxy": "visual_salience_index",
    "layout_hierarchy_proxy": "focal_hierarchy_index",
}
TARGETS = list(TARGET_MAP.values())
SALIENCY_REPAIR_COLUMNS = [
    "rule_of_thirds", "center_offset_x", "center_offset_y", "fg_bg_ratio",
    "saliency_mean", "saliency_std", "num_salient_regions",
    "saliency_entropy", "num_saliency_peaks", "saliency_coverage",
]

# These are the exact historical formulas being audited.  Their names retain
# continuity with the released research object; the manuscript explicitly
# avoids interpreting them as validated human-perception constructs.
METRICS = {
    "visual_complexity_proxy": (
        ["num_dominant_colors", "color_entropy", "edge_density", "edge_orientation_entropy", "gradient_energy", "text_block_count", "whitespace_ratio", "symmetry"],
        [0.20, 0.15, 0.25, 0.15, 0.10, 0.05, -0.05, -0.05],
    ),
    "layout_order_proxy": (
        ["symmetry", "whitespace_ratio", "rule_of_thirds", "edge_density", "text_block_count", "lightness_contrast"],
        [0.25, 0.20, 0.15, -0.25, -0.10, -0.05],
    ),
    "colour_harmony_proxy": (
        ["color_harmony", "warm_color_ratio", "saturation_std", "hue_contrast", "num_dominant_colors"],
        [0.30, 0.25, 0.20, -0.15, -0.10],
    ),
    "visual_intensity_proxy": (
        ["saturation_mean", "lightness_contrast", "gradient_energy", "warm_color_ratio", "saturation_std"],
        [0.35, 0.25, 0.20, 0.10, 0.10],
    ),
    "layout_hierarchy_proxy": (
        ["edge_density", "saliency_std", "fg_bg_ratio", "text_block_count", "has_text", "saliency_mean", "whitespace_ratio"],
        [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, -0.05],
    ),
}

RAW_TARGET_COLUMNS = [
    "n_primitives", "overlap_ratio_param", "edge_density_factor",
    "spacing_irregularity", "actual_whitespace", "alignment_jitter",
    "grid_deviation", "spacing_variance", "hue_dispersion",
    "chroma_dispersion", "palette_incoherence", "contrast_ratio",
    "luminance_contrast", "edge_contrast", "focal_contrast",
    "focal_size_ratio", "secondary_count",
]

BASELINE_CANDIDATES = {
    "visual_complexity_proxy": [
        ("gray_entropy", "gray_entropy", 1.0),
        ("edge_density", "edge_density", 1.0),
        ("contour_density", "contour_density", 1.0),
    ],
    "layout_order_proxy": [
        ("symmetry", "symmetry", 1.0),
        ("spacing_regularity", "font_size_cv", -1.0),
        ("whitespace_ratio", "whitespace_ratio", 1.0),
    ],
    "colour_harmony_proxy": [
        ("colour_harmony", "color_harmony", 1.0),
        ("negative_colour_entropy", "color_entropy", -1.0),
        ("negative_colourfulness", "colourfulness", -1.0),
    ],
    "visual_intensity_proxy": [
        ("rms_contrast", "lightness_contrast", 1.0),
        ("luminance_spread", "value_std", 1.0),
        ("mean_saturation", "saturation_mean", 1.0),
    ],
    "layout_hierarchy_proxy": [
        ("saliency_dispersion", "saliency_std", 1.0),
        ("salient_area", "fg_bg_ratio", 1.0),
        ("edge_coverage", "edge_density", 1.0),
    ],
}


def rho(x, y) -> float:
    value = spearmanr(np.asarray(x), np.asarray(y)).statistic
    return float(value) if np.isfinite(value) else 0.0


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def development_frozen_targets(frame: pd.DataFrame):
    dev = frame.loc[frame["split"].eq("G-dev")]
    rows = []
    constants = {}
    for column in RAW_TARGET_COLUMNS:
        mean = float(dev[column].mean())
        sd = float(dev[column].std(ddof=1))
        if not np.isfinite(sd) or sd <= 0:
            raise RuntimeError(f"target input {column} is constant on G-dev")
        constants[column] = (mean, sd)
        rows.append({
            "raw_variable": column, "fit_split": "G-dev", "n_fit_rows": len(dev),
            "mean": mean, "sample_std": sd,
        })

    def z(column):
        mean, sd = constants[column]
        return (frame[column].to_numpy(float) - mean) / sd

    targets = pd.DataFrame({"image_id": frame["image_id"]})
    targets["structural_clutter_index"] = 100 * ndtr(
        (z("n_primitives") + 2*z("overlap_ratio_param") + z("edge_density_factor")
         + z("spacing_irregularity") - z("actual_whitespace")) / 4
    )
    targets["geometric_order_index"] = 100 * ndtr(
        (-z("alignment_jitter") - z("grid_deviation") - z("spacing_variance")
         - z("overlap_ratio_param")) / 4
    )
    targets["palette_coherence_index"] = 100 * ndtr(
        (-z("hue_dispersion") - z("chroma_dispersion")
         - z("palette_incoherence")) / 3
    )
    targets["visual_salience_index"] = 100 * ndtr(
        (z("contrast_ratio") + z("luminance_contrast") + z("edge_contrast")
         + z("focal_contrast")) / 4
    )
    targets["focal_hierarchy_index"] = 100 * ndtr(
        (z("focal_size_ratio") + z("focal_contrast") - z("secondary_count")
         - z("actual_whitespace")) / 4
    )
    return targets, pd.DataFrame(rows)


def cluster_draws(frame: pd.DataFrame):
    seeds = np.array(sorted(frame["seed"].unique()))
    by_seed = {
        seed: np.flatnonzero(frame["seed"].to_numpy() == seed) for seed in seeds
    }
    rng = np.random.default_rng(RNG_SEED)
    return [
        np.concatenate([by_seed[seed] for seed in rng.choice(seeds, len(seeds), replace=True)])
        for _ in range(N_BOOT)
    ]


def interval(x, y, draws):
    x, y = np.asarray(x), np.asarray(y)
    observed = rho(x, y)
    boot = np.array([rho(x[index], y[index]) for index in draws])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return observed, float(lo), float(hi)


def fit_and_score(frame: pd.DataFrame, version: str, strict: bool):
    dev_mask = frame["split"].eq("G-dev").to_numpy()
    scaler_rows, health_rows, bounds_rows, diagnostic_rows = [], [], [], []
    scores = pd.DataFrame(index=frame.index)
    raw_scores = pd.DataFrame(index=frame.index)
    for proxy, (features, weights) in METRICS.items():
        raw = np.zeros(len(frame), dtype=float)
        clipped_component_count = np.zeros(len(frame), dtype=int)
        retained_weight = 0.0
        total_weight = float(np.abs(weights).sum())
        for feature, weight in zip(features, weights):
            values = frame[feature].to_numpy(float)
            mean = float(values[dev_mask].mean())
            sd = float(values[dev_mask].std(ddof=1))
            retained = bool(np.isfinite(sd) and sd > 1e-12)
            if strict and not retained and weight != 0:
                raise RuntimeError(
                    f"active feature {feature} for {proxy} is constant on G-dev"
                )
            if retained:
                unbounded = (values - mean) / sd
                clipped_component_count += (np.abs(unbounded) > 4).astype(int)
                raw += np.clip(unbounded, -4, 4) * weight
                retained_weight += abs(weight)
            scaler_rows.append({
                "version": version, "proxy": proxy, "feature": feature,
                "fit_split": "G-dev", "mean": mean, "sample_std": sd,
                "retained": retained,
            })
            health_rows.append({
                "version": version, "proxy": proxy, "feature": feature,
                "nominal_weight": weight, "absolute_weight": abs(weight),
                "retained": retained,
                "effective_weight": weight if retained else 0.0,
                "n_unique_all": int(frame[feature].nunique(dropna=False)),
                "n_unique_gdev": int(frame.loc[dev_mask, feature].nunique(dropna=False)),
                "gdev_sample_std": sd,
            })
        low, high = np.quantile(raw[dev_mask], [0.005, 0.995])
        if high - low <= 1e-12:
            raise RuntimeError(f"raw composite {proxy} is constant on G-dev")
        unbounded_score = 100 * (raw - low) / (high - low)
        scores[proxy] = np.clip(unbounded_score, 0, 100)
        raw_scores[proxy] = raw
        bounds_rows.append({
            "version": version, "proxy": proxy, "fit_split": "G-dev",
            "raw_p0_5": low, "raw_p99_5": high,
            "retained_abs_weight_fraction": retained_weight / total_weight,
        })
        for split in ["G-dev", "G-validation", "G-test", "interaction-test"]:
            mask = frame["split"].eq(split).to_numpy()
            values = scores.loc[mask, proxy]
            counts = values.value_counts(normalize=True)
            diagnostic_rows.append({
                "version": version, "proxy": proxy, "split": split,
                "n": int(mask.sum()),
                "feature_z_clip_fraction": float(
                    clipped_component_count[mask].sum() / (mask.sum() * len(features))
                ),
                "final_score_clip_fraction": float(
                    ((unbounded_score[mask] < 0) | (unbounded_score[mask] > 100)).mean()
                ),
                "n_unique_scores": int(values.nunique(dropna=False)),
                "largest_exact_tie_fraction": float(counts.iloc[0]),
            })
    return (
        scores, raw_scores, pd.DataFrame(scaler_rows), pd.DataFrame(health_rows),
        pd.DataFrame(bounds_rows), pd.DataFrame(diagnostic_rows),
    )


def baseline_values(frame: pd.DataFrame, source: str, direction: float):
    if source == "contour_density":
        values = frame["edge_density"] * (1 - frame["whitespace_ratio"])
    elif source == "colourfulness":
        values = np.hypot(frame["saturation_mean"], frame["saturation_std"])
    else:
        values = frame[source]
    return values.to_numpy(float) * direction


def baseline_analysis(frame, scores, draws):
    dev = frame["split"].eq("G-dev").to_numpy()
    test = frame["split"].eq("G-test").to_numpy()
    candidate_rows, selected_rows = [], []
    for proxy, target in TARGET_MAP.items():
        candidates = []
        development_candidate_scores = {}
        for name, source, direction in BASELINE_CANDIDATES[proxy]:
            values = baseline_values(frame, source, direction)
            development_candidate_scores[name] = values[dev]
            dev_rho = rho(values[dev], frame.loc[dev, target])
            test_rho = rho(values[test], frame.loc[test, target])
            candidates.append((dev_rho, name, source, direction, values, test_rho))
        equivalents = rank_equivalent_pairs(pd.DataFrame(development_candidate_scores))
        if not equivalents.empty:
            detail = equivalents.to_dict(orient="records")
            raise RuntimeError(f"rank-equivalent baseline candidates for {proxy}: {detail}")
        candidates.sort(key=lambda value: (-value[0], value[1]))
        selected_name = candidates[0][1]
        for dev_rho, name, source, direction, _, test_rho in candidates:
            candidate_rows.append({
                "proxy": proxy, "target": target, "baseline": name,
                "source_feature": source, "direction": direction,
                "selection_split": "G-dev", "development_rho": dev_rho,
                "gtest_rho_diagnostic": test_rho,
                "selected_on_development": name == selected_name,
            })
        _, name, source, direction, values, baseline_rho = candidates[0]
        proposed = scores.loc[test, proxy].to_numpy(float)
        target_values = frame.loc[test, target].to_numpy(float)
        deltas = np.array([
            rho(proposed[index], target_values[index])
            - rho(values[test][index], target_values[index])
            for index in draws
        ])
        proposed_rho = rho(proposed, target_values)
        selected_rows.append({
            "proxy": proxy, "target": target, "baseline": name,
            "source_feature": source, "direction": direction,
            "selected_on": "G-dev only", "evaluated_on": "G-test only",
            "n_images": int(test.sum()), "n_seed_clusters": frame.loc[test, "seed"].nunique(),
            "proposed_rho": proposed_rho, "baseline_rho": baseline_rho,
            "delta_rho": proposed_rho - baseline_rho,
            "cluster_delta_ci_lower": float(np.quantile(deltas, 0.025)),
            "cluster_delta_ci_upper": float(np.quantile(deltas, 0.975)),
        })
    return pd.DataFrame(candidate_rows), pd.DataFrame(selected_rows)


def ridge_fit(x, y, alpha):
    median = np.median(x, axis=0)
    q25, q75 = np.quantile(x, [0.25, 0.75], axis=0)
    scale = q75 - q25
    scale[scale <= 1e-12] = 1.0
    design = np.clip((x - median) / scale, -8, 8)
    design = np.column_stack([np.ones(len(design)), design])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    if not np.isfinite(beta).all():
        raise RuntimeError("non-finite ridge coefficients")
    return median, scale, beta


def ridge_predict(model, x):
    median, scale, beta = model
    design = np.clip((x - median) / scale, -8, 8)
    design = np.column_stack([np.ones(len(design)), design])
    prediction = design @ beta
    if not np.isfinite(prediction).all():
        raise RuntimeError("non-finite ridge prediction")
    return prediction


def positive_controls(frame, features):
    numeric = list(features.select_dtypes(include=[np.number]).columns)
    dev = frame["split"].eq("G-dev").to_numpy()
    validation = frame["split"].eq("G-validation").to_numpy()
    train = dev | validation
    retained = [
        column for column in numeric
        if np.isfinite(frame.loc[dev, column]).all()
        and frame.loc[dev, column].std(ddof=1) > 1e-12
    ]
    rows = []
    for target in TARGETS:
        candidates = []
        for alpha in [0.1, 1.0, 10.0, 100.0]:
            model = ridge_fit(
                frame.loc[dev, retained].to_numpy(float),
                frame.loc[dev, target].to_numpy(float), alpha,
            )
            validation_rho = rho(
                ridge_predict(model, frame.loc[validation, retained].to_numpy(float)),
                frame.loc[validation, target],
            )
            candidates.append((validation_rho, alpha))
        validation_rho, alpha = max(candidates, key=lambda value: (value[0], -value[1]))
        model = ridge_fit(
            frame.loc[train, retained].to_numpy(float),
            frame.loc[train, target].to_numpy(float), alpha,
        )
        for split in ["G-dev", "G-validation", "G-test", "interaction-test"]:
            mask = frame["split"].eq(split).to_numpy()
            rows.append({
                "target": target, "split": split, "n": int(mask.sum()),
                "n_candidate_numeric_features": len(numeric),
                "n_nonconstant_development_features": len(retained),
                "selected_alpha_on_gvalidation": alpha,
                "validation_rho_used_for_selection": validation_rho,
                "training_splits": "G-dev+G-validation after alpha selection",
                "rho": rho(
                    ridge_predict(model, frame.loc[mask, retained].to_numpy(float)),
                    frame.loc[mask, target],
                ),
            })
    return pd.DataFrame(rows), retained


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(TARGET_METADATA)
    pre_features = pd.read_csv(PRE_FEATURES)
    post_features = pd.read_csv(POST_FEATURES)
    if len(metadata) != 16425 or len(pre_features) != 16425 or len(post_features) != 16425:
        raise RuntimeError("protocol v5 requires exactly 16,425 rows in every primary input")
    target_values, target_constants = development_frozen_targets(metadata)
    metadata = metadata.drop(columns=TARGETS).merge(
        target_values, on="image_id", validate="one_to_one"
    )
    pre = metadata.merge(pre_features, on="image_id", validate="one_to_one")
    post = metadata.merge(post_features, on="image_id", validate="one_to_one")
    saliency_only_features = pre_features.copy().set_index("image_id")
    post_by_id = post_features.set_index("image_id")
    saliency_only_features.loc[:, SALIENCY_REPAIR_COLUMNS] = post_by_id.loc[
        saliency_only_features.index, SALIENCY_REPAIR_COLUMNS
    ].to_numpy()
    saliency_only_features = saliency_only_features.reset_index()
    saliency_only = metadata.merge(
        saliency_only_features, on="image_id", validate="one_to_one"
    )

    development_mask = post["split"].eq("G-dev").to_numpy()
    semantic_gate = assert_weighted_features_healthy(
        post, METRICS, development_mask
    )
    semantic_gate.insert(0, "fit_split", "G-dev")
    semantic_gate.to_csv(OUT / "semantic_health_gate.csv", index=False)

    pre_result = fit_and_score(pre, "pre_fix_silent_fallback", strict=False)
    saliency_only_result = fit_and_score(
        saliency_only, "post_fix_saliency_columns_only", strict=True
    )
    post_result = fit_and_score(post, "post_fix_spectral_residual", strict=True)
    pre_scores, _, pre_scaler, pre_health, pre_bounds, pre_diag = pre_result
    repair_scores, _, repair_scaler, repair_health, repair_bounds, repair_diag = saliency_only_result
    post_scores, _, post_scaler, post_health, post_bounds, post_diag = post_result

    expected_bad = {
        "saliency_mean", "saliency_std", "fg_bg_ratio", "rule_of_thirds",
        "center_offset_x", "center_offset_y",
    }
    for feature in expected_bad:
        if pre[feature].nunique(dropna=False) != 1:
            raise RuntimeError(f"pre-fix control {feature} is unexpectedly nonconstant")
        if post[feature].nunique(dropna=False) <= 1:
            raise RuntimeError(f"post-fix semantic-health gate failed for {feature}")

    target_constants.to_csv(OUT / "development_target_constants.csv", index=False)
    target_values.to_csv(OUT / "development_frozen_targets.csv", index=False)
    pd.concat([pre_scaler, repair_scaler, post_scaler]).to_csv(
        OUT / "development_feature_scalers.csv", index=False
    )
    health = pd.concat([pre_health, repair_health, post_health], ignore_index=True)
    health.to_csv(OUT / "weighted_feature_health.csv", index=False)
    pd.concat([pre_bounds, repair_bounds, post_bounds]).to_csv(
        OUT / "development_score_bounds.csv", index=False
    )
    pd.concat([pre_diag, repair_diag, post_diag]).to_csv(
        OUT / "feature_score_diagnostics.csv", index=False
    )

    test_mask = post["split"].eq("G-test").to_numpy()
    test = post.loc[test_mask].reset_index(drop=True)
    test_scores = post_scores.loc[test_mask].reset_index(drop=True)
    pre_test_scores = pre_scores.loc[test_mask].reset_index(drop=True)
    repair_test_scores = repair_scores.loc[test_mask].reset_index(drop=True)
    draws = cluster_draws(test)
    all_rows, comparison_rows = [], []
    for proxy, nominal_target in TARGET_MAP.items():
        for target in TARGETS:
            observed, lo, hi = interval(test_scores[proxy], test[target], draws)
            all_rows.append({
                "proxy": proxy, "target": target, "n_images": len(test),
                "n_seed_clusters": test["seed"].nunique(), "rho": observed,
                "cluster_ci_lower": lo, "cluster_ci_upper": hi,
                "is_nominal_target": target == nominal_target,
                "target_scaling": "G-dev mean/std frozen",
                "feature_scaling": "G-dev mean/std and score bounds frozen",
            })
        pre_rho, pre_lo, pre_hi = interval(
            pre_test_scores[proxy], test[nominal_target], draws
        )
        repair_rho, repair_lo, repair_hi = interval(
            repair_test_scores[proxy], test[nominal_target], draws
        )
        post_rho, post_lo, post_hi = interval(
            test_scores[proxy], test[nominal_target], draws
        )
        pre_retained = pre_bounds.loc[pre_bounds["proxy"].eq(proxy), "retained_abs_weight_fraction"].iloc[0]
        post_retained = post_bounds.loc[post_bounds["proxy"].eq(proxy), "retained_abs_weight_fraction"].iloc[0]
        comparison_rows.append({
            "proxy": proxy, "target": nominal_target,
            "pre_fix_rho": pre_rho, "pre_fix_ci_lower": pre_lo,
            "pre_fix_ci_upper": pre_hi,
            "post_fix_rho": repair_rho, "post_fix_ci_lower": repair_lo,
            "post_fix_ci_upper": repair_hi,
            "rho_change": repair_rho - pre_rho,
            "comparison_scope": "only saliency-derived columns replaced",
            "full_pipeline_post_fix_rho": post_rho,
            "full_pipeline_post_fix_ci_lower": post_lo,
            "full_pipeline_post_fix_ci_upper": post_hi,
            "full_pipeline_rho_change": post_rho - pre_rho,
            "pre_fix_retained_abs_weight_fraction": pre_retained,
            "post_fix_retained_abs_weight_fraction": post_retained,
        })
    full = pd.DataFrame(all_rows)
    full.to_csv(OUT / "full_25_proxy_target_cluster.csv", index=False)
    primary = full.loc[full["is_nominal_target"]].copy()
    non = (
        full.loc[~full["is_nominal_target"]]
        .assign(abs_rho=lambda value: value["rho"].abs())
        .sort_values("abs_rho").groupby("proxy").tail(1).set_index("proxy")
    )
    primary["max_abs_non_target_rho"] = primary["proxy"].map(non["abs_rho"])
    primary["max_abs_non_target_name"] = primary["proxy"].map(non["target"])
    primary["selectivity_margin"] = (
        primary["rho"].abs() - primary["max_abs_non_target_rho"]
    )
    primary.to_csv(OUT / "primary_cluster_results.csv", index=False)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT / "pre_post_nominal_comparison.csv", index=False)

    candidates, selected = baseline_analysis(post, post_scores, draws)
    candidates.to_csv(OUT / "baseline_candidate_development_selection.csv", index=False)
    selected.to_csv(OUT / "paired_baseline_cluster.csv", index=False)
    controls, retained_features = positive_controls(post, post_features)
    controls.to_csv(OUT / "positive_control_results.csv", index=False)
    (OUT / "positive_control_feature_list.txt").write_text(
        "\n".join(retained_features) + "\n", encoding="utf-8"
    )

    correlation_rows = []
    for split in ["G-dev", "G-validation", "G-test", "interaction-test"]:
        matrix = post.loc[post["split"].eq(split), TARGETS].corr(method="spearman")
        for row_target in TARGETS:
            for column_target in TARGETS:
                correlation_rows.append({
                    "split": split, "row_target": row_target,
                    "column_target": column_target,
                    "spearman_rho": matrix.loc[row_target, column_target],
                })
    pd.DataFrame(correlation_rows).to_csv(
        OUT / "target_correlation_by_split.csv", index=False
    )
    post.groupby(["split", "family"], dropna=False).agg(
        n_images=("image_id", "size"), n_seed_clusters=("seed", "nunique"),
        min_seed=("seed", "min"), max_seed=("seed", "max"),
    ).reset_index().to_csv(OUT / "analysis_units.csv", index=False)

    labels = ["Complexity", "Order", "Harmony", "Intensity", "Hierarchy"]
    x = np.arange(5)
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.bar(x - 0.18, comparison["pre_fix_retained_abs_weight_fraction"], 0.36,
           label="Silent-fallback release")
    ax.bar(x + 0.18, comparison["post_fix_retained_abs_weight_fraction"], 0.36,
           label="Semantic-health repair")
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.08); ax.set_ylabel("Retained absolute formula weight")
    ax.set_title("Semantic health exposes inactive weighted features")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(OUT / "semantic_health_weight_retention.png", dpi=300,
                metadata={"Software": "Matplotlib protocol-v5 renderer"})
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.bar(x - 0.18, comparison["pre_fix_rho"], 0.36, label="Pre-fix")
    ax.bar(x + 0.18, comparison["post_fix_rho"], 0.36, label="Post-fix")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("G-test nominal Spearman rho")
    ax.set_title("Conclusion sensitivity to a silent feature fallback")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(OUT / "pre_post_nominal_correlations.png", dpi=300,
                metadata={"Software": "Matplotlib protocol-v5 renderer"})
    plt.close(fig)

    run_metadata = {
        "protocol": "5.0-semantic-health-failure-analysis",
        "run_id": "run_20260720_semantic_health_v5",
        "rng_seed": RNG_SEED, "n_cluster_bootstrap": N_BOOT,
        "primary_evaluation_split": "G-test",
        "target_scaler_fit_split": "G-dev",
        "feature_scaler_fit_split": "G-dev",
        "baseline_selection_split": "G-dev",
        "baseline_evaluation_split": "G-test",
        "saliency_algorithm": "spectral residual; Hou and Zhang 2007; DOI 10.1109/CVPR.2007.383267",
        "pre_post_causal_comparison": "only saliency-derived columns replaced; full repaired pipeline reported separately",
        "saliency_repair_columns": SALIENCY_REPAIR_COLUMNS,
        "composite_role": "legacy software artefacts under audit; not validated perceptual scales",
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_image": skimage.__version__,
            "opencv": cv2.__version__, "matplotlib": matplotlib.__version__,
        },
        "inputs": {
            str(PRE_FEATURES.relative_to(ROOT)): file_hash(PRE_FEATURES),
            str(POST_FEATURES.relative_to(ROOT)): file_hash(POST_FEATURES),
            str(TARGET_METADATA.relative_to(ROOT)): file_hash(TARGET_METADATA),
        },
    }
    (OUT / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "artifact_hashes.csv":
            rows.append({
                "file": path.name, "sha256": file_hash(path),
                "bytes": path.stat().st_size,
            })
    pd.DataFrame(rows).to_csv(OUT / "artifact_hashes.csv", index=False)
    print(comparison.to_string(index=False))
    print("\nDevelopment-selected baselines evaluated on G-test:\n")
    print(selected.to_string(index=False))


if __name__ == "__main__":
    main()
