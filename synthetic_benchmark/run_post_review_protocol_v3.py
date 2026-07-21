#!/usr/bin/env python3
"""Deterministic post-review reanalysis of the frozen synthetic benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/no_human/runs/run_20260718_protocol_v3/post_review_v3"
FEATURES = ROOT / "synthetic_benchmark/metadata/synthetic_traditional_features.csv"
TARGETS = ROOT / "synthetic_benchmark/metadata/target_metadata.csv"
N_BOOT = 2000
RNG_SEED = 260718

TARGET_MAP = {
    "visual_complexity_proxy": "structural_clutter_index",
    "layout_order_proxy": "geometric_order_index",
    "colour_harmony_proxy": "palette_coherence_index",
    "visual_intensity_proxy": "visual_salience_index",
    "layout_hierarchy_proxy": "focal_hierarchy_index",
}
TARGETS_5 = list(TARGET_MAP.values())

BASELINE_MAP = {
    "visual_complexity_proxy": ("gray_entropy", 1.0, "baseline_complexity_gray_entropy"),
    "layout_order_proxy": ("symmetry", 1.0, "baseline_order_symmetry"),
    "colour_harmony_proxy": ("colourfulness", -1.0, "baseline_harmony_colourfulness"),
    "visual_intensity_proxy": ("value_std", 1.0, "baseline_intensity_luminance_range"),
    "layout_hierarchy_proxy": ("edge_density", 1.0, "baseline_hierarchy_edge_coverage"),
}

METRICS = {
    "visual_complexity_proxy": (["num_dominant_colors", "color_entropy", "edge_density", "edge_orientation_entropy", "gradient_energy", "text_block_count", "whitespace_ratio", "symmetry"], [0.20, 0.15, 0.25, 0.15, 0.10, 0.05, -0.05, -0.05]),
    "layout_order_proxy": (["symmetry", "whitespace_ratio", "rule_of_thirds", "edge_density", "text_block_count", "lightness_contrast"], [0.25, 0.20, 0.15, -0.25, -0.10, -0.05]),
    "colour_harmony_proxy": (["color_harmony", "warm_color_ratio", "saturation_std", "hue_contrast", "num_dominant_colors"], [0.30, 0.25, 0.20, -0.15, -0.10]),
    "visual_intensity_proxy": (["saturation_mean", "lightness_contrast", "gradient_energy", "warm_color_ratio", "saturation_std"], [0.35, 0.25, 0.20, 0.10, 0.10]),
    "layout_hierarchy_proxy": (["edge_density", "saliency_std", "fg_bg_ratio", "text_block_count", "has_text", "saliency_mean", "whitespace_ratio"], [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, -0.05]),
}


def rho(x, y) -> float:
    value = spearmanr(np.asarray(x), np.asarray(y)).statistic
    return float(value) if np.isfinite(value) else 0.0


def cluster_draws(df: pd.DataFrame) -> list[np.ndarray]:
    seeds = np.array(sorted(df["seed"].unique()))
    by_seed = {seed: np.flatnonzero(df["seed"].to_numpy() == seed) for seed in seeds}
    rng = np.random.default_rng(RNG_SEED)
    draws = []
    for _ in range(N_BOOT):
        selected = rng.choice(seeds, size=len(seeds), replace=True)
        draws.append(np.concatenate([by_seed[s] for s in selected]))
    return draws


def cluster_stats(x, y, draws) -> tuple[float, float, float]:
    observed = rho(x, y)
    boot = np.array([rho(np.asarray(x)[idx], np.asarray(y)[idx]) for idx in draws])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return observed, float(lo), float(hi)


def proxy_scores(df: pd.DataFrame) -> pd.DataFrame:
    config = json.loads(json.dumps({}))
    del config
    preproc = pd.read_csv(ROOT / "synthetic_benchmark/metadata/synthetic_traditional_features.csv", nrows=0)
    del preproc
    import yaml
    frozen = yaml.safe_load((ROOT / "configs/no_human_preprocessing.yaml").read_text(encoding="utf-8"))
    stats = frozen["normalization_constants"]
    bounds = frozen["percentile_bounds"]
    out = pd.DataFrame(index=df.index)
    for proxy, (features, weights) in METRICS.items():
        raw = np.zeros(len(df), dtype=float)
        for feature, weight in zip(features, weights):
            mean = stats["mean"].get(feature, float(df[feature].mean()))
            sd = stats["std"].get(feature, float(df[feature].std())) or 1.0
            raw += np.clip((df[feature].to_numpy() - mean) / sd, -4, 4) * weight
        low, high = bounds[proxy]["p0.5"], bounds[proxy]["p99.5"]
        out[proxy] = np.clip(100 * (raw - low) / (high - low + 1e-10), 0, 100)
    return out


def colourfulness(df: pd.DataFrame) -> np.ndarray:
    return np.sqrt(np.square(df["saturation_mean"].to_numpy()) + np.square(df["saturation_std"].to_numpy()))


def independent_target_audit(df: pd.DataFrame) -> pd.DataFrame:
    def z(column):
        a = df[column].to_numpy(dtype=float)
        return (a - a.mean()) / a.std(ddof=1)
    recomputed = {
        "structural_clutter_index": 100 * ndtr((z("n_primitives") + 2*z("overlap_ratio_param") + z("edge_density_factor") + z("spacing_irregularity") - z("actual_whitespace")) / 4),
        "geometric_order_index": 100 * ndtr((-z("alignment_jitter") - z("grid_deviation") - z("spacing_variance") - z("overlap_ratio_param")) / 4),
        "palette_coherence_index": 100 * ndtr((-z("hue_dispersion") - z("chroma_dispersion") - z("palette_incoherence")) / 3),
        "visual_salience_index": 100 * ndtr((z("contrast_ratio") + z("luminance_contrast") + z("edge_contrast") + z("focal_contrast")) / 4),
        "focal_hierarchy_index": 100 * ndtr((z("focal_size_ratio") + z("focal_contrast") - z("secondary_count") - z("actual_whitespace")) / 4),
    }
    return pd.DataFrame([{
        "target": target,
        # Platform BLAS reductions can differ at roughly 1e-14.  The audit is a
        # 1e-10 tolerance test, so serialise at 1e-12 precision for stable files.
        "max_abs_difference": round(float(np.max(np.abs(values - df[target].to_numpy()))), 12),
        "passed_1e_10": bool(np.max(np.abs(values - df[target].to_numpy())) < 1e-10),
    } for target, values in recomputed.items()])


def ridge_fit(x_train, y_train, alpha=10.0):
    median = np.nanmedian(x_train, axis=0)
    q25 = np.nanquantile(x_train, 0.25, axis=0)
    q75 = np.nanquantile(x_train, 0.75, axis=0)
    scale = q75 - q25
    scale[scale == 0] = 1.0
    xt = np.clip(np.nan_to_num((x_train - median) / scale), -8, 8)
    xt = np.column_stack([np.ones(len(xt)), xt])
    penalty = np.eye(xt.shape[1]) * alpha
    penalty[0, 0] = 0.0
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        beta = np.linalg.solve(xt.T @ xt + penalty, xt.T @ y_train)
    if not np.isfinite(beta).all():
        raise RuntimeError("Non-finite ridge coefficients")
    return median, scale, beta


def ridge_predict(model, x_eval):
    median, scale, beta = model
    xe = np.clip(np.nan_to_num((x_eval - median) / scale), -8, 8)
    xe = np.column_stack([np.ones(len(xe)), xe])
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        prediction = xe @ beta
    if not np.isfinite(prediction).all():
        raise RuntimeError("Non-finite ridge predictions")
    return prediction


def main():
    print("Protocol v3.2: stable-seed generator 2.1 inputs; ten-seed cluster reanalysis; no bootstrap p/q values")
    OUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(FEATURES)
    targets = pd.read_csv(TARGETS)
    if len(features) != 16425 or len(targets) != 16425:
        raise RuntimeError("Protocol v3 requires exactly 16,425 frozen rows")
    df = targets.merge(features, on="image_id", how="inner", validate="one_to_one")
    scores = proxy_scores(df)
    gtest = df["split"].eq("G-test")
    test_df = df.loc[gtest].reset_index(drop=True)
    test_scores = scores.loc[gtest].reset_index(drop=True)
    draws = cluster_draws(test_df)

    full_rows = []
    for proxy in TARGET_MAP:
        for target in TARGETS_5:
            observed, lo, hi = cluster_stats(test_scores[proxy], test_df[target], draws)
            full_rows.append({"proxy": proxy, "target": target, "n_images": len(test_df), "n_seed_clusters": test_df.seed.nunique(), "rho": observed, "cluster_ci_lower": lo, "cluster_ci_upper": hi, "is_nominal_target": target == TARGET_MAP[proxy]})
    full = pd.DataFrame(full_rows)
    full.to_csv(OUT / "full_25_proxy_target_cluster.csv", index=False)

    primary = full[full["is_nominal_target"]].copy()
    max_non = full[~full["is_nominal_target"]].assign(abs_rho=lambda x: x.rho.abs()).sort_values("abs_rho").groupby("proxy").tail(1).set_index("proxy")
    primary["max_abs_non_target_rho"] = primary["proxy"].map(max_non["abs_rho"])
    primary["max_abs_non_target_name"] = primary["proxy"].map(max_non["target"])
    primary["selectivity_margin"] = primary["rho"].abs() - primary["max_abs_non_target_rho"]
    primary.to_csv(OUT / "primary_cluster_results.csv", index=False)

    per_seed = []
    for seed, group in test_df.groupby("seed"):
        idx = group.index
        for proxy in TARGET_MAP:
            for target in TARGETS_5:
                per_seed.append({"seed": int(seed), "proxy": proxy, "target": target, "rho": rho(test_scores.loc[idx, proxy], group[target]), "is_nominal_target": target == TARGET_MAP[proxy]})
    pd.DataFrame(per_seed).to_csv(OUT / "per_seed_proxy_target.csv", index=False)

    base_rows = []
    for proxy, target in TARGET_MAP.items():
        col, direction, label = BASELINE_MAP[proxy]
        baseline = colourfulness(test_df) * direction if col == "colourfulness" else test_df[col].to_numpy() * direction
        proposed = test_scores[proxy].to_numpy()
        y = test_df[target].to_numpy()
        prop_rho, base_rho = rho(proposed, y), rho(baseline, y)
        deltas = np.array([rho(proposed[idx], y[idx]) - rho(baseline[idx], y[idx]) for idx in draws])
        base_rows.append({"proxy": proxy, "target": target, "baseline": label, "n_seed_clusters": test_df.seed.nunique(), "proposed_rho": prop_rho, "baseline_rho": base_rho, "delta_rho": prop_rho-base_rho, "cluster_delta_ci_lower": np.quantile(deltas, .025), "cluster_delta_ci_upper": np.quantile(deltas, .975)})
    pd.DataFrame(base_rows).to_csv(OUT / "paired_baseline_cluster.csv", index=False)

    numeric_features = list(features.select_dtypes(include=[np.number]).columns)
    train = df["split"].isin(["G-dev", "G-validation"])
    x_train = df.loc[train, numeric_features].to_numpy(dtype=float)
    rng = np.random.default_rng(RNG_SEED)
    positive, negative = [], []
    for target in TARGETS_5:
        y_train = df.loc[train, target].to_numpy(dtype=float)
        positive_model = ridge_fit(x_train, y_train)
        permutation_models = [ridge_fit(x_train, rng.permutation(y_train)) for _ in range(100)]
        for split in ["G-dev", "G-validation", "G-test", "interaction-test"]:
            mask = df["split"].eq(split)
            x_eval = df.loc[mask, numeric_features].to_numpy(dtype=float)
            actual = df.loc[mask, target].to_numpy(dtype=float)
            positive.append({"target": target, "split": split, "n": int(mask.sum()), "rho": rho(ridge_predict(positive_model, x_eval), actual), "training_splits": "G-dev+G-validation", "model": "ridge_alpha_10_median_iqr_clip8"})
            null_rhos = np.array([rho(ridge_predict(model, x_eval), actual) for model in permutation_models])
            negative.append({"target": target, "split": split, "n": int(mask.sum()), "n_permutations": len(null_rhos), "mean_rho": null_rhos.mean(), "rho_q025": np.quantile(null_rhos, .025), "rho_q975": np.quantile(null_rhos, .975), "max_abs_rho": np.max(np.abs(null_rhos)), "training_splits": "G-dev+G-validation", "model": "permuted_target_ridge"})
    positive_df, negative_df = pd.DataFrame(positive), pd.DataFrame(negative)
    positive_df.to_csv(OUT / "positive_control_results.csv", index=False)
    negative_df.to_csv(OUT / "negative_control_results.csv", index=False)

    construct_target = {
        "visual_complexity": ("structural_clutter_index", "increase"),
        "layout_order": ("geometric_order_index", "decrease"),
        "colour_harmony": ("palette_coherence_index", "decrease"),
        "visual_intensity": ("visual_salience_index", "increase"),
        "layout_hierarchy": ("focal_hierarchy_index", "increase"),
    }
    mono = []
    for split in ["G-dev", "G-validation", "G-test"]:
        for construct, (target, expected_direction) in construct_target.items():
            sub = df[df["split"].eq(split) & df["construct"].eq(construct)]
            means = sub.groupby("level")[target].mean().sort_index()
            diffs = np.diff(means.values)
            passed = np.all(diffs > 0) if expected_direction == "increase" else np.all(diffs < 0)
            mono.append({"split": split, "family": sub.family.iloc[0], "construct": construct, "target": target, "n": len(sub), "expected_direction": expected_direction, "level_target_rho": rho(means.index, means.values), "passed_expected_monotonicity": bool(passed), **{f"level_{int(k)}_mean": v for k, v in means.items()}})
    pd.DataFrame(mono).to_csv(OUT / "target_level_monotonicity.csv", index=False)

    corr_rows = []
    for split in ["G-dev", "G-validation", "G-test", "interaction-test"]:
        matrix = df[df["split"].eq(split)][TARGETS_5].corr(method="spearman")
        for row_target in TARGETS_5:
            for col_target in TARGETS_5:
                corr_rows.append({"split": split, "row_target": row_target, "column_target": col_target, "spearman_rho": matrix.loc[row_target, col_target]})
    pd.DataFrame(corr_rows).to_csv(OUT / "target_correlation_by_split.csv", index=False)
    independent_target_audit(df).to_csv(OUT / "independent_target_recomputation.csv", index=False)

    units = df.groupby(["split", "family"], dropna=False).agg(n_images=("image_id", "size"), n_seeds=("seed", "nunique"), min_seed=("seed", "min"), max_seed=("seed", "max")).reset_index()
    units.to_csv(OUT / "analysis_units.csv", index=False)

    matrix = test_df[TARGETS_5].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.4, 7.0))
    image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    labels = ["Clutter", "Order", "Palette", "Salience", "Hierarchy"]
    ax.set_xticks(range(5), labels, rotation=35, ha="right")
    ax.set_yticks(range(5), labels)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{matrix.iloc[i,j]:.2f}", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, label="Spearman rho")
    ax.set_title("G-test scene-graph target dependence")
    fig.tight_layout(); fig.savefig(
        OUT / "target_correlation_gtest.png", dpi=300,
        metadata={"Software": "Matplotlib deterministic protocol-v3 renderer"},
    ); plt.close(fig)

    plot = positive_df[positive_df["split"].isin(["G-test", "interaction-test"])].pivot(index="target", columns="split", values="rho").loc[TARGETS_5]
    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    x = np.arange(5); width = .36
    ax.bar(x-width/2, plot["G-test"], width, label="G-test")
    ax.bar(x+width/2, plot["interaction-test"], width, label="Interaction test")
    ax.axhline(.60, color="black", ls="--", lw=1, label="rho = 0.60 diagnostic line")
    ax.set_xticks(x, labels, rotation=25, ha="right"); ax.set_ylabel("Spearman rho")
    ax.set_ylim(-.1, 1); ax.legend(frameon=False); ax.set_title("Pixel-side ridge recoverability control")
    fig.tight_layout(); fig.savefig(
        OUT / "positive_control_gtest.png", dpi=300,
        metadata={"Software": "Matplotlib deterministic protocol-v3 renderer"},
    ); plt.close(fig)

    metadata = {"protocol": "3.2-post-review", "generator_version": "2.1", "run_id": "run_20260718_protocol_v3", "n_boot": N_BOOT, "rng_seed": RNG_SEED, "bootstrap_p_values": False, "bh_q_values": False, "inputs": {str(FEATURES.relative_to(ROOT)): hashlib.sha256(FEATURES.read_bytes()).hexdigest(), str(TARGETS.relative_to(ROOT)): hashlib.sha256(TARGETS.read_bytes()).hexdigest()}}
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    hash_rows = []
    for path in sorted(OUT.iterdir()):
        if path.name != "artifact_hashes.csv" and path.is_file():
            hash_rows.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size})
    pd.DataFrame(hash_rows).to_csv(OUT / "artifact_hashes.csv", index=False)
    print(primary[["proxy", "rho", "cluster_ci_lower", "cluster_ci_upper", "selectivity_margin"]].to_string(index=False))
    print("\nPositive controls:\n", positive_df[positive_df.split.isin(["G-test", "interaction-test"])].to_string(index=False))


if __name__ == "__main__":
    main()
