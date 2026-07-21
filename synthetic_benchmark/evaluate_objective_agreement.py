#!/usr/bin/env python3
"""
Evaluate objective agreement between proposed proxies and synthetic benchmark targets.

Uses a frozen evaluation split (G-dev / G-validation / G-test / interaction-test)
and frozen decision rules from synthetic_benchmark/configs/benchmark_decision_rules.yaml.

Usage:
    python synthetic_benchmark/evaluate_objective_agreement.py --run-id <new-run-id> --evaluation-split G-test
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
import seaborn as sns
import yaml
from scipy import stats
from scipy.stats import spearmanr, kendalltau

import sys
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.no_human.fdr_bh import fdr_control

METRICS = {
    "visual_complexity_proxy": {
        "features": ["num_dominant_colors", "color_entropy", "edge_density",
                     "edge_orientation_entropy", "gradient_energy",
                     "text_block_count", "whitespace_ratio", "symmetry"],
        "weights": [0.20, 0.15, 0.25, 0.15, 0.10, 0.05, -0.05, -0.05],
    },
    "layout_order_proxy": {
        "features": ["symmetry", "whitespace_ratio", "rule_of_thirds", "edge_density",
                     "text_block_count", "lightness_contrast"],
        "weights": [0.25, 0.20, 0.15, -0.25, -0.10, -0.05],
    },
    "colour_harmony_proxy": {
        "features": ["color_harmony", "warm_color_ratio", "saturation_std",
                     "hue_contrast", "num_dominant_colors"],
        "weights": [0.30, 0.25, 0.20, -0.15, -0.10],
    },
    "visual_intensity_proxy": {
        "features": ["saturation_mean", "lightness_contrast", "gradient_energy",
                     "warm_color_ratio", "saturation_std"],
        "weights": [0.35, 0.25, 0.20, 0.10, 0.10],
    },
    "layout_hierarchy_proxy": {
        "features": ["edge_density", "saliency_std", "fg_bg_ratio",
                     "text_block_count", "has_text", "saliency_mean", "whitespace_ratio"],
        "weights": [0.25, 0.20, 0.15, 0.15, 0.10, 0.10, -0.05],
    },
}

TARGET_MAP = {
    "visual_complexity_proxy": "structural_clutter_index",
    "layout_order_proxy": "geometric_order_index",
    "colour_harmony_proxy": "palette_coherence_index",
    "visual_intensity_proxy": "visual_salience_index",
    "layout_hierarchy_proxy": "focal_hierarchy_index",
}

NON_TARGET_COLS = list(TARGET_MAP.values())


def load_decision_rules():
    rules_path = PROJECT_ROOT / "synthetic_benchmark" / "configs" / "benchmark_decision_rules.yaml"
    return yaml.safe_load(rules_path.read_text(encoding="utf-8"))


def load_frozen_stats():
    preproc_path = PROJECT_ROOT / "configs" / "no_human_preprocessing.yaml"
    preproc = yaml.safe_load(preproc_path.read_text(encoding="utf-8"))
    stats = preproc["normalization_constants"]
    percentile_bounds = preproc["percentile_bounds"]
    return stats, percentile_bounds


def standardize_features(df, feature_cols, stats):
    z = pd.DataFrame(index=df.index, columns=feature_cols, dtype=float)
    for col in feature_cols:
        mean = stats["mean"].get(col, df[col].mean())
        std = stats["std"].get(col, df[col].std())
        if std == 0:
            std = 1.0
        z[col] = ((df[col] - mean) / std).clip(-4, 4).fillna(0)
    return z


def compute_proxy_scores(z_df, percentile_bounds):
    scores = pd.DataFrame(index=z_df.index)
    for metric, spec in METRICS.items():
        raw = np.zeros(len(z_df))
        for feat, w in zip(spec["features"], spec["weights"]):
            if feat in z_df.columns:
                raw += z_df[feat].values * w
        pb = percentile_bounds[metric]
        p005, p995 = pb["p0.5"], pb["p99.5"]
        score = 100 * (raw - p005) / (p995 - p005 + 1e-10)
        score = np.clip(score, 0, 100)
        scores[metric] = score
    return scores


def bootstrap_ci(x, y, metric="spearman", n_boot=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x)
    estimates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if metric == "spearman":
            estimates.append(spearmanr(x[idx], y[idx])[0])
        else:
            estimates.append(kendalltau(x[idx], y[idx])[0])
    estimates = np.array(estimates)
    lo = np.percentile(estimates, 100 * alpha / 2)
    hi = np.percentile(estimates, 100 * (1 - alpha / 2))
    return lo, hi


def evaluate_association(rho, ci_lower, q, rules):
    assoc = rules["association"]
    if (rho >= assoc["strong"]["min_rho"] and
        (not assoc["strong"]["require_ci_lower_above_zero"] or ci_lower > 0) and
        (not assoc["strong"].get("require_q_below") or q < assoc["strong"]["require_q_below"])):
        return "strong"
    if (rho >= assoc["partial"]["min_rho"] and
        rho < assoc["partial"]["max_rho_for_partial"] and
        (not assoc["partial"]["require_ci_lower_above_zero"] or ci_lower > 0) and
        (not assoc["partial"].get("require_q_below") or q < assoc["partial"]["require_q_below"])):
        return "partial"
    return "failure"


def evaluate_selectivity(selectivity_margin, rules):
    sel = rules["selectivity"]
    if selectivity_margin >= sel["strong"]["min_margin"]:
        return "strong"
    if selectivity_margin > 0:
        return "partial"
    return "failure"


def evaluate_family_generalization(family_association_statuses, rules):
    # statuses is dict family -> association status on that family's data
    if all(s in ("strong", "partial") for s in family_association_statuses.values()):
        return "strong"
    if any(s in ("strong", "partial") for s in family_association_statuses.values()) and \
       not all(s in ("strong", "partial") for s in family_association_statuses.values()):
        return "partial"
    return "failure"


def evaluate_monotonicity(df_sub, scores, proxy, target, rules):
    """Check mean proxy score trend across severity levels for the target construct."""
    construct_map = {
        "structural_clutter_index": "visual_complexity",
        "geometric_order_index": "layout_order",
        "palette_coherence_index": "colour_harmony",
        "visual_salience_index": "visual_intensity",
        "focal_hierarchy_index": "layout_hierarchy",
    }
    construct = construct_map[target]
    base = df_sub[(df_sub["construct"] == construct) & (df_sub["split"] != "interaction-test")].copy()
    if base.empty:
        return "failure", []
    base["proxy"] = scores.loc[base.index, proxy]
    means = base.groupby("level")["proxy"].mean().sort_index().values
    diffs = np.diff(means)
    # For layout order target is reversed (higher order = more orderly), but proxy is expected to increase with target.
    # The target increases with level for all constructs by design.
    increasing = np.mean(diffs > 0)
    decreasing = np.mean(diffs < 0)
    if increasing >= 0.75:
        status = "strong"
    elif increasing > decreasing:
        status = "partial"
    else:
        status = "failure"
    return status, means.tolist()


def compute_family_results(df_full, scores_full, proxy, target, q, rules):
    """Compute per-family association status across all predefined families."""
    family_statuses = {}
    family_rhos = {}
    for family in sorted(df_full["family"].unique()):
        mask = df_full["family"] == family
        if mask.sum() < 10:
            continue
        fam_idx = df_full.loc[mask].index
        fr, _ = spearmanr(scores_full.loc[fam_idx, proxy].values, df_full.loc[fam_idx, target].values)
        family_ci_lo, _ = bootstrap_ci(
            scores_full.loc[fam_idx, proxy].values,
            df_full.loc[fam_idx, target].values,
            seed=hash(family) % (2**32),
        )
        family_statuses[family] = evaluate_association(fr, family_ci_lo, q, rules)
        family_rhos[family] = fr
    return family_rhos, family_statuses


def evaluate_proxy_target(df, scores, proxy, target, q, rules, family_rhos, family_statuses):
    x = scores.loc[df.index, proxy].values
    y = df[target].values
    rho, pval = spearmanr(x, y)
    tau, _ = kendalltau(x, y)
    ci_lo, ci_hi = bootstrap_ci(x, y, metric="spearman")

    # Non-target correlations (exclude the proxy's own target)
    non_target_records = []
    max_abs = -np.inf
    max_abs_name = None
    for nt in NON_TARGET_COLS:
        if nt == target:
            continue
        nt_rho, _ = spearmanr(x, df[nt].values)
        non_target_records.append({"target_variable": nt, "spearman_rho": nt_rho})
        if abs(nt_rho) > max_abs:
            max_abs = abs(nt_rho)
            max_abs_name = nt
    if max_abs == -np.inf:
        max_abs = 0.0
        max_abs_name = ""
    selectivity_margin = abs(rho) - max_abs

    family_support_count = sum(1 for s in family_statuses.values() if s in ("strong", "partial"))
    family_total_count = len(family_statuses)

    assoc_status = evaluate_association(rho, ci_lo, q, rules)
    sel_status = evaluate_selectivity(selectivity_margin, rules)
    family_gen_status = evaluate_family_generalization(family_statuses, rules)
    mono_status, mono_means = evaluate_monotonicity(df, scores, proxy, target, rules)

    # Overall status
    overall = "failure_or_limitation"
    if assoc_status == "strong" and sel_status == "strong" and family_gen_status == "strong":
        overall = "strong_support"
    elif assoc_status == "partial" and selectivity_margin > 0 and \
         family_gen_status in ("strong", "partial") and family_gen_status != "strong":
        overall = "partial_support"
    elif assoc_status in ("strong", "partial") and selectivity_margin > 0 and family_gen_status == "partial":
        overall = "partial_support"
    elif assoc_status in ("strong", "partial") and selectivity_margin <= 0:
        overall = "failure_or_limitation"

    return {
        "proxy": proxy,
        "target": target,
        "n": len(x),
        "target_rho": rho,
        "target_ci_lower": ci_lo,
        "target_ci_upper": ci_hi,
        "target_p": pval,
        "target_q": q,
        "kendall_tau": tau,
        "max_abs_non_target_rho": max_abs,
        "max_abs_non_target_name": max_abs_name,
        "selectivity_margin": selectivity_margin,
        "family_support_count": family_support_count,
        "family_total_count": family_total_count,
        "monotonicity_status": mono_status,
        "association_status": assoc_status,
        "selectivity_status": sel_status,
        "family_generalization_status": family_gen_status,
        "overall_support_status": overall,
    }, non_target_records, family_rhos, family_statuses


def evaluate_split(df_full, scores, split_name, rules, run_dir, pooled=False):
    """Evaluate one split and write tables. Returns primary results DataFrame."""
    if split_name == "pooled":
        df = df_full.copy()
    else:
        df = df_full[df_full["split"] == split_name].copy()

    if df.empty:
        raise ValueError(f"No images found for split={split_name}")

    rows = []
    mono_rows = []
    family_rows = []
    non_target_all = []

    p_values = []
    for proxy, target in TARGET_MAP.items():
        x = scores.loc[df.index, proxy].values
        y = df[target].values
        _, pval = spearmanr(x, y)
        p_values.append(pval)

    q_values = fdr_control(np.array(p_values))

    for (proxy, target), q in zip(TARGET_MAP.items(), q_values):
        family_rhos, family_statuses = compute_family_results(df_full, scores, proxy, target, q, rules)
        result, nt_records, family_rhos, family_statuses = evaluate_proxy_target(
            df, scores, proxy, target, q, rules, family_rhos, family_statuses
        )
        rows.append(result)

        # Monotonicity rows
        construct_map = {
            "structural_clutter_index": "visual_complexity",
            "geometric_order_index": "layout_order",
            "palette_coherence_index": "colour_harmony",
            "visual_salience_index": "visual_intensity",
            "focal_hierarchy_index": "layout_hierarchy",
        }
        construct = construct_map[target]
        base = df[(df["construct"] == construct) & (df["split"] != "interaction-test")].copy()
        if not base.empty:
            base["proxy"] = scores.loc[base.index, proxy]
            for level, grp in base.groupby("level"):
                mono_rows.append({
                    "proxy": proxy,
                    "target": target,
                    "level": int(level),
                    "mean_proxy": grp["proxy"].mean(),
                    "n": len(grp),
                })

        # Family rows
        for family, fr in family_rhos.items():
            family_rows.append({
                "proxy": proxy,
                "target": target,
                "family": family,
                "spearman_rho": fr,
                "association_status": family_statuses[family],
            })

        # Non-target rows
        for rec in nt_records:
            rec["proxy"] = proxy
            rec["is_primary_target"] = rec["target_variable"] == target
            non_target_all.append(rec)

    results = pd.DataFrame(rows)
    mono_df = pd.DataFrame(mono_rows)
    family_df = pd.DataFrame(family_rows)
    non_target_df = pd.DataFrame(non_target_all)

    tables_dir = run_dir / "synthetic_benchmark" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    split_file_key = {
        "pooled": "pooled_exploratory",
        "G-dev": "gdev",
        "G-validation": "gvalidation",
        "G-test": "gtest",
        "interaction-test": "interaction_test",
    }.get(split_name, split_name.lower().replace('-', '_'))
    fname = f"objective_benchmark_{split_file_key}.csv"
    results.to_csv(tables_dir / fname, index=False)

    if split_name != "pooled":
        family_df.to_csv(tables_dir / f"family_stratified_{split_file_key}.csv", index=False)

    return results, mono_df, family_df, non_target_df


def plot_calibration(df, scores, run_dir, split_name="G-test"):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, (proxy, target) in enumerate(TARGET_MAP.items()):
        ax = axes[idx]
        x = scores[proxy].values
        y = df[target].values
        n_bins = 10
        q_bins = pd.qcut(x, q=n_bins, duplicates="drop")
        means = df.loc[scores.index].groupby(q_bins)[target].mean()
        ax.scatter(x, y, alpha=0.2, s=8)
        q_bins_obj = pd.qcut(x, q=n_bins, duplicates="drop")
        bin_mids = [(interval.left + interval.right) / 2 for interval in q_bins_obj.categories]
        ax.plot(bin_mids, means.values, "ro-")
        ax.set_xlabel(proxy.replace("_proxy", " proxy"))
        ax.set_ylabel(target.replace("_index", ""))
        ax.set_title(f"{proxy} vs {target} ({split_name})")
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / "synthetic_benchmark" / "figures" / "calibration.pdf")
    plt.close(fig)


def plot_monotonicity(mono_df, run_dir):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, (proxy, target) in enumerate(TARGET_MAP.items()):
        ax = axes[idx]
        sub = mono_df[(mono_df["proxy"] == proxy) & (mono_df["target"] == target)]
        if not sub.empty:
            ax.plot(sub["level"], sub["mean_proxy"], "o-")
            ax.set_xlabel("Severity level")
            ax.set_ylabel(f"Mean {proxy}")
            ax.set_title(f"Monotonicity: {proxy}")
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / "synthetic_benchmark" / "figures" / "monotonicity.pdf")
    plt.close(fig)


def plot_interaction(df, scores, run_dir):
    interaction_df = df[df["construct"] == "interaction"].copy()
    if interaction_df.empty:
        return
    for proxy in TARGET_MAP.keys():
        interaction_df[proxy] = scores.loc[interaction_df.index, proxy]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    proxies = list(TARGET_MAP.keys())
    for idx, proxy in enumerate(proxies):
        ax = axes[idx]
        sns.boxplot(data=interaction_df, x="interaction", y=proxy, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        ax.set_title(proxy)
    axes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(run_dir / "synthetic_benchmark" / "figures" / "interaction.pdf")
    plt.close(fig)


def write_protocol_hashes(run_dir):
    """Record hashes of protocol-related source files."""
    files = [
        PROJECT_ROOT / "synthetic_benchmark" / "PROTOCOL.md",
        PROJECT_ROOT / "synthetic_benchmark" / "configs" / "benchmark_decision_rules.yaml",
        PROJECT_ROOT / "synthetic_benchmark" / "splits" / "frozen_splits.csv",
        PROJECT_ROOT / "synthetic_benchmark" / "generator" / "generate_layouts.py",
        PROJECT_ROOT / "synthetic_benchmark" / "generator" / "compute_objective_targets.py",
        PROJECT_ROOT / "synthetic_benchmark" / "evaluate_objective_agreement.py",
    ]
    records = []
    for f in files:
        if f.exists():
            records.append({"file": f.name, "path": str(f.relative_to(PROJECT_ROOT)),
                            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()})
    pd.DataFrame(records).to_csv(run_dir / "synthetic_benchmark" / "manifests" / "protocol_hashes.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="New run directory name under results/no_human/runs/")
    parser.add_argument("--evaluation-split", required=True,
                        choices=["G-dev", "G-validation", "G-test", "interaction-test", "pooled"],
                        help="Evaluation split required by protocol.")
    parser.add_argument("--features", default="synthetic_benchmark/metadata/synthetic_traditional_features.csv")
    parser.add_argument("--targets", default="synthetic_benchmark/metadata/target_metadata.csv")
    args = parser.parse_args()

    start = time.time()
    run_dir = PROJECT_ROOT / "results" / "no_human" / "runs" / args.run_id
    (run_dir / "synthetic_benchmark" / "figures").mkdir(parents=True, exist_ok=True)
    (run_dir / "synthetic_benchmark" / "tables").mkdir(parents=True, exist_ok=True)
    (run_dir / "synthetic_benchmark" / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "synthetic_benchmark" / "manifests").mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "synthetic_benchmark" / "logs" / "run.log"
    log = open(log_path, "w", encoding="utf-8")
    log.write(f"Objective agreement evaluation started at {pd.Timestamp.now().isoformat()}\n")
    log.write(f"Run ID: {args.run_id}\n")
    log.write(f"Evaluation split: {args.evaluation_split}\n")

    rules = load_decision_rules()
    log.write(f"Decision rules version: {rules.get('version', 'unknown')}\n")

    features = pd.read_csv(args.features)
    targets = pd.read_csv(args.targets)
    df = features.merge(targets, on="image_id", how="inner")
    log.write(f"Loaded {len(df)} images with features and targets.\n")

    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in NON_TARGET_COLS + ["image_id", "category", "split", "seed"]]
    stats, percentile_bounds = load_frozen_stats()
    z_df = standardize_features(df, feature_cols, stats)
    scores = compute_proxy_scores(z_df, percentile_bounds)
    log.write(f"Computed proxy scores using frozen standardization.\n")

    # Evaluate requested split as primary
    primary_results, mono_df, family_df, non_target_df = evaluate_split(
        df, scores, args.evaluation_split, rules, run_dir
    )
    log.write(f"Primary evaluation on {args.evaluation_split} complete.\n")

    # Always write pooled exploratory and by-split summary if G-test is primary
    if args.evaluation_split == "G-test":
        pooled_results, _, _, _ = evaluate_split(df, scores, "pooled", rules, run_dir, pooled=True)
        by_split_rows = []
        for split_name in ["G-dev", "G-validation", "G-test", "interaction-test"]:
            if split_name == args.evaluation_split:
                res = primary_results
            else:
                res, _, _, _ = evaluate_split(df, scores, split_name, rules, run_dir)
            for _, row in res.iterrows():
                r = row.to_dict()
                r["split"] = split_name
                by_split_rows.append(r)
        by_split_df = pd.DataFrame(by_split_rows)
        by_split_df.to_csv(run_dir / "synthetic_benchmark" / "tables" / "objective_benchmark_by_split.csv", index=False)

    non_target_df.to_csv(run_dir / "synthetic_benchmark" / "tables" / "non_target_association_results.csv", index=False)

    # Failure cases on primary split
    primary_df = df[df["split"] == args.evaluation_split].copy() if args.evaluation_split != "pooled" else df.copy()
    failure_rows = []
    for proxy, target in TARGET_MAP.items():
        sub = primary_df[["image_id", "family", "construct", "level", target]].copy()
        sub["proxy"] = scores.loc[sub.index, proxy]
        sub["abs_error"] = np.abs(sub["proxy"] - sub[target])
        failures = sub.nlargest(20, "abs_error")
        failures["proxy_name"] = proxy
        failures["target_name"] = target
        failure_rows.append(failures)
    failures_df = pd.concat(failure_rows, ignore_index=True)
    failures_df.to_csv(run_dir / "synthetic_benchmark" / "tables" / "failure_cases.csv", index=False)

    write_protocol_hashes(run_dir)

    plot_calibration(primary_df, scores.loc[primary_df.index], run_dir, split_name=args.evaluation_split)
    plot_monotonicity(mono_df, run_dir)
    plot_interaction(df, scores, run_dir)
    log.write("Figures saved.\n")

    elapsed = time.time() - start
    log.write(f"Elapsed: {elapsed:.1f}s\n")
    log.close()

    print("Objective benchmark results:")
    print(primary_results[["proxy", "target", "target_rho", "target_q", "selectivity_margin",
                            "association_status", "selectivity_status",
                            "family_generalization_status", "overall_support_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
