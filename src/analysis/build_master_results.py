#!/usr/bin/env python3
"""
Step 17 — Rebuild the Statistical Reporting
Create one master results file from which manuscript statements are generated.
"""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "redesign" / "tables"
RESULTS.mkdir(parents=True, exist_ok=True)

MASTER = RESULTS / "master_results.csv"
SEED = 42


def add_meta(row, dataset_role, model_version, split_id, sample_size, metric_name, timestamp):
    return {
        "dataset_role": dataset_role,
        "model_version": model_version,
        "split_id": split_id,
        "sample_size": sample_size,
        "metric_name": metric_name,
        "timestamp_utc": timestamp,
        "random_seed": SEED,
        **row,
    }


def main():
    timestamp = datetime.now(timezone.utc).isoformat()
    rows = []

    # ICC
    icc = pd.read_csv(RESULTS / "icc_by_dimension.csv")
    for _, r in icc.iterrows():
        rows.append(add_meta({
            "dimension": r["dimension"],
            "metric": "ICC(2,1)",
            "value": r["icc2_1"],
            "ci_low": r["icc2_1_ci_low"],
            "ci_high": r["icc2_1_ci_high"],
            "p": r["icc2_1_p"],
            "p_fdr": None,
        }, "D-human-raw", "NA", "full", r["n_images"] * r["n_raters"], "icc21", timestamp))
        rows.append(add_meta({
            "dimension": r["dimension"],
            "metric": "ICC(2,k)",
            "value": r["icc2_k"],
            "ci_low": r["icc2_k_ci_low"],
            "ci_high": r["icc2_k_ci_high"],
            "p": None,
            "p_fdr": None,
        }, "D-human-raw", "NA", "full", r["n_images"] * r["n_raters"], "icc2k", timestamp))

    # Proxy nested CV
    proxy_cv = pd.read_csv(RESULTS / "proxy_nested_cv_results.csv")
    primary = proxy_cv.iloc[0]["model"]
    for _, r in proxy_cv.iterrows():
        for dim in ["overall", "beauty", "complexity", "order", "emotion"]:
            rows.append(add_meta({
                "dimension": dim,
                "metric": "R^2",
                "value": r[f"{dim}_r2"],
                "ci_low": None,
                "ci_high": None,
                "p": None,
                "p_fdr": None,
                "model": r["model"],
            }, "D-proxy", r["model"], "group-aware-3fold", 15694, "proxy_task", timestamp))

    # Frozen human validation
    frozen = pd.read_csv(RESULTS / "frozen_human_validation.csv")
    for _, r in frozen.iterrows():
        rows.append(add_meta({
            "dimension": r["dimension"],
            "metric": "Spearman_rho",
            "value": r["spearman_rho"],
            "ci_low": r["rho_ci_low"],
            "ci_high": r["rho_ci_high"],
            "p": r["spearman_p"],
            "p_fdr": r["spearman_p_fdr"],
            "model": primary,
        }, "D-human-mean", primary, "frozen-external", r["n_images"], "human_alignment", timestamp))

    # Baseline comparison
    baselines = pd.read_csv(RESULTS / "baseline_comparison_proxy.csv")
    for _, r in baselines.iterrows():
        for dim in ["overall", "beauty", "complexity", "order", "emotion"]:
            rows.append(add_meta({
                "dimension": dim,
                "metric": "R^2",
                "value": r[f"{dim}_r2"],
                "ci_low": None,
                "ci_high": None,
                "p": None,
                "p_fdr": None,
                "model": r["model"],
            }, "D-proxy-subset", r["model"], "group-aware-3fold", 600, "baseline_proxy", timestamp))

    # Robustness
    robust = pd.read_csv(RESULTS / "robustness_summary.csv")
    for _, r in robust.iterrows():
        rows.append(add_meta({
            "dimension": r["dimension"],
            "metric": "Spearman_rho",
            "value": r["frozen_rho"],
            "ci_low": None,
            "ci_high": None,
            "p": None,
            "p_fdr": None,
            "model": primary,
            "ablation": "with_all_30",
        }, "D-human-mean", primary, "frozen-external", 100, "robustness", timestamp))
        rows.append(add_meta({
            "dimension": r["dimension"],
            "metric": "Spearman_rho",
            "value": r["without_ocr_rho"],
            "ci_low": None,
            "ci_high": None,
            "p": None,
            "p_fdr": None,
            "model": "Ridge_no_ocr",
            "ablation": "without_ocr_layout",
        }, "D-human-mean", "Ridge_no_ocr", "frozen-external", 100, "robustness", timestamp))

    master = pd.DataFrame(rows)
    master.to_csv(MASTER, index=False)
    print(f"Wrote {MASTER}")
    print(f"Total rows: {len(master)}")

    # Manuscript number audit
    audit = [
        "# Manuscript Number Audit",
        "",
        "All numbers below must match statements in the manuscript.",
        "",
    ]
    for _, r in frozen.iterrows():
        audit.append(f"- Frozen validation {r['dimension']}: rho={r['spearman_rho']:.3f}, 95% CI=[{r['rho_ci_low']:.3f},{r['rho_ci_high']:.3f}], p={r['spearman_p']:.4g}, p_fdr={r['spearman_p_fdr']:.4g}")
    for _, r in icc.iterrows():
        audit.append(f"- ICC(2,1) {r['dimension']}: {r['icc2_1']:.3f}, 95% CI=[{r['icc2_1_ci_low']:.3f},{r['icc2_1_ci_high']:.3f}]")
    audit.append("")
    (RESULTS / "manuscript_number_audit.md").write_text("\n".join(audit), encoding="utf-8")
    print(f"Wrote {RESULTS / 'manuscript_number_audit.md'}")

    print("Step 17 complete.")


if __name__ == "__main__":
    main()
