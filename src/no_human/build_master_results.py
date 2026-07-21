#!/usr/bin/env python3
"""
Build the master results table for the no-human manuscript.

Outputs:
    results/no_human/tables/master_no_human_results.csv
    results/no_human/tables/manuscript_number_audit.md
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "no_human" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

METRICS = [
    "visual_complexity_proxy",
    "layout_order_proxy",
    "colour_harmony_proxy",
    "visual_intensity_proxy",
    "layout_hierarchy_proxy",
]


def main():
    trace = pd.read_csv(ROOT / "results" / "no_human" / "data" / "per_image_metric_trace.csv")
    rows = []
    for metric in METRICS:
        rows.append({
            "result_type": "full_sample_distribution",
            "metric": metric,
            "statistic": "mean",
            "value": float(trace[metric].mean()),
            "n": len(trace),
            "source_file": "per_image_metric_trace.csv",
        })
        rows.append({
            "result_type": "full_sample_distribution",
            "metric": metric,
            "statistic": "std",
            "value": float(trace[metric].std()),
            "n": len(trace),
            "source_file": "per_image_metric_trace.csv",
        })

    # Stability summary
    stab = pd.read_csv(ROOT / "results" / "no_human" / "tables" / "stability_results.csv")
    for transform, group in stab.groupby("transform"):
        rows.append({
            "result_type": "stability_mean_absolute_difference",
            "metric": "all",
            "statistic": transform,
            "value": float(group["mean_absolute_difference"].mean()),
            "n": int(group["n"].iloc[0]),
            "source_file": "stability_results.csv",
        })

    # Category means
    cat = pd.read_csv(ROOT / "results" / "no_human" / "tables" / "category_generalization.csv")
    cat = cat[~cat["category"].str.startswith("exclude_")]
    for _, row in cat.iterrows():
        for metric in METRICS:
            rows.append({
                "result_type": "category_mean",
                "metric": metric,
                "statistic": row["category"],
                "value": float(row[f"{metric}_mean"]),
                "n": int(row["n"]),
                "source_file": "category_generalization.csv",
            })

    # Convergence summary
    conv = pd.read_csv(ROOT / "results" / "no_human" / "tables" / "computational_convergence.csv")
    for _, row in conv.iterrows():
        rows.append({
            "result_type": "independent_reference_spearman",
            "metric": row["metric"],
            "statistic": row["reference_descriptor"],
            "value": float(row["spearman_rho"]),
            "n": int(row["n"]),
            "source_file": "computational_convergence.csv",
        })

    # Efficiency
    bench = pd.read_csv(ROOT / "results" / "no_human" / "tables" / "efficiency_benchmark.csv")
    for _, row in bench.iterrows():
        rows.append({
            "result_type": "efficiency",
            "metric": row["stage"],
            "statistic": "median_seconds_per_image",
            "value": float(row["median"]),
            "n": int(row["n"]),
            "source_file": "efficiency_benchmark.csv",
        })

    master = pd.DataFrame(rows)
    master.to_csv(OUT / "master_no_human_results.csv", index=False)
    print(f"Wrote {len(master)} rows to master_no_human_results.csv")

    # Manuscript number audit
    audit_lines = ["# Manuscript Number Audit\n", "\n", "Canonical values are taken from `master_no_human_results.csv`.\n", "\n"]
    for _, row in master.iterrows():
        audit_lines.append(f"- `{row['result_type']}` | `{row['metric']}` | `{row['statistic']}` = {row['value']:.4f} (n={row['n']})\n")
    (OUT / "manuscript_number_audit.md").write_text("".join(audit_lines), encoding="utf-8")
    print("Wrote manuscript_number_audit.md")


if __name__ == "__main__":
    main()
