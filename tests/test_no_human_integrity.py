#!/usr/bin/env python3
"""
Step 18 — Integrity tests specific to the no-human design.

Run with:
    python tests/test_no_human_integrity.py
"""

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "results" / "no_human"
CURRENT_LINK = OUTPUT_ROOT / "current"
if CURRENT_LINK.exists() and CURRENT_LINK.is_symlink():
    OUTPUT_ROOT = CURRENT_LINK.resolve()

BANNED_PATH_FRAGMENTS = [
    "盲评问卷",
    "real_human_ratings.csv",
    "human_mean_ratings.csv",
    "icc_by_dimension.csv",
    "icc_by_category.csv",
    "rater_saturation.csv",
    "calibration_label_efficiency.csv",
    "frozen_human_validation.csv",
    "评分表_rater",
]
BANNED_STRINGS = [
    "rater_id",
    "participant",
    "informed_consent",
    "inter-rater reliability",
    "human validation",
    "human mean",
]

ACTIVE_DIRS = [ROOT / "src" / "no_human"]
ACTIVE_PIPELINE = ROOT / "run_no_human_pipeline.py"

METRICS = [
    "visual_complexity_proxy",
    "layout_order_proxy",
    "colour_harmony_proxy",
    "visual_intensity_proxy",
    "layout_hierarchy_proxy",
]


def _pytest_or_script_result(failures):
    """Assert under pytest; return the list for the standalone integrity runner."""
    if "pytest" in sys.modules:
        assert not failures, "\n".join(failures)
        return None
    return failures


def scan_text(path):
    failures = []
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return [f"Could not read {path}: {e}"]
    lower = text.lower()
    for frag in BANNED_PATH_FRAGMENTS:
        if frag in text:
            failures.append(f"{path}: banned fragment '{frag}'")
    for s in BANNED_STRINGS:
        if s in lower:
            failures.append(f"{path}: banned string '{s}'")
    return failures


def scan_string_literals(path):
    failures = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return failures
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for frag in BANNED_PATH_FRAGMENTS:
                if frag in node.value:
                    failures.append(f"{path}: string literal contains '{frag}'")
    return failures


def test_exclusion_of_human_assets():
    failures = []
    for d in ACTIVE_DIRS:
        if not d.exists():
            continue
        for path in d.rglob("*.py"):
            failures.extend(scan_text(path))
            failures.extend(scan_string_literals(path))
    if ACTIVE_PIPELINE.exists():
        failures.extend(scan_text(ACTIVE_PIPELINE))
        failures.extend(scan_string_literals(ACTIVE_PIPELINE))
    return _pytest_or_script_result(failures)


def test_metric_ranges_and_finite_values():
    failures = []
    trace_path = OUTPUT_ROOT / "data" / "per_image_metric_trace.csv"
    if not trace_path.exists():
        return _pytest_or_script_result(["per_image_metric_trace.csv not found"])
    trace = pd.read_csv(trace_path)
    for metric in METRICS:
        if metric not in trace.columns:
            failures.append(f"Missing metric column: {metric}")
            continue
        if not np.isfinite(trace[metric]).all():
            failures.append(f"{metric} contains non-finite values")
        if trace[metric].min() < 0 or trace[metric].max() > 100:
            failures.append(f"{metric} outside [0, 100]")
    return _pytest_or_script_result(failures)


def test_split_integrity():
    failures = []
    splits_path = OUTPUT_ROOT / "quality_control" / "canonical_splits.csv"
    if not splits_path.exists():
        return _pytest_or_script_result(["canonical_splits.csv not found"])
    splits = pd.read_csv(splits_path)
    required_roles = {"D_development", "D_test", "D_stability", "D_case_study", "D_metric"}
    missing = required_roles - set(splits["role"].unique())
    if missing:
        failures.append(f"Missing roles: {missing}")
    if not splits["d_metric_member"].all():
        failures.append("Not all images are marked as D_metric members")
    # Duplicate groups should not span splits
    dup = splits[splits["duplicate_group_id"] != ""]
    for gid, group in dup.groupby("duplicate_group_id"):
        if group["split"].nunique() > 1:
            failures.append(f"Duplicate group {gid} spans splits")
    return _pytest_or_script_result(failures)


def test_stability_table_completeness():
    failures = []
    stability_path = OUTPUT_ROOT / "tables" / "stability_results.csv"
    config_path = ROOT / "configs" / "no_human_stability_transforms.yaml"
    if not stability_path.exists():
        return _pytest_or_script_result(["stability_results.csv not found"])
    if not config_path.exists():
        return _pytest_or_script_result(["no_human_stability_transforms.yaml not found"])
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results = pd.read_csv(stability_path)
    expected = set()
    for transform_name, tspec in config["transformations"].items():
        for level in tspec["params"]:
            expected.add((transform_name, level["level_id"]))
    actual = set(zip(results["transform"], results["level_id"]))
    missing = expected - actual
    if missing:
        failures.append(f"Missing stability transform/level combinations: {missing}")
    for metric in METRICS:
        if metric not in results["metric"].values:
            failures.append(f"Stability results missing metric {metric}")
    return _pytest_or_script_result(failures)


def test_sensitivity_table_completeness():
    failures = []
    sensitivity_path = OUTPUT_ROOT / "tables" / "directional_sensitivity_results.csv"
    if not sensitivity_path.exists():
        return _pytest_or_script_result(["directional_sensitivity_results.csv not found"])
    results = pd.read_csv(sensitivity_path)
    expected_manipulations = {
        "add_shapes", "sharpen", "desaturate", "saturate",
        "add_text_bars", "reduce_symmetry", "add_colour_noise", "reduce_whitespace",
    }
    actual = set(results["manipulation"].unique())
    missing = expected_manipulations - actual
    if missing:
        failures.append(f"Missing sensitivity manipulations: {missing}")
    return _pytest_or_script_result(failures)


def test_no_circular_reconstruction():
    """Primary metrics are direct weighted sums; no R-squared reconstruction of identical inputs."""
    failures = []
    arch_path = OUTPUT_ROOT / "manifests" / "primary_architecture_decision.md"
    if not arch_path.exists():
        arch_path = ROOT / "results" / "no_human" / "manifests" / "primary_architecture_decision.md"
    if not arch_path.exists():
        failures.append("primary_architecture_decision.md not found")
    else:
        text = arch_path.read_text(encoding="utf-8")
        if "Route A: direct deterministic score formulas" not in text:
            failures.append("Primary architecture is not Route A")
    return _pytest_or_script_result(failures)


def test_required_outputs_exist():
    failures = []
    required = [
        OUTPUT_ROOT / "data" / "per_image_metric_trace.csv",
        OUTPUT_ROOT / "tables" / "metric_definition_table.csv",
        OUTPUT_ROOT / "quality_control" / "canonical_splits.csv",
        OUTPUT_ROOT / "quality_control" / "dataset_role_audit.csv",
        OUTPUT_ROOT / "quality_control" / "image_provenance.csv",
        OUTPUT_ROOT / "quality_control" / "duplicate_groups.csv",
        OUTPUT_ROOT / "quality_control" / "split_leakage_report.md",
        OUTPUT_ROOT / "quality_control" / "repeated_extraction_check.csv",
        OUTPUT_ROOT / "data" / "stability_variants_manifest.csv",
        OUTPUT_ROOT / "tables" / "stability_results.csv",
        OUTPUT_ROOT / "data" / "sensitivity_variants_manifest.csv",
        OUTPUT_ROOT / "tables" / "directional_sensitivity_results.csv",
        OUTPUT_ROOT / "tables" / "structural_group_comparison.csv",
        OUTPUT_ROOT / "tables" / "category_generalization.csv",
        OUTPUT_ROOT / "tables" / "source_held_out_results.csv",
        OUTPUT_ROOT / "tables" / "computational_convergence.csv",
        OUTPUT_ROOT / "tables" / "robustness_ablation.csv",
        OUTPUT_ROOT / "tables" / "failure_case_register.csv",
        OUTPUT_ROOT / "tables" / "feature_contribution_audit.csv",
        OUTPUT_ROOT / "tables" / "efficiency_benchmark.csv",
    ]
    for path in required:
        if not path.exists():
            failures.append(f"Missing required output: {path.relative_to(ROOT)}")
    return _pytest_or_script_result(failures)


def main():
    # The standalone runner needs list-return semantics even if pytest happens to
    # be importable in the environment.
    pytest_module = sys.modules.pop("pytest", None)
    all_failures = []
    all_failures.extend(test_exclusion_of_human_assets())
    all_failures.extend(test_metric_ranges_and_finite_values())
    all_failures.extend(test_split_integrity())
    all_failures.extend(test_stability_table_completeness())
    all_failures.extend(test_sensitivity_table_completeness())
    all_failures.extend(test_no_circular_reconstruction())
    all_failures.extend(test_required_outputs_exist())

    if pytest_module is not None:
        sys.modules["pytest"] = pytest_module

    report_path = OUTPUT_ROOT / "logs" / "integrity_test_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        if all_failures:
            f.write("FAILED no-human integrity tests:\n")
            for fail in all_failures:
                f.write(f"  - {fail}\n")
            print("FAILED no-human integrity tests:")
            for fail in all_failures:
                print(f"  - {fail}")
            sys.exit(1)
        else:
            f.write("All no-human integrity tests passed.\n")
            print("All no-human integrity tests passed.")


if __name__ == "__main__":
    main()
