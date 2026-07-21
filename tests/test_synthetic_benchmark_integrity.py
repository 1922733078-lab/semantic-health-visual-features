"""
Regression tests for the synthetic benchmark protocol, decision rules, and table mapping.
"""
import re
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_stable_image_seed_is_process_hash_independent():
    code = (
        "from synthetic_benchmark.generator.generate_layouts import stable_image_seed; "
        "print(stable_image_seed('C_card','layout_order',3,20,7))"
    )
    values = []
    for hash_seed in ("1", "23", "random"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        values.append(subprocess.check_output(
            [sys.executable, "-c", code], cwd=ROOT, env=env, text=True
        ).strip())
    assert len(set(values)) == 1


def test_decision_rules_file_exists():
    path = ROOT / "synthetic_benchmark" / "configs" / "benchmark_decision_rules.yaml"
    assert path.exists(), "benchmark_decision_rules.yaml is missing"
    rules = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert rules["association"]["strong"]["min_rho"] == 0.60
    assert rules["selectivity"]["strong"]["min_margin"] == 0.15


def test_no_conflicting_strong_support_definitions():
    """Active code and manuscript sources must not define rho>=0.50 as strong support."""
    bad_patterns = [
        r"strong[- ]?support[^\n]{0,60}0\.50",
        r"0\.50[^\n]{0,60}strong[- ]?support",
        r"rho\s*>=?\s*0\.50[^0-9][^\n]{0,60}strong",
        r"strong[^\n]{0,60}rho\s*>=?\s*0\.50[^0-9]",
    ]
    search_dirs = [
        ROOT / "synthetic_benchmark",
        ROOT / "paper" / "sections_no_human",
    ]
    violations = []
    for d in search_dirs:
        for p in d.rglob("*.py"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            for pat in bad_patterns:
                if re.search(pat, text, re.IGNORECASE):
                    violations.append(f"{p}: {pat}")
        for p in d.rglob("*.tex"):
            text = p.read_text(encoding="utf-8", errors="ignore")
            for pat in bad_patterns:
                if re.search(pat, text, re.IGNORECASE):
                    violations.append(f"{p}: {pat}")
    assert not violations, "Found conflicting strong-support definitions: " + "; ".join(violations[:10])


def test_selectivity_margin_from_canonical_values():
    """Reproduce CA-03/CA-04 regression case: target .522, non-target .556 -> margin -.034, failure."""
    from synthetic_benchmark.evaluate_objective_agreement import (
        evaluate_proxy_target, evaluate_association, evaluate_selectivity,
    )
    rules = yaml.safe_load((ROOT / "synthetic_benchmark" / "configs" / "benchmark_decision_rules.yaml").read_text())
    # Manually construct a result row with the historical observed values
    assoc = evaluate_association(0.522, 0.500, 0.0, rules)
    sel = evaluate_selectivity(abs(0.522) - abs(0.556), rules)
    assert assoc == "partial"
    assert sel == "failure"


def test_protocol_v5_gtest_is_primary_and_leakage_controlled():
    path = ROOT / "results/no_human/runs/run_20260720_semantic_health_v5/primary_cluster_results.csv"
    assert path.exists(), "protocol-v5 G-test primary table missing"
    df = pd.read_csv(path)
    required = [
        "proxy", "target", "n_images", "n_seed_clusters", "rho",
        "cluster_ci_lower", "cluster_ci_upper", "is_nominal_target",
        "target_scaling", "feature_scaling", "max_abs_non_target_rho",
        "max_abs_non_target_name", "selectivity_margin",
    ]
    assert set(required).issubset(df.columns)
    assert len(df) == 5
    assert df["n_images"].eq(6250).all()
    assert df["n_seed_clusters"].eq(10).all()
    assert df["target_scaling"].eq("G-dev mean/std frozen").all()
    assert df["feature_scaling"].eq(
        "G-dev mean/std and score bounds frozen"
    ).all()


def test_protocol_v5_releases_all_25_gtest_associations():
    path = ROOT / "results/no_human/runs/run_20260720_semantic_health_v5/full_25_proxy_target_cluster.csv"
    frame = pd.read_csv(path)
    assert len(frame) == 25
    assert frame["proxy"].nunique() == 5
    assert frame["target"].nunique() == 5
    assert frame["n_images"].eq(6250).all()


def test_frozen_splits_reproducible():
    path = ROOT / "synthetic_benchmark" / "splits" / "frozen_splits.csv"
    assert path.exists()
    df = pd.read_csv(path)
    assert {"image_id", "family", "seed", "split", "sha256"}.issubset(set(df.columns))
    # Each family maps to exactly one main split (excluding interaction-test)
    base = df[df["split"] != "interaction-test"]
    mapping = base.drop_duplicates(["family", "split"])
    assert len(mapping) == 3, "Family-to-split mapping is not one-to-one"
    expected = {"A_poster": "G-dev", "B_banner": "G-validation", "C_card": "G-test"}
    actual = dict(zip(mapping["family"], mapping["split"]))
    assert actual == expected, f"Unexpected family split mapping: {actual}"


def test_generator_specification_matches_manuscript():
    spec_path = ROOT / "synthetic_benchmark" / "manifests" / "generator_specification.json"
    if not spec_path.exists():
        pytest.skip("generator_specification.json not yet generated")
    manuscript_path = ROOT / "paper" / "sections_no_human" / "04-experiments.tex"
    if not manuscript_path.exists():
        pytest.skip("manuscript source not included in this package")
    import json
    spec = json.loads(spec_path.read_text())
    manuscript = manuscript_path.read_text(encoding="utf-8")
    for family_name, family_spec in spec.get("families", {}).items():
        display = family_spec.get("display_name", family_name)
        assert display in manuscript, f"Family display name {display} missing from manuscript"
    for construct in spec.get("constructs", {}).keys():
        assert construct.replace("_", " ") in manuscript or construct in manuscript, f"Construct {construct} missing from manuscript"


def test_overall_status_criteria():
    """Decision-function unit: strong only when association, selectivity, and family all strong."""
    from synthetic_benchmark.evaluate_objective_agreement import (
        evaluate_association, evaluate_selectivity, evaluate_family_generalization
    )
    rules = yaml.safe_load((ROOT / "synthetic_benchmark" / "configs" / "benchmark_decision_rules.yaml").read_text())
    assert evaluate_association(0.60, 0.01, 0.04, rules) == "strong"
    assert evaluate_association(0.599, 0.01, 0.04, rules) == "partial"
    assert evaluate_association(0.70, 0.01, 0.10, rules) == "failure"  # q too high for strong; rho too high for partial
    assert evaluate_selectivity(0.15, rules) == "strong"
    assert evaluate_selectivity(0.10, rules) == "partial"
    assert evaluate_selectivity(-0.01, rules) == "failure"
    assert evaluate_family_generalization({"A": "strong", "B": "partial", "C": "partial"}, rules) == "strong"
    assert evaluate_family_generalization({"A": "partial", "B": "failure", "C": "partial"}, rules) == "partial"
    assert evaluate_family_generalization({"A": "failure", "B": "failure", "C": "failure"}, rules) == "failure"


def test_protocol_v5_development_controls_are_complete_and_locked():
    root = ROOT / "results/no_human/runs/run_20260720_semantic_health_v5"
    import json
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["target_scaler_fit_split"] == "G-dev"
    assert metadata["feature_scaler_fit_split"] == "G-dev"
    assert metadata["baseline_selection_split"] == "G-dev"
    assert metadata["baseline_evaluation_split"] == "G-test"
    assert len(pd.read_csv(root / "development_target_constants.csv")) == 17
    assert len(pd.read_csv(root / "full_25_proxy_target_cluster.csv")) == 25
    assert len(pd.read_csv(root / "primary_cluster_results.csv")) == 5
    baselines = pd.read_csv(root / "paired_baseline_cluster.csv")
    assert len(baselines) == 5
    assert baselines["selected_on"].eq("G-dev only").all()
    assert baselines["evaluated_on"].eq("G-test only").all()


def test_protocol_v6_preserves_score_collapse_and_nonconstant_features():
    root = ROOT / "results/no_human/runs/run_20260720_external_protocol_v6"
    controls = pd.read_csv(root / "positive_control_cross_generator.csv")
    assert len(controls) == 5
    assert controls["scope"].eq(
        "pixel-side recoverability probe; not external generalization evidence"
    ).all()
    health = pd.read_csv(root / "external_feature_semantic_health.csv")
    assert len(health) == 6
    assert health["passed_non_degenerate"].all()
    score_health = pd.read_csv(root / "external_composite_score_health.csv")
    failed = set(score_health.loc[
        ~score_health["passed_non_degenerate_final_score"], "proxy"
    ])
    assert failed == {"visual_complexity_proxy", "visual_intensity_proxy"}
    crossings = pd.read_csv(root / "mechanism_crossing_intervals.csv")
    assert len(crossings) == 120
