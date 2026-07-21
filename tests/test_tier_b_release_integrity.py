"""Assertion-based integrity tests scoped to the redistributable Tier-B release."""

import ast
import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BANNED_PATH_FRAGMENTS = {
    "盲评问卷", "real_human_ratings.csv", "human_mean_ratings.csv",
    "icc_by_dimension.csv", "icc_by_category.csv", "rater_saturation.csv",
    "calibration_label_efficiency.csv", "frozen_human_validation.csv", "评分表_rater",
}


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_active_code_excludes_historical_human_assets():
    violations = []
    for path in (ROOT / "src" / "no_human").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for fragment in BANNED_PATH_FRAGMENTS:
            if fragment in text or fragment in path.as_posix():
                violations.append(f"{path.relative_to(ROOT)}: {fragment}")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                for fragment in BANNED_PATH_FRAGMENTS:
                    if fragment in node.value:
                        violations.append(f"{path.relative_to(ROOT)} literal: {fragment}")
    assert not violations, "\n".join(violations)


def test_public_manifest_has_exact_frozen_row_count():
    rows = list(csv.DictReader((ROOT / "public_demo" / "dataset_manifest.csv").open(
        newline="", encoding="utf-8")))
    assert len(rows) == 16425
    assert len({row["relative_path"] for row in rows}) == 16425
    assert all(len(row["sha256"]) == 64 for row in rows)


def test_public_manifest_matches_exact_image_set_and_hashes():
    from public_demo.download_data import verify_frozen_dataset
    valid, detail = verify_frozen_dataset()
    assert valid, detail


def test_generator_seed_is_stable_across_python_hash_salts():
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


def test_synthetic_feature_matrix_is_complete_and_finite():
    frame = pd.read_csv(ROOT / "synthetic_benchmark" / "metadata" / "synthetic_traditional_features.csv")
    numeric = frame.select_dtypes(include=[np.number])
    assert len(frame) == 16425
    assert len(numeric.columns) >= 30
    assert np.isfinite(numeric.to_numpy()).all()


def test_target_metadata_is_complete_and_finite():
    frame = pd.read_csv(ROOT / "synthetic_benchmark" / "metadata" / "target_metadata.csv")
    targets = [
        "structural_clutter_index", "geometric_order_index", "palette_coherence_index",
        "visual_salience_index", "focal_hierarchy_index",
    ]
    assert len(frame) == 16425
    assert set(targets).issubset(frame.columns)
    assert np.isfinite(frame[targets].to_numpy()).all()


def test_frozen_split_counts_and_seed_units():
    frame = pd.read_csv(ROOT / "synthetic_benchmark" / "splits" / "frozen_splits.csv")
    assert frame["split"].value_counts().to_dict() == {
        "G-dev": 6250, "G-test": 6250, "G-validation": 3125, "interaction-test": 800,
    }
    assert frame.loc[frame["split"].eq("G-test"), "seed"].nunique() == 10


def test_protocol_v5_artifact_hashes_are_exact():
    root = ROOT / "results/no_human/runs/run_20260720_semantic_health_v5"
    rows = list(csv.DictReader((root / "artifact_hashes.csv").open(newline="", encoding="utf-8")))
    assert len(rows) >= 20
    for row in rows:
        path = root / row["file"]
        assert path.exists(), row["file"]
        assert path.stat().st_size == int(row["bytes"]), row["file"]
        assert file_hash(path) == row["sha256"], row["file"]


def test_external_holdout_v6_is_complete_and_hash_exact():
    root = ROOT / "results/no_human/runs/run_20260720_external_protocol_v6"
    artifact_rows = list(csv.DictReader((root / "artifact_hashes.csv").open(newline="", encoding="utf-8")))
    assert len(artifact_rows) >= 23
    for row in artifact_rows:
        path = root / row["file"]
        assert path.exists(), row["file"]
        assert path.stat().st_size == int(row["bytes"]), row["file"]
        assert file_hash(path) == row["sha256"], row["file"]
    image_rows = list(csv.DictReader((root / "external_image_manifest.csv").open(
        newline="", encoding="utf-8")))
    assert len(image_rows) == 1250
    assert len({row["relative_path"] for row in image_rows}) == 1250
    targets = pd.read_csv(root / "target_correlation_external.csv", index_col=0)
    off_diagonal = targets.to_numpy()[~np.eye(len(targets), dtype=bool)]
    assert np.max(np.abs(off_diagonal)) <= 1e-12
    metadata = __import__("json").loads((root / "run_metadata.json").read_text())
    assert metadata["protocol"] == "6.0-corrected-second-renderer-mechanism-audit"
    assert metadata["force_regenerate"] is True
    assert (root / "cache_bindings.json").exists()


def test_required_reproduction_evidence_exists():
    required = [
        "synthetic_benchmark/PROTOCOL.md",
        "synthetic_benchmark/PROTOCOL_V5_SEMANTIC_HEALTH.md",
        "synthetic_benchmark/run_semantic_health_protocol_v5.py",
        "synthetic_benchmark/PROTOCOL_V6_EXTERNAL_HOLDOUT.md",
        "synthetic_benchmark/run_external_holdout_protocol_v6.py",
        "results/no_human/runs/run_20260720_semantic_health_v5/run_metadata.json",
        "results/no_human/runs/run_20260720_semantic_health_v5/semantic_health_gate.csv",
        "results/no_human/runs/run_20260720_external_protocol_v6/run_metadata.json",
        "results/no_human/runs/run_20260720_external_protocol_v6/external_composite_score_health.csv",
        "requirements-lock.txt",
    ]
    if (ROOT / "logs").exists():
        required.extend([
            "logs/verification_protocol_v5_clean.log",
            "logs/verification_protocol_v6_clean.log",
        ])
    missing = [relative for relative in required if not (ROOT / relative).exists()]
    assert not missing, "missing Tier-B evidence: " + ", ".join(missing)
