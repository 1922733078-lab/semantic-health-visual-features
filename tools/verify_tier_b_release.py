#!/usr/bin/env python3
"""
Verify a Tier B public release package.

Checks:
  - required top-level files/directories exist;
  - no historical human-study terms in code or documentation;
  - public demo data manifest exists and matches on-disk hashes;
  - README declares Tier B and distinguishes restricted vs public data.
  - the comprehensive release manifest is complete and hash-exact;
  - active protocol-v5 and protocol-v6 artifacts are present and hash-exact.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
    "human subset",
    "questionnaire",
    "score workbook",
    "blind questionnaire",
]

REQUIRED_TOP_LEVEL = [
    "README.md",
    "requirements-lock.txt",
    "configs",
    "src",
    "tools",
    "tests",
    "public_demo",
    "synthetic_benchmark",
    "docs",
    "manifests",
]

TEXT_SUFFIXES = {".py", ".md", ".txt", ".csv", ".json", ".yaml", ".yml", ".toml"}


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def scan_text_files(package_dir: Path) -> list:
    failures = []
    # These paths legitimately reference banned terms because they define or test
    # the exclusion policy, or because they describe the restricted corpus.
    skip_prefixes = (
        "docs/",
        "tests/",
        "tools/verify_tier_b_release.py",
        "tools/verify_submission_manifest.py",
        "tools/verify_clean_environment.py",
        "tools/verify_release.py",
    )
    for path in package_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.stat().st_size > 5 * 1024 * 1024:
            continue
        rel = str(path.relative_to(package_dir))
        if any(rel.startswith(pref) for pref in skip_prefixes):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except Exception:
            continue
        lower_name = rel.lower()
        for frag in BANNED_PATH_FRAGMENTS:
            if frag in lower_name or frag in text:
                failures.append(f"banned fragment '{frag}' in {rel}")
        for s in BANNED_STRINGS:
            if s in text:
                failures.append(f"banned string '{s}' in {rel}")
    return failures


def verify_public_demo_manifest(package_dir: Path) -> list:
    failures = []
    manifest = package_dir / "public_demo" / "dataset_manifest.csv"
    if not manifest.exists():
        return ["public_demo/dataset_manifest.csv missing"]
    with open(manifest, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    listed_images = {
        row.get("image_path") or row.get("relative_path")
        for row in rows
        if row.get("image_path") or row.get("relative_path")
    }
    image_dir = package_dir / "synthetic_benchmark" / "images"
    on_disk_images = {
        f"synthetic_benchmark/images/{entry.name}"
        for entry in os.scandir(image_dir)
        if entry.is_file(follow_symlinks=False)
    }
    for rel in sorted(on_disk_images - listed_images)[:20]:
        failures.append(f"unlisted synthetic image on disk: {rel}")
    for rel in sorted(listed_images - on_disk_images)[:20]:
        failures.append(f"listed synthetic image missing on disk: {rel}")

    def verify_row(row):
        rel = row.get("image_path") or row.get("relative_path")
        expected = row.get("sha256")
        if not rel:
            return None
        path = package_dir / rel
        if not path.exists():
            path = package_dir.parent.parent / rel
        if not path.exists():
            return f"demo manifest entry missing on disk: {rel}"
        if expected and hash_file(path) != expected:
            return f"demo hash mismatch for {rel}"
        return None

    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as executor:
        failures.extend(failure for failure in executor.map(verify_row, rows) if failure)
    return failures


def verify_artifact_hash_table(root: Path, label: str) -> list:
    failures = []
    table = root / "artifact_hashes.csv"
    if not table.exists():
        return [f"missing {label} artifact hash table"]
    with table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = root / row["file"]
        if not path.exists():
            failures.append(f"{label} artifact missing: {row['file']}")
        elif path.stat().st_size != int(row["bytes"]):
            failures.append(f"{label} size mismatch: {row['file']}")
        elif hash_file(path) != row["sha256"]:
            failures.append(f"{label} hash mismatch: {row['file']}")
    return failures


def verify_semantic_health_v5(package_dir: Path) -> list:
    failures = []
    root = (package_dir / "results/no_human/runs" /
            "run_20260720_semantic_health_v5")
    required = [
        package_dir / "src/no_human/semantic_health.py",
        package_dir / "tests/test_feature_semantic_health.py",
        package_dir / "synthetic_benchmark/PROTOCOL_V5_SEMANTIC_HEALTH.md",
        package_dir / "synthetic_benchmark/run_semantic_health_protocol_v5.py",
        root / "run_metadata.json",
        root / "semantic_health_gate.csv",
        root / "weighted_feature_health.csv",
        root / "feature_score_diagnostics.csv",
        root / "development_target_constants.csv",
        root / "development_feature_scalers.csv",
        root / "full_25_proxy_target_cluster.csv",
        root / "primary_cluster_results.csv",
        root / "baseline_candidate_development_selection.csv",
        root / "paired_baseline_cluster.csv",
        root / "pre_post_nominal_comparison.csv",
        root / "positive_control_results.csv",
    ]
    failures.extend(
        f"missing protocol-v5 artifact: {path.relative_to(package_dir)}"
        for path in required if not path.exists()
    )
    if failures:
        return failures
    failures.extend(verify_artifact_hash_table(root, "protocol-v5"))
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("protocol") != "5.0-semantic-health-failure-analysis":
        failures.append("protocol-v5 metadata version is incorrect")
    for key in ("target_scaler_fit_split", "feature_scaler_fit_split", "baseline_selection_split"):
        if metadata.get(key) != "G-dev":
            failures.append(f"protocol-v5 {key} is not G-dev")
    gate = list(csv.DictReader((root / "semantic_health_gate.csv").open(
        newline="", encoding="utf-8")))
    if not gate or {row.get("fit_split") for row in gate} != {"G-dev"}:
        failures.append("protocol-v5 semantic gate is not fitted only on G-dev")
    if any(row.get("active_gate_passed") != "True" for row in gate):
        failures.append("protocol-v5 has a failed active weighted-feature gate")
    paired = list(csv.DictReader((root / "paired_baseline_cluster.csv").open(
        newline="", encoding="utf-8")))
    if len(paired) != 5:
        failures.append(f"protocol-v5 has {len(paired)} paired baselines, expected 5")
    if any(row.get("selected_on") != "G-dev only" or row.get("evaluated_on") != "G-test only"
           for row in paired):
        failures.append("protocol-v5 baseline selection/evaluation split is incorrect")
    comparison = list(csv.DictReader((root / "pre_post_nominal_comparison.csv").open(
        newline="", encoding="utf-8")))
    if len(comparison) != 5:
        failures.append("protocol-v5 pre/post comparison must have five rows")
    if any(float(row["post_fix_retained_abs_weight_fraction"]) < 0.999999
           for row in comparison):
        failures.append("protocol-v5 post-fix formulas retain less than all active weight")
    return failures


def verify_external_v6(package_dir: Path) -> list:
    failures = []
    root = (package_dir / "results/no_human/runs" /
            "run_20260720_external_protocol_v6")
    required = [
        package_dir / "synthetic_benchmark/PROTOCOL_V6_EXTERNAL_HOLDOUT.md",
        package_dir / "synthetic_benchmark/run_external_holdout_protocol_v6.py",
        root / "run_metadata.json",
        root / "external_image_manifest.csv",
        root / "external_feature_semantic_health.csv",
        root / "external_composite_score_health.csv",
        root / "raw_composite_external_holdout.csv",
        root / "mechanism_crossing_intervals.csv",
        root / "renderer_factor_balance.csv",
        root / "composite_external_holdout.csv",
        root / "positive_control_cross_generator.csv",
    ]
    failures.extend(
        f"missing protocol-v6 artifact: {path.relative_to(package_dir)}"
        for path in required if not path.exists()
    )
    if failures:
        return failures
    failures.extend(verify_artifact_hash_table(root, "protocol-v6"))
    metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("protocol") != "6.0-corrected-second-renderer-mechanism-audit":
        failures.append("protocol-v6 metadata version is incorrect")
    if metadata.get("force_regenerate") is not True:
        failures.append("protocol-v6 was not forced to rerender and recompute")
    with (root / "external_image_manifest.csv").open(newline="", encoding="utf-8") as handle:
        images = list(csv.DictReader(handle))
    if len(images) != 1250:
        failures.append(f"protocol-v6 image manifest has {len(images)} rows, expected 1250")

    def verify_image(row):
        relative = row.get("relative_path")
        path = root / relative if relative else None
        if path is None or not path.exists():
            return f"protocol-v6 image missing: {relative}"
        if row.get("sha256") and hash_file(path) != row["sha256"]:
            return f"protocol-v6 image hash mismatch: {relative}"
        return None

    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as executor:
        failures.extend(failure for failure in executor.map(verify_image, images) if failure)
    health = list(csv.DictReader((root / "external_feature_semantic_health.csv").open(
        newline="", encoding="utf-8")))
    if len(health) != 6 or any(row.get("passed_non_degenerate") != "True" for row in health):
        failures.append("protocol-v6 external semantic-health gate did not pass all six fields")
    score_health = list(csv.DictReader((root / "external_composite_score_health.csv").open(
        newline="", encoding="utf-8")))
    if len(score_health) != 5:
        failures.append("protocol-v6 composite score-health table must have five rows")
    if not any(row.get("passed_non_degenerate_final_score") == "False"
               for row in score_health):
        failures.append("protocol-v6 did not preserve the observed final-score collapse")
    crossing = list(csv.DictReader((root / "mechanism_crossing_intervals.csv").open(
        newline="", encoding="utf-8")))
    if len(crossing) != 120:
        failures.append(f"protocol-v6 mechanism crossing has {len(crossing)} rows, expected 120")
    if {row.get("renderer") for row in crossing} != {"Pillow_G-test", "internal_OpenCV"}:
        failures.append("protocol-v6 mechanism crossing does not contain both renderers")
    controls = list(csv.DictReader((root / "positive_control_cross_generator.csv").open(
        newline="", encoding="utf-8")))
    interval_columns = {
        "gtest_cluster_ci_lower", "gtest_cluster_ci_upper",
        "internal_opencv_cluster_ci_lower", "internal_opencv_cluster_ci_upper",
    }
    if len(controls) != 5 or not interval_columns.issubset(controls[0] if controls else {}):
        failures.append("protocol-v6 controls lack five rows with both cluster intervals")
    return failures


def verify_release_manifest(package_dir: Path) -> list:
    failures = []
    manifest = package_dir / "manifests" / "release_manifest.csv"
    if not manifest.exists():
        return ["manifests/release_manifest.csv missing"]
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    listed = [row["path"] for row in rows]
    if len(listed) != len(set(listed)):
        failures.append("release manifest contains duplicate paths")
    excluded = {"manifests/release_manifest.csv"}
    on_disk = {
        path.relative_to(package_dir).as_posix()
        for path in package_dir.rglob("*")
        if path.is_file()
        and path.relative_to(package_dir).as_posix() not in excluded
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and path.suffix != ".pyc"
    }
    listed_set = set(listed)
    for rel in sorted(on_disk - listed_set)[:20]:
        failures.append(f"unlisted release file: {rel}")
    for rel in sorted(listed_set - on_disk)[:20]:
        failures.append(f"release manifest entry missing: {rel}")

    def verify_row(row):
        path = package_dir / row["path"]
        if not path.exists():
            return None
        if row.get("bytes") and path.stat().st_size != int(row["bytes"]):
            return f"release size mismatch: {row['path']}"
        if hash_file(path) != row["sha256"]:
            return f"release hash mismatch: {row['path']}"
        return None

    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as executor:
        failures.extend(failure for failure in executor.map(verify_row, rows) if failure)
    return failures


def main():
    parser = argparse.ArgumentParser(description="Verify a Tier B release package")
    parser.add_argument("--package", type=Path, required=True,
                        help="Path to tier_b_release directory")
    args = parser.parse_args()

    pkg = args.package
    failures = []

    for name in REQUIRED_TOP_LEVEL:
        if not (pkg / name).exists():
            failures.append(f"missing required top-level item: {name}")

    readme = pkg / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf-8").lower()
        if "tier b" not in text:
            failures.append("README.md does not declare Tier B")
        if "restricted" not in text:
            failures.append("README.md does not mention restricted corpus")
        if "synthetic" not in text:
            failures.append("README.md does not mention synthetic benchmark")
    else:
        failures.append("README.md missing")

    failures.extend(scan_text_files(pkg))
    failures.extend(verify_public_demo_manifest(pkg))
    failures.extend(verify_semantic_health_v5(pkg))
    failures.extend(verify_external_v6(pkg))
    failures.extend(verify_release_manifest(pkg))

    if failures:
        print("Tier B release verification FAILED:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("Tier B release verification passed.")


if __name__ == "__main__":
    main()
