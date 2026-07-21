#!/usr/bin/env python3
"""Ensure historical human-study assets cannot enter the active Route-C release."""
import ast
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT.parent / "99-投稿包" / "no_human"

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
    "metadata.csv",
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
    "in_human_subset",
]

HISTORICAL_ROOTS = [
    ROOT / "archive",
    ROOT / "results" / "redesign",
]


def is_under_historical_root(path: Path) -> bool:
    try:
        return any(path.resolve().is_relative_to(hr.resolve()) for hr in HISTORICAL_ROOTS if hr.exists())
    except Exception:
        return False


# Files that legitimately reference historical inputs for provenance or manifest construction
EXCLUDED_PATHS = {
    "build_canonical_manifest.py",
    "input_hashes.csv",
}


def _is_excluded_path(path: Path) -> bool:
    return any(path.name == name for name in EXCLUDED_PATHS)


def scan_text(path: Path) -> list:
    failures = []
    if is_under_historical_root(path) or _is_excluded_path(path):
        return failures
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


def scan_string_literals(path: Path) -> list:
    failures = []
    if is_under_historical_root(path) or _is_excluded_path(path):
        return failures
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


def main():
    failures = []

    # Active Python code
    active_dirs = [ROOT / "src" / "no_human"]
    for d in active_dirs:
        if not d.exists():
            continue
        for path in d.rglob("*.py"):
            failures.extend(scan_text(path))
            failures.extend(scan_string_literals(path))

    active_pipeline = ROOT / "run_no_human_pipeline.py"
    if active_pipeline.exists():
        failures.extend(scan_text(active_pipeline))
        failures.extend(scan_string_literals(active_pipeline))

    # Active configurations
    config_dir = ROOT / "configs"
    if config_dir.exists():
        for path in config_dir.glob("no_human_*.yaml"):
            failures.extend(scan_text(path))

    # Active manuscript source and generated tables
    manuscript_dirs = [
        ROOT / "paper" / "sections_no_human",
        PACKAGE / "manuscript_source",
        PACKAGE / "manuscript_source" / "sections_no_human",
    ]
    for manuscript_dir in manuscript_dirs:
        if manuscript_dir.exists():
            for path in manuscript_dir.glob("*.tex"):
                failures.extend(scan_text(path))
    for main_tex in [
        ROOT / "paper" / "main_no_human.tex",
        PACKAGE / "manuscript_source" / "main_no_human.tex",
    ]:
        if main_tex.exists():
            failures.extend(scan_text(main_tex))

    # Active manifests and audit
    for subdir in ["manifests", "quality_control"]:
        d = ROOT / "results" / "no_human" / subdir
        if d.exists():
            for path in d.rglob("*"):
                if path.is_file() and path.suffix in {".csv", ".md", ".txt", ".json", ".yaml"}:
                    failures.extend(scan_text(path))

    # Active release README and manifested paths. The obsolete ZIP release is
    # intentionally not accepted as a substitute for the promoted package.
    if not PACKAGE.exists():
        failures.append(f"Active Route-C package not found: {PACKAGE}")
    else:
        readme = PACKAGE / "README.md"
        if readme.exists():
            failures.extend(scan_text(readme))
        else:
            failures.append(f"Active Route-C README not found: {readme}")
        manifest = PACKAGE / "manifests" / "submission_manifest.csv"
        if not manifest.exists():
            failures.append(f"Active Route-C manifest not found: {manifest}")
        else:
            with manifest.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    relative = row["relative_path"]
                    for fragment in BANNED_PATH_FRAGMENTS:
                        if fragment == "metadata.csv":
                            continue
                        if fragment in relative:
                            failures.append(
                                f"{manifest}: banned path fragment "
                                f"'{fragment}' in {relative}"
                            )

    if failures:
        print("FAILED no-human exclusion test:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("No-human exclusion test passed.")


def test_no_human_exclusion():
    """Pytest-collectable wrapper around the no-human exclusion scan."""
    main()


if __name__ == "__main__":
    main()
