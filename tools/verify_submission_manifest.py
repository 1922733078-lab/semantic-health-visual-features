#!/usr/bin/env python3
"""Verify the active protocol-v5/v6 Route-C submission package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd


BANNED_NAMES = {
    "real_human_ratings.csv", "human_mean_ratings.csv", "icc_by_dimension.csv",
    "icc_by_category.csv", "rater_saturation.csv",
    "calibration_label_efficiency.csv", "frozen_human_validation.csv",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dir_sha(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        if "__pycache__" in file.parts or ".pytest_cache" in file.parts or file.suffix == ".pyc":
            continue
        digest.update(file.relative_to(path).as_posix().encode("utf-8"))
        digest.update(file.read_bytes())
    return digest.hexdigest()


def check_submission_manifest(pkg: Path, issues: list[str]) -> None:
    manifest = pkg / "manifests/submission_manifest.csv"
    if not manifest.exists():
        issues.append("submission manifest missing"); return
    rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    listed = {row["relative_path"] for row in rows}
    excluded = {
        "manifest_sha256.txt", "manifests/submission_manifest.csv",
        "manifests/canonical_comparison.csv", "manifests/release_metadata.json",
    }
    actual = {
        p.relative_to(pkg).as_posix() for p in pkg.rglob("*")
        if p.is_file()
        and p.relative_to(pkg).as_posix() not in excluded
        and not p.relative_to(pkg).as_posix().startswith("logs/")
        and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts
        and p.suffix != ".pyc"
    }
    if listed != actual:
        issues.extend(f"unlisted payload: {x}" for x in sorted(actual - listed)[:10])
        issues.extend(f"missing manifested payload: {x}" for x in sorted(listed - actual)[:10])
    for row in rows:
        path = pkg / row["relative_path"]
        if not path.exists(): continue
        if path.stat().st_size != int(row["bytes"]):
            issues.append(f"size mismatch: {row['relative_path']}")
        elif sha(path) != row["sha256"]:
            issues.append(f"hash mismatch: {row['relative_path']}")
    control = pkg / "manifest_sha256.txt"
    if not control.exists():
        issues.append("manifest_sha256.txt missing")
    else:
        for line in control.read_text(encoding="utf-8").splitlines():
            expected, relative = line.split(maxsplit=1)
            path = pkg / relative.strip()
            if not path.exists() or sha(path) != expected:
                issues.append(f"control hash mismatch: {relative.strip()}")


def check_canonical(pkg: Path, root: Path, issues: list[str]) -> None:
    register = root / "results/no_human/quality_control/route_c_canonical_artifact_register.csv"
    rows = list(csv.DictReader(register.open(newline="", encoding="utf-8")))
    seen = set()
    for row in rows:
        role = row["role"]
        if role in seen: issues.append(f"duplicate canonical role: {role}")
        seen.add(role)
        canonical = root / row["canonical_path"]
        promoted = pkg / row["promoted_path"]
        if not canonical.exists(): issues.append(f"canonical missing: {row['canonical_path']}"); continue
        if not promoted.exists(): issues.append(f"promoted missing: {row['promoted_path']}"); continue
        if role == "tier_b_release":
            c_hash = sha(canonical / "manifests/release_manifest.csv")
            p_hash = sha(promoted / "manifests/release_manifest.csv")
        else:
            c_hash = dir_sha(canonical) if canonical.is_dir() else sha(canonical)
            p_hash = dir_sha(promoted) if promoted.is_dir() else sha(promoted)
        if c_hash != row["expected_sha256"]:
            issues.append(f"register stale for role {role}")
        if p_hash != row["expected_sha256"]:
            issues.append(f"canonical/promoted mismatch for role {role}")
    comparison = pd.read_csv(pkg / "manifests/canonical_comparison.csv")
    if len(comparison) != len(rows) or not comparison["matches"].all():
        issues.append("canonical comparison is incomplete or contains a mismatch")


def check_scientific_evidence(pkg: Path, issues: list[str]) -> None:
    tier = pkg / "tier_b_release"
    v5 = tier / "results/no_human/runs/run_20260720_semantic_health_v5"
    v6 = tier / "results/no_human/runs/run_20260720_external_protocol_v6"
    required = [
        v5 / "semantic_health_gate.csv", v5 / "full_25_proxy_target_cluster.csv",
        v5 / "paired_baseline_cluster.csv", v5 / "pre_post_nominal_comparison.csv",
        v6 / "external_image_manifest.csv", v6 / "external_feature_semantic_health.csv",
        v6 / "external_composite_score_health.csv", v6 / "raw_composite_external_holdout.csv",
        v6 / "mechanism_crossing_intervals.csv",
        tier / "logs/verification_protocol_v5_clean.log",
        tier / "logs/verification_protocol_v6_clean.log",
    ]
    for path in required:
        if not path.exists(): issues.append(f"active evidence missing: {path.relative_to(pkg)}")
    if any(not path.exists() for path in required): return
    gate = pd.read_csv(v5 / "semantic_health_gate.csv")
    if not gate["active_gate_passed"].all(): issues.append("v5 active feature gate failed")
    if len(pd.read_csv(v5 / "full_25_proxy_target_cluster.csv")) != 25:
        issues.append("v5 does not release 25 associations")
    paired = pd.read_csv(v5 / "paired_baseline_cluster.csv")
    if len(paired) != 5 or (paired["delta_rho"] > 0).sum() != 1:
        issues.append("v5 paired-baseline pattern is not one gain and four losses")
    comparison = pd.read_csv(v5 / "pre_post_nominal_comparison.csv")
    if not comparison["post_fix_retained_abs_weight_fraction"].eq(1.0).all():
        issues.append("v5 repaired formulas do not retain full active weight")
    images = pd.read_csv(v6 / "external_image_manifest.csv")
    if len(images) != 1250: issues.append("v6 image manifest row count is not 1,250")
    health = pd.read_csv(v6 / "external_feature_semantic_health.csv")
    if len(health) != 6 or not health["passed_non_degenerate"].all():
        issues.append("v6 repaired feature health failed")
    score = pd.read_csv(v6 / "external_composite_score_health.csv")
    failures = set(score.loc[~score["passed_non_degenerate_final_score"], "proxy"])
    if failures != {"visual_complexity_proxy", "visual_intensity_proxy"}:
        issues.append("v6 score-collapse audit changed")
    if len(pd.read_csv(v6 / "mechanism_crossing_intervals.csv")) != 120:
        issues.append("v6 mechanism crossing row count is not 120")
    for log in required[-2:]:
        if "RESULT: PASS" not in log.read_text(encoding="utf-8"):
            issues.append(f"clean verification log is not a pass: {log.name}")


def check_manuscript_and_claims(pkg: Path, issues: list[str]) -> None:
    tex_paths = [pkg / "manuscript_source/main_no_human.tex", *(pkg / "manuscript_source/sections_no_human").glob("*.tex")]
    text = "\n".join(path.read_text(encoding="utf-8") for path in tex_paths)
    required = [
        "Semantic Health Checks", "spectral-residual", "constant 0.5",
        "G-dev", "separately implemented internal OpenCV renderer",
        "not a perceptual scale", "not a natural-image validation set",
        "raw weighted sums", "score-distribution gate",
    ]
    for phrase in required:
        if phrase.lower() not in text.lower(): issues.append(f"manuscript wording missing: {phrase}")
    for stale in ["run_20260718_protocol_v3", "run_20260718_protocol_v4", "all five matched"]:
        if stale.lower() in text.lower(): issues.append(f"stale manuscript wording: {stale}")

    ledger = pd.read_csv(pkg / "quality_control/claim_evidence_ledger.csv")
    if ledger["claim_id"].duplicated().any(): issues.append("duplicate claim IDs")
    for row in ledger.itertuples(index=False):
        if row.support_status not in {"supported", "qualified"}:
            issues.append(f"invalid claim status: {row.claim_id}")
        if not (pkg / row.evidence_file).exists():
            issues.append(f"claim evidence missing: {row.claim_id} -> {row.evidence_file}")
    audits = list((pkg / "audits").glob("*.md"))
    if len(audits) != 1 or audits[0].name != "pre_submission_audit.md":
        issues.append("package must contain exactly one active audit")


def check_hygiene(pkg: Path, issues: list[str]) -> None:
    for path in pkg.rglob("*"):
        if path.is_file() and path.name.lower() in BANNED_NAMES:
            issues.append(f"excluded human-study asset present: {path.relative_to(pkg)}")
    runs = pkg / "tier_b_release/results/no_human/runs"
    if runs.exists():
        active = {p.name for p in runs.iterdir() if p.is_dir()}
        expected = {"run_20260720_semantic_health_v5", "run_20260720_external_protocol_v6"}
        if active != expected: issues.append(f"unexpected active Tier-B run directories: {sorted(active)}")
    if (pkg / "pre_submission_audit.md").exists():
        issues.append("duplicate audit at package root")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    args = parser.parse_args()
    pkg, root = args.package.resolve(), args.canonical_root.resolve()
    issues: list[str] = []
    check_submission_manifest(pkg, issues)
    check_canonical(pkg, root, issues)
    check_scientific_evidence(pkg, issues)
    check_manuscript_and_claims(pkg, issues)
    check_hygiene(pkg, issues)
    if issues:
        print("Submission package verification FAILED:")
        for issue in issues[:100]: print(f"  - {issue}")
        sys.exit(1)
    print("Submission package verification passed.")


if __name__ == "__main__":
    main()
