#!/usr/bin/env python3
"""Atomically replace the active Route-C v5/v6 package without keeping duplicates."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_ROOT = ROOT.parent / "99-投稿包"
PACKAGE = SUBMISSION_ROOT / "no_human"
REGISTER = ROOT / "results/no_human/quality_control/route_c_canonical_artifact_register.csv"


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


def copy_artifact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(
            source, destination,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
        )
    else:
        shutil.copy2(source, destination)


def package_readme() -> str:
    return """# JVCIR Route-C submission package

**Manuscript:** *Semantic Health Checks for Reproducible Visual-Feature Benchmarks: A Two-Renderer Failure Analysis*  
**Target journal:** *Journal of Visual Communication and Image Representation*  
**Active evidence:** protocol v5 and protocol v6 only

This package reports a traced software failure analysis. A missing optional
OpenCV saliency component had produced a deterministic constant map; the repair
restores feature responsiveness and exposes both changed conclusions and a
separate cross-renderer score-clipping failure. The five legacy formulas are
not presented as perceptual scales, and the OpenCV renderer is an internal
stress test rather than natural-image or external validation.

## Contents

- `manuscript_output/`: submission PDF and editable DOCX
- `manuscript_source/`: LaTeX, bibliography, sections, and generated tables
- `documents/`: highlights, cover letter, scope, data, and reproduction notes
- `figures/`: the four figures used by the manuscript
- `quality_control/`: active claim/evidence and corrective-issue ledgers
- `audits/`: the single active pre-submission audit
- `tier_b_release/`: complete runnable object with 17,675 images, tests, hashes,
  protocol-v5/v6 outputs, and clean-environment records
- `manifests/`: staged-file hashes, canonical comparisons, and release metadata

Before external submission, deposit the exact Tier B object in a persistent
repository, insert its DOI/URL and immutable tag, and confirm the corresponding
author's e-mail in the submission system.
"""


def main() -> None:
    global PACKAGE
    final_package = PACKAGE
    staging = SUBMISSION_ROOT / ".no_human_building"
    previous = SUBMISSION_ROOT / ".no_human_previous"

    with REGISTER.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("empty Route-C register")

    if staging.exists():
        shutil.rmtree(staging)
    PACKAGE = staging
    PACKAGE.mkdir(parents=True)

    tier_row = next(row for row in rows if row["role"] == "tier_b_release")
    copy_artifact(ROOT / tier_row["canonical_path"], PACKAGE / tier_row["promoted_path"])
    for row in rows:
        if row["role"] == "tier_b_release" or row["promoted_path"].startswith("tier_b_release/"):
            continue
        copy_artifact(ROOT / row["canonical_path"], PACKAGE / row["promoted_path"])

    (PACKAGE / "README.md").write_text(package_readme(), encoding="utf-8")
    register_dst = PACKAGE / "quality_control/route_c_canonical_artifact_register.csv"
    register_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REGISTER, register_dst)

    comparisons = []
    for row in rows:
        promoted = PACKAGE / row["promoted_path"]
        actual = (
            sha(promoted / "manifests/release_manifest.csv")
            if row["role"] == "tier_b_release"
            else dir_sha(promoted) if promoted.is_dir() else sha(promoted)
        )
        comparisons.append({
            "role": row["role"],
            "canonical_path": row["canonical_path"],
            "promoted_path": row["promoted_path"],
            "expected_sha256": row["expected_sha256"],
            "promoted_sha256": actual,
            "matches": actual == row["expected_sha256"],
        })
    if not all(item["matches"] for item in comparisons):
        raise RuntimeError("canonical comparison failed during staging")

    manifest_dir = PACKAGE / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = manifest_dir / "canonical_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparisons[0].keys())
        writer.writeheader(); writer.writerows(comparisons)

    with (PACKAGE / "tier_b_release/manifests/release_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        tier_b_manifest_entries = sum(1 for _ in csv.DictReader(handle))
    release_metadata = {
        "release_run_id": "route_c_20260720_round7_candidate",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_journal": "Journal of Visual Communication and Image Representation",
        "active_protocols": ["5.0", "6.0"],
        "canonical_role_count": len(rows),
        "tier_b_manifest_entries": tier_b_manifest_entries,
        "administrative_holds": ["persistent DOI/URL", "corresponding-author email"],
    }
    metadata_path = manifest_dir / "release_metadata.json"
    metadata_path.write_text(json.dumps(release_metadata, indent=2) + "\n", encoding="utf-8")

    excluded = {
        "manifest_sha256.txt",
        "manifests/submission_manifest.csv",
        "manifests/canonical_comparison.csv",
        "manifests/release_metadata.json",
    }
    payload = []
    for path in sorted(p for p in PACKAGE.rglob("*") if p.is_file()):
        rel = path.relative_to(PACKAGE).as_posix()
        if rel in excluded or rel.startswith("logs/"):
            continue
        payload.append({"relative_path": rel, "sha256": sha(path), "bytes": path.stat().st_size})
    manifest_path = manifest_dir / "submission_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=payload[0].keys())
        writer.writeheader(); writer.writerows(payload)

    controls = [comparison_path, metadata_path, manifest_path]
    (PACKAGE / "manifest_sha256.txt").write_text(
        "".join(f"{sha(path)}  {path.relative_to(PACKAGE).as_posix()}\n" for path in controls),
        encoding="utf-8",
    )
    if previous.exists():
        shutil.rmtree(previous)
    if final_package.exists():
        shutil.move(str(final_package), str(previous))
    try:
        shutil.move(str(PACKAGE), str(final_package))
    except Exception:
        if previous.exists() and not final_package.exists():
            shutil.move(str(previous), str(final_package))
        raise
    if previous.exists():
        shutil.rmtree(previous)
    PACKAGE = final_package
    print(f"Built {PACKAGE} with {len(payload)} manifested payload files")


if __name__ == "__main__":
    main()
