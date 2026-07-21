#!/usr/bin/env python3
"""
Build the Tier B reproducibility release package.

Copies only the no-human-evaluation code, configs, tests, documentation,
and the two synthetic benchmark implementations into
results/no_human/tier_b_release/.  The frozen images are copied so the archive
can be verified and inspected without relying on a network or regeneration.
"""
import hashlib
import csv
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_DIR = ROOT / "results" / "no_human" / "tier_b_release"

# Explicit inclusion list.  Human-study code under src/analysis, src/data,
# src/figures, src/models, src/utils is intentionally omitted.
COPY_ITEMS = [
    # Core no-human evaluation code
    ("src/no_human", "src/no_human"),
    ("src/features/traditional_features.py", "src/features/traditional_features.py"),
    # Configurations
    ("configs", "configs"),
    # Release-specific tools (avoid copying the broader toolset that references
    # historical human-study files)
    ("tools/build_tier_b_release.py", "tools/build_tier_b_release.py"),
    ("tools/verify_tier_b_release.py", "tools/verify_tier_b_release.py"),
    ("tools/verify_release.py", "tools/verify_release.py"),
    ("tools/regenerate_tier_b_manifest.py", "tools/regenerate_tier_b_manifest.py"),
    # Tests
    ("tests/test_tier_b_release_integrity.py", "tests/test_tier_b_release_integrity.py"),
    ("tests/test_feature_semantic_health.py", "tests/test_feature_semantic_health.py"),
    ("tests/test_manuscript_consistency.py", "tests/test_manuscript_consistency.py"),
    # Public demo and synthetic benchmark
    ("public_demo", "public_demo"),
    ("docs/FEATURE_DATA_DICTIONARY.md", "docs/FEATURE_DATA_DICTIONARY.md"),
    ("synthetic_benchmark/generator", "synthetic_benchmark/generator"),
    ("synthetic_benchmark/evaluate_objective_agreement.py", "synthetic_benchmark/evaluate_objective_agreement.py"),
    ("synthetic_benchmark/run_baselines_ablations.py", "synthetic_benchmark/run_baselines_ablations.py"),
    ("synthetic_benchmark/PROTOCOL.md", "synthetic_benchmark/PROTOCOL.md"),
    # Compatibility engines reused by protocol v6. They remain executable
    # dependencies, but their superseded protocol documents/outputs are omitted.
    ("synthetic_benchmark/run_post_review_protocol_v3.py", "synthetic_benchmark/run_post_review_protocol_v3.py"),
    ("synthetic_benchmark/run_external_holdout_protocol_v4.py", "synthetic_benchmark/run_external_holdout_protocol_v4.py"),
    ("synthetic_benchmark/PROTOCOL_V5_SEMANTIC_HEALTH.md", "synthetic_benchmark/PROTOCOL_V5_SEMANTIC_HEALTH.md"),
    ("synthetic_benchmark/run_semantic_health_protocol_v5.py", "synthetic_benchmark/run_semantic_health_protocol_v5.py"),
    ("synthetic_benchmark/PROTOCOL_V6_EXTERNAL_HOLDOUT.md", "synthetic_benchmark/PROTOCOL_V6_EXTERNAL_HOLDOUT.md"),
    ("synthetic_benchmark/run_external_holdout_protocol_v6.py", "synthetic_benchmark/run_external_holdout_protocol_v6.py"),
    ("synthetic_benchmark/configs", "synthetic_benchmark/configs"),
    ("synthetic_benchmark/splits", "synthetic_benchmark/splits"),
    ("synthetic_benchmark/manifests", "synthetic_benchmark/manifests"),
    ("synthetic_benchmark/metadata", "synthetic_benchmark/metadata"),
    ("synthetic_benchmark/images", "synthetic_benchmark/images"),
    ("results/no_human/runs/run_20260720_semantic_health_v5", "results/no_human/runs/run_20260720_semantic_health_v5"),
    ("results/no_human/runs/run_20260720_external_protocol_v6", "results/no_human/runs/run_20260720_external_protocol_v6"),
    ("paper/main_no_human.tex", "paper/main_no_human.tex"),
    ("paper/references.bib", "paper/references.bib"),
    ("paper/sections_no_human", "paper/sections_no_human"),
    ("paper/figures_route_c", "paper/figures_route_c"),
    # Tests and verification tools
    ("tests/test_synthetic_benchmark_integrity.py", "tests/test_synthetic_benchmark_integrity.py"),
    ("tools/verify_clean_environment.py", "tools/verify_clean_environment.py"),
    ("tools/verify_submission_manifest.py", "tools/verify_submission_manifest.py"),
    # Pipeline entry point and dependency manifests
    ("run_no_human_pipeline.py", "run_no_human_pipeline.py"),
    ("requirements.txt", "requirements.txt"),
    ("requirements-lock.txt", "requirements-lock.txt"),
    ("requirements_no_human.txt", "requirements_no_human.txt"),
    ("pytest.ini", "pytest.ini"),
]

PRESERVE_ITEMS = [
    "README.md",
    "LICENSES.md",
    "CITATION.cff",
    "environment.yml",
    "docs",
    "logs",
]


def copy_tree(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src, dst,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
    )


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def prune_unlisted_public_images() -> None:
    """Keep exactly the 16,425 files in the frozen public image manifest."""
    manifest = RELEASE_DIR / "public_demo" / "dataset_manifest.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        listed = {
            row["relative_path"]
            for row in csv.DictReader(handle)
            if row.get("relative_path")
        }
    image_root = RELEASE_DIR / "synthetic_benchmark" / "images"
    removed = 0
    for path in image_root.iterdir():
        relative = path.relative_to(RELEASE_DIR).as_posix()
        if path.is_file() and relative not in listed:
            path.unlink()
            removed += 1
    print(f"Pruned {removed} unlisted synthetic images; retained {len(listed)} frozen images.")


def main():
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    # Remove stale copied content but preserve hand-written release files
    for child in RELEASE_DIR.iterdir():
        if child.name in PRESERVE_ITEMS:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()

    for rel_src, rel_dst in COPY_ITEMS:
        src = ROOT / rel_src
        dst = RELEASE_DIR / rel_dst
        if not src.exists():
            print(f"[SKIP] source not found: {rel_src}")
            continue
        if src.is_dir():
            copy_tree(src, dst)
        else:
            copy_file(src, dst)
        print(f"Copied {rel_src} -> {rel_dst}")

    prune_unlisted_public_images()

    import pandas as pd
    manifest_dir = RELEASE_DIR / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Software bill of materials
    sbom_path = manifest_dir / "software_bill_of_materials.txt"
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True, text=True, check=True
        )
        sbom_path.write_text(result.stdout, encoding="utf-8")
        print("Wrote software bill of materials.")
    except Exception as e:
        sbom_path.write_text(f"# SBOM generation failed: {e}\n", encoding="utf-8")
        print(f"SBOM generation failed: {e}")

    # Build the manifest last so it covers the SBOM and every release artifact.
    manifest_path = manifest_dir / "release_manifest.csv"
    manifest_rows = []
    for path in sorted(RELEASE_DIR.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rel = path.relative_to(RELEASE_DIR)
        if "__pycache__" in rel.parts or ".pytest_cache" in rel.parts or path.suffix == ".pyc":
            continue
        manifest_rows.append({
            "path": str(rel),
            "sha256": hash_file(path),
            "bytes": path.stat().st_size,
        })
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Wrote release manifest ({len(manifest_rows)} files).")

    print(f"Tier B release built at {RELEASE_DIR}")


if __name__ == "__main__":
    main()
