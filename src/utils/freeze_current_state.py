#!/usr/bin/env python3
"""
Step 0 — Freeze and Preserve the Current Study
Generate hashes, environment record, and source-version manifest for the redesign.
"""
import hashlib
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REDESIGN = ROOT / "results" / "redesign"
MANIFESTS = REDESIGN / "manifests"
ARCHIVE = REDESIGN / "archive"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_target_files() -> list[Path]:
    files = []

    # Manuscript and submission files
    files.extend([
        ROOT / "paper" / "main.tex",
        ROOT / "paper" / "main.pdf",
        ROOT / "paper" / "main.docx",
        ROOT / "paper" / "main_v2.docx",
        ROOT / "paper" / "main_v3.docx",
        ROOT / "paper" / "references.bib",
    ])

    # Human rating data
    rating_dir = ROOT / "盲评问卷" / "ratings"
    files.append(rating_dir / "real_human_ratings.csv")
    for i in range(1, 31):
        files.append(rating_dir / f"评分表_rater_{i:02d}.xlsx")

    files.append(ROOT / "盲评问卷" / "图像清单.csv")

    # Blind-evaluation images
    image_dir = ROOT / "盲评问卷" / "盲评图像"
    if image_dir.exists():
        files.extend(sorted(image_dir.glob("*.jpg")))

    # Feature matrices and metadata
    files.extend([
        ROOT / "data" / "features" / "traditional_features.csv",
        ROOT / "data" / "features" / "enhanced_features.csv",
        ROOT / "data" / "processed" / "metadata.csv",
        ROOT / "data" / "ratings" / "raw_ratings.csv",
        ROOT / "data" / "ratings" / "aggregated_ratings.csv",
        ROOT / "data" / "ratings" / "aesthetic_scores.csv",
    ])

    # Core analysis scripts
    scripts = [
        "analyze_real_ratings.py",
        "calc_all_metrics.py",
        "enhanced_experiment.py",
        "enhanced_final.py",
        "enhancement_experiment.py",
        "final_analysis.py",
        "final_metrics.py",
        "final_phase2.py",
        "final_phase2_v2.py",
        "generate_figures.py",
        "phase1_correct.py",
        "phase2_analysis.py",
        "phase2_clean.py",
        "run_final.py",
        "sota_experiment.py",
    ]
    files.extend(ROOT / s for s in scripts)

    return [f for f in files if f.exists()]


def write_input_hashes(files: list[Path]):
    rows = []
    for f in files:
        rel = f.relative_to(ROOT)
        rows.append({
            "path": str(rel),
            "sha256": sha256_file(f),
            "bytes": f.stat().st_size,
        })
    out = MANIFESTS / "input_hashes.csv"
    with open(out, "w", encoding="utf-8") as fp:
        fp.write("path,sha256,bytes\n")
        for r in rows:
            fp.write(f"{r['path']},{r['sha256']},{r['bytes']}\n")
    print(f"Wrote {out}")
    return rows


def write_environment():
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": f"{platform.system()} {platform.release()} {platform.machine()}",
        "python_version": sys.version,
        "python_executable": sys.executable,
    }
    out = MANIFESTS / "environment.txt"
    with open(out, "w", encoding="utf-8") as fp:
        for k, v in info.items():
            fp.write(f"{k}: {v}\n")
    print(f"Wrote {out}")


def write_source_version():
    out = MANIFESTS / "source_version.md"
    lines = [
        "# Source Version Record\n",
        f"\nRedesign start (local): {datetime.now().isoformat()}",
        f"\nWorking directory: {ROOT}",
        "\n## Current manuscript versions",
        "\n- `paper/main.tex`",
        "- `paper/main.pdf`",
        "- `paper/main.docx`",
        "- `paper/main_v2.docx`",
        "- `paper/main_v3.docx`",
        "\n## Notes",
        "\nThis redesign freezes the current study before any result is changed.",
        "All subsequent redesigned outputs are written under `results/redesign/`.",
        "\n",
    ]
    out.write_text("".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def archive_key_files(files: list[Path]):
    archive_sub = ARCHIVE / "source_snapshot"
    archive_sub.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in files:
        rel = f.relative_to(ROOT)
        dest = archive_sub / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        copied += 1
    print(f"Archived {copied} files to {archive_sub}")


def main():
    REDESIGN.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    files = collect_target_files()
    write_input_hashes(files)
    write_environment()
    write_source_version()
    archive_key_files(files)

    print("Step 0 complete.")


if __name__ == "__main__":
    main()
