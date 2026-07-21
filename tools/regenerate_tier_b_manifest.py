#!/usr/bin/env python3
"""Prune public-image debris and regenerate the complete Tier-B manifest."""

import argparse
import csv
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PACKAGE = ROOT / "results" / "no_human" / "tier_b_release"


def hash_row(item):
    package, path = item
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": path.relative_to(package).as_posix(),
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    package = args.package.resolve()

    public_manifest = package / "public_demo" / "dataset_manifest.csv"
    with public_manifest.open(newline="", encoding="utf-8") as handle:
        listed = {
            row["relative_path"]
            for row in csv.DictReader(handle)
            if row.get("relative_path")
        }
    image_root = package / "synthetic_benchmark" / "images"
    removed = 0
    for path in image_root.iterdir():
        relative = path.relative_to(package).as_posix()
        if path.is_file() and relative not in listed:
            path.unlink()
            removed += 1

    manifest = package / "manifests" / "release_manifest.csv"
    files = [
        path for path in package.rglob("*")
        if path.is_file()
        and path != manifest
        and "__pycache__" not in path.parts
        and ".pytest_cache" not in path.parts
        and path.suffix != ".pyc"
    ]
    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as executor:
        rows = list(executor.map(hash_row, ((package, path) for path in sorted(files))))
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256", "bytes"])
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"package={package} listed_images={len(listed)} removed_unlisted={removed} "
        f"manifest_files={len(rows)}"
    )


if __name__ == "__main__":
    main()
