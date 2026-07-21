#!/usr/bin/env python3
"""
Build the canonical no-human image manifest from permitted non-human inputs.

Inputs:
    data/processed/metadata.csv
    results/no_human/quality_control/image_provenance.csv
    results/no_human/quality_control/canonical_splits.csv

Output:
    data/processed/no_human_image_manifest.csv

The manifest contains only image_id, relative_path, category,
split_or_subset_role, sha256, and status. No rater-derived fields are included.
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

METADATA_PATH = ROOT / "data" / "processed" / "metadata.csv"
PROVENANCE_PATH = ROOT / "results" / "no_human" / "quality_control" / "image_provenance.csv"
SPLITS_PATH = ROOT / "results" / "no_human" / "quality_control" / "canonical_splits.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "no_human_image_manifest.csv"


def read_csv_dict(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main():
    if not METADATA_PATH.exists():
        print(f"Metadata not found: {METADATA_PATH}", file=sys.stderr)
        sys.exit(1)

    metadata = {row["image_id"]: row for row in read_csv_dict(METADATA_PATH)}
    provenance = {row["image_id"]: row for row in read_csv_dict(PROVENANCE_PATH)}
    splits = {row["image_id"]: row for row in read_csv_dict(SPLITS_PATH)}

    rows = []
    for image_id, meta in metadata.items():
        prov = provenance.get(image_id, {})
        split = splits.get(image_id, {})
        rows.append({
            "image_id": image_id,
            "relative_path": meta.get("standardized_path", ""),
            "category": meta.get("category", ""),
            "split_or_subset_role": split.get("role", ""),
            "sha256": prov.get("sha256", ""),
            "status": prov.get("status", ""),
        })

    rows.sort(key=lambda r: r["image_id"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_id", "relative_path", "category", "split_or_subset_role", "sha256", "status"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
