#!/usr/bin/env python3
"""Remove Tier-B image debris not listed in the frozen public-demo manifest."""

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MANIFEST = ROOT / "results/no_human/tier_b_release/public_demo/dataset_manifest.csv"
ACTIVE_MANIFEST = ROOT.parent / "99-投稿包/no_human/manifests/submission_manifest.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging", type=Path)
    args = parser.parse_args()

    with PUBLIC_MANIFEST.open(newline="") as handle:
        listed = {row["relative_path"] for row in csv.DictReader(handle)}
    with ACTIVE_MANIFEST.open(newline="") as handle:
        candidates = {
            row["relative_path"].removeprefix("tier_b_release/")
            for row in csv.DictReader(handle)
            if row["relative_path"].startswith("tier_b_release/synthetic_benchmark/images/")
        }

    extras = sorted(candidates - listed)
    roots = [ROOT / "results/no_human/tier_b_release"]
    if args.staging:
        roots.append(args.staging.resolve() / "tier_b_release")
    deleted = 0
    for package_root in roots:
        for relative in extras:
            path = package_root / relative
            if path.exists():
                path.unlink()
                deleted += 1

    print(f"listed={len(listed)} candidates={len(candidates)} extras={len(extras)} deleted={deleted}")


if __name__ == "__main__":
    main()
