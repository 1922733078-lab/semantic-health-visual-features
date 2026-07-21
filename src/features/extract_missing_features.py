#!/usr/bin/env python3
"""
Extract traditional features for images missing from traditional_features.csv
and merge them back into the canonical feature matrix.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.traditional_features import process_single_image


def main():
    existing = pd.read_csv(ROOT / "data" / "features" / "traditional_features.csv")
    meta = pd.read_csv(ROOT / "data" / "processed" / "metadata.csv")

    missing = meta[~meta["image_id"].isin(existing["image_id"])].copy()
    print(f"Missing traditional features: {len(missing)} images")

    new_rows = []
    for _, row in missing.iterrows():
        img_path = ROOT / row["standardized_path"]
        result = process_single_image((str(img_path), row["image_id"], row["category"]))
        if result is not None:
            nan_keys = [k for k, v in result.items() if isinstance(v, float) and v != v]
            if nan_keys:
                print(f"  Partial ({row['image_id']}): NaN in {nan_keys}")
            new_rows.append(result)
        else:
            print(f"  Failed: {row['image_id']}")

    new_df = pd.DataFrame(new_rows)
    # Keep only columns present in the existing canonical feature matrix
    new_df = new_df[[c for c in existing.columns if c in new_df.columns]]
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined[existing.columns]
    combined.to_csv(ROOT / "data" / "features" / "traditional_features.csv", index=False)
    print(f"Updated traditional_features.csv: {len(combined)} images")


if __name__ == "__main__":
    main()
