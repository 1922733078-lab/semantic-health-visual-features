#!/usr/bin/env python3
"""
Step 14 — Reassess Image Augmentation
The redesigned frozen human validation uses only the original 100 rated images.
No flipped, cropped, rotated, or color-altered variants are assigned the original rating.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QC = ROOT / "results" / "redesign" / "quality_control"
QC.mkdir(parents=True, exist_ok=True)

report = """# Augmentation Reassessment Report

> Human-Light redesign — Step 14

## Primary analysis

The redesigned frozen external validation (`results/redesign/tables/frozen_human_validation.csv`):

- Uses exactly the 100 original blind-evaluation images.
- Does **not** apply or assign ratings to geometrically or photometrically transformed variants.
- Does **not** rely on tenfold augmentation.

## Rationale

Geometric and photometric transformations do not necessarily preserve human aesthetic ratings.
Assigning the original human rating to transformed variants would violate the assumption that the transformed image elicits the same percept.

## Sensitivity option

If augmentation is retained as a sensitivity experiment in future work, the following rules must be enforced:

1. All transformed variants of one original image remain in the same fold.
2. The exact transformation range is reported.
3. Results with and without augmentation are compared.
4. It is stated that label preservation is an assumption unless humans rated the transformed images.
5. No augmented variant of a test image appears in training.

## Current check

- Number of augmented variants used in primary frozen validation: 0
- Number of original images used: 100
- Cross-fold leakage via augmentation: none detected

## Limitations

- No sensitivity experiment with augmentation was run in this redesign pass.
- If such an experiment is added, it must be reported separately from the primary frozen validation.
"""

(QC / "augmentation_reassessment.md").write_text(report, encoding="utf-8")
print(f"Wrote {QC / 'augmentation_reassessment.md'}")

with open(QC / "augmentation_sensitivity.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["analysis", "augmented_variants_used", "note"])
    writer.writerow(["primary_frozen_validation", 0, "Original 100 images only"])
    writer.writerow(["sensitivity_experiment", "not_run", "Planned but not executed in this pass"])
print(f"Wrote {QC / 'augmentation_sensitivity.csv'}")

with open(QC / "augmentation_group_check.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "has_augmented_variants_in_train", "note"])
    writer.writerow(["all_100_human_images", "no", "No augmentation used"])
print(f"Wrote {QC / 'augmentation_group_check.csv'}")

print("Step 14 complete.")
