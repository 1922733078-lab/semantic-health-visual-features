#!/usr/bin/env python3
"""
Step 6 — Freeze the Feature Set and Preprocessing Pipeline
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "data" / "documentation"
QC = ROOT / "results" / "redesign" / "quality_control"
DOC.mkdir(parents=True, exist_ok=True)
QC.mkdir(parents=True, exist_ok=True)

FEATURE_MANIFEST = DOC / "feature_manifest.csv"
FEATURE_QUALITY_REPORT = QC / "feature_quality_report.md"

features = [
    # Core 30 features (human-independent)
    ("num_dominant_colors", "Number of dominant colors (k-means clusters covering >5% of pixels)", "count", "k=5 clusters, threshold=0.05", "src/features/traditional_features.py", "no"),
    ("hue_mean", "Mean hue in HSV color space", "degrees (0-180)", "HSV conversion", "src/features/traditional_features.py", "no"),
    ("saturation_mean", "Mean saturation in HSV color space", "0-255", "HSV conversion", "src/features/traditional_features.py", "no"),
    ("saturation_std", "Standard deviation of saturation", "0-255", "HSV conversion", "src/features/traditional_features.py", "no"),
    ("value_mean", "Mean value (brightness) in HSV", "0-255", "HSV conversion", "src/features/traditional_features.py", "no"),
    ("value_std", "Standard deviation of value", "0-255", "HSV conversion", "src/features/traditional_features.py", "no"),
    ("lightness_contrast", "Root-mean-square contrast in CIELAB L* channel", "0-255", "CIELAB conversion", "src/features/traditional_features.py", "no"),
    ("color_entropy", "Entropy of 3D HSV color histogram", "bits", "12x8x8 bins", "src/features/traditional_features.py", "no"),
    ("warm_color_ratio", "Proportion of warm hue pixels", "0-1", "warm=[0,30]U[150,180]", "src/features/traditional_features.py", "no"),
    ("hue_contrast", "Mean angular distance among top-3 hue bins", "0-1", "36 bins, normalized", "src/features/traditional_features.py", "no"),
    ("color_harmony", "Harmony score based on top-2 hue bin difference", "0-1", "harmonic angles {0,30,60,120,180}", "src/features/traditional_features.py", "no"),
    ("edge_density", "Mean Canny edge magnitude", "0-1", "Canny thresholds 50/150", "src/features/traditional_features.py", "no"),
    ("edge_orientation_entropy", "Entropy of strong-edge orientation histogram", "bits", "8 bins, 70th percentile threshold", "src/features/traditional_features.py", "no"),
    ("gray_mean", "Mean grayscale intensity", "0-255", "BGR2GRAY", "src/features/traditional_features.py", "no"),
    ("gray_std", "Standard deviation of grayscale intensity", "0-255", "BGR2GRAY", "src/features/traditional_features.py", "no"),
    ("gray_entropy", "Entropy of grayscale histogram", "bits", "256 bins", "src/features/traditional_features.py", "no"),
    ("gradient_energy", "Mean Sobel gradient magnitude", "0-", "Sobel 3x3", "src/features/traditional_features.py", "no"),
    ("symmetry", "Structural similarity between image and horizontal flip", "0-1", "SSIM, adaptive win_size", "src/features/traditional_features.py", "no"),
    ("rule_of_thirds", "Proximity of saliency centroid to nearest rule-of-thirds intersection", "0-1", "saliency centroid", "src/features/traditional_features.py", "no"),
    ("center_offset_x", "Horizontal offset of saliency centroid from image center", "-1 to 1", "saliency centroid", "src/features/traditional_features.py", "no"),
    ("center_offset_y", "Vertical offset of saliency centroid from image center", "-1 to 1", "saliency centroid", "src/features/traditional_features.py", "no"),
    ("whitespace_ratio", "Proportion of low-edge-density regions", "0-1", "32x32 box filter, threshold=0.02", "src/features/traditional_features.py", "no"),
    ("fg_bg_ratio", "Proportion of pixels above mean saliency", "0-1", "mean saliency threshold", "src/features/traditional_features.py", "no"),
    ("diagonal_energy_ratio", "Ratio of diagonal to anti-diagonal gradient energy", "0-", "Sobel magnitude", "src/features/traditional_features.py", "no"),
    ("text_coverage", "Edge-density proxy for text coverage", "0-1", "Canny thresholds 50/150", "src/features/traditional_features.py", "no"),
    ("text_block_count", "Number of horizontal projection crossings above mean", "count", "mean crossing", "src/features/traditional_features.py", "no"),
    ("font_size_cv", "Coefficient of variation of local contrast", "0-", "64x64 blocks", "src/features/traditional_features.py", "no"),
    ("has_text", "Binary flag for high-contrast text regions", "0/1", "Otsu threshold, dark_ratio in (0.02,0.5)", "src/features/traditional_features.py", "no"),
    ("saliency_mean", "Mean fine-grained saliency map value", "0-1", "cv2.saliency.StaticSaliencyFineGrained", "src/features/traditional_features.py", "no"),
    ("saliency_std", "Standard deviation of saliency map", "0-1", "cv2.saliency.StaticSaliencyFineGrained", "src/features/traditional_features.py", "no"),

    # Enhanced 14 features (human-informed)
    ("golden_ratio", "Proximity of strongest gradient peak to golden-ratio point", "0-1", "resize 256x256", "enhancement_experiment.py", "yes"),
    ("face_presence", "Skin-tone pixel presence proxy for faces", "0-1", "HSV skin range", "enhancement_experiment.py", "yes"),
    ("face_size", "Relative skin-tone pixel area", "0-1", "HSV skin range", "enhancement_experiment.py", "yes"),
    ("color_harmony_adv", "Moon-Spencer harmony over 7 angles", "0-1", "36 bins", "enhancement_experiment.py", "yes"),
    ("warm_cool_balance", "Warm-to-cool pixel ratio", "0-1", "warm=[0,30]U[150,180], cool=[90,150]", "enhancement_experiment.py", "yes"),
    ("fractal_dim", "Box-counting dimension estimate from edge density", "1-2", "resize 128x128", "enhancement_experiment.py", "yes"),
    ("figure_ground", "Center-border intensity difference", "0-1", "center vs border quarter mean", "enhancement_experiment.py", "yes"),
    ("repetition_fft", "Low-frequency FFT energy concentration", "0-1", "resize 128x128, 20x20 low-freq window", "enhancement_experiment.py", "yes"),
    ("grid_alignment", "Proportion of edge pixels aligned to 1/3,1/2,2/3 grid", "0-1", "3-pixel tolerance", "enhancement_experiment.py", "yes"),
    ("quadrant_balance", "1 - normalized quadrant intensity variance", "0-1", "4 quadrants", "enhancement_experiment.py", "yes"),
    ("lr_similarity", "1 / (1 + KL divergence of left/right histograms)", "0-1", "32 bins", "enhancement_experiment.py", "yes"),
    ("top_bottom_ratio", "Ratio of top/bottom grayscale entropy", "0-1", "64 bins", "enhancement_experiment.py", "yes"),
    ("proximity_cv", "1 - CV of nearest-neighbor centroid distances", "0-1", "top 50 edge components", "enhancement_experiment.py", "yes"),
    ("alignment_score", "Proportion of Hough lines near horizontal/vertical", "0-1", "HoughLinesP", "enhancement_experiment.py", "yes"),
]

with open(FEATURE_MANIFEST, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["feature_name", "definition", "unit", "parameters", "code_location", "human_informed"])
    writer.writerows(features)
print(f"Wrote {FEATURE_MANIFEST}")

report = """# Feature Quality and Freeze Report

> Human-Light redesign — Step 6

## Frozen feature set

- **Core features (human-independent)**: 30 features from `src/features/traditional_features.py`.
- **Enhanced features (human-informed)**: 14 additional features from `enhancement_experiment.py`.

## Preprocessing pipeline

| Step | Setting | Note |
|---|---|---|
| Load | `cv2.imread` | BGR order |
| Resize | `cv2.resize(img, (512, 512))` for core features; `(256, 256)` for enhanced features | This difference is a known inconsistency to be harmonized or reported |
| Color spaces | BGR → RGB, HSV, CIELAB, grayscale | Deterministic conversions |
| Saliency | `cv2.saliency.StaticSaliencyFineGrained_create()` | Same OpenCV implementation must be used |
| Edge detection | Canny 50/150 | Fixed thresholds |
| Scalers / PCA | Must be fit inside training folds only | Verified in analysis scripts |

## Human-informed feature audit

The 14 enhanced features were introduced in `enhancement_experiment.py`, whose stated purpose is "Retrain on human ratings with Stacking" and includes a "Layer 4: Order special" designed to improve order/hierarchy predictions on the human dataset. Therefore these features are classified as **human-informed**.

Implications:
- The 30-core feature set is the primary human-independent representation.
- The 44-feature enhanced set is evaluated only within a declared calibration or sensitivity analysis; it is not used for the primary frozen external-validation claim.

## Quality checks

- Determinism: feature extraction uses fixed random seeds (`cv2.KMEANS_RANDOM_CENTERS` in core features) and fixed thresholds.
- Missing values: current extraction silently catches exceptions and returns partial feature vectors. This should be hardened for the redesigned pipeline.
- Image-order invariance: extraction is per-image; order does not affect values.

## Required actions for full compliance

1. Harmonize resize resolution between core (512) and enhanced (256) extraction, or treat the two resolutions as separate preprocessing pipelines.
2. Add explicit NaN handling and per-feature range validation.
3. Replace silent `except: pass` blocks with logged fallbacks and unit tests.
"""

FEATURE_QUALITY_REPORT.write_text(report, encoding="utf-8")
print(f"Wrote {FEATURE_QUALITY_REPORT}")

print("Step 6 complete.")
