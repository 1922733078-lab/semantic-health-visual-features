#!/usr/bin/env python3
"""
Step 1 — Establish a Canonical Data Dictionary
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "data" / "documentation"
DOC.mkdir(parents=True, exist_ok=True)

DATA_DICTIONARY = DOC / "data_dictionary.md"
DATASET_ROLES = DOC / "dataset_roles.csv"

data_dictionary = """# Canonical Data Dictionary

> Project: Interpretable Lightweight Assessment of Visual Complexity and Aesthetic Perception in Art and Design Images
> Generated during Step 1 of the Human-Light redesign.

## 1. Datasets

### D-proxy: Large-scale proxy-development dataset

| Field | Type | Description | Analytical role |
|---|---|---|---|
| `image_id` | string | Unique image identifier, e.g. `painting_0000`, `ui_0196` | Primary key |
| `category` | categorical | Image category: `painting`, `ui`, `packaging`, `poster`, `banner` | Stratification, category analysis |
| `standardized_path` | string | Local path to resized/processed image file | Image loading |
| `in_human_subset` | boolean | Whether the image is one of the 100 human-rated images | Leakage control |
| Source column | string | Original dataset source (inferred from `image_id` prefix) | Source-held-out evaluation |

**Location**: `data/processed/metadata.csv`  
**Size**: 17,337 rows (images).  
**Permitted use**: Feature development, model selection, ablation, robustness testing.  
**Prohibited use**: Direct claims about human-perception accuracy without frozen validation.

### D-human-raw: Individual human ratings

| Field | Type | Description | Analytical role |
|---|---|---|---|
| `image_id` | string | Image identifier | Foreign key to metadata |
| `category` | categorical | Image category | Stratification |
| `rater_id` | string | Rater identifier, `rater_01` to `rater_30` | Reliability analysis |
| `dimension` | categorical | Rated dimension in Chinese: `视觉复杂度` (Complexity), `美感吸引力` (Beauty), `秩序感` (Order), `视觉层级清晰度` (Hierarchy), `情感强度` (Emotion) | Dimension-specific analysis |
| `rating` | integer 1–7 | Human rating on a 7-point Likert scale | Dependent variable |

**Location**: `盲评问卷/ratings/real_human_ratings.csv`  
**Size**: 15,000 rows = 100 images × 30 raters × 5 dimensions.  
**Permitted use**: Reliability analysis, rater-saturation, frozen validation, optional calibration with declared split.  
**Prohibited use**: Model tuning without a declared split.

### D-human-mean: Aggregated human ratings

| Field | Type | Description |
|---|---|---|
| `image_id` | string | Image identifier |
| `category` | categorical | Image category |
| `{dimension}_mean` | float | Mean rating across 30 raters for each dimension |

**Location**: computed from `real_human_ratings.csv`; saved under `results/redesign/tables/human_mean_ratings.csv`.  
**Permitted use**: Frozen validation or split calibration.  
**Prohibited use**: Simultaneous tuning and final testing.

### D-public-human (optional)

Public human-rated aesthetic datasets may be used for external development or transfer learning.  
**Prohibited use**: Mixing with local test data without source tracking.

## 2. Feature matrices

### Traditional features (30 core features)

**Location**: `data/features/traditional_features.csv`

| Feature group | Features |
|---|---|
| Color | `num_dominant_colors`, `hue_mean`, `saturation_mean`, `saturation_std`, `value_mean`, `value_std`, `lightness_contrast`, `color_entropy`, `warm_color_ratio`, `hue_contrast`, `color_harmony` |
| Edge / texture | `edge_density`, `edge_orientation_entropy`, `gray_mean`, `gray_std`, `gray_entropy`, `gradient_energy` |
| Composition | `symmetry`, `rule_of_thirds`, `center_offset_x`, `center_offset_y`, `whitespace_ratio`, `fg_bg_ratio`, `diagonal_energy_ratio` |
| Text / OCR | `text_coverage`, `text_block_count`, `font_size_cv`, `has_text` |
| Saliency | `saliency_mean`, `saliency_std` |
| Identifiers | `image_id`, `category` |

### Enhanced features (44 features)

**Location**: `data/features/enhanced_features.csv`

Includes the 30 traditional features plus 14 additional design-theory features:

| Additional features |
|---|
| `golden_ratio`, `face_presence`, `face_size`, `color_harmony_adv`, `warm_cool_balance`, `fractal_dim`, `figure_ground`, `repetition_fft`, `grid_alignment`, `quadrant_balance`, `lr_similarity`, `top_bottom_ratio`, `proximity_cv`, `alignment_score` |

## 3. Proxy labels

**Location**: `data/ratings/aesthetic_scores.csv`

| Field | Type | Description |
|---|---|---|
| `image_id` | string | Image identifier |
| `category` | categorical | Image category |
| `overall` | float | Overall aesthetic/complexity proxy score |
| `beauty` | float | Beauty proxy score |
| `complexity` | float | Complexity proxy score |
| `order` | float | Order proxy score |
| `emotion` | float | Emotion proxy score |

**Note**: These are computational proxy targets. They must not be labeled "human ground truth."

## 4. Data roles

| Role | Definition |
|---|---|
| Feature development | Use only D-proxy and permitted public data. |
| Model selection | Nested cross-validation on D-proxy; human ratings must not influence choices. |
| Frozen external validation | Apply a pre-fitted model once to D-human-mean. |
| Calibration | Optional mapping learned on a held-out split of D-human; final test split untouched. |
| Testing | Held-out images/folds, never used for fitting or feature design. |

## 5. Scale and anchors

Human ratings use a 7-point Likert scale. Anchors used in the questionnaire should be recorded in `盲评问卷/盲评问卷.xlsx` and the rating protocol. The scale is treated as ordinal for rank-based metrics and interval for ICC under absolute agreement.
"""

DATA_DICTIONARY.write_text(data_dictionary, encoding="utf-8")
print(f"Wrote {DATA_DICTIONARY}")

dataset_roles_csv = """Dataset,Content,PermittedUse,ProhibitedUse
D-proxy,17,337 images with proxy scores,Feature development; model selection; ablation; robustness,Direct claims about human accuracy
D-human-raw,30 x 100 x 5 individual ratings,Reliability and audit,Model tuning without a declared split
D-human-mean,Mean rating per image and dimension,Frozen validation or split calibration,Simultaneous tuning and final testing
D-public-human (optional),Public human-rated aesthetic dataset,External development or transfer learning,Mixing with local test data without source tracking
"""

DATASET_ROLES.write_text(dataset_roles_csv, encoding="utf-8")
print(f"Wrote {DATASET_ROLES}")

if __name__ == "__main__":
    print("Step 1 complete.")
