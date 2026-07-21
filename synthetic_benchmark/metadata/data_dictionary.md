# Synthetic Benchmark Data Dictionary

## Image metadata columns

| Column | Type | Description |
|---|---|---|
| `image_id` | string | Unique identifier: `{family}_{construct}_L{level}_S{seed}_I{index:04d}` |
| `family` | categorical | Generator family: `A_poster`, `B_banner`, `C_card` |
| `construct` | categorical | Construct manipulation or `interaction` |
| `level` | integer | Severity level 0–4; `-1` for interaction conditions |
| `seed` | integer | Generator seed 0–2 |
| `image_index` | integer | Within-cell sample index |
| `canvas_width` | integer | Image width in pixels |
| `canvas_height` | integer | Image height in pixels |
| `n_primitives` | integer | Number of non-focal primitives |
| `overlap_ratio_param` | float | Target overlap ratio parameter |
| `edge_density_factor` | float | Edge-density control parameter |
| `spacing_irregularity` | float | Spacing irregularity parameter |
| `alignment_jitter` | float | Alignment jitter parameter |
| `grid_deviation` | float | Grid deviation parameter |
| `spacing_variance` | float | Spacing variance parameter |
| `hue_dispersion` | float | Hue dispersion parameter |
| `chroma_dispersion` | float | Chroma dispersion parameter |
| `palette_incoherence` | float | Palette incoherence parameter |
| `contrast_ratio` | float | Colour contrast ratio parameter |
| `luminance_contrast` | float | Luminance contrast parameter |
| `edge_contrast` | float | Edge contrast parameter |
| `focal_contrast` | float | Focal-object contrast parameter |
| `focal_size_ratio` | float | Focal-object area ratio parameter |
| `secondary_count` | integer | Number of secondary elements |
| `whitespace_fraction_param` | float | Target whitespace fraction parameter |
| `has_text_bars` | boolean | Whether text-like bars are drawn |
| `text_bar_count` | integer | Number of text-like bars |
| `actual_overlap_ratio` | float | Realized overlap area ratio from scene graph |
| `actual_whitespace` | float | Realized whitespace fraction from scene graph |
| `image_path` | string | Relative path to rendered PNG |
| `sha256` | string | SHA-256 hash of rendered PNG |
| `interaction` | string | Interaction condition name (if applicable) |

## Objective target columns

| Column | Range | Description |
|---|---|---|
| `structural_clutter_index` | [0, 100] | Visual complexity target |
| `geometric_order_index` | [0, 100] | Layout order target |
| `palette_coherence_index` | [0, 100] | Colour harmony target |
| `visual_salience_index` | [0, 100] | Visual intensity target |
| `focal_hierarchy_index` | [0, 100] | Layout hierarchy target |
| `target_formula_version` | string | Hash of the target formula specification |

## Units and missing-value behavior

- All pixel dimensions are integers.
- All parameter ratios and dispersion values are floats in [0, 1] unless otherwise noted.
- Target indices are scaled to [0, 100] to match proxy score range.
- Missing values are not expected; if any metadata row lacks a rendered image, the row is excluded before target computation.
