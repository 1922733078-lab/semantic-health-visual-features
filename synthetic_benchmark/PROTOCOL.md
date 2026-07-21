# Synthetic Benchmark Protocol

**Project:** Interpretable, label-free visual-complexity and layout-order proxy measurement  
**Target journal:** *Journal of Visual Communication and Image Representation* (JVCIR)  
**Date:** 2026-07-14  
**Version:** 2.0  
**Generator version hash:** computed at runtime from `synthetic_benchmark/generator/generate_layouts.py` SHA-256  

## 1. Purpose

Demonstrate that each proposed proxy responds to independently specified, objectively controlled image properties. The benchmark evaluates whether the method recovers known generation parameters on held-out compositions and seeds.

## 2. Non-negotiable constraints

1. No human ratings, human raters, expert judgments, or historical human-rating files are used.
2. All synthetic generator parameters are treated as **objective generation metadata**, not perception labels.
3. Target variables are computed from the rendering parameters and geometric scene graph, not from the pixel features used by the proposed proxy.
4. Protocol is frozen before final benchmark generation.

## 3. Generator design

### 3.1 Output families

Three procedural generator families are used for out-of-distribution compositional testing:

- **Family A (poster-like):** 768×1024 canvas, 6–14 primitives, large focal object allowed.
- **Family B (banner-like):** 1024×384 canvas, 4–10 primitives, horizontal emphasis.
- **Family C (card-like):** 512×768 canvas, 3–8 primitives, compact centered layout.

### 3.2 Parameter ranges and severity levels

Five severity levels (0–4) are defined for each construct manipulation. Level 0 is the low/regular condition; level 4 is the high/perturbed condition.

| Construct | Manipulation parameters | Level 0 | Level 4 |
|---|---|---|---|
| Visual complexity (clutter) | `n_primitives`, `overlap_ratio`, `edge_density_factor`, `spacing_irregularity` | low counts, no overlap, low edge | high counts, high overlap, high edge, irregular spacing |
| Layout order | `alignment_jitter`, `grid_deviation`, `spacing_variance`, `overlap_ratio` | aligned, grid-regular, uniform spacing | jittered, off-grid, irregular spacing |
| Colour harmony | `hue_dispersion`, `chroma_dispersion`, `palette_incoherence` | restricted palette, low dispersion | many hues, high chroma variance, incoherent palette |
| Visual intensity | `contrast_ratio`, `luminance_contrast`, `edge_contrast`, `focal_contrast` | low contrast, flat | high contrast, strong edges, salient focal object |
| Layout hierarchy | `focal_size_ratio`, `focal_contrast`, `secondary_count`, `whitespace_fraction` | flat composition | dominant focal object, few secondary, moderate whitespace |

### 3.3 Interaction conditions

Four held-out interaction conditions are generated to test selectivity:

1. high clutter + high alignment
2. low clutter + high colour noise
3. strong focal hierarchy + large whitespace
4. symmetric composition + high edge density

### 3.4 Fixed factors

- Canvas sizes per family are fixed.
- Primitive shapes are rectangles and circles.
- Background colour is sampled from a fixed low-saturation palette.
- Text-like elements are rendered as horizontal high-contrast bars (no linguistic content, no font redistribution issue).
- All randomness is controlled by documented seeds.

## 4. Target variable definitions

Targets are computed from the scene graph and rendering instructions. They are standardized to zero mean and unit standard deviation within the full generated set for reporting.

### 4.1 Structural clutter index (visual complexity target)

```
clutter = z(n_primitives)
          + 2 * z(overlap_ratio)
          + z(edge_density_factor)
          + z(spacing_irregularity)
          - z(whitespace_fraction)
structural_clutter_index = standard_normal_cdf(clutter / 4)
```

### 4.2 Geometric order index (layout order target)

```
order = -z(alignment_jitter)
        - z(grid_deviation)
        - z(spacing_variance)
        - z(overlap_ratio)
geometric_order_index = standard_normal_cdf(order / 4)
```

### 4.3 Palette coherence index (colour harmony target)

```
coherence = -z(hue_dispersion)
            - z(chroma_dispersion)
            - z(palette_incoherence)
palette_coherence_index = standard_normal_cdf(coherence / 3)
```

### 4.4 Visual salience index (visual intensity target)

```
salience = z(contrast_ratio)
           + z(luminance_contrast)
           + z(edge_contrast)
           + z(focal_contrast)
visual_salience_index = standard_normal_cdf(salience / 4)
```

### 4.5 Focal hierarchy index (layout hierarchy target)

```
hierarchy = z(focal_size_ratio)
            + z(focal_contrast)
            - z(secondary_count)
            - z(whitespace_fraction)
focal_hierarchy_index = standard_normal_cdf(hierarchy / 4)
```

`z(x)` denotes the standardized value of parameter `x` over the full generated set. `standard_normal_cdf` maps to [0,1]; final target indices are multiplied by 100 for reporting parity with proxy scores.

## 5. Random seeds and splits

### 5.1 Seeds

- G-dev: generator family A + seeds 0–9 (documentation/debugging only)
- G-validation: generator family B + seeds 10–14 (predefined diagnostic checks)
- G-test: generator family C + seeds 15–24 (final reported benchmark)

### 5.2 Sample size

Base set (seeds are split by family):

```text
5 constructs × 5 severity levels × (10 + 5 + 10) seeds × 25 images
= 15,625 images
```

Interaction set (held out as interaction-test):

```text
4 interaction types × (10 + 5 + 10) seeds × 8 images
= 800 images
```

Total: **16,425 images**.

This yields at least 1,250 base images per construct manipulation in each family, satisfying the practical minimum of 1,000 per construct.

## 6. Primary and secondary metrics

For each proxy–target pair:

- Spearman correlation (ρ) and 95% bootstrap confidence interval;
- Kendall’s τ;
- monotonic trend across severity levels (mean proxy score per level);
- calibration plot: proxy score against objective target quantiles;
- error distribution and failure-case identification;
- correlation with non-target objective variables;
- results separately by held-out generator family.

FDR correction: Benjamini–Hochberg across all planned proxy–target tests (25 tests).

## 7. Success criteria

| Criterion | Strong support | Partial support | Failure / limitation |
|---|---|---|---|
| Target association on G-test | Spearman ρ ≥ 0.60; 95% CI excludes 0; BH q < .05 | ρ 0.30–0.59 or heterogeneous by family | ρ < 0.30, reversed, or q ≥ .05 |
| Monotonicity across levels | correct direction in all families | correct in a majority but not all families | reversed or inconsistent |
| Selectivity | target association exceeds every specified non-target association by ≥0.15 | smaller separation | no separation / stronger non-target association |
| Family generalization | criterion holds in all held-out families | holds in some families | fails broadly |

## 8. Multiple-testing correction

All planned proxy–target correlations (5 proxies × 5 targets = 25) are corrected with Benjamini–Hochberg FDR. Secondary interaction and family-stratified analyses are reported as exploratory unless pre-specified above.

## 9. Required outputs

- `synthetic_benchmark/manifests/generator_hashes.csv`
- `synthetic_benchmark/manifests/image_manifest.csv`
- `synthetic_benchmark/metadata/data_dictionary.md`
- `synthetic_benchmark/metadata/target_metadata.csv`
- `results/no_human/runs/<new-run-id>/synthetic_benchmark/tables/objective_benchmark_results.csv`
- `results/no_human/runs/<new-run-id>/synthetic_benchmark/tables/non_target_association_results.csv`
- `results/no_human/runs/<new-run-id>/synthetic_benchmark/tables/failure_cases.csv`
- `results/no_human/runs/<new-run-id>/synthetic_benchmark/figures/calibration.pdf`
- `results/no_human/runs/<new-run-id>/synthetic_benchmark/figures/monotonicity.pdf`
- `results/no_human/runs/<new-run-id>/synthetic_benchmark/figures/interaction.pdf`
- `results/no_human/runs/<new-run-id>/synthetic_benchmark/logs/run.log`
