# Protocol v5: semantic-health failure analysis

## Scientific role

Protocol v5 treats the five weighted composites as legacy software artefacts
under audit. It does not treat their names as evidence that they measure human
perception, aesthetics, or universal image attributes. The primary question is
whether a reproducible feature pipeline can be numerically stable yet
semantically inactive, and whether that defect changes benchmark conclusions.

## Defect and repair

The pre-fix extractor called the optional `cv2.saliency` contrib namespace.
When the namespace was absent, every exception was suppressed and a constant
0.5 map was returned. Across all 16,425 Pillow images this made
`saliency_mean`, `saliency_std`, `fg_bg_ratio`, `rule_of_thirds`, and both
centroid offsets constant. The OpenCV holdout showed the same failure.

The repaired extractor implements the spectral-residual method of Hou and
Zhang (CVPR 2007, DOI `10.1109/CVPR.2007.383267`) with NumPy FFT and OpenCV
core. A non-uniform image that produces a non-finite or constant map raises.
Missing images, decode failures, incomplete rows, duplicate identifiers, and
non-finite feature values also raise. OpenCV k-means is seeded from image bytes
so worker scheduling does not change colour features.

## Frozen analysis order

1. Read the preserved pre-fix feature matrix and newly extracted post-fix
   matrix.
2. Build a causal repair matrix by replacing only the ten saliency-derived
   columns in the pre-fix matrix. Use this matrix for the primary pre/post
   conclusion-change estimate. Report the fully re-extracted pipeline
   separately because deterministic k-means seeding is also repaired there.
3. Fit all scene-graph target means and standard deviations on G-dev only
   (6,250 images); apply them unchanged to every split.
4. Fit each active feature mean/standard deviation and the composite 0.5th and
   99.5th percentile bounds on G-dev only; apply them unchanged elsewhere.
5. A weighted feature with zero G-dev variance is an error in the post-fix
   analysis. The pre-fix control assigns it zero effective contribution and
   reports lost absolute weight.
6. Evaluate all 25 proxy--target associations on G-test (6,250 images, ten
   seed clusters). Report seed-cluster bootstrap percentile intervals only;
   do not derive zero-null p-values from bootstrap sign frequencies.
7. Select one of three non-rank-duplicate elementary baselines for each target
   using G-dev direction-adjusted Spearman correlation. Lock the selected name
   and direction before the paired G-test comparison.
8. Select ridge regularisation on G-validation, refit on G-dev plus
   G-validation, and report G-test and interaction-test recoverability. Remove
   G-dev-constant features before fitting and report the effective count.

## Required semantic gates

- The six saliency-dependent fields must be finite and nonconstant after repair.
- Every nonzero-weight formula input must vary on G-dev.
- Target and feature constants must identify G-dev as their only fit split.
- The baseline-selection table must identify G-dev, and the paired comparison
  must identify G-test.
- Exact-score tie rates and feature/score clipping rates must be released for
  every split and both pre/post versions.

Run with:

```bash
python synthetic_benchmark/generator/extract_synthetic_features.py
python synthetic_benchmark/run_semantic_health_protocol_v5.py
```
