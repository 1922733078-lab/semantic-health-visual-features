# Protocol v6: corrected second-renderer mechanism audit

Protocol v6 rerenders 1,250 images from 125 pairwise-orthogonal five-level
factor combinations under ten seeds using the separate OpenCV drawing program.
It is an internally controlled second implementation, not an external
natural-image validation set.

The repaired spectral-residual extractor is forced across the OpenCV images.
Protocol-v5 G-dev feature constants and score bounds are applied unchanged.
Pillow targets use G-dev-frozen target constants; OpenCV factor targets are the
direct orthogonal levels specified before rendering.

The release includes:

- semantic-health ranges for all six saliency-dependent fields;
- final-score boundary, tie, and non-degeneration checks for every composite;
- raw weighted-sum associations alongside clipped 0--100 score associations,
  so cross-renderer boundary collapse cannot be mistaken for absent raw signal;
- nominal and non-target correlations for every operational descriptor and
  saliency-dependent descriptor, with ten-seed cluster intervals;
- the complete composite cross-target table;
- target-factor balance and explicit scope language;
- ridge-probe point estimates and cluster intervals in both renderers, with
  G-dev-constant features removed and the effective count reported.

No restricted permutation output acts as a hypothesis-test null or decision
gate. The analysis concerns controlled image response and cross-implementation
failure behaviour only.

Run after protocol v5 with:

```bash
python synthetic_benchmark/run_external_holdout_protocol_v6.py --force-regenerate
```
