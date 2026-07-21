# Restricted Image Redistribution Notice

## Status of source images

The computational experiments in this repository use art and design images collected from several public web sources and third-party datasets (e.g., WikiArt samples, AVA, UI screenshots, packaging photographs, posters, and banners). The licensing terms and redistribution policies of these sources vary, and in many cases do not permit redistribution of the original images or of derived high-resolution copies.

Because of these restrictions, **the original and preprocessed image files are not included in the public submission package**.

## What is provided

- `data/processed/metadata.csv`: canonical list of image IDs, categories, and relative paths used in the no-human pipeline (17,337 rows).
- `results/no_human/quality_control/canonical_splits.csv`: frozen split assignments.
- `data/documentation/no_human_data_dictionary.md`: definitions of dataset roles and construction rules.
- Feature-level outputs in `results/no_human/tables/` and per-image traces in `results/no_human/traces/` (numeric values only, no pixel data).
- The full source code and frozen configuration files needed to reproduce the feature extraction and scoring pipeline.

## How to obtain comparable images

Researchers who wish to replicate or extend this work can acquire images from the original public sources. The raw download directories in `data/raw/` correspond to the following source families:

- `wikiart_sample`, `painting-style-classification`, `Baroque_style` → painting category
- `Phone_Photosho_UI`, `ShowUI-desktop`, `idea_bench`, `user_manuals` → UI category
- `packaging`, `packaging_drinks` → packaging category
- `poster` → poster category
- `banner`, `banner_brazilian`, `banner_korean`, `banner_kratos`, `banner_realestate` → banner category

Exact acquisition URLs, dataset identifiers, and any applicable licences should be requested from the authors.

## Synthetic and controlled variants

Benign-transformation variants (JPEG, brightness, contrast, resize, noise) and controlled structural manipulations (added shapes, sharpening, desaturation, text bars, symmetry reduction, colour noise, whitespace reduction) were generated programmatically from the canonical images. These variants are intermediate research artefacts and are also not redistributed.

## Contact

For questions about image provenance, licensing, or to request the restricted image set under a data-use agreement, please contact the corresponding author.
