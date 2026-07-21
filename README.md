# Semantic Health Checks for Reproducible Visual-Feature Pipelines

**Paper:** *Semantic Health Checks for Reproducible Visual-Feature Pipelines: A Two-Renderer Failure Case Study*  
**Target Journal:** Journal of Visual Communication and Image Representation (JVCIR), Elsevier  
**Authors:** Ming Zhu (Henan University of Engineering), Zihang Xue (Qilu University of Technology)

## Overview

This repository contains the code, data, and paper source for a two-renderer failure case study demonstrating that byte-level reproducibility does not guarantee semantic activity in visual-feature pipelines. A silent optional-dependency fallback (OpenCV `cv2.saliency`) produced constant saliency features across 16,425 images while all conventional integrity checks passed.

### Key Contributions

1. **Four layered semantic-health gates** for deterministic visual-feature pipelines:
   - Weighted-feature responsiveness
   - Retained formula weight
   - Rank-equivalent baseline rejection
   - Exported-score boundary-mass audit

2. **Spectral-residual saliency repair** (Hou & Zhang 2007) with fail-fast semantics

3. **Leakage-controlled evaluation protocol** (G-dev-only scaling, development-selected baselines)

4. **Two-renderer failure analysis** (16,425 Pillow + 1,250 OpenCV images)

## Repository Structure

```
├── src/                    # Source code (feature extraction, models, analysis)
├── paper/                  # LaTeX source, PDF, figures, references
│   ├── main_no_human.tex   # Main manuscript
│   ├── sections_no_human/  # Section files
│   └── references.bib      # Bibliography (41 entries)
├── tests/                  # Test suite (56 tests)
├── tools/                  # Verification and release tools
├── configs/                # Experiment configurations
├── synthetic_benchmark/    # 16,425 Pillow images + metadata + targets
├── results/no_human/runs/  # Protocol v5/v6 experiment results
├── requirements-lock.txt   # Pinned dependencies (protocol v5 runtime)
└── run_no_human_pipeline.py # Main experiment entry point
```

## Reproduction

### Environment

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-lock.txt
```

Python 3.11, OpenCV 5.0 (headless), NumPy 2.4, SciPy 1.17, pandas 3.0.

### Run experiments

```bash
python run_no_human_pipeline.py
```

### Run tests

```bash
python -m pytest tests/ -q
```

## Key Results

| Composite | Target ρ (G-test) | Baseline ρ | Selective? |
|-----------|-------------------|------------|------------|
| Complexity | 0.623 | 0.673 | No |
| Order | 0.275 | 0.229 | No |
| Harmony | −0.098 | 0.193 | No |
| Intensity | 0.187 | 0.324 | No |
| Hierarchy | 0.260 | 0.750 | No |

None of the five legacy composites is selective for its nominal rendering-parameter target.

## Citation

```bibtex
@article{zhu2026semantic,
  title={Semantic Health Checks for Reproducible Visual-Feature Pipelines: A Two-Renderer Failure Case Study},
  author={Zhu, Ming and Xue, Zihang},
  journal={Journal of Visual Communication and Image Representation},
  year={2026},
  publisher={Elsevier}
}
```

## License

Code: MIT License. Synthetic images: CC BY 4.0.
