#!/usr/bin/env python3
"""Extract traditional features from synthetic benchmark images."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.features.traditional_features import extract_all_traditional_features

if __name__ == "__main__":
    extract_all_traditional_features(
        str(ROOT / "synthetic_benchmark" / "metadata" / "feature_extraction_metadata.csv"),
        str(ROOT / "synthetic_benchmark" / "metadata" / "synthetic_traditional_features.csv"),
        n_workers=6,
    )
