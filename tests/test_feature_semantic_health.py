"""Semantic-health regression tests for deterministic image features."""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.features.traditional_features import (
    _compute_saliency,
    extract_composition_features,
    extract_saliency_features,
    process_single_image,
)
from src.no_human.baselines import best_baseline_for_target
from src.no_human.semantic_health import (
    assert_weighted_features_healthy,
    rank_equivalent_pairs,
    score_distribution_health,
)


ROOT = Path(__file__).resolve().parents[1]
SALIENCY_DEPENDENT = [
    "saliency_mean",
    "saliency_std",
    "fg_bg_ratio",
    "rule_of_thirds",
    "center_offset_x",
    "center_offset_y",
]


def controlled_image(x: int, y: int) -> np.ndarray:
    image = np.full((256, 256, 3), 232, dtype=np.uint8)
    cv2.circle(image, (x, y), 24, (20, 20, 20), -1, cv2.LINE_AA)
    cv2.rectangle(image, (18, 188), (226, 204), (175, 175, 175), -1)
    return image


def test_spectral_saliency_is_finite_bounded_and_non_degenerate():
    saliency = _compute_saliency(controlled_image(64, 64))
    assert saliency.shape == (256, 256)
    assert np.isfinite(saliency).all()
    assert 0 <= saliency.min() <= saliency.max() <= 1
    assert saliency.std() > 1e-3


def test_controlled_focal_positions_change_saliency_composition_features():
    left = controlled_image(64, 64)
    right = controlled_image(192, 64)
    left_features = {
        **extract_composition_features(left),
        **extract_saliency_features(left),
    }
    right_features = {
        **extract_composition_features(right),
        **extract_saliency_features(right),
    }
    assert left_features["center_offset_x"] < right_features["center_offset_x"]
    changed = [
        name for name in SALIENCY_DEPENDENT
        if not np.isclose(left_features[name], right_features[name], atol=1e-7)
    ]
    assert len(changed) >= 2


def test_uniform_image_has_explicit_zero_saliency_not_legacy_half_map():
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    saliency = _compute_saliency(image)
    assert np.array_equal(saliency, np.zeros((64, 64), dtype=np.float32))


def test_process_single_image_fails_on_missing_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        process_single_image((str(tmp_path / "missing.png"), "missing", "test"))


def test_best_baseline_selects_finite_maximum_absolute_correlation():
    frame = pd.DataFrame({
        "weak": [0, 1, 3, 2, 4],
        "strong": [4, 3, 2, 1, 0],
    })
    target = np.arange(5)
    name, correlation = best_baseline_for_target(
        frame, target, ["weak", "strong"]
    )
    assert name == "strong"
    assert correlation == pytest.approx(-1.0)


def test_weighted_feature_gate_rejects_constant_active_input():
    frame = pd.DataFrame({"active": [0.5, 0.5, 0.5], "healthy": [0.0, 1.0, 2.0]})
    specs = {"metric": (["active", "healthy"], [0.2, 0.8])}
    with pytest.raises(RuntimeError, match="metric/active"):
        assert_weighted_features_healthy(frame, specs, np.ones(3, dtype=bool))


def test_rank_equivalence_gate_detects_sign_reversed_duplicate():
    frame = pd.DataFrame({
        "symmetry": [0.1, 0.4, 0.2, 0.8],
        "one_minus_symmetry": [0.9, 0.6, 0.8, 0.2],
        "independent": [0.2, 0.9, 0.1, 0.4],
    })
    duplicates = rank_equivalent_pairs(frame)
    assert len(duplicates) == 1
    assert duplicates.iloc[0]["relationship"] == "reversed_ranking"


def test_score_distribution_health_reports_clipping_and_ties():
    audit = score_distribution_health([0, 0, 20, 100])
    assert audit["n_unique"] == 3
    assert audit["largest_exact_tie_fraction"] == pytest.approx(0.5)
    assert audit["lower_boundary_fraction"] == pytest.approx(0.5)
    assert audit["upper_boundary_fraction"] == pytest.approx(0.25)


def test_released_feature_matrix_has_non_degenerate_saliency_fields():
    frame = pd.read_csv(
        ROOT / "synthetic_benchmark" / "metadata" / "synthetic_traditional_features.csv"
    )
    for column in SALIENCY_DEPENDENT:
        assert frame[column].nunique(dropna=False) > 1, column
        assert np.isfinite(frame[column]).all(), column
