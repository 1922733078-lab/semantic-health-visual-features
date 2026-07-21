"""Reusable semantic-health gates for deterministic visual-feature benchmarks.

Hash equality and finite-value checks establish byte-level reproducibility, but
they cannot show that a weighted feature is responsive or that two candidate
baselines are not the same ranking under opposite signs.  These checks operate
on the development split before any held-out evaluation is interpreted.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def weighted_feature_health(
    frame: pd.DataFrame,
    metric_specs: dict,
    fit_mask,
    variance_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Audit whether every nonzero-weight input is finite and responsive."""
    mask = np.asarray(fit_mask, dtype=bool)
    if len(mask) != len(frame) or not mask.any():
        raise ValueError("fit_mask must select at least one row and match the frame")
    rows = []
    for metric, (features, weights) in metric_specs.items():
        if len(features) != len(weights):
            raise ValueError(f"feature/weight length mismatch for {metric}")
        for feature, weight in zip(features, weights):
            if feature not in frame:
                raise KeyError(f"missing weighted feature: {metric}/{feature}")
            values = frame.loc[mask, feature].to_numpy(float)
            finite = bool(np.isfinite(values).all())
            sample_std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            responsive = bool(finite and sample_std > variance_tolerance)
            rows.append({
                "metric": metric, "feature": feature, "weight": weight,
                "fit_rows": len(values), "finite": finite,
                "n_unique": int(pd.Series(values).nunique(dropna=False)),
                "sample_std": sample_std, "responsive": responsive,
                "active_gate_passed": bool(weight == 0 or responsive),
            })
    return pd.DataFrame(rows)


def assert_weighted_features_healthy(
    frame: pd.DataFrame,
    metric_specs: dict,
    fit_mask,
    variance_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Return the audit table or raise with every failed active input."""
    audit = weighted_feature_health(
        frame, metric_specs, fit_mask, variance_tolerance=variance_tolerance
    )
    failed = audit.loc[~audit["active_gate_passed"]]
    if not failed.empty:
        detail = ", ".join(
            f"{row.metric}/{row.feature}" for row in failed.itertuples()
        )
        raise RuntimeError(f"degenerate or non-finite weighted features: {detail}")
    return audit


def rank_equivalent_pairs(
    candidate_scores: pd.DataFrame,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Find candidate columns with identical or reversed rank order."""
    rows = []
    for left, right in combinations(candidate_scores.columns, 2):
        value = spearmanr(candidate_scores[left], candidate_scores[right]).statistic
        if np.isfinite(value) and abs(abs(value) - 1.0) <= tolerance:
            rows.append({
                "left": left, "right": right, "spearman_rho": float(value),
                "relationship": "same_ranking" if value > 0 else "reversed_ranking",
            })
    return pd.DataFrame(
        rows, columns=["left", "right", "spearman_rho", "relationship"]
    )


def score_distribution_health(values, lower: float = 0.0, upper: float = 100.0):
    """Return exact-tie and boundary diagnostics for one score vector."""
    series = pd.Series(np.asarray(values, dtype=float))
    finite = bool(np.isfinite(series).all())
    counts = series.value_counts(normalize=True, dropna=False)
    return {
        "n": len(series), "finite": finite,
        "n_unique": int(series.nunique(dropna=False)),
        "sample_std": float(series.std(ddof=1)),
        "largest_exact_tie_fraction": float(counts.iloc[0]),
        "lower_boundary_fraction": float((series <= lower).mean()),
        "upper_boundary_fraction": float((series >= upper).mean()),
    }

