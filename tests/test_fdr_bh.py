#!/usr/bin/env python3
"""
Unit tests for Benjamini-Hochberg FDR correction used in the no-human pipeline.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "no_human"))

from fdr_bh import fdr_control


def test_fixture_four_pvalues():
    pvals = np.array([0.001, 0.010, 0.050, 0.500])
    expected = np.array([0.004, 0.020, 0.06666666666666667, 0.500])
    q = fdr_control(pvals)
    np.testing.assert_allclose(q, expected, rtol=1e-12)


def test_against_statsmodels():
    """If statsmodels is available, verify our BH matches multipletests."""
    pytest.importorskip("statsmodels")
    from statsmodels.stats.multitest import multipletests

    rng = np.random.default_rng(42)
    pvals = rng.uniform(0.0001, 0.9, size=20)
    q = fdr_control(pvals)
    sm_q = multipletests(pvals, method="fdr_bh")[1]
    np.testing.assert_allclose(q, sm_q, rtol=1e-12)


def test_q_values_vary():
    pvals = np.array([0.001, 0.010, 0.050, 0.500])
    q = fdr_control(pvals)
    assert len(np.unique(q)) == len(q), "q-values collapsed to a constant"


def test_q_in_unit_interval():
    pvals = np.array([0.0, 0.001, 0.5, 1.0])
    q = fdr_control(pvals)
    assert q.min() >= 0.0
    assert q.max() <= 1.0


def test_order_restored():
    pvals = np.array([0.5, 0.001, 0.05, 0.01])
    q = fdr_control(pvals)
    # The ranking of q-values should match the ranking of p-values
    assert np.argsort(pvals).tolist() == np.argsort(q).tolist()


def test_nonsignificant_not_reported_as_lt_001():
    pvals = np.array([0.001, 0.5])
    q = fdr_control(pvals)
    assert q[1] >= 0.05, "A non-significant raw p-value was reported as highly significant after FDR"


def test_empty_input():
    q = fdr_control([])
    assert len(q) == 0


def test_nan_rejected():
    with pytest.raises(ValueError):
        fdr_control([0.01, np.nan, 0.05])


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
