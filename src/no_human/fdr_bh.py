#!/usr/bin/env python3
"""
Benjamini-Hochberg FDR correction.

Isolated in its own module so tests do not need to import heavy pipeline
dependencies.
"""
import numpy as np


def fdr_control(pvals, method="bh"):
    """Benjamini-Hochberg FDR control.

    Parameters
    ----------
    pvals : array-like
        Raw p-values.
    method : str, optional
        Currently only "bh" is supported.

    Returns
    -------
    np.ndarray
        BH-adjusted q-values in the original input order, clipped to [0, 1].

    Raises
    ------
    ValueError
        If p-values contain NaN or if an unsupported method is requested.
    """
    if method != "bh":
        raise ValueError(f"Unsupported FDR method: {method}")

    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return np.array([])
    if np.isnan(pvals).any():
        raise ValueError("fdr_control does not accept NaN p-values")

    order = np.argsort(pvals)
    ranked = pvals[order]
    # Step-up values p_(i) * m / i
    raw = ranked * n / np.arange(1, n + 1)
    # Reverse cumulative minimum enforces monotonicity in ascending p-value order
    adjusted = np.minimum.accumulate(raw[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    result = np.empty(n)
    result[order] = adjusted
    return result
