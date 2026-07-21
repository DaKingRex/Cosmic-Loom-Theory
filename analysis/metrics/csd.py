"""
Critical slowing down (CSD) / early-warning-signal metrics for CLT.

As a dynamical system approaches a tipping point (a fold/saddle-node bifurcation),
its recovery from perturbations slows: the dominant relaxation time diverges. This
manifests statistically as **rising lag-1 autocorrelation and rising variance** in
a fluctuating observable. These are the standard early-warning signals from the
critical-transitions literature (Scheffer et al.), applied in CLT to detect
approach to the chaos or rigidity boundary.

Empirical note: CSD is well-supported for anesthesia induction and depression
transitions, but contested for epileptic seizures — so treat these indicators as a
hypothesis-testing tool, not a settled predictor, especially for seizure scenarios.

Implements:
- rolling_autocorr : lag-1 autocorrelation in a sliding window
- rolling_variance : variance in a sliding window
- csd_indicators   : both of the above, bundled
- kendall_tau_trend: monotonic-trend statistic (positive ⇒ rising ⇒ approaching)
"""

from typing import Dict

import numpy as np

_EPS = 1e-12


def _sliding_windows(series: np.ndarray, window: int) -> np.ndarray:
    """Return a 2-D view of overlapping windows of the given length."""
    series = np.asarray(series, dtype=float).ravel()
    if window < 2:
        raise ValueError("window must be at least 2.")
    if series.size < window:
        raise ValueError("series shorter than the requested window.")
    n_windows = series.size - window + 1
    idx = np.arange(window)[None, :] + np.arange(n_windows)[:, None]
    return series[idx]


def rolling_variance(series: np.ndarray, window: int) -> np.ndarray:
    """
    Variance within each sliding window.

    Returns an array of length (len(series) - window + 1).
    """
    windows = _sliding_windows(series, window)
    return np.var(windows, axis=1)


def rolling_autocorr(series: np.ndarray, window: int) -> np.ndarray:
    """
    Lag-1 autocorrelation within each sliding window.

    Returns an array of length (len(series) - window + 1). Windows with no
    variance yield an autocorrelation of 0.0.
    """
    windows = _sliding_windows(series, window)
    out = np.zeros(windows.shape[0])
    for i, w in enumerate(windows):
        w = w - w.mean()
        denom = np.sum(w * w)
        if denom <= _EPS:
            out[i] = 0.0
        else:
            out[i] = np.sum(w[:-1] * w[1:]) / denom
    return out


def csd_indicators(series: np.ndarray, window: int) -> Dict[str, np.ndarray]:
    """
    Bundle the two canonical early-warning indicators.

    Returns:
        Dict with 'ac1' (lag-1 autocorrelation) and 'variance', each an array
        of length (len(series) - window + 1).
    """
    return {
        "ac1": rolling_autocorr(series, window),
        "variance": rolling_variance(series, window),
    }


def kendall_tau_trend(x: np.ndarray) -> float:
    """
    Kendall rank-correlation of a series against time, in [-1, 1].

    A standard early-warning-signal summary: a positive tau means the indicator
    is rising over time (approaching a transition); negative means it is falling.
    Pure-numpy O(n²) implementation (indicator series are short).

    Args:
        x: 1-D series (e.g. rolling AC1 or variance over time).

    Returns:
        Kendall's tau (float). Returns 0.0 for series shorter than 2.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = x.size
    if n < 2:
        return 0.0
    concordant_minus_discordant = 0
    for i in range(n - 1):
        diff = np.sign(x[i + 1:] - x[i])
        concordant_minus_discordant += np.sum(diff)
    return float(concordant_minus_discordant / (0.5 * n * (n - 1)))
