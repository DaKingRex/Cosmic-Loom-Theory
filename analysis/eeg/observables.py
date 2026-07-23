"""
CLT observables computed from real EEG recordings (Phase 4, deliverable B).

Turns a multi-channel EEG epoch into the CLT observables used to test the theory's
published predictions on open data. This module deliberately *reuses* the canonical
metrics in ``analysis.metrics`` (Lempel-Ziv complexity, spectral entropy,
critical-slowing-down indicators, Kuramoto order, the éR formula, the viable window);
it only adapts them to real multi-channel EEG and adds the EEG-based **éR proxy**.

Honest scope (stated openly in the paper as well): the éR proxy uses a *spectral-power
surrogate* for Energy Present (EP) — it is NOT a metabolic measurement. Absolute éR is
therefore proxy-scaled. Regime classification against a fixed simulation-scale window is
deferred to the calibration phase, where a per-recording viable window is derived from a
baseline (awake) state; see the analysis harness. This module returns the raw physical
quantities so that calibration decision is made downstream, not frozen here.

The per-epoch readout mirrors the repo-wide ``map_to_er_space()`` dict convention
(``energy_present``, ``frequency``, ``energy_resistance``, ``coherence``) used by the
substrate simulators, e.g. ``simulations/emergence/kuramoto_network.py``.
"""

import os
import sys
from typing import Dict

import numpy as np
from scipy.signal import hilbert, welch

# Allow direct execution: ensure the project root is importable for analysis.metrics.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis.metrics.complexity import lz_complexity, spectral_entropy  # noqa: E402
from analysis.metrics.coherence import calculate_er  # noqa: E402


def _as_2d(epoch) -> np.ndarray:
    """Coerce an epoch to a (n_channels, n_samples) float array."""
    arr = np.asarray(epoch, dtype=float)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    if arr.ndim != 2:
        raise ValueError("epoch must be 1-D (samples) or 2-D (channels x samples).")
    return arr


def phase_coherence(epoch) -> float:
    """
    Spatial phase coherence across channels — a proxy for the CLT metric Q.

    Takes the analytic-signal (Hilbert) phase of each channel and returns the mean,
    over time, of the instantaneous Kuramoto order parameter across channels
    (R = abs(mean(e^{iθ}))). 1.0 = channels phase-locked, ~0 = incoherent.
    """
    arr = _as_2d(epoch)
    if arr.shape[0] < 2:
        return 1.0  # a single channel is trivially "coherent" with itself
    phases = np.angle(hilbert(arr, axis=-1))          # (n_channels, n_samples)
    r_t = np.abs(np.mean(np.exp(1j * phases), axis=0))  # order parameter per sample
    return float(np.mean(r_t))


def dominant_frequency(epoch, fs: float) -> float:
    """
    Dominant frequency f (Hz) — the spectral peak of the channel-averaged PSD.

    Uses Welch's method; f enters the éR proxy as éR = EP / f². Falls back to the
    full segment when the signal is short.
    """
    arr = _as_2d(epoch)
    nperseg = int(min(arr.shape[-1], 256))
    freqs, psd = welch(arr, fs=fs, nperseg=nperseg, axis=-1)
    psd_mean = psd.mean(axis=0)
    # Ignore the DC bin so a non-zero mean does not dominate the peak.
    if freqs.size > 1:
        peak = int(np.argmax(psd_mean[1:])) + 1
    else:
        peak = 0
    return float(freqs[peak])


def energy_present(epoch) -> float:
    """
    Energy-present surrogate EP — mean band power (per-channel variance, averaged).

    A spectral-power surrogate for metabolic energy, NOT a metabolic measurement.
    Scales with the square of signal amplitude.
    """
    arr = _as_2d(epoch)
    return float(np.mean(np.var(arr, axis=-1)))


def er_proxy(epoch, fs: float) -> Dict[str, float]:
    """
    éR proxy for one EEG epoch: EP surrogate, dominant frequency, éR = EP/f², coherence.

    Returns the repo-standard (partial) map_to_er_space dict. Regime classification is
    intentionally omitted here — it requires the per-recording calibrated window built by
    the analysis harness (the éR scale is proxy-dependent).
    """
    ep = energy_present(epoch)
    f = dominant_frequency(epoch, fs)
    er = calculate_er(ep, f)
    return {
        "energy_present": ep,
        "frequency": f,
        "energy_resistance": float(er),
        "coherence": phase_coherence(epoch),
    }


def complexity(epoch, fs: float) -> Dict[str, float]:
    """
    Channel-averaged temporal-complexity observables: Lempel-Ziv and spectral entropy.

    These are the paper's primary ``∂L/∂t`` proxies (the anesthesia/psychedelic
    LZc / entropy literature). Both are normalized to ~[0, 1].
    """
    arr = _as_2d(epoch)
    lz = float(np.mean([lz_complexity(ch) for ch in arr]))
    se = float(np.mean([spectral_entropy(ch, fs) for ch in arr]))
    return {"lz_complexity": lz, "spectral_entropy": se}


def epoch_observables(epoch, fs: float) -> Dict[str, float]:
    """
    All CLT observables for one EEG epoch: éR proxy + complexity in a single dict.

    This is the per-epoch feature vector the analysis harness aggregates into
    state/time trajectories to test CLT's predictions.
    """
    out = er_proxy(epoch, fs)
    out.update(complexity(epoch, fs))
    return out
