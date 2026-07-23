"""
Analysis harness for the CLT Phase 4 open-data validation (deliverable B).

Aggregates the per-epoch CLT observables (``analysis.eeg.observables``) computed from a
loaded recording (``analysis.eeg.loader``) into per-recording summaries, calibrates a
*per-recording* viable window from the subject's own awake baseline, and assembles the
state regime-trajectory that tests CLT's anesthesia predictions (P1: anesthesia =
rigidity boundary; P3: induction traces toward éR_max).

Why a per-recording calibrated window: the éR proxy uses a spectral-power surrogate for
EP, so its absolute scale is arbitrary and the simulation-scale BASELINE_WINDOW
(0.5–5.0) is meaningless for EEG. Calibrating the window from the awake-baseline éR
distribution judges each subsequent state *against that subject's own conscious
baseline* — a departure toward rigidity is then a within-subject, scale-free statement.

Preprocessing note: anti-aliased downsampling is applied here (not in the pure-I/O
loader). The complexity observable (LZc) still needs a surrogate/bandpass preprocessing
recipe to reproduce the literature's collapse — that is a tracked calibration sub-task;
the éR/regime trajectory below does not depend on it.
"""

import os
import sys
from typing import Dict, Optional

import numpy as np

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis.eeg.loader import load_recording, epoch_data  # noqa: E402
from analysis.eeg.observables import epoch_observables  # noqa: E402
from analysis.metrics.coherence import ViableWindow  # noqa: E402


def analyze_recording(
    path: str,
    epoch_seconds: float = 10.0,
    resample: Optional[float] = None,
    picks: Optional[str] = "eeg",
    max_epochs: Optional[int] = None,
) -> Dict:
    """
    Load one recording and compute per-epoch CLT observables + a mean summary.

    Args:
        path: Recording path (see loader.load_recording).
        epoch_seconds: Epoch length.
        resample: Target sampling rate (Hz) for anti-aliased downsampling before
            analysis; None keeps the native rate. (Speeds LZc and de-aliases.)
        picks: Channel selection passed to the loader.
        max_epochs: Cap on the number of epochs analyzed (bounds compute).

    Returns:
        dict with: summary (mean per observable), per_epoch (arrays per observable),
        fs (analysis rate), n_epochs, info (loader metadata).
    """
    data, info = load_recording(path, picks=picks)
    fs = info["fs"]
    if resample is not None and resample < fs:
        from scipy.signal import decimate
        factor = int(round(fs / resample))
        if factor > 1:
            data = decimate(data, factor, axis=-1, ftype="fir")
            fs = fs / factor
    epochs = epoch_data(data, fs, epoch_seconds)
    if max_epochs is not None:
        epochs = epochs[:max_epochs]
    rows = [epoch_observables(ep, fs) for ep in epochs]
    keys = list(rows[0].keys()) if rows else []
    per_epoch = {k: np.array([r[k] for r in rows]) for k in keys}
    summary = {k: float(np.mean(per_epoch[k])) for k in keys}
    return {
        "summary": summary,
        "per_epoch": per_epoch,
        "fs": fs,
        "n_epochs": len(epochs),
        "info": info,
    }


def calibrate_window_from_baseline(
    baseline_er,
    lo_q: float = 5.0,
    hi_q: float = 95.0,
) -> ViableWindow:
    """
    Build a per-recording ViableWindow from the awake-baseline éR distribution.

    The awake baseline is 'viable' by construction: the window spans its lo_q–hi_q
    percentiles, so a later state whose éR exceeds hi_q reads as 'rigidity' and one
    below lo_q as 'chaos'. Falls back to a band around the median if the percentiles
    are degenerate (e.g. non-positive).
    """
    er = np.asarray(baseline_er, dtype=float)
    er_min = float(np.percentile(er, lo_q))
    er_max = float(np.percentile(er, hi_q))
    if er_min <= 0 or er_min >= er_max:
        med = max(float(np.median(er)), 1e-30)
        er_min, er_max = 0.5 * med, 1.5 * med
    return ViableWindow(er_min, er_max)


def state_regime_trajectory(
    analyses_by_state: Dict[str, Dict],
    baseline_key: str = "baseline",
) -> Dict:
    """
    Calibrate the window from the baseline state and classify every state's mean éR.

    Args:
        analyses_by_state: {state_label: analyze_recording(...) result}.
        baseline_key: Which state is the awake baseline used to calibrate the window.

    Returns:
        dict with 'window' (the calibrated ViableWindow) and 'states' — for each state,
        its mean éR, dominant frequency, coherence, LZc, and the classified regime.
    """
    if baseline_key not in analyses_by_state:
        raise KeyError(f"baseline state '{baseline_key}' not in analyses.")
    window = calibrate_window_from_baseline(
        analyses_by_state[baseline_key]["per_epoch"]["energy_resistance"])
    states = {}
    for state, a in analyses_by_state.items():
        s = a["summary"]
        er = s["energy_resistance"]
        # Per-epoch regime occupancy: the rigidity signature under propofol is
        # intermittent (alternating alpha and slow-wave epochs), so the *fraction*
        # of epochs in each regime discriminates depth better than the mean alone.
        er_epochs = a["per_epoch"]["energy_resistance"]
        regimes = np.array([window.classify(float(e)) for e in er_epochs])
        occupancy = {r: float(np.mean(regimes == r))
                     for r in ("chaos", "viable", "rigidity")}
        states[state] = {
            "energy_resistance": er,
            "frequency": s["frequency"],
            "coherence": s["coherence"],
            "lz_complexity": s["lz_complexity"],
            "regime": window.classify(er),
            "occupancy": occupancy,
        }
    return {"window": window, "states": states}
