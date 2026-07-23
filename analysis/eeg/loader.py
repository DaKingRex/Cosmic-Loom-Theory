"""
EEG ingestion for the CLT Phase 4 open-data validation (deliverable B).

A thin, format-dispatching reader over ``mne`` that turns an open EEG recording
(BrainVision / EDF / EEGLAB / FIF, as found in OpenNeuro / BIDS / FieldTrip shares)
into plain numpy arrays plus minimal metadata, and splits a recording into
fixed-length epochs. Deliberately pure I/O: computing CLT observables from the epochs
is the job of ``analysis.eeg.observables`` and the analysis harness, so the loader has
no scientific opinions and no dependency on the metrics.

Units note: ``mne`` returns data in volts; every CLT observable used here is either
scale-invariant (phase coherence, dominant frequency, median-binarized LZ, spectral
entropy) or proxy-scaled (the éR surrogate), so no unit rescaling is applied.
"""

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

# Allow direct execution: ensure the project root is importable.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Map file extension -> the mne.io reader that handles it.
_READERS = {
    ".vhdr": "read_raw_brainvision",
    ".edf": "read_raw_edf",
    ".bdf": "read_raw_bdf",
    ".set": "read_raw_eeglab",
    ".fif": "read_raw_fif",
}


def _read_raw_any(path: str):
    """Dispatch to the appropriate mne.io reader by file extension."""
    import mne  # imported lazily so the rest of the repo needn't have mne installed

    ext = os.path.splitext(path)[1].lower()
    if path.endswith(".fif.gz"):
        ext = ".fif"
    reader_name = _READERS.get(ext)
    if reader_name is None:
        raise ValueError(
            f"Unsupported EEG format '{ext}'. Supported: {sorted(_READERS)}.")
    reader = getattr(mne.io, reader_name)
    return reader(path, preload=True, verbose="ERROR")


def load_recording(
    path: str,
    picks: Optional[str] = "eeg",
) -> Tuple[np.ndarray, Dict]:
    """
    Load one EEG recording into a (n_channels, n_samples) array + metadata.

    Args:
        path: Path to the recording (.vhdr / .edf / .bdf / .set / .fif). For a BIDS
            dataset, point at the recording file inside the tree (e.g. the .vhdr).
        picks: mne channel selection to keep (default "eeg"); None keeps all channels.

    Returns:
        (data, info) where data is float (n_channels, n_samples) and info is a dict
        with keys: fs, ch_names, n_channels, n_samples, duration.
    """
    raw = _read_raw_any(path)
    if picks is not None:
        raw.pick(picks)
    data = np.asarray(raw.get_data(), dtype=float)
    fs = float(raw.info["sfreq"])
    return data, {
        "fs": fs,
        "ch_names": list(raw.ch_names),
        "n_channels": int(data.shape[0]),
        "n_samples": int(data.shape[1]),
        "duration": float(data.shape[1] / fs),
    }


def epoch_data(
    data: np.ndarray,
    fs: float,
    epoch_seconds: float = 10.0,
    overlap: float = 0.0,
) -> np.ndarray:
    """
    Split a (n_channels, n_samples) recording into fixed-length epochs.

    Args:
        data: (n_channels, n_samples) array.
        fs: Sampling rate (Hz).
        epoch_seconds: Epoch length in seconds (Chennu shares use 10 s epochs).
        overlap: Fractional overlap between consecutive epochs, in [0, 1).

    Returns:
        (n_epochs, n_channels, epoch_len) array. Empty (0, n_channels, epoch_len) if the
        recording is shorter than one epoch.
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1).")
    win = int(round(epoch_seconds * fs))
    if win <= 0:
        raise ValueError("epoch_seconds * fs must be positive.")
    step = max(1, int(round(win * (1.0 - overlap))))
    n_samples = data.shape[1]
    starts = range(0, n_samples - win + 1, step)
    epochs = [data[:, s:s + win] for s in starts]
    if not epochs:
        return np.empty((0, data.shape[0], win))
    return np.stack(epochs)
