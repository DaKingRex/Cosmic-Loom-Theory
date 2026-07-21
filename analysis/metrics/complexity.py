"""
Signal complexity / diversity metrics for Cosmic Loom Theory.

Complexity is the unifying empirical observable across Phase 3 phenomena: it is
high in conscious/healthy/wake/REM/psychedelic states and low in anesthesia,
deep sleep, and depression. In CLT regime terms, low complexity indicates drift
toward rigidity, excess entropy indicates drift toward chaos, and peak flexible
complexity marks the viable window (near-criticality).

Implements:
- Lempel-Ziv complexity (LZ76), normalized — the standard EEG/MEG diversity index
  used in the anesthesia (PCI/LZc) and psychedelic ("entropic brain") literature.
- Spectral entropy — normalized Shannon entropy of the power spectrum.
"""

from typing import Union

import numpy as np

_EPS = 1e-12


# =============================================================================
# LEMPEL-ZIV COMPLEXITY
# =============================================================================

def _binarize(signal: np.ndarray, method: str) -> np.ndarray:
    """Convert a real-valued signal to a binary sequence."""
    signal = np.asarray(signal, dtype=float).ravel()
    if method == "median":
        threshold = np.median(signal)
    elif method == "mean":
        threshold = np.mean(signal)
    else:
        raise ValueError("binarize method must be 'median' or 'mean'.")
    return (signal > threshold).astype(np.uint8)


def _lz76(sequence: np.ndarray) -> int:
    """
    Raw Lempel-Ziv (1976) complexity: the number of distinct substrings
    encountered when scanning the binary sequence left to right.
    """
    n = sequence.size
    if n == 0:
        return 0
    # Work with a Python string for fast substring membership.
    s = "".join(str(int(b)) for b in sequence)
    i, complexity, prefix_len, match_len = 0, 1, 1, 1
    while prefix_len + match_len <= n:
        if s[i:i + match_len] == s[prefix_len:prefix_len + match_len]:
            match_len += 1
        else:
            if match_len > 1:
                i += 1
                if i == prefix_len:
                    complexity += 1
                    prefix_len += match_len
                    i, match_len = 0, 1
            else:
                complexity += 1
                prefix_len += 1
                i, match_len = 0, 1
    if match_len != 1:
        complexity += 1
    return complexity


def lz_complexity(signal: Union[np.ndarray, list], binarize: str = "median") -> float:
    """
    Normalized Lempel-Ziv complexity of a 1-D signal, in ~[0, 1].

    The raw LZ count is normalized by n / log2(n), its asymptotic value for a
    random binary sequence, so that ~1.0 indicates maximal diversity (random)
    and values near 0 indicate a highly regular/periodic signal.

    Args:
        signal: 1-D real-valued time series.
        binarize: Threshold rule, 'median' (default) or 'mean'.

    Returns:
        Normalized LZ complexity (float).
    """
    binary = _binarize(signal, binarize)
    n = binary.size
    if n < 2:
        return 0.0
    raw = _lz76(binary)
    norm = n / np.log2(n)
    return float(raw / norm)


# =============================================================================
# SPECTRAL ENTROPY
# =============================================================================

def spectral_entropy(signal: Union[np.ndarray, list], fs: float = 1.0) -> float:
    """
    Normalized spectral (Shannon) entropy of a 1-D signal, in [0, 1].

    The power spectral density is treated as a probability distribution over
    frequency; its Shannon entropy is normalized by log(n_freqs). ~1.0 indicates
    a flat, broadband (noise-like) spectrum; low values indicate a narrowband
    (periodic) signal. fs is accepted for API symmetry (result is scale-free).

    Args:
        signal: 1-D real-valued time series.
        fs: Sampling frequency (unused in the normalized result; kept for clarity).

    Returns:
        Normalized spectral entropy (float).
    """
    signal = np.asarray(signal, dtype=float).ravel()
    n = signal.size
    if n < 2:
        return 0.0
    signal = signal - np.mean(signal)
    power = np.abs(np.fft.rfft(signal)) ** 2
    total = np.sum(power)
    if total <= _EPS:
        return 0.0
    p = power / total
    p = p[p > _EPS]
    entropy = -np.sum(p * np.log(p))
    return float(entropy / np.log(len(power)))
