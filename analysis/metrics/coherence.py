"""
Coherence and Energy Resistance metrics for Cosmic Loom Theory.

Canonical implementations of the CLT v1.1 §7 observables that were previously
re-implemented per substrate module. This is the single source of truth for:

- Energy Resistance:   éR = EP / f²   (CLT v1.1 §7, ERP)
- Regime classification against a (dynamic) viable window
- Kuramoto order parameter R = |mean(e^{iθ})|  (spatial/phase coherence)

Key concepts:
- The **viable window** [er_min, er_max] separates the chaos regime (éR too low,
  decoherence) from the rigidity regime (éR too high, frozen dynamics). Healthy
  biological coherence lives between them — the CLT reading of the brain-criticality
  / "edge of chaos" hypothesis.
- The window is **not static**. Pathology contracts/shifts it (boundary collapse,
  CLT §7.7) and healing widens/restores it. Model it as a first-class parameter.
"""

from dataclasses import dataclass
from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]

# =============================================================================
# CONSTANTS
# =============================================================================

# Baseline human viable window (the deliberate, live default used across the repo:
# EnergyResistanceVisualizer.DEFAULT_ER_MIN/MAX and bioelectric map_to_er_space).
BASELINE_ER_MIN = 0.5   # Below this: chaos / decoherence
BASELINE_ER_MAX = 5.0   # Above this: rigidity / frozen dynamics

_EPS = 1e-9


# =============================================================================
# ENERGY RESISTANCE
# =============================================================================

def calculate_er(ep: ArrayLike, freq: ArrayLike) -> ArrayLike:
    """
    Energy Resistance:  éR = EP / f².

    Args:
        ep: Energy present (metabolic/pattern energy), scalar or array.
        freq: Characteristic frequency (Hz), scalar or array.

    Returns:
        éR value(s), matching the broadcast shape of the inputs.
    """
    ep = np.asarray(ep, dtype=float)
    freq = np.asarray(freq, dtype=float)
    er = ep / (freq ** 2 + _EPS)
    # Return a plain float when both inputs are scalar.
    if er.ndim == 0:
        return float(er)
    return er


# =============================================================================
# VIABLE WINDOW
# =============================================================================

@dataclass
class ViableWindow:
    """
    The viable éR window separating chaos (below) from rigidity (above).

    Attributes:
        er_min: Chaos threshold. Below this, coherence fragments.
        er_max: Rigidity threshold. Above this, dynamics freeze.
    """

    er_min: float = BASELINE_ER_MIN
    er_max: float = BASELINE_ER_MAX

    def __post_init__(self):
        if self.er_min <= 0 or self.er_max <= 0:
            raise ValueError("Viable window bounds must be positive.")
        if self.er_min >= self.er_max:
            raise ValueError("er_min must be strictly less than er_max.")

    @property
    def width(self) -> float:
        """Linear width of the window."""
        return self.er_max - self.er_min

    @property
    def center(self) -> float:
        """Arithmetic center of the window."""
        return 0.5 * (self.er_min + self.er_max)

    def contains(self, er: float) -> bool:
        """True if éR falls inside the viable window (inclusive)."""
        return self.er_min <= er <= self.er_max

    def classify(self, er: float) -> str:
        """Classify an éR value as 'chaos', 'viable', or 'rigidity'."""
        if er < self.er_min:
            return "chaos"
        if er > self.er_max:
            return "rigidity"
        return "viable"

    def scaled(self, factor: float) -> "ViableWindow":
        """
        Narrow (factor<1, pathology) or widen (factor>1, healing) the window
        about its geometric center. Boundary collapse vs. re-coupling (CLT §7.7).

        Geometric (log-space) scaling keeps both bounds positive for any factor,
        which suits éR spanning an order of magnitude.
        """
        if factor <= 0:
            raise ValueError("scale factor must be positive.")
        log_center = 0.5 * (np.log(self.er_min) + np.log(self.er_max))
        log_half = 0.5 * (np.log(self.er_max) - np.log(self.er_min)) * factor
        return ViableWindow(float(np.exp(log_center - log_half)),
                            float(np.exp(log_center + log_half)))

    def shifted(self, factor: float) -> "ViableWindow":
        """
        Slide the whole window along the éR axis by a multiplicative factor
        (factor>1 drifts toward rigidity, factor<1 toward chaos).
        """
        if factor <= 0:
            raise ValueError("shift factor must be positive.")
        return ViableWindow(self.er_min * factor, self.er_max * factor)


# The canonical baseline window. Pass a custom ViableWindow to model
# pathology-contracted or healing-widened states.
BASELINE_WINDOW = ViableWindow(BASELINE_ER_MIN, BASELINE_ER_MAX)


def classify_regime(er: float, window: ViableWindow = BASELINE_WINDOW) -> str:
    """Classify éR as 'chaos', 'viable', or 'rigidity' against a viable window."""
    return window.classify(er)


# =============================================================================
# PHASE COHERENCE (KURAMOTO ORDER PARAMETER)
# =============================================================================

def kuramoto_order(phases: np.ndarray) -> float:
    """
    Kuramoto order parameter R = |mean(e^{iθ})| ∈ [0, 1].

    R = 0 for fully incoherent (uniformly distributed) phases; R = 1 for
    perfect phase locking. The canonical CLT phase-synchrony observable.

    Args:
        phases: Array of oscillator phases (radians), any shape.

    Returns:
        Order parameter R in [0, 1].
    """
    phases = np.asarray(phases, dtype=float)
    if phases.size == 0:
        return 0.0
    return float(np.abs(np.mean(np.exp(1j * phases))))
