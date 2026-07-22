"""
Healing scenarios for CLT Phase 3.3 — return to and widening of the viable window.

Healing is the mirror of pathology (see pathology.py): each scenario is a scripted
`TimeCourse` (see scenario.py) that moves one of the two engines *back toward — or
beyond* — the viable window, while the window itself *widens* (re-coupling and
restored capacity, CLT §7.7) rather than contracting. Set:

- meditation   (kuramoto) — self-induced gamma coherence that stays flexible (viable).
- psychedelics (regime)   — raised entropy/diversity + softened boundaries.
- sleep_wake   (regime)   — slow, reversible cycling across the regime axis.
- therapy      (regime)   — injury → intervention → recovery to a deeper attractor.

Pathology contracts the window; healing widens it. That opposition is the visible
spine of Phase 3. See docs/theory/phase3_empirical_grounding.md for the signatures.
"""

import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis.metrics.coherence import BASELINE_WINDOW  # noqa: E402
from simulations.emergence.regime_system import fold_b  # noqa: E402
from simulations.emergence.kuramoto_network import critical_coupling, DEFAULT_GAMMA  # noqa: E402
from simulations.emergence.scenario import TimeCourse, smoothstep  # noqa: E402

_FB = fold_b(1.0)                       # fold of the a=1 cusp
_KC = critical_coupling(DEFAULT_GAMMA)  # Kuramoto critical coupling
_K_PARTIAL = 1.3 * _KC                  # partial-sync (viable) baseline
_K_MED = 2.0 * _KC                      # meditation: stronger, still-flexible sync


# =============================================================================
# MEDITATION (kuramoto) — coherence enhancement that stays inside the window
# =============================================================================

def meditation() -> TimeCourse:
    """Self-induced gamma coherence building over time, staying flexible (viable)."""
    def control(p):
        # Coupling rises from the partial-sync baseline toward stronger — but still
        # flexible — synchrony, short of the rigid hypersynchronous lock. The window
        # widens with it, so the heightened coherence remains inside the viable band.
        return {"coupling": _K_PARTIAL + (_K_MED - _K_PARTIAL) * smoothstep(p),
                "noise": 0.05}

    def window(p):
        return BASELINE_WINDOW.scaled(1.0 + 0.5 * smoothstep(p))

    return TimeCourse(
        name="meditation", target="kuramoto",
        description="Self-induced gamma coherence that stays flexible.",
        signature=("Long-term meditators build high-amplitude gamma synchrony and "
                   "frontoparietal coherence over time — coherence enhancement that "
                   "stays inside the viable window, not rigid hypersynchrony."),
        frames=240, control=control, window=window, steps_per_frame=3)


# =============================================================================
# PSYCHEDELICS (regime) — raised entropy/diversity + softened boundaries
# =============================================================================

def psychedelics() -> TimeCourse:
    """Rising signal diversity and softened boundaries — toward the chaos edge."""
    def control(p):
        # Noise (signal diversity/entropy) climbs while the healthy landscape is
        # retained: the state jitters and explores rather than collapsing.
        return {"a": 1.0, "b": 0.0, "sigma": 0.10 + 0.28 * smoothstep(p)}

    def window(p):
        # Softened boundaries: the window widens symmetrically (both bounds relax).
        return BASELINE_WINDOW.scaled(1.0 + 0.6 * smoothstep(p))

    return TimeCourse(
        name="psychedelics", target="regime",
        description="Raised entropy/diversity with softened boundaries.",
        signature=("The entropic brain: increased signal diversity/entropy and "
                   "dissolved boundaries — an expanded but less-stable coherence "
                   "pushed toward the chaos edge of a widened window."),
        frames=240, control=control, window=window, steps_per_frame=3)


# =============================================================================
# SLEEP / WAKE (regime) — slow, reversible cyclic traversal
# =============================================================================

_SW_AMP = 2.5 * _FB     # drive amplitude (decisively crosses the fold each half-cycle)
_SW_CYCLES = 2.0        # number of wake→sleep→wake cycles over the scenario


def sleep_wake() -> TimeCourse:
    """Slow periodic drive cycling wake → deep sleep → wake — fully reversible."""
    def control(p):
        # A slow sinusoidal drive carries the state across the regime axis and back:
        # the reversibility (vs. depression's one-way tip) is the healthy signature.
        b = _SW_AMP * np.sin(2.0 * np.pi * _SW_CYCLES * p)
        return {"a": 1.0, "b": float(b), "sigma": 0.06}

    def window(p):
        # Restorative: the window recovers slightly over the night.
        return BASELINE_WINDOW.scaled(1.0 + 0.2 * smoothstep(p))

    return TimeCourse(
        name="sleep_wake", target="regime",
        description="Slow, reversible cycling across the regime axis.",
        signature=("Complexity is graded by state (wake high → slow-wave sleep low → "
                   "back), a slow reversible traversal — unlike a pathological tip, "
                   "the healthy cycle always returns."),
        frames=300, control=control, window=window, steps_per_frame=6)


# =============================================================================
# THERAPY (regime) — injury → intervention → recovery to a deeper attractor
# =============================================================================

def _therapy_drive(p):
    """Injury drives past the fold; a sustained intervention pulls the state out."""
    up = 2.0 * _FB
    if p < 0.12:                                          # injury (drive past fold)
        return up * smoothstep(p / 0.12)
    if p < 0.25:                                          # injured state holds
        return up
    if p < 0.38:                                          # intervention down through 0
        return up * (1.0 - 2.0 * smoothstep((p - 0.25) / 0.13))
    if p < 0.68:                                          # sustained pull-out (climb + settle)
        return -up
    if p < 0.82:                                          # release the intervention
        return -up * (1.0 - smoothstep((p - 0.68) / 0.14))
    return 0.0                                            # settled in the restored well


def _therapy_bistability(p):
    """The recovered attractor is deeper/more resilient than the pre-injury baseline."""
    # Deepen only after the pull-out completes, so it never re-traps the climbing state.
    return 1.0 + 0.8 * smoothstep(float(np.clip((p - 0.68) / 0.32, 0.0, 1.0)))


def _therapy_window_factor(p):
    """Contract during injury, then widen past baseline as healing settles."""
    if p < 0.25:
        return 1.0 - 0.4 * smoothstep(p / 0.25)          # injury contracts the window
    return 0.6 + 0.7 * smoothstep((p - 0.25) / 0.75)     # healing widens it past baseline


def therapy() -> TimeCourse:
    """Injury, intervention, and recovery to a modified, more resilient attractor."""
    def control(p):
        return {"a": _therapy_bistability(p), "b": _therapy_drive(p), "sigma": 0.05}

    def window(p):
        return BASELINE_WINDOW.scaled(_therapy_window_factor(p))

    return TimeCourse(
        name="therapy", target="regime",
        description="Injury, intervention, and recovery to a deeper attractor.",
        signature=("Re-coupling and restoration: the state is driven into collapse, "
                   "an intervention pulls it back out, and it settles into a deeper, "
                   "more resilient well than the pre-injury baseline (a widened "
                   "window, not the old one)."),
        frames=280, control=control, window=window, steps_per_frame=5)


# Registry of healing scenarios.
HEALING = {
    "meditation": meditation,
    "psychedelics": psychedelics,
    "sleep_wake": sleep_wake,
    "therapy": therapy,
}
