"""
Pathology scenarios for CLT Phase 3.2 — departures from the viable window.

Each pathology is a scripted `TimeCourse` (see scenario.py) that drives one of the
two engines out of the viable window along its empirically-grounded trajectory, while
the viable window *contracts* (boundary collapse, CLT §7.7). Flagship set:

- depression   (regime)   — gradual slide into a collapsed mood state; CSD precedes it.
- anesthesia   (regime)   — induction to unconsciousness with hysteretic emergence.
- seizure      (kuramoto) — onset desynchronization → runaway hypersynchrony → relax.

See docs/theory/phase3_empirical_grounding.md for the signatures these reproduce.
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
_K_HYPER = 8.0                          # hypersynchronous coupling
_K_PARTIAL = 1.3 * _KC                  # partial-sync (viable) baseline


# =============================================================================
# DEPRESSION (regime)
# =============================================================================

def depression() -> TimeCourse:
    """Gradual drive toward the collapsed well; viable window contracts."""
    def control(p):
        # Ramp the drive to its maximum by p=0.75, then hold so the collapsed state settles.
        return {"a": 1.0, "b": 2.0 * _FB * smoothstep(min(p / 0.75, 1.0)), "sigma": 0.06}

    def window(p):
        return BASELINE_WINDOW.scaled(1.0 - 0.5 * smoothstep(min(p / 0.75, 1.0)))

    return TimeCourse(
        name="depression", target="regime",
        description="Gradual slide into a collapsed mood state.",
        signature=("Reduced complexity and a bistable mood; critical slowing down "
                   "(rising autocorrelation/variance) precedes the tip into the "
                   "collapsed well."),
        frames=260, control=control, window=window, steps_per_frame=5)


# =============================================================================
# ANESTHESIA (regime) — the hysteresis / neural-inertia signature
# =============================================================================

def _anesthesia_drive(p):
    """
    Induction (0→0.30) ramps drive up; sustained unconsciousness holds it (0.30→0.55);
    emergence (0.55→1) ramps it down through zero to negative, so recovery requires the
    drive to fall well below the induction threshold — the hysteresis.
    """
    up = 2.0 * _FB
    if p < 0.12:                                          # induction (drive up)
        return up * smoothstep(p / 0.12)
    if p < 0.42:                                          # sustained unconsciousness
        return up
    if p < 0.55:                                          # emergence (drive down past 0)
        return up * (1.0 - 2.0 * smoothstep((p - 0.42) / 0.13))
    if p < 0.85:                                          # hold negative — recovery completes
        return -up
    return -up * (1.0 - smoothstep((p - 0.85) / 0.15))    # relax back to rest


def anesthesia() -> TimeCourse:
    """
    Induction to unconsciousness and hysteretic emergence: the state collapses once
    drive exceeds the fold, but only recovers once drive falls well below it —
    the neural-inertia hysteresis loop.
    """
    def control(p):
        return {"a": 1.0, "b": _anesthesia_drive(p), "sigma": 0.03}

    def window(p):
        return BASELINE_WINDOW.scaled(1.0 - 0.6 * np.sin(np.pi * p))

    return TimeCourse(
        name="anesthesia", target="regime",
        description="Induction to unconsciousness and hysteretic emergence.",
        signature=("Complexity collapses under drive; recovery is hysteretic — "
                   "consciousness returns only once the drive falls well below the "
                   "level at which it was lost (neural inertia)."),
        frames=280, control=control, window=window, steps_per_frame=5)


# =============================================================================
# SEIZURE (kuramoto) — desync → hypersync → relax
# =============================================================================

def _seizure_coupling(p):
    """Partial baseline → onset desync dip → runaway hypersync → relax → post-ictal."""
    if p < 0.10:                                   # partial-sync baseline
        return _K_PARTIAL
    if p < 0.22:                                   # onset desynchronization
        return _K_PARTIAL * (1.0 - 0.9 * smoothstep((p - 0.10) / 0.12))
    if p < 0.44:                                   # runaway hypersynchrony
        low = _K_PARTIAL * 0.10
        return low + (_K_HYPER - low) * smoothstep((p - 0.22) / 0.22)
    if p < 0.58:                                   # sustained seizure
        return _K_HYPER
    if p < 0.76:                                   # relaxation + post-ictal suppression
        return _K_HYPER + (0.4 * _K_PARTIAL - _K_HYPER) * smoothstep((p - 0.58) / 0.18)
    return 0.4 * _K_PARTIAL + 0.6 * _K_PARTIAL * smoothstep((p - 0.76) / 0.24)  # recover


def _seizure_hypersync_level(p):
    """0 at baseline/partial, 1 at full hypersync — used to contract the window."""
    return float(np.clip((_seizure_coupling(p) - _K_PARTIAL) / (_K_HYPER - _K_PARTIAL),
                         0.0, 1.0))


def seizure() -> TimeCourse:
    """Onset desynchronization, runaway hypersynchrony, then relaxation to baseline."""
    def control(p):
        return {"coupling": _seizure_coupling(p), "noise": 0.05}

    def window(p):
        return BASELINE_WINDOW.scaled(1.0 - 0.5 * _seizure_hypersync_level(p))

    return TimeCourse(
        name="seizure", target="kuramoto",
        description="Onset desynchronization, runaway hypersynchrony, then relaxation.",
        signature=("Seizure as loss of the flexible near-critical regime: "
                   "desynchronization → hypersynchrony (rigidity) → recovery. "
                   "(Critical slowing down before seizures is empirically contested.)"),
        frames=260, control=control, window=window, steps_per_frame=3)


# Registry of flagship pathologies.
PATHOLOGIES = {
    "depression": depression,
    "anesthesia": anesthesia,
    "seizure": seizure,
}
