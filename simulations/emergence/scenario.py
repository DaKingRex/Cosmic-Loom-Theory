"""
Scenario time-course driver for CLT Phase 3.2 / 3.3.

Several pathology and healing phenomena are not static parameter sets but *scripted
time-courses* — anesthesia induction→emergence (hysteresis), seizure desync→hypersync,
sleep/wake cycling, therapy injury→recovery. This module provides a `TimeCourse`: a
schedule of engine control parameters AND a viable-window schedule over normalized
progress p ∈ [0, 1], which the engines and interactive visualizers both "play".

The moving viable window is the visible spine of Phase 3: pathology contracts it,
healing widens it. A TimeCourse's `window(p)` returns the ViableWindow at each moment,
and the visualizer redraws the éR-panel bands as it plays.
"""

import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis.metrics.coherence import ViableWindow  # noqa: E402
from simulations.emergence.regime_system import RegimeSystem  # noqa: E402
from simulations.emergence.kuramoto_network import KuramotoNetwork  # noqa: E402


def smoothstep(p: float) -> float:
    """Smooth 0→1 ramp with zero slope at the ends (Hermite)."""
    p = float(np.clip(p, 0.0, 1.0))
    return p * p * (3.0 - 2.0 * p)


# =============================================================================
# TIME COURSE
# =============================================================================

@dataclass
class TimeCourse:
    """
    A scripted scenario over normalized progress p ∈ [0, 1].

    Attributes:
        name: Short identifier (e.g. "anesthesia").
        target: Which engine it drives — "regime" or "kuramoto".
        description: One-line human description.
        signature: The empirical fingerprint this reproduces.
        frames: Number of frames the scenario plays over.
        control: p -> dict of engine control params (keys are engine attributes,
            e.g. {"a", "b", "sigma"} for regime; {"coupling", "noise"} for kuramoto).
        window: p -> ViableWindow at that moment (contract=pathology, widen=healing).
        steps_per_frame: Integration steps advanced per frame.
    """

    name: str
    target: str
    description: str
    signature: str
    frames: int
    control: Callable[[float], Dict[str, float]]
    window: Callable[[float], ViableWindow]
    steps_per_frame: int = 3

    def __post_init__(self):
        if self.target not in ("regime", "kuramoto"):
            raise ValueError("target must be 'regime' or 'kuramoto'.")

    def apply(self, engine, progress: float) -> None:
        """Set the engine's control params and viable window for this progress."""
        for key, val in self.control(progress).items():
            setattr(engine, key, float(val))
        engine.window = self.window(progress)


# =============================================================================
# ENGINE FACTORY + HEADLESS PLAYER
# =============================================================================

def make_engine(tc: TimeCourse, seed: int = 0):
    """Construct the engine a TimeCourse targets, initialized at its p=0 state."""
    if tc.target == "regime":
        engine = RegimeSystem(seed=seed)
        tc.apply(engine, 0.0)
        # Re-seat the regime system in its starting well after params are applied.
        engine.x = float(np.sqrt(engine.a)) if engine.a > 0 else 0.0
        engine.x_history = [engine.x]
    else:
        engine = KuramotoNetwork(seed=seed)
        tc.apply(engine, 0.0)
        # Warm up from random phases to the steady state at the starting coupling.
        engine.step(1200)
        engine.coherence_history = [engine.order_parameter()]
    return engine


def run_time_course(tc: TimeCourse, seed: int = 0) -> Dict[str, np.ndarray]:
    """
    Play a TimeCourse headlessly through its engine, recording the trajectory.

    Returns a dict of per-frame arrays: progress, order (order parameter: x for
    regime, R for kuramoto), energy_resistance, regime (str), er_min, er_max.
    """
    engine = make_engine(tc, seed=seed)
    n = tc.frames
    progress = np.zeros(n)
    order = np.zeros(n)
    er = np.zeros(n)
    er_min = np.zeros(n)
    er_max = np.zeros(n)
    regimes: List[str] = []
    for i in range(n):
        p = i / max(n - 1, 1)
        tc.apply(engine, p)
        engine.step(tc.steps_per_frame)
        mapping = engine.map_to_er_space()
        progress[i] = p
        order[i] = mapping["order_parameter"]
        er[i] = mapping["energy_resistance"]
        er_min[i] = engine.window.er_min
        er_max[i] = engine.window.er_max
        regimes.append(mapping["regime"])
    return {
        "progress": progress,
        "order": order,
        "energy_resistance": er,
        "regime": np.array(regimes),
        "er_min": er_min,
        "er_max": er_max,
    }
