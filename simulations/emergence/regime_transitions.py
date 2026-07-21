"""
Regime-transition scenarios for Cosmic Loom Theory — Phase 3.1.

Ties the two 3.1 primitives (RegimeSystem, KuramotoNetwork) together into named
scenarios that reproduce the four coherence-regime-transition phenomena, and
bridges their output into the existing éR phase-space visualizer.

Scenarios:
- threshold_crossing   : ramp the drive past the fold; the order parameter jumps
                         (saddle-node / boundary crossing).
- hysteresis_loop       : up-then-down sweep; induction and release follow
                         different thresholds (neural-inertia analogue).
- critical_slowing_down : fluctuation statistics (lag-1 autocorrelation, variance)
                         measured in the healthy well at increasing drive — rising
                         indicators signal approach to the fold.
- sync_transition       : ramp Kuramoto coupling; desynchronization → partial
                         (chimera-like) sync → hypersynchrony.

Empirical grounding: threshold/hysteresis follow the tipping-point normal form
(Scheffer et al.); hysteresis is anchored in anesthesia "neural inertia" (loss
threshold > recovery threshold); critical slowing down is well-supported for
anesthesia/depression but *contested* for seizures — treat it as a
hypothesis-testing tool, not a settled predictor.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

# np.trapezoid replaces the deprecated np.trapz in newer numpy.
_trapezoid = getattr(np, "trapezoid", np.trapz)

# Allow direct execution (python simulations/emergence/regime_transitions.py) by
# ensuring the project root is importable for the shared analysis.metrics package.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis.metrics.coherence import calculate_er  # noqa: E402
from analysis.metrics.csd import kendall_tau_trend  # noqa: E402
from simulations.emergence.regime_system import (  # noqa: E402
    RegimeSystem, fold_b, equilibria, is_stable,
)
from simulations.emergence.kuramoto_network import KuramotoNetwork  # noqa: E402

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_SWEEP_POINTS = 80        # control-parameter samples per sweep
DEFAULT_STEPS_PER_POINT = 80     # integration steps held at each control value
DEFAULT_CSD_SEGMENTS = 8         # number of increasing-drive segments for CSD
DEFAULT_CSD_SETTLE = 2000        # transient steps discarded per CSD segment
DEFAULT_CSD_SAMPLE = 6000        # sampled steps per CSD segment


# =============================================================================
# RESULT
# =============================================================================

@dataclass
class ScenarioResult:
    """
    Standard scenario output. `ep` and `freq` are éR-phase-space coordinate arrays
    ready for EnergyResistanceVisualizer.add_trajectory; `data` holds the
    scenario-specific series and summary statistics.
    """

    name: str
    ep: np.ndarray
    freq: np.ndarray
    er: np.ndarray
    data: Dict = field(default_factory=dict)


# =============================================================================
# SCENARIOS
# =============================================================================

def run_threshold_crossing(
    a: float = 1.0,
    sigma: float = 0.03,
    seed: Optional[int] = 0,
    points: int = DEFAULT_SWEEP_POINTS,
    steps_per_point: int = DEFAULT_STEPS_PER_POINT,
) -> ScenarioResult:
    """Ramp the drive b from 0 past the fold; the order parameter jumps wells."""
    fb = fold_b(a)
    b_values = np.linspace(0.0, 1.5 * fb, points)
    system = RegimeSystem(a=a, b=0.0, sigma=sigma, x0=np.sqrt(a), seed=seed)
    sweep = system.sweep_control(b_values, "b", steps_per_point)
    traj = system.er_trajectory()

    # Crossing point: where x first drops below 0 (jump to the collapsed well).
    x = sweep["x"]
    crossed = np.where(x < 0)[0]
    crossing_b = float(b_values[crossed[0]]) if crossed.size else None

    return ScenarioResult(
        name="threshold_crossing",
        ep=traj["ep"], freq=traj["freq"], er=traj["er"],
        data={
            "control_b": b_values, "x": x, "fold_b": fb,
            "crossing_b": crossing_b,
        },
    )


def run_hysteresis_loop(
    a: float = 1.0,
    sigma: float = 0.01,
    seed: Optional[int] = 0,
    points: int = DEFAULT_SWEEP_POINTS,
    steps_per_point: int = DEFAULT_STEPS_PER_POINT,
) -> ScenarioResult:
    """
    Up-then-down sweep of the drive. The healthy well collapses on the up-sweep at
    a higher drive than it recovers on the down-sweep, tracing a hysteresis loop —
    the anesthesia neural-inertia signature (induction threshold > release).
    """
    fb = fold_b(a)
    up = np.linspace(0.0, 1.6 * fb, points)
    down = up[::-1]
    system = RegimeSystem(a=a, b=0.0, sigma=sigma, x0=np.sqrt(a), seed=seed)

    res_up = system.sweep_control(up, "b", steps_per_point)
    res_down = system.sweep_control(down, "b", steps_per_point)
    traj = system.er_trajectory()

    x_up = res_up["x"]
    x_down = res_down["x"][::-1]  # realign to ascending b
    loop_area = float(_trapezoid(np.abs(x_up - x_down), up))

    return ScenarioResult(
        name="hysteresis_loop",
        ep=traj["ep"], freq=traj["freq"], er=traj["er"],
        data={
            "control_b": up, "x_up": x_up, "x_down": x_down,
            "loop_area": loop_area, "fold_b": fb,
        },
    )


def run_critical_slowing_down(
    a: float = 1.0,
    sigma: float = 0.05,
    seed: Optional[int] = 0,
    segments: int = DEFAULT_CSD_SEGMENTS,
    settle: int = DEFAULT_CSD_SETTLE,
    sample: int = DEFAULT_CSD_SAMPLE,
) -> ScenarioResult:
    """
    Measure early-warning indicators as the drive approaches the fold.

    At each of `segments` increasing drive values, the system settles in the
    healthy well and its detrended fluctuations are sampled; lag-1 autocorrelation
    and variance rise as the fold nears. Uses the standard stationary-segment
    method (robust against the well's mean drift).
    """
    fb = fold_b(a)
    b_values = np.linspace(0.0, 0.9 * fb, segments)
    rng = np.random.default_rng(seed)

    ac1 = np.zeros(segments)
    variance = np.zeros(segments)
    for i, b in enumerate(b_values):
        system = RegimeSystem(a=a, b=float(b), sigma=sigma,
                              seed=int(rng.integers(0, 2**31 - 1)))
        stable = [x for x in equilibria(a, float(b)) if is_stable(x, a)]
        system.x = max(stable) if stable else np.sqrt(a)
        system.x_history = [system.x]
        system.step(settle + sample)
        seg = np.asarray(system.x_history[settle:], dtype=float)
        seg = seg - seg.mean()
        denom = np.sum(seg * seg)
        ac1[i] = np.sum(seg[:-1] * seg[1:]) / denom if denom > 1e-12 else 0.0
        variance[i] = np.var(seg)

    # éR trajectory across the drive sweep (mean state per segment).
    coherence = 1.0 / (1.0 + np.exp(-1.5 * np.sqrt(np.maximum(a - b_values, 0.0))))
    ep = 1.5 + 0.5 * coherence
    freq = 0.6 + 1.7 * (1.0 - coherence)
    er = calculate_er(ep, freq)

    return ScenarioResult(
        name="critical_slowing_down",
        ep=ep, freq=freq, er=np.asarray(er, dtype=float),
        data={
            "control_b": b_values, "ac1": ac1, "variance": variance,
            "tau_ac1": kendall_tau_trend(ac1),
            "tau_variance": kendall_tau_trend(variance),
            "fold_b": fb,
        },
    )


def run_sync_transition(
    n: int = 200,
    gamma: float = 0.5,
    seed: Optional[int] = 0,
    points: int = 40,
    settle_time: float = 8.0,
) -> ScenarioResult:
    """
    Ramp Kuramoto coupling from sub- to super-critical; the order parameter climbs
    from desynchronization through partial (chimera-like) sync to hypersynchrony —
    the seizure-arc primitive for Phase 3.2.
    """
    k_values = np.linspace(0.0, 8.0, points)
    net = KuramotoNetwork(n, coupling=0.0, gamma=gamma, seed=seed)
    order = np.zeros(points)
    ep = np.zeros(points)
    freq = np.zeros(points)
    for i, k in enumerate(k_values):
        net.set_coupling(float(k))
        net.run(settle_time)
        er = net.map_to_er_space()
        order[i] = er["order_parameter"]
        ep[i] = er["energy_present"]
        freq[i] = er["frequency"]
    er_arr = calculate_er(ep, freq)

    return ScenarioResult(
        name="sync_transition",
        ep=ep, freq=freq, er=np.asarray(er_arr, dtype=float),
        data={"coupling_K": k_values, "order_parameter": order},
    )


# Registry of 3.1 scenarios.
SCENARIOS: Dict[str, Callable[..., ScenarioResult]] = {
    "threshold_crossing": run_threshold_crossing,
    "hysteresis_loop": run_hysteresis_loop,
    "critical_slowing_down": run_critical_slowing_down,
    "sync_transition": run_sync_transition,
}


# =============================================================================
# éR VISUALIZER BRIDGE
# =============================================================================

def add_scenario_to_er_visualizer(
    visualizer,
    result: ScenarioResult,
    label: Optional[str] = None,
    color: str = "#FFD93D",
) -> None:
    """
    Render a scenario's emergent trajectory in an EnergyResistanceVisualizer,
    via its existing add_trajectory API.
    """
    visualizer.add_trajectory(
        result.ep, result.freq,
        label=label or result.name,
        color=color,
    )


# =============================================================================
# DEMO
# =============================================================================

def demo() -> None:
    """Run all four 3.1 scenarios and print their key statistics."""
    print("Regime-transition scenarios — CLT Phase 3.1")

    tc = run_threshold_crossing()
    print(f"  threshold_crossing : crossing at b={tc.data['crossing_b']} (fold={tc.data['fold_b']:.4f})")

    hl = run_hysteresis_loop()
    print(f"  hysteresis_loop    : loop area={hl.data['loop_area']:.3f}")

    cs = run_critical_slowing_down()
    print(f"  critical_slowing   : tau(AC1)={cs.data['tau_ac1']:.3f}  tau(var)={cs.data['tau_variance']:.3f}")

    st = run_sync_transition()
    print(f"  sync_transition    : R: {st.data['order_parameter'][0]:.3f} -> {st.data['order_parameter'][-1]:.3f}")


if __name__ == "__main__":
    demo()
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python -m simulations.emergence.regime_transitions")
