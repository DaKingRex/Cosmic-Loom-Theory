"""
RegimeSystem — coherence regime-transition primitive for Cosmic Loom Theory.

Implements CLT Phase 3.1 (Coherence Regime Transitions) with a low-dimensional
stochastic dynamical system: an order parameter x evolving in a control-tunable
cusp-catastrophe potential. This is the standard normal form for tipping points
(Scheffer et al.) and yields all four 3.1 phenomena from one model:

    V(x) = x⁴/4 − a·x²/2 + b·x
    dx/dt = −∂V/∂x + noise = (−x³ + a·x − b) + σ·ξ(t)

- **a** (bistability/splitting): a>0 gives two macrostates — a "healthy/integrated"
  well (positive x) and a "collapsed/fragmented" well (negative x); a≤0 gives a
  single well.
- **b** (drive/stress): the stress / anesthetic / mood knob, oriented so that
  *increasing b shallows the healthy well*. Raising b to the fold b=fold_b(a)
  produces a saddle-node bifurcation (threshold crossing / regime transition); the
  up-vs-down sweep traces a hysteresis loop; and near the fold the relaxation time
  diverges, giving critical slowing down (rising autocorrelation + variance).

CLT reading: the two wells are the boundary-collapse macrostates of §7.7; the
healthy well sits near the critical point (edge of chaos). The order parameter maps
into éR phase space via the shared metrics so trajectories render alongside the
substrate simulators.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

# Allow direct execution (python simulations/emergence/regime_system.py) by
# ensuring the project root is importable for the shared analysis.metrics package.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis.metrics.coherence import (  # noqa: E402
    calculate_er,
    ViableWindow,
    BASELINE_WINDOW,
)

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_A = 1.0        # bistability / splitting (a>0 ⇒ double well)
DEFAULT_B = 0.0        # drive / asymmetry (symmetric at 0)
DEFAULT_SIGMA = 0.10   # noise amplitude
DEFAULT_DT = 0.01      # integration timestep

# éR-mapping calibration (chosen so the healthy well lands mid-viable and the
# collapsed well drifts to a boundary; see map_to_er_space).
_COHERENCE_GAIN = 1.5
_EP_BASE = 1.5
_EP_GAIN = 0.5
_FREQ_BASE = 0.6
_FREQ_GAIN = 1.7


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def fold_b(a: float) -> float:
    """
    Magnitude of the drive b at the saddle-node fold for a given a>0.

    The bistable region is |b| < fold_b(a); beyond it only one well remains.
    """
    if a <= 0:
        return 0.0
    return 2.0 * (a / 3.0) ** 1.5


def equilibria(a: float, b: float) -> List[float]:
    """Real fixed points of −x³ + a·x − b = 0 (i.e. roots of x³ − a·x + b)."""
    roots = np.roots([1.0, 0.0, -a, b])
    return sorted(float(r.real) for r in roots if abs(r.imag) < 1e-9)


def is_stable(x: float, a: float) -> bool:
    """A fixed point is stable where the potential curvature U''(x)=3x²−a > 0."""
    return (3.0 * x ** 2 - a) > 0.0


# =============================================================================
# STATE
# =============================================================================

@dataclass
class RegimeSnapshot:
    """A single recorded step of the regime system."""

    time: float
    x: float
    a: float
    b: float


# =============================================================================
# REGIME SYSTEM
# =============================================================================

class RegimeSystem:
    """
    Stochastic cusp/double-well order-parameter model of coherence regimes.

    Parameters:
        a: Bistability parameter (a>0 ⇒ two wells).
        b: Control/drive parameter (asymmetry).
        sigma: Noise amplitude (σ).
        dt: Integration timestep.
        x0: Initial order parameter (defaults to the positive/healthy well).
        window: ViableWindow used for regime classification of the éR readout.
        seed: Optional RNG seed (else uses the global numpy RNG for test parity).
    """

    def __init__(
        self,
        a: float = DEFAULT_A,
        b: float = DEFAULT_B,
        sigma: float = DEFAULT_SIGMA,
        dt: float = DEFAULT_DT,
        x0: Optional[float] = None,
        window: Optional[ViableWindow] = None,
        seed: Optional[int] = None,
    ):
        self.a = a
        self.b = b
        self.sigma = sigma
        self.dt = dt
        self.window = window if window is not None else BASELINE_WINDOW
        self._rng = np.random.default_rng(seed) if seed is not None else None

        # Start in the positive (healthy) well by default.
        if x0 is None:
            x0 = np.sqrt(a) if a > 0 else 0.0
        self.x0 = float(x0)

        self.time = 0.0
        self.x = self.x0
        self.x_history: List[float] = [self.x]
        self.time_history: List[float] = [0.0]
        self.b_history: List[float] = [self.b]

    # ---- dynamics -----------------------------------------------------------

    def force(self, x: Optional[float] = None) -> float:
        """Deterministic force −∂V/∂x = −x³ + a·x − b."""
        if x is None:
            x = self.x
        return -x ** 3 + self.a * x - self.b

    def _normal(self) -> float:
        if self._rng is not None:
            return float(self._rng.standard_normal())
        return float(np.random.standard_normal())

    def step(self, n_steps: int = 1) -> None:
        """Advance the system by n_steps using Euler–Maruyama integration."""
        for _ in range(n_steps):
            noise = self.sigma * np.sqrt(self.dt) * self._normal()
            self.x = self.x + self.dt * self.force() + noise
            self.time += self.dt
            self.x_history.append(self.x)
            self.time_history.append(self.time)
            self.b_history.append(self.b)

    def run(self, duration: float) -> None:
        """Advance for a duration of simulated time."""
        self.step(int(round(duration / self.dt)))

    # ---- control ------------------------------------------------------------

    def set_control(self, a: Optional[float] = None, b: Optional[float] = None) -> None:
        """Update the bistability (a) and/or drive (b) parameters."""
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b

    def sweep_control(
        self,
        values: np.ndarray,
        param: str = "b",
        steps_per_value: int = 50,
    ) -> Dict[str, np.ndarray]:
        """
        Quasi-statically sweep a control parameter through `values`, integrating
        `steps_per_value` steps at each setting. Returns the control values paired
        with the mean order parameter reached at each — the raw material for
        hysteresis loops and critical-slowing-down analysis.
        """
        if param not in ("a", "b"):
            raise ValueError("param must be 'a' or 'b'.")
        recorded_x = np.zeros(len(values))
        for i, v in enumerate(values):
            self.set_control(**{param: float(v)})
            self.step(steps_per_value)
            recorded_x[i] = self.x
        return {"control": np.asarray(values, dtype=float), "x": recorded_x}

    def reset(self) -> None:
        """Restore the initial state and clear history."""
        self.time = 0.0
        self.x = self.x0
        self.b = self.b_history[0]
        self.x_history = [self.x]
        self.time_history = [0.0]
        self.b_history = [self.b]

    # ---- readouts -----------------------------------------------------------

    def coherence(self) -> float:
        """
        Coherence ∈ [0, 1] from the order parameter: high in the positive
        (healthy) well, low in the negative (collapsed) well.
        """
        return float(_sigmoid(_COHERENCE_GAIN * self.x))

    def relaxation_time(self) -> float:
        """
        Local relaxation timescale 1/|U''(x*)| at the nearest stable equilibrium.
        Diverges near the fold — the mechanistic basis of critical slowing down.
        """
        eqs = [x for x in equilibria(self.a, self.b) if is_stable(x, self.a)]
        if not eqs:
            return float("inf")
        x_star = min(eqs, key=lambda e: abs(e - self.x))
        curvature = 3.0 * x_star ** 2 - self.a
        return float("inf") if curvature <= 1e-9 else 1.0 / curvature

    def map_to_er_space(self) -> Dict[str, float]:
        """
        Map the current state to CLT éR phase space.

        The healthy well maps mid-viable; drift toward the collapsed well raises
        the effective frequency and lowers éR toward the chaos boundary.
        """
        coherence = self.coherence()
        ep = _EP_BASE + _EP_GAIN * coherence
        freq = _FREQ_BASE + _FREQ_GAIN * (1.0 - coherence)
        er = calculate_er(ep, freq)
        return {
            "energy_present": float(ep),
            "frequency": float(freq),
            "energy_resistance": float(er),
            "coherence": float(coherence),
            "order_parameter": float(self.x),
            "regime": self.window.classify(float(er)),
        }

    def er_trajectory(self) -> Dict[str, np.ndarray]:
        """
        Recompute the éR readout across the recorded x-history, returning arrays
        suitable for EnergyResistanceVisualizer.add_trajectory (ep, freq).
        """
        x_arr = np.asarray(self.x_history, dtype=float)
        coherence = _sigmoid(_COHERENCE_GAIN * x_arr)
        ep = _EP_BASE + _EP_GAIN * coherence
        freq = _FREQ_BASE + _FREQ_GAIN * (1.0 - coherence)
        er = calculate_er(ep, freq)
        return {
            "ep": ep,
            "freq": freq,
            "er": np.asarray(er, dtype=float),
            "coherence": coherence,
        }


# =============================================================================
# PRESETS
# =============================================================================

def create_bistable_system(sigma: float = DEFAULT_SIGMA, seed: Optional[int] = None) -> RegimeSystem:
    """A symmetric double-well system poised in the healthy well."""
    return RegimeSystem(a=1.0, b=0.0, sigma=sigma, seed=seed)


def create_monostable_system(sigma: float = DEFAULT_SIGMA, seed: Optional[int] = None) -> RegimeSystem:
    """A single-well (a<0) system — no alternative stable state."""
    return RegimeSystem(a=-1.0, b=0.0, sigma=sigma, x0=0.0, seed=seed)


def create_near_fold_system(sigma: float = DEFAULT_SIGMA, seed: Optional[int] = None) -> RegimeSystem:
    """
    A bistable system driven close to the fold, sitting in the shallow (healthy)
    well that is about to disappear — the critical-slowing-down regime.
    """
    a = 1.0
    b = 0.95 * fold_b(a)
    # Occupied well = the shallow positive equilibrium nearest the fold.
    stable = [x for x in equilibria(a, b) if is_stable(x, a)]
    x0 = max(stable) if stable else np.sqrt(a)
    return RegimeSystem(a=a, b=b, sigma=sigma, x0=x0, seed=seed)


# =============================================================================
# DEMO
# =============================================================================

def demo() -> None:
    """Print a short demonstration of the three 3.1 phenomena."""
    print("RegimeSystem demo — CLT Phase 3.1")
    a = 1.0
    print(f"  fold at |b| = {fold_b(a):.4f}")
    print(f"  equilibria at b=0: {equilibria(a, 0.0)}")

    sys_ = create_bistable_system(seed=0)
    sys_.run(5.0)
    er = sys_.map_to_er_space()
    print(f"  healthy well éR = {er['energy_resistance']:.3f} ({er['regime']})")


if __name__ == "__main__":
    demo()
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python -m simulations.emergence.regime_system")
