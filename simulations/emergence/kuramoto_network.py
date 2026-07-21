"""
KuramotoNetwork — coupled-oscillator synchronization primitive for CLT.

A clean, standalone Kuramoto model of N phase oscillators. Serves the
synchronization-based Phase 3 phenomena (seizure desync→hypersync in 3.2) and
provides a sync-based view of the coherence regimes:

    dθ_i/dt = ω_i + (K/N) Σ_j sin(θ_j − θ_i) + noise

- Low coupling K / high noise  → incoherent phases        → **chaos**
- Moderate K (just above K_c)  → partial synchronization  → **viable window**
- Very high K                  → near-complete phase lock → **rigidity**

The order parameter R = abs(mean(e^{iθ})) is the shared CLT phase-coherence metric.
Ramping K reproduces the seizure-like arc: desynchronization → partial (chimera-
like) sync → hypersynchrony. Kept dependency-free so the microtubule simulator can
later be refactored onto it (Phase 1/2 retrofit).
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np

# Allow direct execution (python simulations/emergence/kuramoto_network.py) by
# ensuring the project root is importable for the shared analysis.metrics package.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from analysis.metrics.coherence import (  # noqa: E402
    calculate_er,
    kuramoto_order,
    ViableWindow,
    BASELINE_WINDOW,
)

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_N = 200          # number of oscillators
DEFAULT_K = 1.5          # coupling strength
DEFAULT_GAMMA = 0.5      # natural-frequency spread (std of normal distribution)
DEFAULT_NOISE = 0.05     # phase-noise amplitude (D)
DEFAULT_DT = 0.01        # integration timestep

# éR-mapping calibration: R→0 maps to chaos, R~0.5 to viable, R→1 to rigidity.
_EP_BASE = 1.0
_EP_GAIN = 4.0
_FREQ_BASE = 2.5
_FREQ_GAIN = 1.8


def critical_coupling(gamma: float) -> float:
    """
    Approximate mean-field critical coupling K_c for a normal frequency spread.

    For a Gaussian g(ω) of std γ, K_c = 2 / (π g(0)) = 2·√(2π)·γ / √(2π) ... the
    standard result is K_c ≈ √(8/π)·γ ≈ 1.596·γ.
    """
    return float(np.sqrt(8.0 / np.pi) * gamma)


# =============================================================================
# KURAMOTO NETWORK
# =============================================================================

class KuramotoNetwork:
    """
    Mean-field (or nearest-neighbor lattice) Kuramoto oscillator network.

    Parameters:
        n_oscillators: Number of oscillators N.
        coupling: Coupling strength K.
        gamma: Std of the natural-frequency distribution (normal, mean 0).
        noise: Phase-noise amplitude D.
        dt: Integration timestep.
        coupling_mode: 'mean_field' (all-to-all) or 'lattice' (nearest neighbor).
        window: ViableWindow for regime classification of the éR readout.
        seed: Optional RNG seed for reproducible frequencies/noise.
    """

    def __init__(
        self,
        n_oscillators: int = DEFAULT_N,
        coupling: float = DEFAULT_K,
        gamma: float = DEFAULT_GAMMA,
        noise: float = DEFAULT_NOISE,
        dt: float = DEFAULT_DT,
        coupling_mode: str = "mean_field",
        window: Optional[ViableWindow] = None,
        seed: Optional[int] = None,
    ):
        if coupling_mode not in ("mean_field", "lattice"):
            raise ValueError("coupling_mode must be 'mean_field' or 'lattice'.")
        self.n = n_oscillators
        self.coupling = coupling
        self.gamma = gamma
        self.noise = noise
        self.dt = dt
        self.coupling_mode = coupling_mode
        self.window = window if window is not None else BASELINE_WINDOW
        self._rng = np.random.default_rng(seed)

        # Natural frequencies (fixed) and initial phases.
        self.omega = self._rng.normal(0.0, gamma, self.n)
        self.phases = self._rng.uniform(0.0, 2 * np.pi, self.n)

        self.time = 0.0
        self.coherence_history: List[float] = [self.order_parameter()]
        self.time_history: List[float] = [0.0]

    # ---- dynamics -----------------------------------------------------------

    def _coupling_term(self) -> np.ndarray:
        """Coupling contribution to each oscillator's phase velocity."""
        if self.coupling_mode == "mean_field":
            # Mean-field form: (K/N) Σ_j sin(θ_j − θ_i) = K·R·sin(ψ − θ_i).
            mean_field = np.mean(np.exp(1j * self.phases))
            r = np.abs(mean_field)
            psi = np.angle(mean_field)
            return self.coupling * r * np.sin(psi - self.phases)
        # Nearest-neighbor ring lattice.
        coupling = np.sin(np.roll(self.phases, 1) - self.phases)
        coupling += np.sin(np.roll(self.phases, -1) - self.phases)
        return 0.5 * self.coupling * coupling

    def step(self, n_steps: int = 1) -> None:
        """Advance the network by n_steps (Euler with additive phase noise)."""
        for _ in range(n_steps):
            dtheta = self.omega + self._coupling_term()
            noise = self.noise * np.sqrt(self.dt) * self._rng.standard_normal(self.n)
            self.phases = self.phases + self.dt * dtheta + noise
            self.phases = np.mod(self.phases, 2 * np.pi)
            self.time += self.dt
            self.coherence_history.append(self.order_parameter())
            self.time_history.append(self.time)

    def run(self, duration: float) -> None:
        """Advance for a duration of simulated time."""
        self.step(int(round(duration / self.dt)))

    # ---- control ------------------------------------------------------------

    def set_coupling(self, coupling: float) -> None:
        """Set the coupling strength K (used for time-varying drive)."""
        self.coupling = coupling

    def synchronize(self, level: float = 1.0) -> None:
        """
        Force phases toward a common value (level=1 ⇒ fully locked, 0 ⇒ unchanged).
        Convenience for initializing hypersynchronous states.
        """
        level = float(np.clip(level, 0.0, 1.0))
        target = self.phases[0]
        self.phases = np.mod((1 - level) * self.phases + level * target, 2 * np.pi)

    def reset(self, seed: Optional[int] = None) -> None:
        """Re-randomize phases and clear history (frequencies preserved)."""
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        self.phases = rng.uniform(0.0, 2 * np.pi, self.n)
        self.time = 0.0
        self.coherence_history = [self.order_parameter()]
        self.time_history = [0.0]

    # ---- readouts -----------------------------------------------------------

    def order_parameter(self) -> float:
        """Kuramoto order parameter R = abs(mean(e^{iθ})) ∈ [0, 1]."""
        return kuramoto_order(self.phases)

    def map_to_er_space(self) -> Dict[str, float]:
        """
        Map the current synchronization state to CLT éR phase space.

        Incoherent (R→0) maps to chaos; partial sync (R~0.5) to the viable window;
        near-complete lock (R→1) to rigidity.
        """
        r = self.order_parameter()
        ep = _EP_BASE + _EP_GAIN * r
        freq = _FREQ_BASE - _FREQ_GAIN * r
        er = calculate_er(ep, freq)
        return {
            "energy_present": float(ep),
            "frequency": float(freq),
            "energy_resistance": float(er),
            "coherence": float(r),
            "order_parameter": float(r),
            "regime": self.window.classify(float(er)),
        }


# =============================================================================
# PRESETS
# =============================================================================

def create_incoherent_network(n: int = DEFAULT_N, seed: Optional[int] = None) -> KuramotoNetwork:
    """Sub-critical coupling — desynchronized (chaos-side) network."""
    return KuramotoNetwork(n, coupling=0.2, gamma=DEFAULT_GAMMA, seed=seed)


def create_partial_sync_network(n: int = DEFAULT_N, seed: Optional[int] = None) -> KuramotoNetwork:
    """Coupling just above K_c — partial synchronization (viable-window analogue)."""
    k = 1.3 * critical_coupling(DEFAULT_GAMMA)
    return KuramotoNetwork(n, coupling=k, gamma=DEFAULT_GAMMA, seed=seed)


def create_hypersync_network(n: int = DEFAULT_N, seed: Optional[int] = None) -> KuramotoNetwork:
    """Strong coupling — near-complete phase lock (rigidity/seizure analogue)."""
    return KuramotoNetwork(n, coupling=8.0, gamma=DEFAULT_GAMMA, seed=seed)


# =============================================================================
# DEMO
# =============================================================================

def demo() -> None:
    """Print order parameter across coupling regimes."""
    print("KuramotoNetwork demo — CLT Phase 3.1")
    print(f"  K_c ≈ {critical_coupling(DEFAULT_GAMMA):.3f}")
    for label, net in [
        ("incoherent", create_incoherent_network(seed=0)),
        ("partial", create_partial_sync_network(seed=0)),
        ("hypersync", create_hypersync_network(seed=0)),
    ]:
        net.run(20.0)
        er = net.map_to_er_space()
        print(f"  {label:11s} R={er['order_parameter']:.3f}  éR={er['energy_resistance']:.3f}  ({er['regime']})")


if __name__ == "__main__":
    demo()
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Usage: python -m simulations.emergence.kuramoto_network")
