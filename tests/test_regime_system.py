"""
Tests for RegimeSystem — the cusp/double-well regime-transition primitive
(CLT Phase 3.1).
"""

import numpy as np
import pytest


class TestRegimeSystemBasics:
    """Construction, state, stepping, reset."""

    def test_import_and_init(self):
        from simulations.emergence.regime_system import RegimeSystem
        s = RegimeSystem(a=1.0, b=0.0, seed=0)
        assert s.a == 1.0
        # Default start is the positive (healthy) well at x = sqrt(a).
        assert np.isclose(s.x, 1.0)

    def test_step_advances_time(self):
        from simulations.emergence.regime_system import RegimeSystem
        s = RegimeSystem(seed=0)
        t0 = s.time
        s.step(10)
        assert s.time > t0
        assert len(s.x_history) == 11

    def test_run_duration(self):
        from simulations.emergence.regime_system import RegimeSystem
        s = RegimeSystem(dt=0.01, seed=0)
        s.run(1.0)
        assert len(s.x_history) == 101

    def test_reset(self):
        from simulations.emergence.regime_system import RegimeSystem
        s = RegimeSystem(seed=0)
        s.run(2.0)
        s.reset()
        assert s.time == 0.0
        assert len(s.x_history) == 1
        assert np.isclose(s.x, s.x0)


class TestPotential:
    """Fold, equilibria, and stability of the cusp potential."""

    def test_fold_positive_for_bistable(self):
        from simulations.emergence.regime_system import fold_b
        assert fold_b(1.0) > 0.0
        assert fold_b(-1.0) == 0.0

    def test_bistable_has_three_equilibria(self):
        from simulations.emergence.regime_system import equilibria
        eqs = equilibria(1.0, 0.0)
        assert len(eqs) == 3
        assert np.allclose(sorted(eqs), [-1.0, 0.0, 1.0], atol=1e-6)

    def test_monostable_has_one_equilibrium(self):
        from simulations.emergence.regime_system import equilibria
        assert len(equilibria(-1.0, 0.0)) == 1

    def test_stability(self):
        from simulations.emergence.regime_system import is_stable
        # Wells at ±1 are stable; the ridge at 0 is not (for a=1).
        assert is_stable(1.0, 1.0)
        assert is_stable(-1.0, 1.0)
        assert not is_stable(0.0, 1.0)

    def test_beyond_fold_single_well(self):
        from simulations.emergence.regime_system import equilibria, fold_b, is_stable
        b = 1.5 * fold_b(1.0)
        stable = [x for x in equilibria(1.0, b) if is_stable(x, 1.0)]
        assert len(stable) == 1


class TestHysteresis:
    """Drive sweeps and path-dependence."""

    def test_up_sweep_collapses_healthy_well(self):
        from simulations.emergence.regime_system import RegimeSystem, fold_b
        fb = fold_b(1.0)
        s = RegimeSystem(a=1.0, b=0.0, sigma=0.01, x0=1.0, seed=1)
        s.sweep_control(np.linspace(0, 1.6 * fb, 60), "b", 80)
        # After driving past the fold the system sits in the collapsed well.
        assert s.x < 0

    def test_sweep_invalid_param(self):
        from simulations.emergence.regime_system import RegimeSystem
        s = RegimeSystem(seed=0)
        with pytest.raises(ValueError):
            s.sweep_control(np.array([0.0]), "c", 10)


class TestCriticalSlowingDown:
    """Relaxation time diverges near the fold."""

    def test_relaxation_longer_near_fold(self):
        from simulations.emergence.regime_system import (
            create_near_fold_system, create_bistable_system,
        )
        near = create_near_fold_system(seed=2).relaxation_time()
        healthy = create_bistable_system(seed=2).relaxation_time()
        assert near > healthy


class TestERMapping:
    """éR phase-space readout."""

    def test_keys(self):
        from simulations.emergence.regime_system import RegimeSystem
        er = RegimeSystem(seed=0).map_to_er_space()
        for key in ("energy_present", "frequency", "energy_resistance",
                    "coherence", "order_parameter", "regime"):
            assert key in er

    def test_er_formula(self):
        from simulations.emergence.regime_system import RegimeSystem
        er = RegimeSystem(seed=0).map_to_er_space()
        assert np.isclose(er["energy_resistance"],
                          er["energy_present"] / er["frequency"] ** 2, rtol=1e-3)

    def test_healthy_well_is_viable(self):
        from simulations.emergence.regime_system import create_bistable_system
        s = create_bistable_system(seed=3)
        s.run(6.0)
        assert s.map_to_er_space()["regime"] == "viable"

    def test_collapsed_well_is_chaos(self):
        from simulations.emergence.regime_system import RegimeSystem
        s = RegimeSystem(a=1.0, b=0.0, sigma=0.05, x0=-1.0, seed=3)
        s.run(6.0)
        assert s.map_to_er_space()["regime"] == "chaos"

    def test_er_trajectory_shapes(self):
        from simulations.emergence.regime_system import RegimeSystem
        s = RegimeSystem(seed=0)
        s.run(1.0)
        traj = s.er_trajectory()
        assert len(traj["ep"]) == len(traj["freq"]) == len(s.x_history)
        assert np.all(traj["freq"] > 0)


class TestPresets:
    """Preset factory functions."""

    def test_bistable_preset(self):
        from simulations.emergence.regime_system import create_bistable_system
        assert create_bistable_system(seed=0).a > 0

    def test_monostable_preset(self):
        from simulations.emergence.regime_system import create_monostable_system, equilibria
        s = create_monostable_system(seed=0)
        assert len(equilibria(s.a, s.b)) == 1

    def test_near_fold_preset(self):
        from simulations.emergence.regime_system import create_near_fold_system, fold_b
        s = create_near_fold_system(seed=0)
        assert s.b > 0.5 * fold_b(s.a)
