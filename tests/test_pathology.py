"""
Tests for the Phase 3.2 scenario time-course driver and flagship pathologies.

Pathologies are asserted by their empirical *arc* (shape over time), not exact
values, so they stay robust across RNG seeds.
"""

import numpy as np


class TestTimeCourse:
    """The TimeCourse dataclass and its application to an engine."""

    def test_construction_and_target_validation(self):
        import pytest
        from simulations.emergence.scenario import TimeCourse
        from analysis.metrics import BASELINE_WINDOW
        tc = TimeCourse(name="t", target="regime", description="", signature="",
                        frames=10, control=lambda p: {"b": p},
                        window=lambda p: BASELINE_WINDOW)
        assert tc.target == "regime"
        with pytest.raises(ValueError):
            TimeCourse(name="t", target="bogus", description="", signature="",
                       frames=10, control=lambda p: {}, window=lambda p: BASELINE_WINDOW)

    def test_apply_sets_params_and_window(self):
        from simulations.emergence.scenario import TimeCourse
        from simulations.emergence.regime_system import RegimeSystem
        from analysis.metrics import ViableWindow
        tc = TimeCourse(name="t", target="regime", description="", signature="",
                        frames=10, control=lambda p: {"b": 0.3, "a": 1.0, "sigma": 0.05},
                        window=lambda p: ViableWindow(1.0, 3.0))
        eng = RegimeSystem(seed=0)
        tc.apply(eng, 0.5)
        assert eng.b == 0.3 and eng.a == 1.0
        assert eng.window.er_min == 1.0 and eng.window.er_max == 3.0


class TestScenarioDriver:
    """make_engine and run_time_course."""

    def test_make_engine_types(self):
        from simulations.emergence.scenario import make_engine
        from simulations.emergence.pathology import depression, seizure
        from simulations.emergence.regime_system import RegimeSystem
        from simulations.emergence.kuramoto_network import KuramotoNetwork
        assert isinstance(make_engine(depression(), seed=0), RegimeSystem)
        assert isinstance(make_engine(seizure(), seed=0), KuramotoNetwork)

    def test_run_time_course_shapes(self):
        from simulations.emergence.scenario import run_time_course
        from simulations.emergence.pathology import depression
        tc = depression()
        out = run_time_course(tc, seed=0)
        for key in ("progress", "order", "energy_resistance", "regime", "er_min", "er_max"):
            assert key in out
            assert len(out[key]) == tc.frames


class TestDepression:
    """Gradual tip into the collapsed well + window contraction."""

    def test_tips_and_window_contracts(self):
        from simulations.emergence.scenario import run_time_course
        from simulations.emergence.pathology import depression
        for seed in (0, 2, 7):
            r = run_time_course(depression(), seed=seed)
            assert r["order"][0] > 0.5          # starts healthy
            assert r["order"][-1] < 0.0         # tips into collapsed well
            w0 = r["er_max"][0] - r["er_min"][0]
            w1 = r["er_max"][-1] - r["er_min"][-1]
            assert w1 < w0                       # window contracted


class TestAnesthesia:
    """Induction to unconsciousness and hysteretic emergence."""

    def test_induction_and_hysteretic_emergence(self):
        from simulations.emergence.scenario import run_time_course
        from simulations.emergence.pathology import anesthesia
        for seed in (0, 1, 7):
            x = run_time_course(anesthesia(), seed=seed)["order"]
            mid = len(x) // 2
            collapse_i = int(np.argmax(x < 0))
            recover_i = len(x) - 1 - int(np.argmax(x[::-1] > 0.5))
            assert x[0] > 0.5                    # starts conscious
            assert x[mid] < 0                    # unconscious mid-scenario
            assert x[-1] > 0.5                   # emerges
            # hysteresis: collapse on the way in, recovery later on the way out
            assert collapse_i < mid < recover_i


class TestSeizure:
    """Desync → runaway hypersync → recovery arc."""

    def test_hypersync_arc(self):
        from simulations.emergence.scenario import run_time_course
        from simulations.emergence.pathology import seizure
        for seed in (0, 1, 42):
            r = run_time_course(seizure(), seed=seed)
            R = r["order"]
            peak_frac = int(np.argmax(R)) / len(R)
            assert R.max() > 0.95                       # reaches hypersync
            assert R[0] < R.max() - 0.1                 # started below hypersync
            assert R[-1] < R.max() - 0.1                # recovered from hypersync
            assert 0.2 < peak_frac < 0.8                # peak is mid-scenario
            assert "rigidity" in set(r["regime"])       # entered the rigidity regime


class TestNeurodegeneration:
    """Progressive coupling decay → falling coherence + contracting domain."""

    def test_coherence_decays_and_window_contracts(self):
        from simulations.emergence.scenario import run_time_course
        from simulations.emergence.pathology import neurodegeneration
        for seed in (0, 1, 7):
            r = run_time_course(neurodegeneration(), seed=seed)
            R = r["order"]
            assert R[0] > 0.35                   # starts partially synchronized (viable)
            assert R[-1] < 0.35                  # decays toward incoherence
            assert R[0] > R[-1] + 0.2            # monotone-ish decline, no recovery
            w0 = r["er_max"][0] - r["er_min"][0]
            w1 = r["er_max"][-1] - r["er_min"][-1]
            assert w1 < w0                        # coherence domain contracted
            assert r["regime"][-1] == "chaos"     # ends outside the (contracted) window


class TestRegistry:
    """The PATHOLOGIES registry."""

    def test_keys_and_callables(self):
        from simulations.emergence.pathology import PATHOLOGIES
        assert set(PATHOLOGIES.keys()) == {
            "depression", "anesthesia", "seizure", "neurodegeneration"}
        assert all(callable(fn) for fn in PATHOLOGIES.values())

    def test_targets(self):
        from simulations.emergence.pathology import PATHOLOGIES
        assert PATHOLOGIES["depression"]().target == "regime"
        assert PATHOLOGIES["anesthesia"]().target == "regime"
        assert PATHOLOGIES["seizure"]().target == "kuramoto"
        assert PATHOLOGIES["neurodegeneration"]().target == "kuramoto"
