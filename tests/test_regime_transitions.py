"""
Tests for the Phase 3.1 regime-transition scenarios and the éR-visualizer bridge.
"""

import numpy as np


class TestScenarioResult:
    """The standard scenario output container."""

    def test_arrays_aligned(self):
        from simulations.emergence.regime_transitions import run_threshold_crossing
        r = run_threshold_crossing(points=40, steps_per_point=40)
        assert len(r.ep) == len(r.freq) == len(r.er)
        assert np.all(r.freq > 0)
        assert np.all(np.isfinite(r.ep))


class TestThresholdCrossing:
    """Ramping the drive past the fold jumps the order parameter."""

    def test_crosses_near_fold(self):
        from simulations.emergence.regime_transitions import run_threshold_crossing
        r = run_threshold_crossing(seed=0, points=60, steps_per_point=60)
        assert r.data["crossing_b"] is not None
        # The jump occurs at or beyond the fold (finite sweep speed).
        assert r.data["crossing_b"] > 0.8 * r.data["fold_b"]


class TestHysteresisLoop:
    """Up-vs-down sweep encloses a hysteresis loop."""

    def test_loop_area_positive(self):
        from simulations.emergence.regime_transitions import run_hysteresis_loop
        r = run_hysteresis_loop(seed=0, points=60, steps_per_point=60)
        assert r.data["loop_area"] > 0.5

    def test_up_and_down_branches_differ(self):
        from simulations.emergence.regime_transitions import run_hysteresis_loop
        r = run_hysteresis_loop(seed=0, points=60, steps_per_point=60)
        gap = np.abs(r.data["x_up"] - r.data["x_down"])
        assert gap.max() > 1.0


class TestCriticalSlowingDown:
    """Early-warning indicators rise as the fold is approached."""

    def test_indicators_trend_up(self):
        from simulations.emergence.regime_transitions import run_critical_slowing_down
        r = run_critical_slowing_down(seed=0, segments=8, settle=1500, sample=4000)
        assert r.data["tau_ac1"] > 0.3
        assert r.data["tau_variance"] > 0.3

    def test_arrays_length(self):
        from simulations.emergence.regime_transitions import run_critical_slowing_down
        r = run_critical_slowing_down(seed=0, segments=6, settle=1000, sample=3000)
        assert len(r.data["ac1"]) == 6
        assert len(r.data["variance"]) == 6


class TestSyncTransition:
    """Kuramoto coupling ramp reproduces the desync→hypersync arc."""

    def test_order_rises(self):
        from simulations.emergence.regime_transitions import run_sync_transition
        r = run_sync_transition(seed=0, points=30, settle_time=6.0)
        order = r.data["order_parameter"]
        assert order[0] < 0.3
        assert order[-1] > 0.9


class TestScenarioRegistry:
    """The SCENARIOS registry."""

    def test_keys(self):
        from simulations.emergence.regime_transitions import SCENARIOS
        assert set(SCENARIOS.keys()) == {
            "threshold_crossing", "hysteresis_loop",
            "critical_slowing_down", "sync_transition",
        }

    def test_callables(self):
        from simulations.emergence.regime_transitions import SCENARIOS
        assert all(callable(fn) for fn in SCENARIOS.values())


class TestERBridge:
    """The bridge into EnergyResistanceVisualizer.add_trajectory."""

    def test_add_scenario_calls_add_trajectory(self):
        from simulations.emergence.regime_transitions import (
            run_threshold_crossing, add_scenario_to_er_visualizer,
        )

        class FakeVisualizer:
            def __init__(self):
                self.calls = []

            def add_trajectory(self, ep, freq, label="", color="white"):
                self.calls.append((ep, freq, label, color))

        viz = FakeVisualizer()
        result = run_threshold_crossing(points=30, steps_per_point=30)
        add_scenario_to_er_visualizer(viz, result, label="test", color="#fff")
        assert len(viz.calls) == 1
        ep, freq, label, color = viz.calls[0]
        assert label == "test"
        assert len(ep) == len(freq)


class TestVisualizerImport:
    """The Phase 3.1 visualizer imports."""

    def test_regime_visualizer_imports(self):
        from visualizations.interactive import RegimeVisualizer
        assert RegimeVisualizer is not None


class TestConstants:
    """Module-level constants exist and are sane."""

    def test_scenario_constants(self):
        from simulations.emergence import regime_transitions as rt
        assert rt.DEFAULT_SWEEP_POINTS > 0
        assert rt.DEFAULT_CSD_SEGMENTS > 1
        assert rt.DEFAULT_CSD_SAMPLE > rt.DEFAULT_CSD_SETTLE
