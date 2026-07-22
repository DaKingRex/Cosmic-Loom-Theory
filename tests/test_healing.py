"""
Tests for the Phase 3.3 healing scenarios.

Healing is the mirror of pathology: the state returns toward (or beyond) the viable
window while the window *widens* rather than contracts. As in test_pathology, each
scenario is asserted by its empirical *arc* (shape over time), not exact values, so
it stays robust across RNG seeds.
"""

import numpy as np


class TestMeditation:
    """Gamma-coherence enhancement that stays flexible (inside the window)."""

    def test_coherence_rises_and_stays_viable(self):
        from simulations.emergence.scenario import run_time_course
        from simulations.emergence.healing import meditation
        for seed in (0, 1, 7, 42):
            r = run_time_course(meditation(), seed=seed)
            R = r["order"]
            assert R[-1] > R[0] + 0.03            # coherence builds over time
            assert R[-1] > 0.85                    # reaches strong synchrony
            assert "rigidity" not in set(r["regime"])   # never rigid — stays flexible
            assert r["regime"][-1] == "viable"
            w0 = r["er_max"][0] - r["er_min"][0]
            w1 = r["er_max"][-1] - r["er_min"][-1]
            assert w1 > w0                          # window widened (healing)


class TestPsychedelics:
    """Raised diversity/entropy + softened boundaries, without collapse."""

    def test_noise_and_window_widen_without_collapse(self):
        from simulations.emergence.scenario import run_time_course
        from simulations.emergence.healing import psychedelics
        tc = psychedelics()
        assert tc.control(1.0)["sigma"] > tc.control(0.0)["sigma"]   # entropy climbs
        for seed in (0, 1, 7, 42):
            r = run_time_course(psychedelics(), seed=seed)
            o = r["order"]
            assert o.mean() > 0.3                   # stays predominantly coherent
            assert "rigidity" not in set(r["regime"])
            w0 = r["er_max"][0] - r["er_min"][0]
            w1 = r["er_max"][-1] - r["er_min"][-1]
            assert w1 > w0                          # softened/expanded boundaries


class TestSleepWake:
    """Slow, reversible traversal wake → deep sleep → wake."""

    def test_traverses_and_returns(self):
        from simulations.emergence.scenario import run_time_course
        from simulations.emergence.healing import sleep_wake
        for seed in (0, 1, 7, 42):
            r = run_time_course(sleep_wake(), seed=seed)
            o = r["order"]
            assert o[0] > 0.5                       # starts awake
            assert o.min() < -0.7                   # reaches deep sleep (other well)
            assert o[-1] > 0.5                      # returns to wake (reversible)


class TestTherapy:
    """Injury → intervention → recovery to a deeper, more resilient attractor."""

    def test_recovers_deeper_and_window_widens(self):
        from simulations.emergence.scenario import run_time_course
        from simulations.emergence.healing import therapy
        tc = therapy()
        assert tc.control(1.0)["a"] > 1.5           # deepened (modified) attractor
        for seed in (0, 1, 7, 42):
            r = run_time_course(therapy(), seed=seed)
            o = r["order"]
            assert o[0] > 0.5                       # starts healthy
            assert o.min() < 0.0                    # injury drives it into collapse
            assert o[-1] > o[0]                      # recovers to a deeper well
            assert o[-1] > 1.1                       # deeper than the pre-injury baseline
            w0 = r["er_max"][0] - r["er_min"][0]
            w1 = r["er_max"][-1] - r["er_min"][-1]
            assert w1 > w0                          # window widened past baseline


class TestHealingRegistry:
    """The HEALING registry."""

    def test_keys_and_callables(self):
        from simulations.emergence.healing import HEALING
        assert set(HEALING.keys()) == {
            "meditation", "psychedelics", "sleep_wake", "therapy"}
        assert all(callable(fn) for fn in HEALING.values())

    def test_targets(self):
        from simulations.emergence.healing import HEALING
        assert HEALING["meditation"]().target == "kuramoto"
        assert HEALING["psychedelics"]().target == "regime"
        assert HEALING["sleep_wake"]().target == "regime"
        assert HEALING["therapy"]().target == "regime"

    def test_windows_widen_end_to_end(self):
        """Every healing scenario ends with a window at least as wide as it began."""
        from simulations.emergence.scenario import run_time_course
        from simulations.emergence.healing import HEALING
        for name, fn in HEALING.items():
            r = run_time_course(fn(), seed=0)
            w0 = r["er_max"][0] - r["er_min"][0]
            w1 = r["er_max"][-1] - r["er_min"][-1]
            assert w1 >= w0, f"{name} window should not contract"
