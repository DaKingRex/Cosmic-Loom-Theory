"""
Interactive-visualizer tests for CLT Phase 3.1.

These follow the "verify like a user" principle: they build each visualizer headless
and DRIVE its controls (sliders, presets, click) exactly as a user would, asserting
the live state responds correctly — not merely that the class imports.
"""

import matplotlib
matplotlib.use("Agg")  # headless: no display needed

import numpy as np  # noqa: E402


def _fake_click(ax, xdata):
    """Minimal event object mimicking a matplotlib button_press_event."""
    return type("Event", (), {"inaxes": ax, "xdata": xdata, "ydata": 0.0})()


class TestRegimeVisualizerInteractive:
    """Driving the RegimeSystem ball-in-potential explorer."""

    def test_drive_past_fold_collapses_ball(self):
        from visualizations.interactive.regime_transitions import RegimeVisualizer
        viz = RegimeVisualizer(seed=1)
        viz._build()
        # Settle in the healthy well.
        for _ in range(150):
            viz._update(0)
        assert viz.sim.map_to_er_space()["regime"] == "viable"
        assert viz.sim.x > 0.5
        # User drags 'drive b' past the fold (~0.385).
        viz._on_b(0.55)
        for _ in range(500):
            viz._update(0)
        assert viz.sim.x < 0, "ball should fall into the collapsed well"
        assert viz.sim.map_to_er_space()["regime"] == "chaos"

    def test_scenario_playback_moves_window(self):
        from visualizations.interactive.regime_transitions import RegimeVisualizer
        viz = RegimeVisualizer(seed=1)
        viz._build()
        w0 = viz.sim.window.er_max - viz.sim.window.er_min
        viz._on_scenario("Depression")
        viz._on_run(None)
        tc = viz.active_tc
        for _ in range(tc.frames):
            viz._update(0)
        assert viz.sim.x < 0                                      # tipped into collapse
        assert (viz.sim.window.er_max - viz.sim.window.er_min) < w0  # window contracted
        assert viz.active_tc is None                             # scenario finished

    def test_click_to_kick(self):
        from visualizations.interactive.regime_transitions import RegimeVisualizer
        viz = RegimeVisualizer(seed=1)
        viz._build()
        viz._on_click(_fake_click(viz.ax_pot, 1.0))
        assert abs(viz.sim.x - 1.0) < 1e-6

    def test_preset_switch(self):
        from visualizations.interactive.regime_transitions import RegimeVisualizer
        from simulations.emergence.regime_system import fold_b
        viz = RegimeVisualizer(seed=1)
        viz._build()
        viz._on_preset("Near-fold")
        assert viz.sim.a == 1.0
        assert viz.sim.b > 0.5 * fold_b(1.0)
        viz._on_preset("Monostable")
        assert viz.sim.a < 0

    def test_pause_toggle(self):
        from visualizations.interactive.regime_transitions import RegimeVisualizer
        viz = RegimeVisualizer(seed=1)
        viz._build()
        assert viz.running
        viz._on_play(None)
        assert not viz.running

    def test_static_figure(self):
        from visualizations.interactive.regime_transitions import RegimeVisualizer
        fig = RegimeVisualizer.create_static_figure(seed=0)
        assert fig is not None
        assert len(fig.axes) >= 4


class TestKuramotoVisualizerInteractive:
    """Driving the Kuramoto phase-circle explorer."""

    def test_raise_coupling_locks_phases(self):
        from visualizations.interactive.kuramoto_sync import KuramotoSyncVisualizer
        viz = KuramotoSyncVisualizer(seed=1)
        viz._build()
        # Low coupling -> incoherent.
        viz._on_k(0.15)
        for _ in range(300):
            viz._update(0)
        assert viz.net.order_parameter() < 0.35
        assert viz.net.map_to_er_space()["regime"] == "chaos"
        # User drags coupling up -> phases lock.
        viz._on_k(6.0)
        for _ in range(500):
            viz._update(0)
        assert viz.net.order_parameter() > 0.9
        assert viz.net.map_to_er_space()["regime"] == "rigidity"

    def test_seizure_scenario_playback(self):
        from visualizations.interactive.kuramoto_sync import KuramotoSyncVisualizer
        viz = KuramotoSyncVisualizer(seed=1)
        viz._build()
        viz._on_scenario("Seizure")
        viz._on_run(None)
        tc = viz.active_tc
        peak = 0.0
        for _ in range(tc.frames):
            viz._update(0)
            peak = max(peak, viz.net.order_parameter())
        assert peak > 0.95                    # reached hypersynchrony
        assert viz.active_tc is None          # scenario finished

    def test_partial_preset_is_viable(self):
        from visualizations.interactive.kuramoto_sync import KuramotoSyncVisualizer
        viz = KuramotoSyncVisualizer(seed=1)
        viz._build()
        viz._on_preset("Partial")
        for _ in range(400):
            viz._update(0)
        r = viz.net.order_parameter()
        assert 0.35 < r < 0.95

    def test_static_figure(self):
        from visualizations.interactive.kuramoto_sync import KuramotoSyncVisualizer
        fig = KuramotoSyncVisualizer.create_static_figure(seed=0)
        assert fig is not None
        assert len(fig.axes) >= 4


class TestVisualizerRegistration:
    """Both visualizers export from the interactive package."""

    def test_exports(self):
        from visualizations.interactive import (
            RegimeVisualizer, KuramotoSyncVisualizer,
        )
        assert RegimeVisualizer is not None
        assert KuramotoSyncVisualizer is not None
