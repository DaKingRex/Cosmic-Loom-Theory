"""
RegimeVisualizer — interactive explorer for CLT Phase 3.1 regime transitions.

Its design emerges from what regime transitions need to show: they are a *hybrid*
of reshaping a landscape and watching a state evolve. So the hero panel is a
**state "ball" rolling and bouncing under noise inside a live, reshaping double-well
potential** (the RegimeSystem cusp model). Drive the system past the fold and watch
the ball roll over a vanishing barrier into the collapsed well — boundary collapse
made visible; near the fold the ball goes sluggish — critical slowing down made
visible.

Interaction (matplotlib real-time FuncAnimation, like the substrate visualizers):
- Sliders: drive b, bistability a, noise σ.
- Preset RadioButtons: Bistable / Near-fold / Monostable.
- Buttons: Start/Pause, Reset.
- Click the potential panel to kick the ball to a new position.
Live panels: potential + ball, éR phase space + regime label, order-parameter x(t),
and rolling critical-slowing-down indicators (variance, lag-1 autocorrelation).

`create_static_figure` keeps a publication 4-panel summary for the paper.
"""

import os
import sys
from collections import deque
from typing import Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.animation as animation

# Allow direct execution (python visualizations/interactive/regime_transitions.py)
# by ensuring the project root is importable.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from simulations.emergence.regime_system import (  # noqa: E402
    RegimeSystem, fold_b, DEFAULT_SIGMA,
)
from simulations.emergence.regime_transitions import (  # noqa: E402
    run_hysteresis_loop,
    run_critical_slowing_down,
    run_threshold_crossing,
)
from simulations.emergence.pathology import depression, anesthesia  # noqa: E402

# Palette (dark theme, consistent with the interactive family).
_BG = "#1a1a2e"
_AX = "#16213e"
_C_HEALTHY = "#00C2A8"
_C_COLLAPSE = "#E4572E"
_C_DRIVE = "#4C6EF5"
_C_ACCENT = "#F2B705"
_C_BALL = "#FFFFFF"
_REGIME_COLORS = {"chaos": _C_COLLAPSE, "viable": _C_HEALTHY, "rigidity": _C_ACCENT}

_HISTORY = 400          # scrolling window length for x(t) and CSD
_CSD_WINDOW = 120        # samples used for the live variance / autocorrelation


class RegimeVisualizer:
    """Interactive ball-in-a-reshaping-potential explorer for regime transitions."""

    def __init__(self, a: float = 1.0, b: float = 0.0,
                 sigma: float = DEFAULT_SIGMA, seed: Optional[int] = 0,
                 steps_per_frame: int = 3):
        self.sim = RegimeSystem(a=a, b=b, sigma=sigma, seed=seed)
        self.steps_per_frame = steps_per_frame
        self.running = True

        # Scenario playback state (pathology time-courses that target the regime engine).
        self.scenarios = {"Depression": depression, "Anesthesia": anesthesia}
        self.selected_scenario = "Depression"
        self.active_tc = None
        self.scenario_frame = 0

        # Scrolling buffers.
        self.x_trace = deque([self.sim.x], maxlen=_HISTORY)
        self.var_trace = deque([0.0], maxlen=_HISTORY)
        self.ac1_trace = deque([0.0], maxlen=_HISTORY)

        self.fig = None
        self.anim = None

    # ---- potential ----------------------------------------------------------

    def _potential(self, x):
        """V(x) = x⁴/4 − a·x²/2 + b·x for the current parameters."""
        a, b = self.sim.a, self.sim.b
        return x ** 4 / 4 - a * x ** 2 / 2 + b * x

    # ---- figure construction ------------------------------------------------

    def _build(self):
        self.fig = plt.figure(figsize=(14, 8.5), facecolor=_BG)
        self.fig.suptitle("Coherence Regime Transitions — CLT Phase 3.1 (interactive)",
                          color="white", fontsize=14, fontweight="bold")
        gs = GridSpec(2, 2, figure=self.fig, left=0.30, right=0.97,
                      top=0.91, bottom=0.10, hspace=0.35, wspace=0.28)

        self.ax_pot = self.fig.add_subplot(gs[:, 0])   # hero: potential + ball
        self.ax_er = self.fig.add_subplot(gs[0, 1])    # éR phase space
        self.ax_x = self.fig.add_subplot(gs[1, 1])     # x(t) + CSD (twin)
        for ax in (self.ax_pot, self.ax_er, self.ax_x):
            ax.set_facecolor(_AX)
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#33415c")

        self._init_potential()
        self._init_er()
        self._init_timeseries()
        self._add_controls()

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    def _init_potential(self):
        ax = self.ax_pot
        xs = np.linspace(-1.9, 1.9, 400)
        (self.pot_line,) = ax.plot(xs, self._potential(xs), color=_C_DRIVE, lw=2.5)
        (self.ball,) = ax.plot([self.sim.x], [self._potential(self.sim.x)],
                               "o", color=_C_BALL, markersize=16,
                               markeredgecolor="black", zorder=5)
        self.fold_txt = ax.text(0.03, 0.95, "", transform=ax.transAxes,
                                color="white", fontsize=9, va="top")
        self.status_txt = ax.text(0.98, 0.96, "", transform=ax.transAxes,
                                  ha="right", va="top", color=_C_ACCENT, fontsize=10,
                                  fontweight="bold")
        ax.set_title("Potential landscape + state ball  (click to kick)",
                     color="white", fontsize=11, fontweight="bold")
        ax.set_xlabel("order parameter x", color="white")
        ax.set_ylabel("V(x)", color="white")
        ax.set_xlim(-1.9, 1.9)
        ax.grid(alpha=0.15)

    def _draw_er_window(self):
        """(Re)draw the viable-window bands from the engine's current window."""
        for art in getattr(self, "_er_artists", []):
            art.remove()
        w = self.sim.window
        ax = self.ax_er
        self._er_artists = [
            ax.axhspan(0.0, w.er_min, color=_C_COLLAPSE, alpha=0.12),
            ax.axhspan(w.er_min, w.er_max, color=_C_HEALTHY, alpha=0.12),
            ax.axhspan(w.er_max, 20.0, color=_C_ACCENT, alpha=0.12),
            ax.axhline(w.er_min, color=_C_COLLAPSE, ls="--", lw=1),
            ax.axhline(w.er_max, color=_C_ACCENT, ls="--", lw=1),
        ]
        self._last_window = (w.er_min, w.er_max)

    def _init_er(self):
        ax = self.ax_er
        self._er_artists = []
        self._draw_er_window()   # movable viable-window bands
        (self.er_star,) = ax.plot([0.5], [1.0], "*", color=_C_BALL, markersize=18,
                                  markeredgecolor="black", zorder=5)
        self.regime_txt = ax.text(0.5, 0.92, "", transform=ax.transAxes,
                                  ha="center", color="white", fontsize=11,
                                  fontweight="bold")
        ax.set_title("éR phase position + regime (window moves with scenario)",
                     color="white", fontsize=10, fontweight="bold")
        ax.set_ylabel("éR = EP / f²", color="white")
        ax.set_xticks([])
        ax.set_ylim(0, 12)
        ax.set_xlim(0, 1)

    def _init_timeseries(self):
        ax = self.ax_x
        (self.x_line,) = ax.plot([], [], color=_C_HEALTHY, lw=1.5, label="x(t)")
        ax.axhline(0.0, color="#556", lw=0.8)
        ax.set_title("Order parameter x(t) & early-warning signals",
                     color="white", fontsize=11, fontweight="bold")
        ax.set_xlabel("recent steps", color="white")
        ax.set_ylabel("x", color=_C_HEALTHY)
        ax.set_ylim(-1.8, 1.8)
        ax.set_xlim(0, _HISTORY)
        ax.grid(alpha=0.15)

        self.ax_csd = ax.twinx()
        self.ax_csd.set_facecolor("none")
        self.ax_csd.tick_params(colors=_C_ACCENT, labelsize=8)
        (self.var_line,) = self.ax_csd.plot([], [], color=_C_ACCENT, lw=1.5,
                                            label="variance")
        (self.ac1_line,) = self.ax_csd.plot([], [], color=_C_DRIVE, lw=1.2,
                                            alpha=0.8, label="lag-1 autocorr.")
        self.ax_csd.set_ylabel("variance / AC1 (windowed)", color=_C_ACCENT)
        self.ax_csd.set_xlim(0, _HISTORY)
        self.ax_csd.set_ylim(0, 1.0)

    # ---- controls -----------------------------------------------------------

    def _add_controls(self):
        def mk_ax(rect):
            a = self.fig.add_axes(rect, facecolor="#2d3748")
            return a

        self.s_b = Slider(mk_ax([0.06, 0.80, 0.17, 0.025]), "drive b",
                          0.0, 0.6, valinit=self.sim.b, color=_C_COLLAPSE)
        self.s_a = Slider(mk_ax([0.06, 0.75, 0.17, 0.025]), "bistab. a",
                          -1.0, 2.0, valinit=self.sim.a, color=_C_DRIVE)
        self.s_sig = Slider(mk_ax([0.06, 0.70, 0.17, 0.025]), "noise σ",
                            0.0, 0.4, valinit=self.sim.sigma, color=_C_ACCENT)
        for s in (self.s_b, self.s_a, self.s_sig):
            s.label.set_color("white")
            s.valtext.set_color("white")
        self.s_b.on_changed(self._on_b)
        self.s_a.on_changed(self._on_a)
        self.s_sig.on_changed(self._on_sigma)

        # Physics presets.
        self.fig.text(0.065, 0.635, "PRESETS", color="white", fontsize=8, fontweight="bold")
        self.radio = RadioButtons(
            mk_ax([0.06, 0.49, 0.17, 0.13]),
            ("Bistable", "Near-fold", "Monostable"), active=0)
        for lbl in self.radio.labels:
            lbl.set_color("white")
            lbl.set_fontsize(9)
        self.radio.on_clicked(self._on_preset)

        # Pathology scenarios (played by the Run button).
        self.fig.text(0.065, 0.435, "SCENARIO", color=_C_ACCENT, fontsize=8, fontweight="bold")
        self.scenario_radio = RadioButtons(
            mk_ax([0.06, 0.34, 0.17, 0.09]),
            tuple(self.scenarios.keys()), active=0)
        for lbl in self.scenario_radio.labels:
            lbl.set_color("white")
            lbl.set_fontsize(9)
        self.scenario_radio.on_clicked(self._on_scenario)

        self.btn_run = Button(mk_ax([0.06, 0.27, 0.17, 0.045]), "▶ Run scenario",
                              color="#1f6f54", hovercolor="#2a8f6e")
        self.btn_run.label.set_color("white")
        self.btn_run.on_clicked(self._on_run)

        self.btn_play = Button(mk_ax([0.06, 0.21, 0.08, 0.045]), "Pause",
                               color="#2d3748", hovercolor="#3d4758")
        self.btn_reset = Button(mk_ax([0.15, 0.21, 0.08, 0.045]), "Reset",
                                color="#2d3748", hovercolor="#3d4758")
        self.btn_play.label.set_color("white")
        self.btn_reset.label.set_color("white")
        self.btn_play.on_clicked(self._on_play)
        self.btn_reset.on_clicked(self._on_reset)

    # ---- callbacks ----------------------------------------------------------

    def _refresh_potential_curve(self):
        xs = np.linspace(-1.9, 1.9, 400)
        self.pot_line.set_ydata(self._potential(xs))
        lo, hi = self._potential(xs).min(), self._potential(xs).max()
        self.ax_pot.set_ylim(lo - 0.2, hi + 0.2)
        self.fold_txt.set_text(f"fold at b = {fold_b(self.sim.a):.3f}"
                               if self.sim.a > 0 else "single well (a ≤ 0)")

    def _on_b(self, val):
        self.sim.set_control(b=float(val))
        self._refresh_potential_curve()

    def _on_a(self, val):
        self.sim.set_control(a=float(val))
        self._refresh_potential_curve()

    def _on_sigma(self, val):
        self.sim.sigma = float(val)

    def _on_preset(self, label):
        if label == "Bistable":
            a, b = 1.0, 0.0
        elif label == "Near-fold":
            a, b = 1.0, 0.95 * fold_b(1.0)
        else:  # Monostable
            a, b = -1.0, 0.0
        self.sim.set_control(a=a, b=b)
        self.sim.reset()
        self.s_a.set_val(a)
        self.s_b.set_val(b)
        self._reset_traces()
        self._refresh_potential_curve()

    def _on_play(self, _event):
        self.running = not self.running
        self.btn_play.label.set_text("Start" if not self.running else "Pause")

    def _on_scenario(self, label):
        self.selected_scenario = label

    def _on_run(self, _event):
        """Load the selected pathology time-course and start playing it."""
        tc = self.scenarios[self.selected_scenario]()
        self.active_tc = tc
        self.scenario_frame = 0
        tc.apply(self.sim, 0.0)                       # seat control params + window at start
        self.sim.x = float(np.sqrt(self.sim.a)) if self.sim.a > 0 else 0.0
        self.sim.time = 0.0
        self.sim.x_history = [self.sim.x]
        self.running = True
        self._reset_traces()
        self._refresh_potential_curve()

    def _on_reset(self, _event):
        self.active_tc = None
        self.status_txt.set_text("")
        self.sim.reset()
        self._reset_traces()

    def _on_click(self, event):
        # Click the potential panel to kick the ball to a new x.
        if event.inaxes is self.ax_pot and event.xdata is not None:
            self.sim.x = float(np.clip(event.xdata, -1.9, 1.9))

    def _reset_traces(self):
        self.x_trace.clear()
        self.x_trace.append(self.sim.x)
        self.var_trace.clear()
        self.var_trace.append(0.0)
        self.ac1_trace.clear()
        self.ac1_trace.append(0.0)

    # ---- animation ----------------------------------------------------------

    def _windowed_csd(self):
        buf = np.asarray(self.x_trace, dtype=float)
        if buf.size < _CSD_WINDOW:
            return 0.0, 0.0
        seg = buf[-_CSD_WINDOW:] - buf[-_CSD_WINDOW:].mean()
        var = float(np.var(seg))
        denom = np.sum(seg * seg)
        ac1 = float(np.sum(seg[:-1] * seg[1:]) / denom) if denom > 1e-12 else 0.0
        return var, ac1

    def _update(self, _frame):
        if self.active_tc is not None:
            # Play the scripted scenario: apply its control + window, then step.
            tc = self.active_tc
            p = self.scenario_frame / max(tc.frames - 1, 1)
            tc.apply(self.sim, p)
            self._refresh_potential_curve()
            self.status_txt.set_text(f"▶ {tc.name}  ({int(p * 100)}%)")
            self.sim.step(tc.steps_per_frame)
            self.scenario_frame += 1
            if self.scenario_frame >= tc.frames:
                self.active_tc = None
                self.status_txt.set_text(f"{tc.name}: complete")
        elif self.running:
            self.sim.step(self.steps_per_frame)
        x = self.sim.x
        self.x_trace.append(x)
        var, ac1 = self._windowed_csd()
        self.var_trace.append(var)
        self.ac1_trace.append(max(ac1, 0.0))

        # Hero: ball on the potential.
        self.ball.set_data([x], [self._potential(x)])

        # éR panel — redraw the movable viable-window bands if the window changed.
        er = self.sim.map_to_er_space()
        window_now = (self.sim.window.er_min, self.sim.window.er_max)
        if window_now != self._last_window:
            self._draw_er_window()
        self.er_star.set_data([0.5], [min(er["energy_resistance"], 11.8)])
        self.er_star.set_color(_REGIME_COLORS.get(er["regime"], _C_BALL))
        self.regime_txt.set_text(er["regime"].upper())
        self.regime_txt.set_color(_REGIME_COLORS.get(er["regime"], "white"))

        # Time series + CSD.
        n = len(self.x_trace)
        xs = np.arange(n)
        self.x_line.set_data(xs, np.asarray(self.x_trace))
        self.var_line.set_data(xs, np.asarray(self.var_trace))
        self.ac1_line.set_data(xs, np.asarray(self.ac1_trace))
        vmax = max(0.05, float(np.max(self.var_trace)) * 1.2)
        self.ax_csd.set_ylim(0, max(1.0, vmax))
        return []

    # ---- entry points -------------------------------------------------------

    def run(self, interval: int = 30, save_path: Optional[str] = None):
        """Launch the interactive window (or save a single snapshot frame)."""
        if self.fig is None:
            self._build()
        self.anim = animation.FuncAnimation(
            self.fig, self._update, interval=interval,
            blit=False, cache_frame_data=False)
        if save_path:
            self._update(0)
            self.fig.savefig(save_path, dpi=150, facecolor=_BG, bbox_inches="tight")
        else:
            plt.show()
        return self.anim

    # ---- static publication figure -----------------------------------------

    @classmethod
    def create_static_figure(cls, save_path: Optional[str] = None, seed: int = 0):
        """Publication 4-panel RegimeSystem summary (potential, hysteresis, CSD, éR)."""
        fig = plt.figure(figsize=(12, 9))
        fig.suptitle("Coherence Regime Transitions — CLT Phase 3.1",
                     fontsize=14, fontweight="bold")
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

        # Potential shallowing.
        ax = fig.add_subplot(gs[0, 0])
        xs = np.linspace(-1.8, 1.8, 400)
        fb = fold_b(1.0)
        for frac, color in [(0.0, _C_HEALTHY), (0.6, _C_DRIVE), (1.0, _C_COLLAPSE)]:
            b = frac * fb
            ax.plot(xs, xs ** 4 / 4 - xs ** 2 / 2 + b * xs, color=color, lw=2,
                    label=f"b = {b:.2f}")
        ax.set_title("Cusp potential shallows with drive", fontweight="bold")
        ax.set_xlabel("order parameter x")
        ax.set_ylabel("V(x)")
        ax.legend(fontsize=8, title="drive")
        ax.grid(alpha=0.2)

        # Hysteresis.
        ax = fig.add_subplot(gs[0, 1])
        hl = run_hysteresis_loop(seed=seed, points=60, steps_per_point=60)
        b = hl.data["control_b"]
        ax.plot(b, hl.data["x_up"], color=_C_COLLAPSE, lw=2, label="up (induction)")
        ax.plot(b, hl.data["x_down"], color=_C_HEALTHY, lw=2, label="down (release)")
        ax.axvline(hl.data["fold_b"], color="gray", ls="--", lw=1, alpha=0.7)
        ax.set_title(f"Hysteresis loop (area = {hl.data['loop_area']:.2f})",
                     fontweight="bold")
        ax.set_xlabel("drive b")
        ax.set_ylabel("order parameter x")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

        # Critical slowing down.
        ax = fig.add_subplot(gs[1, 0])
        cs = run_critical_slowing_down(seed=seed, segments=8, settle=1500, sample=4000)
        b = cs.data["control_b"]
        ax.plot(b, cs.data["ac1"], "o-", color=_C_DRIVE, lw=2, label="lag-1 autocorr.")
        ax.set_xlabel("drive b")
        ax.set_ylabel("lag-1 autocorrelation", color=_C_DRIVE)
        ax.tick_params(axis="y", labelcolor=_C_DRIVE)
        ax.set_title(f"Critical slowing down (τ={cs.data['tau_variance']:.2f})",
                     fontweight="bold")
        ax.grid(alpha=0.2)
        ax2 = ax.twinx()
        ax2.plot(b, cs.data["variance"], "s-", color=_C_ACCENT, lw=2)
        ax2.set_ylabel("variance", color=_C_ACCENT)
        ax2.tick_params(axis="y", labelcolor=_C_ACCENT)

        # éR trajectory of a threshold crossing.
        ax = fig.add_subplot(gs[1, 1])
        tc = run_threshold_crossing(seed=seed)
        ax.plot(tc.er, color=_C_DRIVE, lw=1.5)
        ax.axhspan(0.0, 0.5, color=_C_COLLAPSE, alpha=0.12)
        ax.axhspan(0.5, 5.0, color=_C_HEALTHY, alpha=0.12)
        ax.axhspan(5.0, ax.get_ylim()[1], color=_C_ACCENT, alpha=0.12)
        ax.set_title("éR along a threshold crossing", fontweight="bold")
        ax.set_xlabel("sweep step")
        ax.set_ylabel("éR = EP/f²")
        ax.grid(alpha=0.2)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig


def demo(save_path: Optional[str] = None) -> None:
    """Launch the interactive RegimeSystem explorer (or save a snapshot)."""
    print("RegimeVisualizer (interactive) — CLT Phase 3.1")
    RegimeVisualizer().run(save_path=save_path)


if __name__ == "__main__":
    if "--static" in sys.argv:
        matplotlib.use("Agg")
        RegimeVisualizer.create_static_figure(
            save_path=os.path.join(_PROJECT_ROOT, "output", "regime_transitions_31.png"))
        print("saved static figure to output/regime_transitions_31.png")
    elif "--save" in sys.argv:
        matplotlib.use("Agg")
        demo(save_path=os.path.join(_PROJECT_ROOT, "output", "regime_interactive_snapshot.png"))
    else:
        demo()
