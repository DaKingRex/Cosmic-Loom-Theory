"""
KuramotoSyncVisualizer — interactive explorer for CLT synchronization regimes.

Its design emerges from what synchronization needs to show: a *temporal* process you
watch. So the hero panel is the canonical **phase ensemble on a unit circle** — a
smeared ring when desynchronized (chaos), tightening to a clump as coupling rises
(partial sync = viable, then hypersync = rigidity) — with the mean-field resultant
vector (length = order parameter R) drawn from the center. This is also the picture
Phase 3.2 uses for seizure (desync → hypersync).

Interaction (matplotlib real-time FuncAnimation):
- Sliders: coupling K, phase noise D.
- Preset RadioButtons: Incoherent / Partial / Hypersync.
- Buttons: Start/Pause, Reset.
Live panels: phase circle + resultant vector, éR phase position + regime label, and
the order parameter R(t).

`create_static_figure` keeps a publication summary (phase circles at three couplings
+ R-vs-K sweep + éR) for the paper.
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

# Allow direct execution by ensuring the project root is importable.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from simulations.emergence.kuramoto_network import (  # noqa: E402
    KuramotoNetwork, critical_coupling, DEFAULT_GAMMA,
)
from simulations.emergence.regime_transitions import run_sync_transition  # noqa: E402
from simulations.emergence.pathology import seizure, neurodegeneration  # noqa: E402
from simulations.emergence.healing import meditation  # noqa: E402

_BG = "#1a1a2e"
_AX = "#16213e"
_C_HEALTHY = "#00C2A8"
_C_COLLAPSE = "#E4572E"
_C_DRIVE = "#4C6EF5"
_C_ACCENT = "#F2B705"
_REGIME_COLORS = {"chaos": _C_COLLAPSE, "viable": _C_HEALTHY, "rigidity": _C_ACCENT}

_HISTORY = 400


class KuramotoSyncVisualizer:
    """Interactive phase-circle explorer for Kuramoto synchronization regimes."""

    def __init__(self, n: int = 120, coupling: float = 1.0,
                 gamma: float = DEFAULT_GAMMA, seed: Optional[int] = 0,
                 steps_per_frame: int = 2):
        self.net = KuramotoNetwork(n_oscillators=n, coupling=coupling,
                                   gamma=gamma, seed=seed)
        self.steps_per_frame = steps_per_frame
        self.running = True
        self.r_trace = deque([self.net.order_parameter()], maxlen=_HISTORY)

        # Scenario playback: pathology (window contracts) + healing (window widens)
        # time-courses targeting the Kuramoto engine.
        self.scenarios = {
            "Seizure": seizure, "Neurodegeneration": neurodegeneration,
            "Meditation": meditation,
        }
        self._healing_scenarios = {"Meditation"}
        self.selected_scenario = "Seizure"
        self.active_tc = None
        self.scenario_frame = 0

        self.fig = None
        self.anim = None

    # ---- figure construction ------------------------------------------------

    def _build(self):
        self.fig = plt.figure(figsize=(14, 8), facecolor=_BG)
        self.fig.suptitle("Kuramoto Synchronization Regimes — CLT (interactive)",
                          color="white", fontsize=14, fontweight="bold")
        gs = GridSpec(2, 2, figure=self.fig, left=0.28, right=0.97,
                      top=0.91, bottom=0.10, hspace=0.35, wspace=0.28)
        self.ax_circle = self.fig.add_subplot(gs[:, 0])
        self.ax_er = self.fig.add_subplot(gs[0, 1])
        self.ax_r = self.fig.add_subplot(gs[1, 1])
        for ax in (self.ax_circle, self.ax_er, self.ax_r):
            ax.set_facecolor(_AX)
            ax.tick_params(colors="white", labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#33415c")

        self._init_circle()
        self._init_er()
        self._init_r()
        self._add_controls()

    def _init_circle(self):
        ax = self.ax_circle
        ring = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(ring), np.sin(ring), color="#33415c", lw=1)
        phases = self.net.phases
        (self.dots,) = ax.plot(np.cos(phases), np.sin(phases), "o",
                               color=_C_HEALTHY, markersize=5, alpha=0.7)
        self.resultant = ax.annotate(
            "", xy=(0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color=_C_ACCENT, lw=2.5))
        self.r_label = ax.text(0, -1.32, "", ha="center", color="white",
                               fontsize=11, fontweight="bold")
        self.status_txt = ax.text(0.98, 0.98, "", transform=ax.transAxes,
                                  ha="right", va="top", color=_C_ACCENT,
                                  fontsize=10, fontweight="bold")
        ax.set_title("Phase ensemble on the unit circle", color="white",
                     fontsize=11, fontweight="bold")
        ax.set_xlim(-1.35, 1.35)
        ax.set_ylim(-1.4, 1.35)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    def _draw_er_window(self):
        """(Re)draw the viable-window bands from the network's current window."""
        for art in getattr(self, "_er_artists", []):
            art.remove()
        w = self.net.window
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
        self._draw_er_window()
        (self.er_star,) = ax.plot([0.5], [1.0], "*", color="white", markersize=18,
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

    def _init_r(self):
        ax = self.ax_r
        (self.r_line,) = ax.plot([], [], color=_C_DRIVE, lw=1.8)
        ax.set_title("Order parameter R(t)", color="white",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("recent steps", color="white")
        ax.set_ylabel("R", color=_C_DRIVE)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(0, _HISTORY)
        ax.grid(alpha=0.15)

    # ---- controls -----------------------------------------------------------

    def _add_controls(self):
        def mk_ax(rect):
            return self.fig.add_axes(rect, facecolor="#2d3748")

        kc = critical_coupling(self.net.gamma)
        self.s_k = Slider(mk_ax([0.05, 0.72, 0.17, 0.03]), "coupling K",
                          0.0, 8.0, valinit=self.net.coupling, color=_C_DRIVE)
        self.s_d = Slider(mk_ax([0.05, 0.66, 0.17, 0.03]), "noise D",
                          0.0, 0.5, valinit=self.net.noise, color=_C_ACCENT)
        for s in (self.s_k, self.s_d):
            s.label.set_color("white")
            s.valtext.set_color("white")
        self.s_k.on_changed(self._on_k)
        self.s_d.on_changed(self._on_d)
        # Mark the critical coupling on the K slider.
        self.s_k.ax.axvline(kc, color=_C_HEALTHY, lw=1.5, alpha=0.8)

        self.fig.text(0.055, 0.635, "PRESETS", color="white", fontsize=8, fontweight="bold")
        self.radio = RadioButtons(
            mk_ax([0.05, 0.49, 0.17, 0.13]),
            ("Incoherent", "Partial", "Hypersync"), active=1)
        for lbl in self.radio.labels:
            lbl.set_color("white")
            lbl.set_fontsize(9)
        self.radio.on_clicked(self._on_preset)

        self.fig.text(0.055, 0.455, "SCENARIO  (red=pathology, teal=healing)",
                      color=_C_ACCENT, fontsize=7, fontweight="bold")
        self.scenario_radio = RadioButtons(
            mk_ax([0.05, 0.31, 0.17, 0.12]),
            tuple(self.scenarios.keys()), active=0)
        for lbl in self.scenario_radio.labels:
            healing = lbl.get_text() in self._healing_scenarios
            lbl.set_color(_C_HEALTHY if healing else _C_COLLAPSE)
            lbl.set_fontsize(9)
        self.scenario_radio.on_clicked(self._on_scenario)

        self.btn_run = Button(mk_ax([0.05, 0.255, 0.17, 0.04]), "▶ Run scenario",
                              color="#1f6f54", hovercolor="#2a8f6e")
        self.btn_run.label.set_color("white")
        self.btn_run.on_clicked(self._on_run)

        self.btn_play = Button(mk_ax([0.05, 0.205, 0.08, 0.04]), "Pause",
                               color="#2d3748", hovercolor="#3d4758")
        self.btn_reset = Button(mk_ax([0.14, 0.205, 0.08, 0.04]), "Reset",
                                color="#2d3748", hovercolor="#3d4758")
        self.btn_play.label.set_color("white")
        self.btn_reset.label.set_color("white")
        self.btn_play.on_clicked(self._on_play)
        self.btn_reset.on_clicked(self._on_reset)

    # ---- callbacks ----------------------------------------------------------

    def _on_k(self, val):
        self.net.set_coupling(float(val))

    def _on_d(self, val):
        self.net.noise = float(val)

    def _on_preset(self, label):
        if label == "Incoherent":
            k = 0.2
        elif label == "Partial":
            k = 1.3 * critical_coupling(self.net.gamma)
        else:  # Hypersync
            k = 8.0
        self.net.set_coupling(k)
        self.net.reset()
        self.s_k.set_val(k)
        self.r_trace.clear()
        self.r_trace.append(self.net.order_parameter())

    def _on_play(self, _event):
        self.running = not self.running
        self.btn_play.label.set_text("Start" if not self.running else "Pause")

    def _on_scenario(self, label):
        self.selected_scenario = label

    def _on_run(self, _event):
        """Load the selected pathology time-course, warm up, and start playing it."""
        tc = self.scenarios[self.selected_scenario]()
        self.active_tc = tc
        self.scenario_frame = 0
        tc.apply(self.net, 0.0)
        self.net.reset()                 # re-randomize phases at the start coupling
        self.net.step(1200)              # warm up to the partial-sync steady state
        self.running = True
        self.r_trace.clear()
        self.r_trace.append(self.net.order_parameter())

    def _on_reset(self, _event):
        self.active_tc = None
        self.status_txt.set_text("")
        self.net.reset()
        self.r_trace.clear()
        self.r_trace.append(self.net.order_parameter())

    # ---- animation ----------------------------------------------------------

    def _update(self, _frame):
        if self.active_tc is not None:
            tc = self.active_tc
            p = self.scenario_frame / max(tc.frames - 1, 1)
            tc.apply(self.net, p)
            self.status_txt.set_text(f"▶ {tc.name}  ({int(p * 100)}%)")
            self.net.step(tc.steps_per_frame)
            self.scenario_frame += 1
            if self.scenario_frame >= tc.frames:
                self.active_tc = None
                self.status_txt.set_text(f"{tc.name}: complete")
        elif self.running:
            self.net.step(self.steps_per_frame)
        phases = self.net.phases
        self.dots.set_data(np.cos(phases), np.sin(phases))

        mean_field = np.mean(np.exp(1j * phases))
        r = float(np.abs(mean_field))
        psi = float(np.angle(mean_field))
        self.resultant.set_position((0, 0))
        self.resultant.xy = (r * np.cos(psi), r * np.sin(psi))
        self.r_label.set_text(f"R = {r:.3f}")

        er = self.net.map_to_er_space()
        window_now = (self.net.window.er_min, self.net.window.er_max)
        if window_now != self._last_window:
            self._draw_er_window()
        self.er_star.set_data([0.5], [min(er["energy_resistance"], 11.8)])
        self.er_star.set_color(_REGIME_COLORS.get(er["regime"], "white"))
        self.regime_txt.set_text(er["regime"].upper())
        self.regime_txt.set_color(_REGIME_COLORS.get(er["regime"], "white"))

        self.r_trace.append(r)
        xs = np.arange(len(self.r_trace))
        self.r_line.set_data(xs, np.asarray(self.r_trace))
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
        """Phase circles at three couplings + the R-vs-K sweep."""
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle("Kuramoto Synchronization Regimes — CLT",
                     fontsize=14, fontweight="bold")
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        regimes = [("Incoherent (chaos)", 0.2, _C_COLLAPSE),
                   ("Partial (viable)", 1.3 * critical_coupling(DEFAULT_GAMMA), _C_HEALTHY),
                   ("Hypersync (rigidity)", 8.0, _C_ACCENT)]
        for i, (title, k, color) in enumerate(regimes):
            ax = fig.add_subplot(gs[0, i])
            net = KuramotoNetwork(n_oscillators=120, coupling=k, seed=seed)
            net.run(25.0)
            ring = np.linspace(0, 2 * np.pi, 200)
            ax.plot(np.cos(ring), np.sin(ring), color="#bbb", lw=1)
            ax.plot(np.cos(net.phases), np.sin(net.phases), "o", color=color,
                    markersize=5, alpha=0.7)
            r = net.order_parameter()
            psi = np.angle(np.mean(np.exp(1j * net.phases)))
            ax.annotate("", xy=(r * np.cos(psi), r * np.sin(psi)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=2))
            ax.set_title(f"{title}\nK={k:.2f}, R={r:.2f}", fontsize=10)
            ax.set_xlim(-1.3, 1.3)
            ax.set_ylim(-1.3, 1.3)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

        ax = fig.add_subplot(gs[1, :])
        st = run_sync_transition(seed=seed, points=40, settle_time=6.0)
        k = st.data["coupling_K"]
        ax.plot(k, st.data["order_parameter"], "o-", color=_C_DRIVE, lw=2)
        ax.axvline(critical_coupling(DEFAULT_GAMMA), color=_C_HEALTHY, ls="--",
                   label="K_c")
        ax.set_title("Synchronization transition: R vs coupling K", fontweight="bold")
        ax.set_xlabel("coupling K")
        ax.set_ylabel("order parameter R")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
        ax.set_ylim(-0.02, 1.02)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig


def demo(save_path: Optional[str] = None) -> None:
    """Launch the interactive Kuramoto synchronization explorer."""
    print("KuramotoSyncVisualizer (interactive) — CLT")
    KuramotoSyncVisualizer().run(save_path=save_path)


if __name__ == "__main__":
    if "--static" in sys.argv:
        matplotlib.use("Agg")
        KuramotoSyncVisualizer.create_static_figure(
            save_path=os.path.join(_PROJECT_ROOT, "output", "kuramoto_sync.png"))
        print("saved static figure to output/kuramoto_sync.png")
    elif "--save" in sys.argv:
        matplotlib.use("Agg")
        demo(save_path=os.path.join(_PROJECT_ROOT, "output", "kuramoto_interactive_snapshot.png"))
    else:
        demo()
