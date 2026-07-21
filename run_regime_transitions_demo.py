#!/usr/bin/env python3
"""
Run the Coherence Regime Transitions dashboard (CLT Phase 3.1).

Visualizes the four regime-transition phenomena in one figure:
  1. the cusp/double-well potential shallowing as the drive rises,
  2. the hysteresis loop (induction threshold > release threshold),
  3. critical-slowing-down indicators (autocorrelation, variance) vs. drive,
  4. the Kuramoto synchronization transition (order parameter vs. coupling).

These are CLT's account of how coherence crosses between the viable window and
the chaos / rigidity regimes.

Usage:
    python run_regime_transitions_demo.py            # interactive window (drive it!)
    python run_regime_transitions_demo.py --static   # save the 300 DPI paper figure
"""

import sys

from visualizations.interactive.regime_transitions import RegimeVisualizer

if __name__ == "__main__":
    if "--static" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        out = "output/regime_transitions_31.png"
        RegimeVisualizer.create_static_figure(save_path=out)
        print(f"Saved regime-transitions paper figure to {out}")
    else:
        RegimeVisualizer().run()
