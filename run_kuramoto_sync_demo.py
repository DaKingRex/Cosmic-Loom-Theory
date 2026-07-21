#!/usr/bin/env python3
"""
Run the Kuramoto Synchronization Regimes explorer (CLT Phase 3.1).

Watch a population of phase oscillators on the unit circle: a smeared ring when
desynchronized (chaos), tightening to a clump as you raise the coupling K through
the critical point (partial sync = viable window) and beyond (hypersync = rigidity).
The mean-field resultant vector's length is the order parameter R. This is also the
picture Phase 3.2 uses for seizure dynamics (desync -> hypersync).

Usage:
    python run_kuramoto_sync_demo.py            # interactive window (drag the K slider!)
    python run_kuramoto_sync_demo.py --static   # save the 300 DPI paper figure
"""

import sys

from visualizations.interactive.kuramoto_sync import KuramotoSyncVisualizer

if __name__ == "__main__":
    if "--static" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")
        out = "output/kuramoto_sync.png"
        KuramotoSyncVisualizer.create_static_figure(save_path=out)
        print(f"Saved Kuramoto sync paper figure to {out}")
    else:
        KuramotoSyncVisualizer().run()
