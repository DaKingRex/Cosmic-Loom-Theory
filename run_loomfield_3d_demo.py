#!/usr/bin/env python3
"""
Run the 3D Loomfield Wave Propagation Visualizer.

The centerpiece 3D visualization for Cosmic Loom Theory, showing the
Loomfield as a calm, living volume in three-dimensional space.

Design Philosophy:
    "The 3D Loomfield should feel like a calm, living volume whose internal
    order becomes visible as coherence increases — not a collection of objects,
    but a continuous system revealing structure through constraint."

The visualization uses:
    - Volumetric rendering (primary): Semi-transparent density showing coherence
    - Isosurfaces (secondary): Threshold surfaces where coherence stabilizes
    - Minimal particles: Only for sources and perturbations

Usage:
    python run_loomfield_3d_demo.py

Or make it executable:
    chmod +x run_loomfield_3d_demo.py
    ./run_loomfield_3d_demo.py
"""

from visualizations.interactive.loomfield_3d import demo

if __name__ == "__main__":
    demo()
