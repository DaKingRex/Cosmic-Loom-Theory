#!/usr/bin/env python3
"""
Run the Real-Time 3D Loomfield Simulator.

An interactive 3D visualization with real-time physics computation.
This is the 3D extension of the 2D loomfield wave simulator.

Features:
    - Real-time 3D wave propagation
    - Volumetric rendering with gold/blue colormap
    - Interactive rotation and zoom
    - Parameter sliders (v_L, κ_L, speed)
    - Preset scenarios (Healthy, Pathology, Healing)
    - Slice plane to view interior structure

Requirements:
    pip install vispy PyQt5 numpy scipy

Usage:
    python run_loomfield_3d_realtime.py

Or make it executable:
    chmod +x run_loomfield_3d_realtime.py
    ./run_loomfield_3d_realtime.py
"""

from visualizations.interactive.loomfield_3d_realtime import demo

if __name__ == "__main__":
    demo()
