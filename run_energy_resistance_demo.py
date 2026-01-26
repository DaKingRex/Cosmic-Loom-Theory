#!/usr/bin/env python3
"""
Run the Energy Resistance (éR) Phase Space Visualizer.

This is a convenience script to launch the interactive visualization
without needing to navigate the module structure.

Usage:
    python run_energy_resistance_demo.py

Or make it executable:
    chmod +x run_energy_resistance_demo.py
    ./run_energy_resistance_demo.py
"""

from visualizations.interactive.energy_resistance import demo

if __name__ == "__main__":
    demo()
