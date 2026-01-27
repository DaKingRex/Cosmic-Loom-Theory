#!/usr/bin/env python3
"""
Run the Energy Resistance (éR) Phase Space Visualizer.

A core teaching tool for Cosmic Loom Theory showing the relationship
between Energy Present (EP), frequency (f), and Energy Resistance:

    éR = EP / f²

Living systems operate in a "viable window" where éR is neither
too low (chaos/decoherence) nor too high (rigidity/frozen dynamics).
This is where biological coherence - and consciousness - can exist.

Usage:
    python run_energy_resistance_demo.py

Or make it executable:
    chmod +x run_energy_resistance_demo.py
    ./run_energy_resistance_demo.py
"""

from visualizations.interactive.energy_resistance import demo

if __name__ == "__main__":
    demo()
