#!/usr/bin/env python3
"""
Run the Loomfield Wave Propagation Simulator.

This is the centerpiece visualization for Cosmic Loom Theory, implementing
the Loomfield dynamics equation from CLT v1.1:

    ∇²L − (1/v²ₗ)(∂²L/∂t²) = κₗ · ρ_coh

Watch consciousness as a propagating wave of coherence.

Usage:
    python run_loomfield_demo.py

Or make it executable:
    chmod +x run_loomfield_demo.py
    ./run_loomfield_demo.py
"""

from visualizations.interactive.loomfield_wave import demo

if __name__ == "__main__":
    demo()
