# Loomfield Dynamics Simulations
"""
Numerical solvers for the Loomfield wave equation.

The Loomfield L(r,t) is the central effective field in CLT, governed by:
    ∇²L − (1/v²ₗ)(∂²L/∂t²) = κₗ · ρ_coh

This module provides:
- 2D/3D finite difference solvers
- Absorbing boundary conditions (Mur ABC)
- Coherence density source terms
- Multi-scale coupling between biological layers
- Wave propagation and interference dynamics

See visualizations/interactive/loomfield_wave.py for the reference implementation.
"""
