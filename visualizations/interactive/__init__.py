# Interactive CLT Visualizations
"""
Interactive visualization tools for Cosmic Loom Theory.

Available visualizers:
- EnergyResistanceVisualizer: éR = EP/f² phase space with viable window
- LoomfieldVisualizer: Real-time Loomfield wave propagation
"""

from .energy_resistance import (
    EnergyResistanceVisualizer,
    calculate_system_er,
    demo as energy_resistance_demo
)

from .loomfield_wave import (
    LoomfieldSimulator,
    LoomfieldVisualizer,
    demo as loomfield_demo
)

__all__ = [
    'EnergyResistanceVisualizer',
    'calculate_system_er',
    'energy_resistance_demo',
    'LoomfieldSimulator',
    'LoomfieldVisualizer',
    'loomfield_demo'
]
