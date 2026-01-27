# Interactive CLT Visualizations
"""
Interactive visualization tools for Cosmic Loom Theory.

Available visualizers:
- EnergyResistanceVisualizer: éR = EP/f² phase space with viable window
- LoomfieldVisualizer: Real-time 2D Loomfield wave propagation
- LoomfieldVisualizer3D: 3D volumetric Loomfield visualization (plotly)
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

from .loomfield_3d import (
    LoomfieldSimulator3D,
    LoomfieldVisualizer3D,
    create_volumetric_figure,
    create_slice_figure,
    create_animated_figure,
    create_healthy_preset,
    create_pathology_preset,
    create_healing_preset,
    demo as loomfield_3d_demo
)

# Real-time 3D visualizer (requires vispy + PyQt5)
try:
    from .loomfield_3d_realtime import (
        LoomfieldSimulator3DRealtime,
        Loomfield3DVisualizer,
        demo as loomfield_3d_realtime_demo
    )
    HAS_REALTIME_3D = True
except ImportError:
    HAS_REALTIME_3D = False

__all__ = [
    # Energy Resistance
    'EnergyResistanceVisualizer',
    'calculate_system_er',
    'energy_resistance_demo',
    # Loomfield 2D
    'LoomfieldSimulator',
    'LoomfieldVisualizer',
    'loomfield_demo',
    # Loomfield 3D
    'LoomfieldSimulator3D',
    'LoomfieldVisualizer3D',
    'create_volumetric_figure',
    'create_slice_figure',
    'create_animated_figure',
    'create_healthy_preset',
    'create_pathology_preset',
    'create_healing_preset',
    'loomfield_3d_demo'
]
