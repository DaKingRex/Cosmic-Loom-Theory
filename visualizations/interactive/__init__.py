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
    # Biological parameter mapping functions
    map_hrv_to_frequency,
    map_metabolic_rate_to_energy,
    map_eeg_band_to_frequency,
    biological_state_to_er,
    # Reference data
    BIOLOGICAL_STATES,
    PATHOLOGY_ZONES,
    # Demo functions
    demo as energy_resistance_demo,
    demo_biological as energy_resistance_demo_biological,
    demo_pathology as energy_resistance_demo_pathology,
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
    'map_hrv_to_frequency',
    'map_metabolic_rate_to_energy',
    'map_eeg_band_to_frequency',
    'biological_state_to_er',
    'BIOLOGICAL_STATES',
    'PATHOLOGY_ZONES',
    'energy_resistance_demo',
    'energy_resistance_demo_biological',
    'energy_resistance_demo_pathology',
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
