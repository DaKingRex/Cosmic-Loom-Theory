# Biological Quantum Coherence Simulations
"""
Quantum coherence in biological substrates per CLT v1.1.

This module implements simulations for:
- Microtubule time crystals (Penrose-Hameroff-Bandyopadhyay)
- Multi-scale oscillations (kHz → THz)
- Triplet resonance patterns (fractal time crystal)
- Decoherence timescales in warm biological environments
- Coherence protection via Floquet driving

Note: CLT does not require quantum mechanics for consciousness,
but quantum coherence may contribute to biological substrate dynamics.

Modules:
- microtubule: Tubulin dipole lattice, time crystal dynamics
"""

# Microtubule time crystal simulation
from .microtubule import (
    # Core classes
    MicrotubuleSimulator,
    MicrotubuleVisualizer,

    # Enums and states
    OscillationScale,
    MicrotubuleState,

    # Physical constants
    N_PROTOFILAMENTS,
    TUBULIN_LENGTH_NM,
    AROMATIC_RINGS_PER_TUBULIN,
    FREQ_CTERMINI,
    FREQ_LATTICE,
    FREQ_WATER_CHANNEL,
    FREQ_AROMATIC,
    TRIPLET_RATIO_1,
    TRIPLET_RATIO_2,
    TRIPLET_RATIO_3,
    TEMPERATURE_BODY,

    # Presets
    create_coherent_mt,
    create_thermal_mt,
    create_floquet_driven_mt,
    create_anesthetized_mt,
    create_cold_mt,

    # Demos
    demo as microtubule_demo,
    demo_thermal as microtubule_thermal_demo,
    demo_floquet as microtubule_floquet_demo,
    demo_anesthesia as microtubule_anesthesia_demo,
    demo_comparison as microtubule_comparison_demo,
)

__all__ = [
    # Core classes
    'MicrotubuleSimulator',
    'MicrotubuleVisualizer',

    # Enums
    'OscillationScale',
    'MicrotubuleState',

    # Constants
    'N_PROTOFILAMENTS',
    'TUBULIN_LENGTH_NM',
    'AROMATIC_RINGS_PER_TUBULIN',
    'FREQ_CTERMINI',
    'FREQ_LATTICE',
    'FREQ_WATER_CHANNEL',
    'FREQ_AROMATIC',
    'TRIPLET_RATIO_1',
    'TRIPLET_RATIO_2',
    'TRIPLET_RATIO_3',
    'TEMPERATURE_BODY',

    # Presets
    'create_coherent_mt',
    'create_thermal_mt',
    'create_floquet_driven_mt',
    'create_anesthetized_mt',
    'create_cold_mt',

    # Demos
    'microtubule_demo',
    'microtubule_thermal_demo',
    'microtubule_floquet_demo',
    'microtubule_anesthesia_demo',
    'microtubule_comparison_demo',
]
