# Field Dynamics Simulations
"""
Biological field dynamics simulations for Cosmic Loom Theory.

This module provides simulators for the physical substrates that
generate and maintain coherent patterns in living systems:

1. **Bioelectric Fields** (CLT v1.1 §4.1)
   - Ion channel networks and membrane potentials
   - Gap junction coupling between cells
   - Bioelectric pattern formation and maintenance
   - Injury/regeneration dynamics

2. **Multi-Layer Tissue Coupling**
   - Cross-tissue coherence (epithelial, neural, mesenchymal)
   - Vertical coupling between tissue layers
   - Hierarchical coherence metrics

3. **Morphogenetic Fields**
   - Target pattern encoding (morphogenetic memory)
   - Pattern attraction dynamics
   - Regeneration after injury
   - Repatterning capabilities (Levin-style)

Available modules:
- bioelectric: Single-layer ion channel networks, gap junctions
- bioelectric_multilayer: Multi-layer tissue coupling
- morphogenetic: Pattern memory, regeneration, morphogenesis
"""

# Single-layer bioelectric simulation
from .bioelectric import (
    # Core simulator
    BioelectricSimulator,
    BioelectricVisualizer,

    # Physical constants
    V_REST,
    V_THRESHOLD,
    V_DEPOLARIZED,
    V_HYPERPOLARIZED,
    V_REVERSAL_NA,
    V_REVERSAL_K,
    G_GAP_DEFAULT,

    # Presets
    create_uniform_preset,
    create_depolarized_region_preset,
    create_bioelectric_pattern_preset,
    create_injured_tissue_preset,
    create_regeneration_preset,

    # Demo functions
    demo as bioelectric_demo,
    demo_pattern_formation as bioelectric_pattern_demo,
    demo_injury_regeneration as bioelectric_injury_demo,
)

# Multi-layer bioelectric simulation
from .bioelectric_multilayer import (
    # Core classes
    MultiLayerBioelectricSimulator,
    MultiLayerVisualizer,

    # Tissue types
    TissueType,
    TissueProperties,
    TISSUE_PRESETS,

    # Presets
    create_default_multilayer,
    create_epithelial_neural_pair,
    create_decoupled_layers,
    create_tightly_coupled_layers,
    create_injured_multilayer,

    # Demos
    demo as multilayer_demo,
    demo_decoupled as multilayer_decoupled_demo,
    demo_tightly_coupled as multilayer_coupled_demo,
)

# Morphogenetic field simulation
from .morphogenetic import (
    # Core classes
    MorphogeneticSimulator,
    MorphogeneticVisualizer,

    # Pattern types
    PatternType,
    generate_pattern,

    # Presets
    create_stable_pattern,
    create_regeneration_scenario,
    create_repatterning_scenario,
    create_cancer_scenario,

    # Demos
    demo as morphogenetic_demo,
    demo_regeneration as morphogenetic_regeneration_demo,
    demo_repatterning as morphogenetic_repattern_demo,
    demo_cancer as morphogenetic_cancer_demo,
)

__all__ = [
    # === Single-layer bioelectric ===
    'BioelectricSimulator',
    'BioelectricVisualizer',
    'V_REST',
    'V_THRESHOLD',
    'V_DEPOLARIZED',
    'V_HYPERPOLARIZED',
    'V_REVERSAL_NA',
    'V_REVERSAL_K',
    'G_GAP_DEFAULT',
    'create_uniform_preset',
    'create_depolarized_region_preset',
    'create_bioelectric_pattern_preset',
    'create_injured_tissue_preset',
    'create_regeneration_preset',
    'bioelectric_demo',
    'bioelectric_pattern_demo',
    'bioelectric_injury_demo',

    # === Multi-layer bioelectric ===
    'MultiLayerBioelectricSimulator',
    'MultiLayerVisualizer',
    'TissueType',
    'TissueProperties',
    'TISSUE_PRESETS',
    'create_default_multilayer',
    'create_epithelial_neural_pair',
    'create_decoupled_layers',
    'create_tightly_coupled_layers',
    'create_injured_multilayer',
    'multilayer_demo',
    'multilayer_decoupled_demo',
    'multilayer_coupled_demo',

    # === Morphogenetic fields ===
    'MorphogeneticSimulator',
    'MorphogeneticVisualizer',
    'PatternType',
    'generate_pattern',
    'create_stable_pattern',
    'create_regeneration_scenario',
    'create_repatterning_scenario',
    'create_cancer_scenario',
    'morphogenetic_demo',
    'morphogenetic_regeneration_demo',
    'morphogenetic_repattern_demo',
    'morphogenetic_cancer_demo',
]
