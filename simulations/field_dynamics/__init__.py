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

4. **Biophoton Emission** (CLT v1.1 §4.2)
   - Ultra-weak photon emission from mitochondria
   - Emission statistics (Poissonian, coherent, squeezed, chaotic)
   - Spatial and temporal coherence metrics
   - Metabolic state coupling (ATP, ROS)
   - LoomSense-compatible output

5. **DNA Constraints** (CLT v1.1 §4.3)
   - Long-timescale constraints on Loomfield topology
   - Genetic constraints on coherence parameters
   - Epigenetic modulation (methylation, environmental factors)
   - Species-specific viable windows in éR space
   - Developmental dynamics across lifespan
   - Pi stack quantum coherence in DNA helix

Available modules:
- bioelectric: Single-layer ion channel networks, gap junctions
- bioelectric_multilayer: Multi-layer tissue coupling
- morphogenetic: Pattern memory, regeneration, morphogenesis
- biophoton: Ultra-weak photon emission, metabolic coherence
- dna_constraints: DNA genetic/epigenetic constraints on coherence
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

# Biophoton emission simulation
from .biophoton import (
    # Core classes
    BiophotonSimulator,
    BiophotonVisualizer,

    # Emission modes and states
    EmissionMode,
    TissueState,

    # Physical constants
    WAVELENGTH_MIN,
    WAVELENGTH_MAX,
    WAVELENGTH_PEAK,
    EMISSION_RATE_BASELINE,
    ATP_BASELINE,
    ROS_BASELINE,

    # Presets
    create_healthy_tissue,
    create_stressed_tissue,
    create_coherent_emission,
    create_meditation_state,
    create_inflammation_model,

    # Demos
    demo as biophoton_demo,
    demo_stressed as biophoton_stressed_demo,
    demo_coherent as biophoton_coherent_demo,
    demo_meditation as biophoton_meditation_demo,
    demo_comparison as biophoton_comparison_demo,
)

# DNA constraint simulation
from .dna_constraints import (
    # Core classes
    DNAConstraintSimulator,
    DNAConstraintVisualizer,

    # Enums and dataclasses
    Gene,
    GeneCategory,
    DevelopmentalStage,
    SpeciesComplexity,
    BasePair,

    # Constants
    PI_STACK_SPACING_NM,
    PI_STACK_COUPLING,
    AROMATIC_RINGS_PER_BASE,
    DEFAULT_GENES,

    # Presets
    create_human_baseline,
    create_high_plasticity,
    create_stressed_aging,
    create_developmental_series,
    create_cross_species,
    create_meditation_epigenetics,

    # Demos
    demo as dna_demo,
    demo_development as dna_development_demo,
    demo_aging as dna_aging_demo,
    demo_comparison as dna_comparison_demo,
    demo_species_comparison as dna_species_demo,
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

    # === Biophoton emission ===
    'BiophotonSimulator',
    'BiophotonVisualizer',
    'EmissionMode',
    'TissueState',
    'WAVELENGTH_MIN',
    'WAVELENGTH_MAX',
    'WAVELENGTH_PEAK',
    'EMISSION_RATE_BASELINE',
    'ATP_BASELINE',
    'ROS_BASELINE',
    'create_healthy_tissue',
    'create_stressed_tissue',
    'create_coherent_emission',
    'create_meditation_state',
    'create_inflammation_model',
    'biophoton_demo',
    'biophoton_stressed_demo',
    'biophoton_coherent_demo',
    'biophoton_meditation_demo',
    'biophoton_comparison_demo',

    # === DNA constraints ===
    'DNAConstraintSimulator',
    'DNAConstraintVisualizer',
    'Gene',
    'GeneCategory',
    'DevelopmentalStage',
    'SpeciesComplexity',
    'BasePair',
    'PI_STACK_SPACING_NM',
    'PI_STACK_COUPLING',
    'AROMATIC_RINGS_PER_BASE',
    'DEFAULT_GENES',
    'create_human_baseline',
    'create_high_plasticity',
    'create_stressed_aging',
    'create_developmental_series',
    'create_cross_species',
    'create_meditation_epigenetics',
    'dna_demo',
    'dna_development_demo',
    'dna_aging_demo',
    'dna_comparison_demo',
    'dna_species_demo',
]
