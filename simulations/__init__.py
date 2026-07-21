# Cosmic Loom Theory - Simulations Module
"""
Loomfield dynamics and biological coherence simulations.

This module contains numerical solvers and simulation frameworks for:
- Loomfield wave equation: ∇²L − (1/v²ₗ)(∂²L/∂t²) = κₗ·ρ_coh
- Coherence regime transitions (chaos ↔ viable ↔ rigidity)
- Biological substrate dynamics (bioelectric, biophoton, microtubule)
- Pathology and healing dynamics

Submodules:
- field_dynamics: Bioelectric fields, gap junctions, pattern formation
  - bioelectric: Single-layer ion channel networks
  - bioelectric_multilayer: Multi-layer tissue coupling
  - morphogenetic: Pattern memory and regeneration
  - biophoton: Ultra-weak photon emission, metabolic coherence
  - dna_constraints: DNA genetic/epigenetic constraints
- emergence: Coherence emergence from substrate interactions
- quantum: Quantum coherence effects (microtubules, etc.)
"""

# Import field dynamics components
from .field_dynamics import (
    # === Single-layer bioelectric ===
    BioelectricSimulator,
    BioelectricVisualizer,
    V_REST,
    V_THRESHOLD,
    V_DEPOLARIZED,
    V_HYPERPOLARIZED,
    V_REVERSAL_NA,
    V_REVERSAL_K,
    G_GAP_DEFAULT,
    create_uniform_preset,
    create_depolarized_region_preset,
    create_bioelectric_pattern_preset,
    create_injured_tissue_preset,
    create_regeneration_preset,
    bioelectric_demo,
    bioelectric_pattern_demo,
    bioelectric_injury_demo,

    # === Multi-layer bioelectric ===
    MultiLayerBioelectricSimulator,
    MultiLayerVisualizer,
    TissueType,
    TissueProperties,
    TISSUE_PRESETS,
    create_default_multilayer,
    create_epithelial_neural_pair,
    create_decoupled_layers,
    create_tightly_coupled_layers,
    create_injured_multilayer,
    multilayer_demo,
    multilayer_decoupled_demo,
    multilayer_coupled_demo,

    # === Morphogenetic fields ===
    MorphogeneticSimulator,
    MorphogeneticVisualizer,
    PatternType,
    generate_pattern,
    create_stable_pattern,
    create_regeneration_scenario,
    create_repatterning_scenario,
    create_cancer_scenario,
    morphogenetic_demo,
    morphogenetic_regeneration_demo,
    morphogenetic_repattern_demo,
    morphogenetic_cancer_demo,

    # === Biophoton emission ===
    BiophotonSimulator,
    BiophotonVisualizer,
    EmissionMode,
    TissueState,
    WAVELENGTH_MIN,
    WAVELENGTH_MAX,
    WAVELENGTH_PEAK,
    EMISSION_RATE_BASELINE,
    ATP_BASELINE,
    ROS_BASELINE,
    create_healthy_tissue,
    create_stressed_tissue,
    create_coherent_emission,
    create_meditation_state,
    create_inflammation_model,
    biophoton_demo,
    biophoton_stressed_demo,
    biophoton_coherent_demo,
    biophoton_meditation_demo,
    biophoton_comparison_demo,

    # === DNA constraints ===
    DNAConstraintSimulator,
    DNAConstraintVisualizer,
    Gene,
    GeneCategory,
    DevelopmentalStage,
    SpeciesComplexity,
    BasePair,
    PI_STACK_SPACING_NM,
    PI_STACK_COUPLING,
    AROMATIC_RINGS_PER_BASE,
    DEFAULT_GENES,
    create_human_baseline,
    create_high_plasticity,
    create_stressed_aging,
    create_developmental_series,
    create_cross_species,
    create_meditation_epigenetics,
    dna_demo,
    dna_development_demo,
    dna_aging_demo,
    dna_comparison_demo,
    dna_species_demo,
)

# Import emergence components (Phase 3.1: coherence regime transitions)
from .emergence import (
    # === Regime system (cusp / double-well) ===
    RegimeSystem,
    fold_b,
    equilibria,
    is_stable,
    create_bistable_system,
    create_monostable_system,
    create_near_fold_system,
    # === Kuramoto network ===
    KuramotoNetwork,
    critical_coupling,
    create_incoherent_network,
    create_partial_sync_network,
    create_hypersync_network,
    # === Regime-transition scenarios ===
    ScenarioResult,
    run_threshold_crossing,
    run_hysteresis_loop,
    run_critical_slowing_down,
    run_sync_transition,
    add_scenario_to_er_visualizer,
    SCENARIOS,
    # === Scenario time-course driver + pathologies (Phase 3.2) ===
    TimeCourse,
    make_engine,
    run_time_course,
    depression,
    anesthesia,
    seizure,
    PATHOLOGIES,
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

    # === Regime system (Phase 3.1) ===
    'RegimeSystem',
    'fold_b',
    'equilibria',
    'is_stable',
    'create_bistable_system',
    'create_monostable_system',
    'create_near_fold_system',

    # === Kuramoto network (Phase 3.1) ===
    'KuramotoNetwork',
    'critical_coupling',
    'create_incoherent_network',
    'create_partial_sync_network',
    'create_hypersync_network',

    # === Regime-transition scenarios (Phase 3.1) ===
    'ScenarioResult',
    'run_threshold_crossing',
    'run_hysteresis_loop',
    'run_critical_slowing_down',
    'run_sync_transition',
    'add_scenario_to_er_visualizer',
    'SCENARIOS',

    # === Scenario time-course driver + pathologies (Phase 3.2) ===
    'TimeCourse',
    'make_engine',
    'run_time_course',
    'depression',
    'anesthesia',
    'seizure',
    'PATHOLOGIES',
]
