"""
DNA Constraint Simulator for Cosmic Loom Theory

Models how DNA provides long-timescale constraints on Loomfield topology.
DNA doesn't directly generate consciousness but CONSTRAINS the parameter
space within which conscious dynamics can occur.

From CLT v1.1 Section 4.3:
While bioelectric, biophoton, and microtubule substrates operate on fast
timescales (ms to ns), DNA operates on developmental and evolutionary
timescales. It's the slow "scaffold" that determines what coherence
patterns are POSSIBLE for a given organism.

DNA determines:
- Protein expression (tubulin variants, ion channels)
- Metabolic machinery (mitochondria → biophotons)
- Tissue architecture (bioelectric patterns)
- Species-specific viable windows in éR space

From Hameroff research:
DNA's aromatic base pairs form a "pi stack" down the helix, creating
potential pathways for coherent electron transfer. DNA may participate
in quantum coherence, not just information storage.

Key concepts:
- Genetic constraints on coherence parameters
- Epigenetic modulation (methylation, environmental effects)
- Species-specific viable windows
- Developmental dynamics across lifespan
- Pi stack quantum coherence in the helix
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, FancyBboxPatch, Wedge
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.animation as animation


# =============================================================================
# CONSTANTS AND PARAMETERS
# =============================================================================

# Base pair types
class BasePair(Enum):
    """DNA base pairs."""
    AT = "A-T"  # Adenine-Thymine (2 H-bonds)
    TA = "T-A"
    GC = "G-C"  # Guanine-Cytosine (3 H-bonds, more stable)
    CG = "C-G"


# Gene categories affecting coherence substrates
class GeneCategory(Enum):
    """Categories of genes relevant to CLT substrates."""
    TUBULIN = "tubulin"           # Microtubule dynamics
    ION_CHANNEL = "ion_channel"   # Bioelectric patterns
    MITOCHONDRIAL = "mitochondrial"  # Biophoton emission
    GAP_JUNCTION = "gap_junction"    # Intercellular coupling
    SIGNALING = "signaling"       # Signal transduction
    METABOLIC = "metabolic"       # Energy metabolism
    STRUCTURAL = "structural"     # Cytoskeletal architecture


# Developmental stages
class DevelopmentalStage(Enum):
    """Stages of development affecting gene expression."""
    EMBRYONIC = "embryonic"
    FETAL = "fetal"
    INFANT = "infant"
    CHILD = "child"
    ADOLESCENT = "adolescent"
    ADULT = "adult"
    MIDDLE_AGE = "middle_age"
    ELDERLY = "elderly"


# Species complexity levels
class SpeciesComplexity(Enum):
    """Species complexity affecting viable window."""
    PROKARYOTE = "prokaryote"     # Bacteria
    SIMPLE_EUKARYOTE = "simple_eukaryote"  # Yeast
    INVERTEBRATE = "invertebrate"  # C. elegans, insects
    SIMPLE_VERTEBRATE = "simple_vertebrate"  # Fish
    MAMMAL = "mammal"             # Mouse, rat
    PRIMATE = "primate"           # Non-human primate
    HUMAN = "human"               # Homo sapiens


# Pi stack parameters
PI_STACK_SPACING_NM = 0.34       # Distance between base pairs (nm)
PI_STACK_COUPLING = 0.05         # Electron coupling between bases (eV)
AROMATIC_RINGS_PER_BASE = 2      # Purine has 2 rings, pyrimidine has 1


# =============================================================================
# GENE DEFINITIONS
# =============================================================================

@dataclass
class Gene:
    """
    A gene that affects coherence-relevant parameters.

    Attributes:
        name: Gene identifier
        category: Which substrate it affects
        expression_level: 0-1 normalized expression
        methylation: 0-1 epigenetic silencing (higher = more silenced)
        effect_on_coherence: How expression maps to coherence parameters
    """
    name: str
    category: GeneCategory
    expression_level: float = 1.0
    methylation: float = 0.0  # Epigenetic silencing
    importance: float = 1.0   # Weight in calculations

    @property
    def effective_expression(self) -> float:
        """Expression accounting for epigenetic silencing."""
        return self.expression_level * (1.0 - self.methylation * 0.9)


# Default human gene set
DEFAULT_GENES = {
    # Tubulin variants (affect microtubule dynamics)
    'TUBA1A': Gene('TUBA1A', GeneCategory.TUBULIN, 1.0, 0.0, 1.2),
    'TUBB3': Gene('TUBB3', GeneCategory.TUBULIN, 0.8, 0.0, 1.0),
    'TUBB4A': Gene('TUBB4A', GeneCategory.TUBULIN, 0.6, 0.1, 0.8),

    # Ion channels (affect bioelectric patterns)
    'SCN1A': Gene('SCN1A', GeneCategory.ION_CHANNEL, 1.0, 0.0, 1.5),  # Nav1.1
    'KCNQ2': Gene('KCNQ2', GeneCategory.ION_CHANNEL, 0.9, 0.0, 1.2),  # Kv7.2
    'CACNA1C': Gene('CACNA1C', GeneCategory.ION_CHANNEL, 0.85, 0.05, 1.0),  # Cav1.2

    # Gap junctions (intercellular coupling)
    'GJA1': Gene('GJA1', GeneCategory.GAP_JUNCTION, 1.0, 0.0, 1.3),  # Cx43
    'GJB2': Gene('GJB2', GeneCategory.GAP_JUNCTION, 0.7, 0.0, 0.9),  # Cx26

    # Mitochondrial genes (biophoton emission)
    'MT-ND1': Gene('MT-ND1', GeneCategory.MITOCHONDRIAL, 1.0, 0.0, 1.4),
    'MT-CO1': Gene('MT-CO1', GeneCategory.MITOCHONDRIAL, 0.95, 0.0, 1.2),
    'MT-ATP6': Gene('MT-ATP6', GeneCategory.MITOCHONDRIAL, 0.9, 0.05, 1.0),

    # Metabolic (energy for coherence)
    'HK1': Gene('HK1', GeneCategory.METABOLIC, 1.0, 0.0, 0.8),
    'PFKM': Gene('PFKM', GeneCategory.METABOLIC, 0.9, 0.0, 0.7),

    # Signaling (coordination)
    'CAMK2A': Gene('CAMK2A', GeneCategory.SIGNALING, 1.0, 0.0, 1.1),
    'PRKACA': Gene('PRKACA', GeneCategory.SIGNALING, 0.85, 0.0, 0.9),
}


# =============================================================================
# DNA CONSTRAINT SIMULATOR
# =============================================================================

class DNAConstraintSimulator:
    """
    Simulates how DNA constrains Loomfield coherence parameters.

    Models the genetic and epigenetic factors that determine what
    coherence patterns are possible for an organism. This is the
    long-timescale scaffold that shapes consciousness capacity.

    Parameters:
        genes: Dictionary of genes to simulate
        species: Species complexity level
        developmental_stage: Current developmental stage
        base_pairs: Number of base pairs to model for pi stack
    """

    def __init__(
        self,
        genes: Dict[str, Gene] = None,
        species: SpeciesComplexity = SpeciesComplexity.HUMAN,
        developmental_stage: DevelopmentalStage = DevelopmentalStage.ADULT,
        base_pairs: int = 100
    ):
        # Copy genes to allow modification
        self.genes = {k: Gene(g.name, g.category, g.expression_level,
                              g.methylation, g.importance)
                     for k, g in (genes or DEFAULT_GENES).items()}
        self.species = species
        self.developmental_stage = developmental_stage
        self.n_base_pairs = base_pairs

        # Initialize DNA sequence (for pi stack)
        self._init_dna_sequence()

        # Apply developmental modifications
        self._apply_developmental_effects()

        # Compute derived parameters
        self._compute_coherence_constraints()

        # History tracking
        self.viable_window_history = []
        self.expression_history = []

    def _init_dna_sequence(self):
        """Initialize DNA base pair sequence for pi stack model."""
        # Random but realistic sequence (GC content ~40-60%)
        self.sequence = []
        for _ in range(self.n_base_pairs):
            if np.random.random() < 0.5:
                bp = np.random.choice([BasePair.AT, BasePair.TA])
            else:
                bp = np.random.choice([BasePair.GC, BasePair.CG])
            self.sequence.append(bp)

        # Pi stack coherence state (phase at each base)
        self.pi_stack_phase = np.random.uniform(0, 2*np.pi, self.n_base_pairs)
        self.pi_stack_amplitude = np.ones(self.n_base_pairs)

    def _apply_developmental_effects(self):
        """Apply developmental stage effects on gene expression."""
        stage_modifiers = {
            DevelopmentalStage.EMBRYONIC: {
                GeneCategory.TUBULIN: 1.3,      # High during growth
                GeneCategory.MITOCHONDRIAL: 0.8,
                GeneCategory.ION_CHANNEL: 0.6,  # Still developing
            },
            DevelopmentalStage.FETAL: {
                GeneCategory.TUBULIN: 1.2,
                GeneCategory.MITOCHONDRIAL: 0.9,
                GeneCategory.ION_CHANNEL: 0.7,
            },
            DevelopmentalStage.INFANT: {
                GeneCategory.TUBULIN: 1.1,
                GeneCategory.SIGNALING: 1.2,    # High plasticity
                GeneCategory.GAP_JUNCTION: 1.1,
            },
            DevelopmentalStage.CHILD: {
                GeneCategory.SIGNALING: 1.15,
                GeneCategory.GAP_JUNCTION: 1.05,
            },
            DevelopmentalStage.ADOLESCENT: {
                GeneCategory.SIGNALING: 1.1,
                GeneCategory.METABOLIC: 1.1,
            },
            DevelopmentalStage.ADULT: {
                # Baseline - no modifications
            },
            DevelopmentalStage.MIDDLE_AGE: {
                GeneCategory.MITOCHONDRIAL: 0.9,
                GeneCategory.METABOLIC: 0.95,
            },
            DevelopmentalStage.ELDERLY: {
                GeneCategory.MITOCHONDRIAL: 0.7,  # Mitochondrial decline
                GeneCategory.TUBULIN: 0.8,
                GeneCategory.METABOLIC: 0.8,
                GeneCategory.ION_CHANNEL: 0.85,
            },
        }

        modifiers = stage_modifiers.get(self.developmental_stage, {})

        for gene in self.genes.values():
            if gene.category in modifiers:
                gene.expression_level *= modifiers[gene.category]
                gene.expression_level = np.clip(gene.expression_level, 0, 2)

    def _compute_coherence_constraints(self):
        """Compute coherence constraints from current genetic state."""
        # Aggregate expression by category
        self.category_expression = {}
        for cat in GeneCategory:
            genes_in_cat = [g for g in self.genes.values() if g.category == cat]
            if genes_in_cat:
                weighted_expr = sum(g.effective_expression * g.importance
                                   for g in genes_in_cat)
                total_importance = sum(g.importance for g in genes_in_cat)
                self.category_expression[cat] = weighted_expr / total_importance
            else:
                self.category_expression[cat] = 0.0

        # Map to coherence substrate parameters
        self._compute_substrate_constraints()

        # Compute viable window
        self._compute_viable_window()

    def _compute_substrate_constraints(self):
        """Compute constraints on each biological substrate."""
        # Microtubule coherence capacity
        self.microtubule_capacity = (
            self.category_expression.get(GeneCategory.TUBULIN, 0.5) * 0.7 +
            self.category_expression.get(GeneCategory.METABOLIC, 0.5) * 0.3
        )

        # Bioelectric pattern capacity
        self.bioelectric_capacity = (
            self.category_expression.get(GeneCategory.ION_CHANNEL, 0.5) * 0.5 +
            self.category_expression.get(GeneCategory.GAP_JUNCTION, 0.5) * 0.3 +
            self.category_expression.get(GeneCategory.SIGNALING, 0.5) * 0.2
        )

        # Biophoton emission capacity
        self.biophoton_capacity = (
            self.category_expression.get(GeneCategory.MITOCHONDRIAL, 0.5) * 0.7 +
            self.category_expression.get(GeneCategory.METABOLIC, 0.5) * 0.3
        )

        # Overall coherence capacity
        self.coherence_capacity = (
            self.microtubule_capacity * 0.35 +
            self.bioelectric_capacity * 0.35 +
            self.biophoton_capacity * 0.30
        )

    def _compute_viable_window(self):
        """Compute the viable window in éR phase space."""
        # Species determines base viable window size
        species_factors = {
            SpeciesComplexity.PROKARYOTE: 0.1,
            SpeciesComplexity.SIMPLE_EUKARYOTE: 0.2,
            SpeciesComplexity.INVERTEBRATE: 0.4,
            SpeciesComplexity.SIMPLE_VERTEBRATE: 0.5,
            SpeciesComplexity.MAMMAL: 0.7,
            SpeciesComplexity.PRIMATE: 0.85,
            SpeciesComplexity.HUMAN: 1.0,
        }

        base_size = species_factors.get(self.species, 0.5)

        # Genetic expression modulates window
        genetic_factor = self.coherence_capacity

        # Viable window boundaries
        # éR has viable region between "chaos" (too low) and "rigidity" (too high)
        self.viable_window = {
            'er_min': 0.1 / (base_size * genetic_factor + 0.1),
            'er_max': 10.0 * base_size * genetic_factor,
            'ep_min': 0.1,
            'ep_max': 10.0 * genetic_factor,
            'freq_min': 0.1,
            'freq_max': 10.0 * genetic_factor,
            'window_area': base_size * genetic_factor,
        }

    # =========================================================================
    # PI STACK DYNAMICS
    # =========================================================================

    def compute_pi_stack_coherence(self) -> float:
        """
        Compute coherence along the DNA pi stack.

        The aromatic bases can support coherent electron delocalization.
        GC pairs (3 H-bonds) are more stable than AT (2 H-bonds).

        Returns:
            Pi stack coherence (0-1)
        """
        # GC content affects stability
        gc_content = sum(1 for bp in self.sequence
                        if bp in [BasePair.GC, BasePair.CG]) / len(self.sequence)

        # Phase coherence (Kuramoto order parameter)
        complex_phases = np.exp(1j * self.pi_stack_phase)
        phase_coherence = np.abs(np.mean(complex_phases))

        # Amplitude factor
        amplitude_factor = np.mean(self.pi_stack_amplitude)

        # Combined coherence
        coherence = phase_coherence * amplitude_factor * (0.5 + 0.5 * gc_content)

        return coherence

    def step_pi_stack(self, n_steps: int = 1, dt: float = 1e-12):
        """
        Evolve pi stack dynamics.

        Parameters:
            n_steps: Number of time steps
            dt: Time step (seconds)
        """
        omega = 2 * np.pi * 1e12  # THz frequency for pi electrons

        for _ in range(n_steps):
            # Neighbor coupling
            coupling = np.zeros_like(self.pi_stack_phase)
            coupling[1:] += PI_STACK_COUPLING * np.sin(
                self.pi_stack_phase[:-1] - self.pi_stack_phase[1:]
            )
            coupling[:-1] += PI_STACK_COUPLING * np.sin(
                self.pi_stack_phase[1:] - self.pi_stack_phase[:-1]
            )

            # GC pairs couple more strongly
            for i, bp in enumerate(self.sequence):
                if bp in [BasePair.GC, BasePair.CG]:
                    coupling[i] *= 1.5

            # Update phases
            self.pi_stack_phase += dt * (omega + coupling)
            self.pi_stack_phase %= (2 * np.pi)

            # Thermal decoherence
            self.pi_stack_amplitude *= 0.9999
            self.pi_stack_amplitude += 0.0001 * np.random.randn(self.n_base_pairs)
            self.pi_stack_amplitude = np.clip(self.pi_stack_amplitude, 0, 1)

    # =========================================================================
    # GENE MANIPULATION
    # =========================================================================

    def set_gene_expression(self, gene_name: str, level: float):
        """
        Set expression level for a gene.

        Parameters:
            gene_name: Name of gene
            level: Expression level (0-2)
        """
        if gene_name in self.genes:
            self.genes[gene_name].expression_level = np.clip(level, 0, 2)
            self._compute_coherence_constraints()

    def set_gene_methylation(self, gene_name: str, level: float):
        """
        Set methylation (epigenetic silencing) for a gene.

        Parameters:
            gene_name: Name of gene
            level: Methylation level (0-1)
        """
        if gene_name in self.genes:
            self.genes[gene_name].methylation = np.clip(level, 0, 1)
            self._compute_coherence_constraints()

    def apply_global_methylation(self, category: GeneCategory, level: float):
        """Apply methylation to all genes in a category."""
        for gene in self.genes.values():
            if gene.category == category:
                gene.methylation = np.clip(level, 0, 1)
        self._compute_coherence_constraints()

    def apply_environmental_stress(self, stress_level: float):
        """
        Apply environmental stress affecting epigenetics.

        Stress increases methylation of metabolic and mitochondrial genes,
        and can shift the viable window.

        Parameters:
            stress_level: 0-1 stress intensity
        """
        # Stress affects mitochondrial genes most
        self.apply_global_methylation(GeneCategory.MITOCHONDRIAL, stress_level * 0.5)
        self.apply_global_methylation(GeneCategory.METABOLIC, stress_level * 0.3)

        # Also affects signaling
        self.apply_global_methylation(GeneCategory.SIGNALING, stress_level * 0.2)

    def set_developmental_stage(self, stage: DevelopmentalStage):
        """Change developmental stage and recompute constraints."""
        self.developmental_stage = stage

        # Reset gene expression to defaults then apply stage effects
        for name, default_gene in DEFAULT_GENES.items():
            if name in self.genes:
                self.genes[name].expression_level = default_gene.expression_level

        self._apply_developmental_effects()
        self._compute_coherence_constraints()

    def set_species(self, species: SpeciesComplexity):
        """Change species and recompute viable window."""
        self.species = species
        self._compute_coherence_constraints()

    # =========================================================================
    # SUBSTRATE PARAMETER OUTPUTS
    # =========================================================================

    def get_microtubule_parameters(self) -> Dict:
        """
        Get parameters for microtubule simulation.

        Returns:
            Dictionary of microtubule-relevant parameters
        """
        tubulin_expr = self.category_expression.get(GeneCategory.TUBULIN, 0.5)

        return {
            'coupling_strength': 0.2 + 0.4 * tubulin_expr,
            'coherence_time_factor': 0.5 + 0.5 * tubulin_expr,
            'aromatic_activity': tubulin_expr,
            'decoherence_rate': 1.0 / (0.5 + 0.5 * tubulin_expr),
        }

    def get_bioelectric_parameters(self) -> Dict:
        """
        Get parameters for bioelectric simulation.

        Returns:
            Dictionary of bioelectric-relevant parameters
        """
        ion_expr = self.category_expression.get(GeneCategory.ION_CHANNEL, 0.5)
        gap_expr = self.category_expression.get(GeneCategory.GAP_JUNCTION, 0.5)

        return {
            'g_Na_factor': 0.5 + 0.5 * ion_expr,
            'g_K_factor': 0.5 + 0.5 * ion_expr,
            'gap_conductance_factor': 0.5 + 0.5 * gap_expr,
            'excitability': ion_expr,
        }

    def get_biophoton_parameters(self) -> Dict:
        """
        Get parameters for biophoton simulation.

        Returns:
            Dictionary of biophoton-relevant parameters
        """
        mito_expr = self.category_expression.get(GeneCategory.MITOCHONDRIAL, 0.5)
        metab_expr = self.category_expression.get(GeneCategory.METABOLIC, 0.5)

        return {
            'emission_rate_factor': 0.5 + 0.5 * mito_expr,
            'atp_production': metab_expr,
            'ros_baseline': 0.1 + 0.1 * (1.0 - mito_expr),  # Lower mito = higher ROS
            'coherence_coupling': 0.3 + 0.4 * mito_expr,
        }

    # =========================================================================
    # ÉR PHASE SPACE MAPPING
    # =========================================================================

    def map_to_er_space(self) -> Dict:
        """
        Map current genetic state to éR phase space.

        Returns:
            Dictionary with viable window and current state
        """
        pi_coherence = self.compute_pi_stack_coherence()

        return {
            # Viable window boundaries
            'viable_window': self.viable_window.copy(),

            # Current genetic capacity
            'coherence_capacity': self.coherence_capacity,
            'microtubule_capacity': self.microtubule_capacity,
            'bioelectric_capacity': self.bioelectric_capacity,
            'biophoton_capacity': self.biophoton_capacity,

            # Pi stack state
            'pi_stack_coherence': pi_coherence,

            # Species and development
            'species': self.species.value,
            'developmental_stage': self.developmental_stage.value,

            # Gene expression summary
            'mean_expression': np.mean([g.effective_expression
                                        for g in self.genes.values()]),
            'mean_methylation': np.mean([g.methylation
                                         for g in self.genes.values()]),
        }

    def get_complete_substrate_state(self) -> Dict:
        """
        Get complete state for all substrates.

        This shows how DNA constrains all other CLT substrates.
        """
        return {
            'microtubule': self.get_microtubule_parameters(),
            'bioelectric': self.get_bioelectric_parameters(),
            'biophoton': self.get_biophoton_parameters(),
            'viable_window': self.viable_window.copy(),
            'coherence_capacity': self.coherence_capacity,
        }

    def reset(self):
        """Reset to default state."""
        self.genes = {k: Gene(g.name, g.category, g.expression_level,
                              g.methylation, g.importance)
                     for k, g in DEFAULT_GENES.items()}
        self.developmental_stage = DevelopmentalStage.ADULT
        self._apply_developmental_effects()
        self._compute_coherence_constraints()
        self._init_dna_sequence()


# =============================================================================
# VISUALIZER
# =============================================================================

class DNAConstraintVisualizer:
    """
    Interactive visualization for DNA constraint simulation.

    Features:
    - DNA helix with base pairs
    - Gene expression panel
    - Epigenetic modification overlay
    - Viable window in éR phase space
    - Substrate parameter outputs
    """

    def __init__(self, simulator: DNAConstraintSimulator):
        self.sim = simulator
        self.running = False

        # Create figure
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('DNA Constraint Simulator - CLT Phase 2.4',
                         fontsize=14, fontweight='bold')

        gs = GridSpec(3, 4, figure=self.fig, height_ratios=[2, 1, 1],
                     hspace=0.35, wspace=0.3)

        # DNA helix / Pi stack display
        self.ax_dna = self.fig.add_subplot(gs[0, 0])
        self.ax_dna.set_title('DNA Pi Stack')

        # Gene expression bars
        self.ax_genes = self.fig.add_subplot(gs[0, 1])
        self.ax_genes.set_title('Gene Expression')

        # Viable window in éR space
        self.ax_er = self.fig.add_subplot(gs[0, 2])
        self.ax_er.set_title('Viable Window (éR Phase Space)')

        # Substrate capacities
        self.ax_substrates = self.fig.add_subplot(gs[0, 3])
        self.ax_substrates.set_title('Substrate Capacities')

        # Category expression
        self.ax_categories = self.fig.add_subplot(gs[1, :2])
        self.ax_categories.set_title('Expression by Category')

        # Info display
        self.ax_info = self.fig.add_subplot(gs[1, 2:])
        self.ax_info.set_title('State Summary')
        self.ax_info.axis('off')

        # Controls
        self.ax_controls = self.fig.add_subplot(gs[2, :])
        self.ax_controls.axis('off')

        # Initialize plots
        self._init_plots()
        self._add_controls()

        # Animation
        self.anim = None

    def _init_plots(self):
        """Initialize all plot elements."""
        # DNA pi stack (show as color-coded line)
        self.pi_stack_line = self.ax_dna.imshow(
            np.zeros((10, self.sim.n_base_pairs)),
            cmap='twilight',
            aspect='auto',
            vmin=0, vmax=2*np.pi
        )
        self.ax_dna.set_xlabel('Base Pair Position')
        self.ax_dna.set_ylabel('Pi Electron Phase')

        # Gene expression bars
        gene_names = list(self.sim.genes.keys())[:10]  # Show first 10
        self.gene_bars = self.ax_genes.barh(
            range(len(gene_names)),
            [self.sim.genes[g].effective_expression for g in gene_names],
            color='steelblue'
        )
        self.ax_genes.set_yticks(range(len(gene_names)))
        self.ax_genes.set_yticklabels(gene_names, fontsize=8)
        self.ax_genes.set_xlim(0, 1.5)
        self.ax_genes.set_xlabel('Expression')

        # Viable window (placeholder)
        self.ax_er.set_xlim(0, 12)
        self.ax_er.set_ylim(0, 12)
        self.ax_er.set_xlabel('Frequency (Hz)')
        self.ax_er.set_ylabel('Energy Present')
        self._draw_viable_window()

        # Substrate capacities
        substrates = ['Microtubule', 'Bioelectric', 'Biophoton', 'Overall']
        capacities = [
            self.sim.microtubule_capacity,
            self.sim.bioelectric_capacity,
            self.sim.biophoton_capacity,
            self.sim.coherence_capacity
        ]
        colors = ['#FFFF66', '#66FF66', '#66FFFF', '#FF66FF']
        self.substrate_bars = self.ax_substrates.bar(
            substrates, capacities, color=colors
        )
        self.ax_substrates.set_ylim(0, 1.5)
        self.ax_substrates.set_ylabel('Capacity')
        self.ax_substrates.tick_params(axis='x', rotation=45)

        # Category expression
        categories = [cat.value for cat in GeneCategory]
        expressions = [self.sim.category_expression.get(cat, 0)
                      for cat in GeneCategory]
        self.category_bars = self.ax_categories.bar(
            categories, expressions,
            color=['#FF6666', '#66FF66', '#6666FF', '#FFFF66',
                   '#FF66FF', '#66FFFF', '#FFAA66']
        )
        self.ax_categories.set_ylim(0, 1.5)
        self.ax_categories.tick_params(axis='x', rotation=45)
        self.ax_categories.set_ylabel('Mean Expression')

        # Info text
        self.info_text = self.ax_info.text(
            0.05, 0.95, '', transform=self.ax_info.transAxes,
            fontsize=9, family='monospace', verticalalignment='top'
        )

    def _draw_viable_window(self):
        """Draw the viable window in éR space."""
        self.ax_er.clear()
        self.ax_er.set_title('Viable Window (éR Phase Space)')
        self.ax_er.set_xlabel('Frequency (Hz)')
        self.ax_er.set_ylabel('Energy Present')
        self.ax_er.set_xlim(0, 12)
        self.ax_er.set_ylim(0, 12)

        vw = self.sim.viable_window

        # Draw chaos region (low éR)
        self.ax_er.fill_between([0, 12], [0, 0], [1, 1],
                                color='red', alpha=0.2, label='Chaos')

        # Draw viable region
        viable_patch = plt.Rectangle(
            (vw['freq_min'], vw['ep_min']),
            vw['freq_max'] - vw['freq_min'],
            vw['ep_max'] - vw['ep_min'],
            facecolor='green', alpha=0.3, edgecolor='green',
            linewidth=2, label='Viable'
        )
        self.ax_er.add_patch(viable_patch)

        # Draw rigidity region (high éR)
        self.ax_er.fill_between([0, 12], [10, 10], [12, 12],
                               color='blue', alpha=0.2, label='Rigidity')

        # Draw éR contours
        freq = np.linspace(0.5, 11, 50)
        for er_val in [0.1, 1.0, 10.0]:
            ep = er_val * freq ** 2
            self.ax_er.plot(freq, ep, 'k--', alpha=0.3, linewidth=0.5)

        self.ax_er.legend(loc='upper left', fontsize=8)

    def _add_controls(self):
        """Add interactive controls."""
        left = 0.08

        # Developmental stage selector
        ax_stage = plt.axes([left, 0.12, 0.12, 0.12])
        stages = ['Embryonic', 'Infant', 'Child', 'Adult', 'Elderly']
        self.radio_stage = RadioButtons(ax_stage, stages, active=3)
        self.radio_stage.on_clicked(self._on_stage_change)

        # Species selector
        ax_species = plt.axes([left + 0.14, 0.12, 0.12, 0.12])
        species_labels = ['Invertebrate', 'Mammal', 'Primate', 'Human']
        self.radio_species = RadioButtons(ax_species, species_labels, active=3)
        self.radio_species.on_clicked(self._on_species_change)

        # Stress slider
        ax_stress = plt.axes([left + 0.30, 0.22, 0.15, 0.02])
        self.slider_stress = Slider(ax_stress, 'Stress', 0.0, 1.0, valinit=0.0)
        self.slider_stress.on_changed(self._on_stress_change)

        # Methylation slider
        ax_methyl = plt.axes([left + 0.30, 0.18, 0.15, 0.02])
        self.slider_methyl = Slider(ax_methyl, 'Methylation', 0.0, 1.0, valinit=0.0)
        self.slider_methyl.on_changed(self._on_methyl_change)

        # Buttons
        ax_reset = plt.axes([left + 0.50, 0.20, 0.06, 0.03])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._on_reset_click)

        ax_pause = plt.axes([left + 0.58, 0.20, 0.06, 0.03])
        self.btn_pause = Button(ax_pause, 'Animate')
        self.btn_pause.on_clicked(self._on_pause_click)

        # Presets
        ax_preset = plt.axes([left + 0.68, 0.12, 0.12, 0.12])
        presets = ['Human Baseline', 'High Plasticity', 'Stressed/Aging', 'Simple Organism']
        self.radio_preset = RadioButtons(ax_preset, presets, active=0)
        self.radio_preset.on_clicked(self._on_preset_change)

        # Gene category checkboxes
        ax_cats = plt.axes([left + 0.82, 0.12, 0.12, 0.12])
        cat_labels = ['Tubulin', 'Ion Chan', 'Mito', 'Gap Junc']
        self.check_cats = CheckButtons(ax_cats, cat_labels, [True]*4)
        self.check_cats.on_clicked(self._on_category_toggle)

    def _on_stage_change(self, label):
        """Handle developmental stage change."""
        stage_map = {
            'Embryonic': DevelopmentalStage.EMBRYONIC,
            'Infant': DevelopmentalStage.INFANT,
            'Child': DevelopmentalStage.CHILD,
            'Adult': DevelopmentalStage.ADULT,
            'Elderly': DevelopmentalStage.ELDERLY,
        }
        self.sim.set_developmental_stage(stage_map.get(label, DevelopmentalStage.ADULT))
        self._update_plots()

    def _on_species_change(self, label):
        """Handle species change."""
        species_map = {
            'Invertebrate': SpeciesComplexity.INVERTEBRATE,
            'Mammal': SpeciesComplexity.MAMMAL,
            'Primate': SpeciesComplexity.PRIMATE,
            'Human': SpeciesComplexity.HUMAN,
        }
        self.sim.set_species(species_map.get(label, SpeciesComplexity.HUMAN))
        self._update_plots()

    def _on_stress_change(self, val):
        """Handle stress level change."""
        self.sim.apply_environmental_stress(val)
        self._update_plots()

    def _on_methyl_change(self, val):
        """Handle global methylation change."""
        for gene in self.sim.genes.values():
            gene.methylation = val
        self.sim._compute_coherence_constraints()
        self._update_plots()

    def _on_reset_click(self, event):
        """Reset simulation."""
        self.sim.reset()
        self.slider_stress.set_val(0)
        self.slider_methyl.set_val(0)
        self._update_plots()

    def _on_pause_click(self, event):
        """Toggle animation."""
        self.running = not self.running
        self.btn_pause.label.set_text('Stop' if self.running else 'Animate')

    def _on_preset_change(self, label):
        """Apply preset configuration."""
        self.sim.reset()

        if label == 'Human Baseline':
            pass  # Default state

        elif label == 'High Plasticity':
            # Enhanced expression of plasticity genes
            self.sim.set_gene_expression('TUBB3', 1.5)
            self.sim.set_gene_expression('SCN1A', 1.3)
            self.sim.set_gene_expression('CAMK2A', 1.4)
            self.sim.set_gene_expression('GJA1', 1.3)
            self.sim._compute_coherence_constraints()

        elif label == 'Stressed/Aging':
            self.sim.set_developmental_stage(DevelopmentalStage.ELDERLY)
            self.sim.apply_environmental_stress(0.6)
            self.slider_stress.set_val(0.6)

        elif label == 'Simple Organism':
            self.sim.set_species(SpeciesComplexity.INVERTEBRATE)
            # Lower expression across the board
            for gene in self.sim.genes.values():
                gene.expression_level *= 0.6
            self.sim._compute_coherence_constraints()

        self._update_plots()

    def _on_category_toggle(self, label):
        """Toggle gene category expression."""
        cat_map = {
            'Tubulin': GeneCategory.TUBULIN,
            'Ion Chan': GeneCategory.ION_CHANNEL,
            'Mito': GeneCategory.MITOCHONDRIAL,
            'Gap Junc': GeneCategory.GAP_JUNCTION,
        }
        cat = cat_map.get(label)
        if cat:
            # Toggle between high and low expression
            current = self.sim.category_expression.get(cat, 0.5)
            new_level = 0.2 if current > 0.5 else 1.0
            for gene in self.sim.genes.values():
                if gene.category == cat:
                    gene.expression_level = new_level
            self.sim._compute_coherence_constraints()
            self._update_plots()

    def _update_plots(self):
        """Update all plots with current state."""
        # Update pi stack display
        phase_display = np.tile(self.sim.pi_stack_phase, (10, 1))
        self.pi_stack_line.set_array(phase_display)

        # Update gene bars
        gene_names = list(self.sim.genes.keys())[:10]
        for i, (bar, name) in enumerate(zip(self.gene_bars, gene_names)):
            bar.set_width(self.sim.genes[name].effective_expression)
            # Color by methylation
            meth = self.sim.genes[name].methylation
            bar.set_color(plt.cm.RdYlGn(1.0 - meth))

        # Update viable window
        self._draw_viable_window()

        # Update substrate bars
        capacities = [
            self.sim.microtubule_capacity,
            self.sim.bioelectric_capacity,
            self.sim.biophoton_capacity,
            self.sim.coherence_capacity
        ]
        for bar, cap in zip(self.substrate_bars, capacities):
            bar.set_height(cap)

        # Update category bars
        for bar, cat in zip(self.category_bars, GeneCategory):
            bar.set_height(self.sim.category_expression.get(cat, 0))

        # Update info text
        er = self.sim.map_to_er_space()
        info_str = (
            f"=== DNA Constraint State ===\n"
            f"Species: {er['species']}\n"
            f"Stage: {er['developmental_stage']}\n"
            f"\nCoherence Capacities:\n"
            f"  Microtubule: {er['microtubule_capacity']:.3f}\n"
            f"  Bioelectric: {er['bioelectric_capacity']:.3f}\n"
            f"  Biophoton:   {er['biophoton_capacity']:.3f}\n"
            f"  Overall:     {er['coherence_capacity']:.3f}\n"
            f"\nViable Window:\n"
            f"  éR range: [{er['viable_window']['er_min']:.2f}, "
            f"{er['viable_window']['er_max']:.2f}]\n"
            f"  Area: {er['viable_window']['window_area']:.3f}\n"
            f"\nPi Stack Coherence: {er['pi_stack_coherence']:.3f}\n"
            f"\nGene Expression:\n"
            f"  Mean: {er['mean_expression']:.3f}\n"
            f"  Methylation: {er['mean_methylation']:.3f}"
        )
        self.info_text.set_text(info_str)

        self.fig.canvas.draw_idle()

    def _update(self, frame):
        """Animation update."""
        if self.running:
            # Evolve pi stack
            self.sim.step_pi_stack(100)
            self._update_plots()

        return []

    def run(self, interval: int = 100):
        """Start the interactive visualization."""
        self._update_plots()
        self.anim = animation.FuncAnimation(
            self.fig, self._update, interval=interval, blit=False,
            cache_frame_data=False
        )
        plt.show()


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

def create_human_baseline() -> DNAConstraintSimulator:
    """Create simulator with typical human gene expression."""
    return DNAConstraintSimulator(
        species=SpeciesComplexity.HUMAN,
        developmental_stage=DevelopmentalStage.ADULT
    )


def create_high_plasticity() -> DNAConstraintSimulator:
    """
    Create simulator with enhanced neuroplasticity genes.

    Models a state of high cognitive flexibility.
    """
    sim = DNAConstraintSimulator(
        species=SpeciesComplexity.HUMAN,
        developmental_stage=DevelopmentalStage.CHILD  # Natural high plasticity
    )

    # Enhance plasticity-related genes
    sim.set_gene_expression('TUBB3', 1.4)
    sim.set_gene_expression('SCN1A', 1.3)
    sim.set_gene_expression('CAMK2A', 1.5)
    sim.set_gene_expression('GJA1', 1.3)

    return sim


def create_stressed_aging() -> DNAConstraintSimulator:
    """
    Create simulator modeling stressed/aging state.

    Shows how stress and aging narrow the viable window.
    """
    sim = DNAConstraintSimulator(
        species=SpeciesComplexity.HUMAN,
        developmental_stage=DevelopmentalStage.ELDERLY
    )

    sim.apply_environmental_stress(0.5)

    return sim


def create_developmental_series() -> List[DNAConstraintSimulator]:
    """
    Create series of simulators across development.

    Returns:
        List of simulators from embryo to elderly
    """
    stages = [
        DevelopmentalStage.EMBRYONIC,
        DevelopmentalStage.INFANT,
        DevelopmentalStage.CHILD,
        DevelopmentalStage.ADOLESCENT,
        DevelopmentalStage.ADULT,
        DevelopmentalStage.MIDDLE_AGE,
        DevelopmentalStage.ELDERLY,
    ]

    return [DNAConstraintSimulator(developmental_stage=stage) for stage in stages]


def create_cross_species() -> Dict[str, DNAConstraintSimulator]:
    """
    Create simulators for different species.

    Returns:
        Dictionary of species name → simulator
    """
    species_list = [
        SpeciesComplexity.INVERTEBRATE,
        SpeciesComplexity.SIMPLE_VERTEBRATE,
        SpeciesComplexity.MAMMAL,
        SpeciesComplexity.PRIMATE,
        SpeciesComplexity.HUMAN,
    ]

    return {sp.value: DNAConstraintSimulator(species=sp) for sp in species_list}


def create_meditation_epigenetics() -> DNAConstraintSimulator:
    """
    Create simulator modeling meditation-induced epigenetic changes.

    Research shows meditation can affect gene expression,
    particularly stress-related genes.
    """
    sim = DNAConstraintSimulator(
        species=SpeciesComplexity.HUMAN,
        developmental_stage=DevelopmentalStage.ADULT
    )

    # Meditation effects: reduced stress methylation
    for gene in sim.genes.values():
        if gene.category in [GeneCategory.MITOCHONDRIAL, GeneCategory.METABOLIC]:
            gene.methylation = max(0, gene.methylation - 0.2)

    # Enhanced signaling genes
    sim.set_gene_expression('CAMK2A', 1.2)

    sim._compute_coherence_constraints()
    return sim


# =============================================================================
# DEMO FUNCTIONS
# =============================================================================

def demo():
    """Run the interactive DNA constraint visualizer."""
    sim = create_human_baseline()
    viz = DNAConstraintVisualizer(sim)
    viz.run()


def demo_development():
    """Demo showing developmental changes."""
    sim = DNAConstraintSimulator(developmental_stage=DevelopmentalStage.INFANT)
    viz = DNAConstraintVisualizer(sim)
    viz.run()


def demo_aging():
    """Demo showing aging effects."""
    sim = create_stressed_aging()
    viz = DNAConstraintVisualizer(sim)
    viz.run()


def demo_comparison():
    """
    Static comparison of different genetic states.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('DNA Constraints Across States', fontsize=14, fontweight='bold')

    configs = [
        ('Human Adult', create_human_baseline),
        ('High Plasticity', create_high_plasticity),
        ('Stressed/Aging', create_stressed_aging),
        ('Meditation', create_meditation_epigenetics),
    ]

    for i, (name, create_fn) in enumerate(configs):
        sim = create_fn()

        # Viable window
        ax = axes[0, i]
        vw = sim.viable_window

        ax.fill_between([0, 10], [0, 0], [1, 1], color='red', alpha=0.2)
        rect = plt.Rectangle(
            (vw['freq_min'], vw['ep_min']),
            vw['freq_max'] - vw['freq_min'],
            vw['ep_max'] - vw['ep_min'],
            facecolor='green', alpha=0.4, edgecolor='green', linewidth=2
        )
        ax.add_patch(rect)
        ax.fill_between([0, 10], [9, 9], [10, 10], color='blue', alpha=0.2)

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('EP')

        # Substrate capacities
        ax2 = axes[1, i]
        substrates = ['MT', 'Bio-E', 'Photon', 'Total']
        capacities = [
            sim.microtubule_capacity,
            sim.bioelectric_capacity,
            sim.biophoton_capacity,
            sim.coherence_capacity
        ]
        colors = ['#FFFF66', '#66FF66', '#66FFFF', '#FF66FF']
        ax2.bar(substrates, capacities, color=colors)
        ax2.set_ylim(0, 1.2)
        ax2.set_ylabel('Capacity')

    plt.tight_layout()
    plt.show()


def demo_species_comparison():
    """Compare viable windows across species."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    fig.suptitle('Viable Windows Across Species', fontsize=14, fontweight='bold')

    species_sims = create_cross_species()

    for ax, (species_name, sim) in zip(axes, species_sims.items()):
        vw = sim.viable_window

        # Draw viable window
        ax.fill_between([0, 10], [0, 0], [1, 1], color='red', alpha=0.2)
        rect = plt.Rectangle(
            (vw['freq_min'], vw['ep_min']),
            vw['freq_max'] - vw['freq_min'],
            vw['ep_max'] - vw['ep_min'],
            facecolor='green', alpha=0.4, edgecolor='green', linewidth=2
        )
        ax.add_patch(rect)
        ax.fill_between([0, 10], [9, 9], [10, 10], color='blue', alpha=0.2)

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_title(f'{species_name}\nArea: {vw["window_area"]:.2f}', fontsize=10)
        ax.set_xlabel('f')
        ax.set_ylabel('EP')

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print("DNA Constraint Simulator - CLT Phase 2.4")
    print("=" * 50)
    print("\nModeling how DNA constrains Loomfield topology")
    print("\nAvailable demos:")
    print("  1. demo()              - Interactive human baseline")
    print("  2. demo_development()  - Developmental stage effects")
    print("  3. demo_aging()        - Aging/stress effects")
    print("  4. demo_comparison()   - Compare genetic states")
    print("  5. demo_species_comparison() - Cross-species comparison")
    print("\nStarting interactive demo...")
    demo()
