"""
Tests for DNA Constraint Simulator.

Tests genetic constraints, epigenetic modulation, species-specific viable
windows, developmental dynamics, and pi stack quantum coherence.
"""

import pytest
import numpy as np


class TestEnums:
    """Test enum definitions."""

    def test_base_pair_enum(self):
        """Test BasePair enum exists with all base pairs."""
        from simulations.field_dynamics.dna_constraints import BasePair

        assert hasattr(BasePair, 'AT')
        assert hasattr(BasePair, 'TA')
        assert hasattr(BasePair, 'GC')
        assert hasattr(BasePair, 'CG')

    def test_gene_category_enum(self):
        """Test GeneCategory enum with all categories."""
        from simulations.field_dynamics.dna_constraints import GeneCategory

        assert hasattr(GeneCategory, 'TUBULIN')
        assert hasattr(GeneCategory, 'ION_CHANNEL')
        assert hasattr(GeneCategory, 'MITOCHONDRIAL')
        assert hasattr(GeneCategory, 'GAP_JUNCTION')
        assert hasattr(GeneCategory, 'SIGNALING')
        assert hasattr(GeneCategory, 'METABOLIC')
        assert hasattr(GeneCategory, 'STRUCTURAL')

    def test_developmental_stage_enum(self):
        """Test DevelopmentalStage enum with all stages."""
        from simulations.field_dynamics.dna_constraints import DevelopmentalStage

        assert hasattr(DevelopmentalStage, 'EMBRYONIC')
        assert hasattr(DevelopmentalStage, 'FETAL')
        assert hasattr(DevelopmentalStage, 'INFANT')
        assert hasattr(DevelopmentalStage, 'CHILD')
        assert hasattr(DevelopmentalStage, 'ADOLESCENT')
        assert hasattr(DevelopmentalStage, 'ADULT')
        assert hasattr(DevelopmentalStage, 'MIDDLE_AGE')
        assert hasattr(DevelopmentalStage, 'ELDERLY')

    def test_species_complexity_enum(self):
        """Test SpeciesComplexity enum with all levels."""
        from simulations.field_dynamics.dna_constraints import SpeciesComplexity

        assert hasattr(SpeciesComplexity, 'PROKARYOTE')
        assert hasattr(SpeciesComplexity, 'SIMPLE_EUKARYOTE')
        assert hasattr(SpeciesComplexity, 'INVERTEBRATE')
        assert hasattr(SpeciesComplexity, 'SIMPLE_VERTEBRATE')
        assert hasattr(SpeciesComplexity, 'MAMMAL')
        assert hasattr(SpeciesComplexity, 'PRIMATE')
        assert hasattr(SpeciesComplexity, 'HUMAN')


class TestGeneDataclass:
    """Test Gene dataclass."""

    def test_gene_creation(self):
        """Test creating a Gene."""
        from simulations.field_dynamics.dna_constraints import Gene, GeneCategory

        gene = Gene(
            name='TEST1',
            category=GeneCategory.TUBULIN,
            expression_level=0.8,
            methylation=0.2,
            importance=1.0
        )

        assert gene.name == 'TEST1'
        assert gene.category == GeneCategory.TUBULIN
        assert gene.expression_level == 0.8
        assert gene.methylation == 0.2
        assert gene.importance == 1.0

    def test_gene_default_values(self):
        """Test Gene default values."""
        from simulations.field_dynamics.dna_constraints import Gene, GeneCategory

        gene = Gene(name='TEST2', category=GeneCategory.METABOLIC)

        assert gene.expression_level == 1.0
        assert gene.methylation == 0.0
        assert gene.importance == 1.0

    def test_effective_expression_no_methylation(self):
        """Test effective expression with no methylation."""
        from simulations.field_dynamics.dna_constraints import Gene, GeneCategory

        gene = Gene(
            name='TEST3',
            category=GeneCategory.ION_CHANNEL,
            expression_level=1.0,
            methylation=0.0
        )

        assert gene.effective_expression == 1.0

    def test_effective_expression_with_methylation(self):
        """Test effective expression is reduced by methylation."""
        from simulations.field_dynamics.dna_constraints import Gene, GeneCategory

        gene = Gene(
            name='TEST4',
            category=GeneCategory.ION_CHANNEL,
            expression_level=1.0,
            methylation=0.5
        )

        # methylation of 0.5 should reduce expression
        assert gene.effective_expression < 1.0
        assert gene.effective_expression > 0.5  # Not completely silenced

    def test_effective_expression_full_methylation(self):
        """Test full methylation significantly reduces expression."""
        from simulations.field_dynamics.dna_constraints import Gene, GeneCategory

        gene = Gene(
            name='TEST5',
            category=GeneCategory.MITOCHONDRIAL,
            expression_level=1.0,
            methylation=1.0
        )

        # Full methylation (90% silencing)
        assert np.isclose(gene.effective_expression, 0.1)


class TestDNAConstraintSimulatorBasics:
    """Test basic DNAConstraintSimulator functionality."""

    def test_simulator_import(self):
        """Test that simulator can be imported."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator
        assert DNAConstraintSimulator is not None

    def test_simulator_initialization(self):
        """Test simulator initializes correctly."""
        from simulations.field_dynamics.dna_constraints import (
            DNAConstraintSimulator, SpeciesComplexity, DevelopmentalStage
        )

        sim = DNAConstraintSimulator()

        assert sim.species == SpeciesComplexity.HUMAN
        assert sim.developmental_stage == DevelopmentalStage.ADULT
        assert len(sim.genes) > 0
        assert sim.n_base_pairs == 100

    def test_custom_base_pairs(self):
        """Test custom number of base pairs."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator(base_pairs=50)

        assert sim.n_base_pairs == 50
        assert len(sim.sequence) == 50
        assert len(sim.pi_stack_phase) == 50

    def test_custom_species(self):
        """Test custom species setting."""
        from simulations.field_dynamics.dna_constraints import (
            DNAConstraintSimulator, SpeciesComplexity
        )

        sim = DNAConstraintSimulator(species=SpeciesComplexity.MAMMAL)

        assert sim.species == SpeciesComplexity.MAMMAL

    def test_custom_developmental_stage(self):
        """Test custom developmental stage."""
        from simulations.field_dynamics.dna_constraints import (
            DNAConstraintSimulator, DevelopmentalStage
        )

        sim = DNAConstraintSimulator(developmental_stage=DevelopmentalStage.CHILD)

        assert sim.developmental_stage == DevelopmentalStage.CHILD

    def test_default_genes_loaded(self):
        """Test default genes are loaded."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator()

        # Check some key genes exist
        assert 'TUBA1A' in sim.genes
        assert 'SCN1A' in sim.genes
        assert 'MT-ND1' in sim.genes
        assert 'GJA1' in sim.genes


class TestPiStackDynamics:
    """Test pi stack quantum coherence model."""

    def test_pi_stack_coherence_range(self):
        """Test pi stack coherence is in valid range."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator(base_pairs=100)

        coherence = sim.compute_pi_stack_coherence()

        assert 0 <= coherence <= 1

    def test_pi_stack_step(self):
        """Test stepping pi stack dynamics."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator(base_pairs=50)
        initial_amplitude = sim.pi_stack_amplitude.copy()

        # Use enough steps to see amplitude decoherence effects
        sim.step_pi_stack(n_steps=100)

        # Amplitudes should have changed due to decoherence
        assert not np.allclose(sim.pi_stack_amplitude, initial_amplitude)

    def test_pi_stack_phase_wrapping(self):
        """Test phases wrap around 2*pi."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator(base_pairs=50)
        sim.step_pi_stack(n_steps=1000)

        # All phases should be in [0, 2*pi)
        assert np.all(sim.pi_stack_phase >= 0)
        assert np.all(sim.pi_stack_phase < 2*np.pi)

    def test_pi_stack_decoherence(self):
        """Test thermal decoherence affects amplitudes."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator(base_pairs=50)
        sim.pi_stack_amplitude = np.ones(50)

        sim.step_pi_stack(n_steps=100)

        # Amplitudes should be bounded
        assert np.all(sim.pi_stack_amplitude >= 0)
        assert np.all(sim.pi_stack_amplitude <= 1)


class TestGeneManipulation:
    """Test gene expression manipulation."""

    def test_set_gene_expression(self):
        """Test setting gene expression level."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator()

        sim.set_gene_expression('TUBA1A', 1.5)

        assert sim.genes['TUBA1A'].expression_level == 1.5

    def test_set_gene_expression_clipping(self):
        """Test gene expression is clipped to valid range."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator()

        sim.set_gene_expression('TUBA1A', 5.0)  # Too high
        assert sim.genes['TUBA1A'].expression_level == 2.0  # Clipped

        sim.set_gene_expression('TUBA1A', -1.0)  # Too low
        assert sim.genes['TUBA1A'].expression_level == 0.0  # Clipped

    def test_set_gene_methylation(self):
        """Test setting gene methylation level."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator()

        sim.set_gene_methylation('SCN1A', 0.5)

        assert sim.genes['SCN1A'].methylation == 0.5

    def test_set_gene_methylation_clipping(self):
        """Test methylation is clipped to [0, 1]."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator()

        sim.set_gene_methylation('SCN1A', 1.5)
        assert sim.genes['SCN1A'].methylation == 1.0

        sim.set_gene_methylation('SCN1A', -0.5)
        assert sim.genes['SCN1A'].methylation == 0.0

    def test_apply_global_methylation(self):
        """Test applying methylation to gene category."""
        from simulations.field_dynamics.dna_constraints import (
            DNAConstraintSimulator, GeneCategory
        )

        sim = DNAConstraintSimulator()

        sim.apply_global_methylation(GeneCategory.TUBULIN, 0.5)

        # All tubulin genes should have 0.5 methylation
        for gene in sim.genes.values():
            if gene.category == GeneCategory.TUBULIN:
                assert gene.methylation == 0.5

    def test_apply_environmental_stress(self):
        """Test environmental stress affects genes."""
        from simulations.field_dynamics.dna_constraints import (
            DNAConstraintSimulator, GeneCategory
        )

        sim = DNAConstraintSimulator()

        sim.apply_environmental_stress(stress_level=1.0)

        # Mitochondrial genes should be most affected
        for gene in sim.genes.values():
            if gene.category == GeneCategory.MITOCHONDRIAL:
                assert gene.methylation > 0


class TestDevelopmentalDynamics:
    """Test developmental stage effects."""

    def test_set_developmental_stage(self):
        """Test changing developmental stage."""
        from simulations.field_dynamics.dna_constraints import (
            DNAConstraintSimulator, DevelopmentalStage
        )

        sim = DNAConstraintSimulator()

        sim.set_developmental_stage(DevelopmentalStage.ELDERLY)

        assert sim.developmental_stage == DevelopmentalStage.ELDERLY

    def test_elderly_reduces_capacity(self):
        """Test elderly stage reduces coherence capacity."""
        from simulations.field_dynamics.dna_constraints import (
            DNAConstraintSimulator, DevelopmentalStage
        )

        sim_adult = DNAConstraintSimulator(developmental_stage=DevelopmentalStage.ADULT)
        sim_elderly = DNAConstraintSimulator(developmental_stage=DevelopmentalStage.ELDERLY)

        # Elderly should have lower coherence capacity
        assert sim_elderly.coherence_capacity < sim_adult.coherence_capacity

    def test_embryonic_tubulin_boost(self):
        """Test embryonic stage has higher tubulin expression."""
        from simulations.field_dynamics.dna_constraints import (
            DNAConstraintSimulator, DevelopmentalStage, GeneCategory
        )

        sim_adult = DNAConstraintSimulator(developmental_stage=DevelopmentalStage.ADULT)
        sim_embryonic = DNAConstraintSimulator(developmental_stage=DevelopmentalStage.EMBRYONIC)

        # Embryonic should have higher tubulin expression (for growth)
        assert (sim_embryonic.category_expression[GeneCategory.TUBULIN] >
                sim_adult.category_expression[GeneCategory.TUBULIN])


class TestSpeciesConstraints:
    """Test species-specific constraints."""

    def test_set_species(self):
        """Test changing species."""
        from simulations.field_dynamics.dna_constraints import (
            DNAConstraintSimulator, SpeciesComplexity
        )

        sim = DNAConstraintSimulator()

        sim.set_species(SpeciesComplexity.INVERTEBRATE)

        assert sim.species == SpeciesComplexity.INVERTEBRATE

    def test_human_has_largest_viable_window(self):
        """Test human has largest viable window."""
        from simulations.field_dynamics.dna_constraints import (
            DNAConstraintSimulator, SpeciesComplexity
        )

        sim_human = DNAConstraintSimulator(species=SpeciesComplexity.HUMAN)
        sim_mammal = DNAConstraintSimulator(species=SpeciesComplexity.MAMMAL)
        sim_invert = DNAConstraintSimulator(species=SpeciesComplexity.INVERTEBRATE)

        assert sim_human.viable_window['window_area'] > sim_mammal.viable_window['window_area']
        assert sim_mammal.viable_window['window_area'] > sim_invert.viable_window['window_area']

    def test_species_progression(self):
        """Test viable window increases with species complexity."""
        from simulations.field_dynamics.dna_constraints import (
            DNAConstraintSimulator, SpeciesComplexity
        )

        species_order = [
            SpeciesComplexity.PROKARYOTE,
            SpeciesComplexity.SIMPLE_EUKARYOTE,
            SpeciesComplexity.INVERTEBRATE,
            SpeciesComplexity.SIMPLE_VERTEBRATE,
            SpeciesComplexity.MAMMAL,
            SpeciesComplexity.PRIMATE,
            SpeciesComplexity.HUMAN,
        ]

        window_areas = []
        for sp in species_order:
            sim = DNAConstraintSimulator(species=sp)
            window_areas.append(sim.viable_window['window_area'])

        # Each species should have larger window than previous
        for i in range(1, len(window_areas)):
            assert window_areas[i] > window_areas[i-1]


class TestSubstrateParameters:
    """Test substrate parameter outputs."""

    def test_get_microtubule_parameters(self):
        """Test getting microtubule parameters."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator()
        params = sim.get_microtubule_parameters()

        assert 'coupling_strength' in params
        assert 'coherence_time_factor' in params
        assert 'aromatic_activity' in params
        assert 'decoherence_rate' in params

    def test_get_bioelectric_parameters(self):
        """Test getting bioelectric parameters."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator()
        params = sim.get_bioelectric_parameters()

        assert 'g_Na_factor' in params
        assert 'g_K_factor' in params
        assert 'gap_conductance_factor' in params
        assert 'excitability' in params

    def test_get_biophoton_parameters(self):
        """Test getting biophoton parameters."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator()
        params = sim.get_biophoton_parameters()

        assert 'emission_rate_factor' in params
        assert 'atp_production' in params
        assert 'ros_baseline' in params
        assert 'coherence_coupling' in params

    def test_get_complete_substrate_state(self):
        """Test getting complete substrate state."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator()
        state = sim.get_complete_substrate_state()

        assert 'microtubule' in state
        assert 'bioelectric' in state
        assert 'biophoton' in state
        assert 'viable_window' in state
        assert 'coherence_capacity' in state


class TestERMapping:
    """Test éR phase space mapping."""

    def test_map_to_er_space(self):
        """Test éR space mapping returns expected keys."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator()
        result = sim.map_to_er_space()

        assert 'viable_window' in result
        assert 'coherence_capacity' in result
        assert 'microtubule_capacity' in result
        assert 'bioelectric_capacity' in result
        assert 'biophoton_capacity' in result
        assert 'pi_stack_coherence' in result
        assert 'species' in result
        assert 'developmental_stage' in result
        assert 'mean_expression' in result
        assert 'mean_methylation' in result

    def test_er_values_reasonable(self):
        """Test éR values are in reasonable ranges."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator()
        result = sim.map_to_er_space()

        assert 0 <= result['coherence_capacity'] <= 2
        assert 0 <= result['microtubule_capacity'] <= 2
        assert 0 <= result['bioelectric_capacity'] <= 2
        assert 0 <= result['biophoton_capacity'] <= 2
        assert 0 <= result['pi_stack_coherence'] <= 1
        assert 0 <= result['mean_expression'] <= 2
        assert 0 <= result['mean_methylation'] <= 1

    def test_viable_window_bounds(self):
        """Test viable window has proper bounds."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator()
        vw = sim.viable_window

        assert vw['er_min'] < vw['er_max']
        assert vw['ep_min'] < vw['ep_max']
        assert vw['freq_min'] < vw['freq_max']
        assert vw['window_area'] > 0


class TestCoherenceCapacity:
    """Test coherence capacity calculations."""

    def test_coherence_capacity_range(self):
        """Test coherence capacity is in valid range."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator()

        assert 0 <= sim.coherence_capacity <= 2
        assert 0 <= sim.microtubule_capacity <= 2
        assert 0 <= sim.bioelectric_capacity <= 2
        assert 0 <= sim.biophoton_capacity <= 2

    def test_methylation_reduces_capacity(self):
        """Test that methylation reduces coherence capacity."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator()
        initial_capacity = sim.coherence_capacity

        # Apply methylation to all genes
        for gene in sim.genes.values():
            gene.methylation = 0.8
        sim._compute_coherence_constraints()

        # Capacity should decrease
        assert sim.coherence_capacity < initial_capacity


class TestReset:
    """Test reset functionality."""

    def test_reset_restores_defaults(self):
        """Test reset restores default state."""
        from simulations.field_dynamics.dna_constraints import (
            DNAConstraintSimulator, DevelopmentalStage
        )

        sim = DNAConstraintSimulator()

        # Modify state
        sim.set_gene_expression('TUBA1A', 0.5)
        sim.apply_environmental_stress(0.8)
        sim.step_pi_stack(100)

        # Reset
        sim.reset()

        assert sim.developmental_stage == DevelopmentalStage.ADULT
        # Methylation should be reset to low values
        assert sim.genes['TUBA1A'].methylation < 0.3


class TestPresets:
    """Test preset configurations."""

    def test_human_baseline_preset(self):
        """Test human baseline preset."""
        from simulations.field_dynamics.dna_constraints import (
            create_human_baseline, SpeciesComplexity, DevelopmentalStage
        )

        sim = create_human_baseline()

        assert sim.species == SpeciesComplexity.HUMAN
        assert sim.developmental_stage == DevelopmentalStage.ADULT

    def test_high_plasticity_preset(self):
        """Test high plasticity preset."""
        from simulations.field_dynamics.dna_constraints import (
            create_high_plasticity, DevelopmentalStage
        )

        sim = create_high_plasticity()

        assert sim.developmental_stage == DevelopmentalStage.CHILD
        # Should have enhanced plasticity genes
        assert sim.genes['CAMK2A'].expression_level > 1.0

    def test_stressed_aging_preset(self):
        """Test stressed/aging preset."""
        from simulations.field_dynamics.dna_constraints import (
            create_stressed_aging, DevelopmentalStage
        )

        sim = create_stressed_aging()

        assert sim.developmental_stage == DevelopmentalStage.ELDERLY
        # Should have stress-induced methylation
        total_methylation = sum(g.methylation for g in sim.genes.values())
        assert total_methylation > 0

    def test_developmental_series_preset(self):
        """Test developmental series preset."""
        from simulations.field_dynamics.dna_constraints import create_developmental_series

        series = create_developmental_series()

        assert len(series) == 7  # 7 stages
        stages = [sim.developmental_stage.value for sim in series]
        assert 'embryonic' in stages
        assert 'elderly' in stages

    def test_cross_species_preset(self):
        """Test cross-species preset."""
        from simulations.field_dynamics.dna_constraints import create_cross_species

        species_dict = create_cross_species()

        assert 'invertebrate' in species_dict
        assert 'mammal' in species_dict
        assert 'human' in species_dict

    def test_meditation_epigenetics_preset(self):
        """Test meditation epigenetics preset."""
        from simulations.field_dynamics.dna_constraints import (
            create_meditation_epigenetics, GeneCategory
        )

        sim = create_meditation_epigenetics()

        # Should have enhanced CAMK2A
        assert sim.genes['CAMK2A'].expression_level >= 1.2


class TestVisualizerImport:
    """Test visualizer import."""

    def test_visualizer_import(self):
        """Test that DNAConstraintVisualizer can be imported."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintVisualizer
        assert DNAConstraintVisualizer is not None


class TestConstants:
    """Test physical constants."""

    def test_pi_stack_constants(self):
        """Test pi stack constants are defined."""
        from simulations.field_dynamics.dna_constraints import (
            PI_STACK_SPACING_NM, PI_STACK_COUPLING, AROMATIC_RINGS_PER_BASE
        )

        assert PI_STACK_SPACING_NM == 0.34
        assert PI_STACK_COUPLING > 0
        assert AROMATIC_RINGS_PER_BASE == 2

    def test_default_genes_constant(self):
        """Test DEFAULT_GENES is defined."""
        from simulations.field_dynamics.dna_constraints import DEFAULT_GENES

        assert len(DEFAULT_GENES) > 10  # Should have multiple genes
        assert 'TUBA1A' in DEFAULT_GENES
        assert 'SCN1A' in DEFAULT_GENES


class TestCategoryExpression:
    """Test category-level expression calculations."""

    def test_category_expression_computed(self):
        """Test category expression is computed."""
        from simulations.field_dynamics.dna_constraints import (
            DNAConstraintSimulator, GeneCategory
        )

        sim = DNAConstraintSimulator()

        assert GeneCategory.TUBULIN in sim.category_expression
        assert GeneCategory.ION_CHANNEL in sim.category_expression
        assert GeneCategory.MITOCHONDRIAL in sim.category_expression
        assert GeneCategory.GAP_JUNCTION in sim.category_expression

    def test_category_expression_values(self):
        """Test category expression values are reasonable."""
        from simulations.field_dynamics.dna_constraints import (
            DNAConstraintSimulator, GeneCategory
        )

        sim = DNAConstraintSimulator()

        for cat in GeneCategory:
            expr = sim.category_expression.get(cat, 0)
            assert 0 <= expr <= 2


class TestDNASequence:
    """Test DNA sequence generation."""

    def test_sequence_length(self):
        """Test sequence has correct length."""
        from simulations.field_dynamics.dna_constraints import DNAConstraintSimulator

        sim = DNAConstraintSimulator(base_pairs=75)

        assert len(sim.sequence) == 75

    def test_sequence_contains_valid_base_pairs(self):
        """Test sequence contains only valid base pairs."""
        from simulations.field_dynamics.dna_constraints import (
            DNAConstraintSimulator, BasePair
        )

        sim = DNAConstraintSimulator(base_pairs=100)

        valid_bps = {BasePair.AT, BasePair.TA, BasePair.GC, BasePair.CG}
        for bp in sim.sequence:
            assert bp in valid_bps

    def test_gc_content_reasonable(self):
        """Test GC content is in reasonable range (40-60%)."""
        from simulations.field_dynamics.dna_constraints import (
            DNAConstraintSimulator, BasePair
        )

        # Use larger sequence for statistical reliability
        sim = DNAConstraintSimulator(base_pairs=1000)

        gc_count = sum(1 for bp in sim.sequence
                       if bp in [BasePair.GC, BasePair.CG])
        gc_content = gc_count / len(sim.sequence)

        # Should be roughly around 50% (may vary due to randomness)
        assert 0.3 <= gc_content <= 0.7
