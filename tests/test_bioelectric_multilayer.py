"""
Tests for Multi-Layer Bioelectric Field Simulator.

Tests cross-tissue coherence coupling, tissue type presets,
and hierarchical coherence metrics.
"""

import pytest
import numpy as np


class TestMultiLayerSimulatorBasics:
    """Test basic MultiLayerBioelectricSimulator functionality."""

    def test_simulator_import(self):
        """Test that simulator can be imported."""
        from simulations.field_dynamics import MultiLayerBioelectricSimulator
        assert MultiLayerBioelectricSimulator is not None

    def test_simulator_initialization(self):
        """Test simulator initializes with correct dimensions."""
        from simulations.field_dynamics import MultiLayerBioelectricSimulator

        sim = MultiLayerBioelectricSimulator(grid_size=(20, 20), n_layers=3)

        assert sim.rows == 20
        assert sim.cols == 20
        assert sim.n_layers == 3
        assert len(sim.layers) == 3

    def test_default_layer_types(self):
        """Test that default layer types are set correctly."""
        from simulations.field_dynamics import MultiLayerBioelectricSimulator, TissueType

        sim = MultiLayerBioelectricSimulator(grid_size=(20, 20), n_layers=3)

        assert sim.layer_types[0] == TissueType.EPITHELIAL
        assert sim.layer_types[1] == TissueType.NEURAL
        assert sim.layer_types[2] == TissueType.MESENCHYMAL

    def test_custom_layer_types(self):
        """Test setting custom layer types."""
        from simulations.field_dynamics import MultiLayerBioelectricSimulator, TissueType

        sim = MultiLayerBioelectricSimulator(
            grid_size=(20, 20),
            n_layers=2,
            layer_types=[TissueType.NEURAL, TissueType.NEURAL]
        )

        assert sim.layer_types[0] == TissueType.NEURAL
        assert sim.layer_types[1] == TissueType.NEURAL


class TestTissueProperties:
    """Test tissue type properties."""

    def test_tissue_presets_exist(self):
        """Test that tissue presets are defined."""
        from simulations.field_dynamics import TISSUE_PRESETS, TissueType

        assert TissueType.EPITHELIAL in TISSUE_PRESETS
        assert TissueType.NEURAL in TISSUE_PRESETS
        assert TissueType.MESENCHYMAL in TISSUE_PRESETS

    def test_tissue_properties_have_required_fields(self):
        """Test that tissue properties have all required fields."""
        from simulations.field_dynamics import TISSUE_PRESETS

        required_fields = ['name', 'gap_conductance', 'vertical_conductance',
                          'g_Na', 'g_K', 'g_leak', 'excitable', 'v_rest', 'color']

        for tissue_type, props in TISSUE_PRESETS.items():
            for field in required_fields:
                assert hasattr(props, field), f"TissueProperties missing field: {field}"

    def test_neural_tissue_is_excitable(self):
        """Test that neural tissue is marked as excitable."""
        from simulations.field_dynamics import TISSUE_PRESETS, TissueType

        neural = TISSUE_PRESETS[TissueType.NEURAL]
        assert neural.excitable == True

    def test_epithelial_tissue_not_excitable(self):
        """Test that epithelial tissue is not excitable."""
        from simulations.field_dynamics import TISSUE_PRESETS, TissueType

        epithelial = TISSUE_PRESETS[TissueType.EPITHELIAL]
        assert epithelial.excitable == False


class TestVerticalCoupling:
    """Test inter-layer coupling."""

    def test_vertical_coupling_shape(self):
        """Test vertical coupling array has correct shape."""
        from simulations.field_dynamics import MultiLayerBioelectricSimulator

        sim = MultiLayerBioelectricSimulator(grid_size=(20, 20), n_layers=3)

        # Should have n_layers-1 coupling arrays
        assert sim.g_vertical.shape[0] == 2  # 3-1 = 2
        assert sim.g_vertical.shape[1] == 20
        assert sim.g_vertical.shape[2] == 20

    def test_set_vertical_coupling(self):
        """Test setting vertical coupling strength."""
        from simulations.field_dynamics import MultiLayerBioelectricSimulator

        sim = MultiLayerBioelectricSimulator(grid_size=(20, 20), n_layers=3)

        sim.set_vertical_coupling(0, 2.5)

        assert np.allclose(sim.g_vertical[0], 2.5)

    def test_through_injury_breaks_vertical_connections(self):
        """Test that through-injury breaks vertical coupling."""
        from simulations.field_dynamics import MultiLayerBioelectricSimulator

        sim = MultiLayerBioelectricSimulator(grid_size=(20, 20), n_layers=3)

        center = (10, 10)
        radius = 3
        sim.create_through_injury(center, radius)

        # Check vertical coupling is broken in injury region
        assert sim.g_vertical[0, center[0], center[1]] == 0.0
        assert sim.g_vertical[1, center[0], center[1]] == 0.0


class TestCrossLayerCoherence:
    """Test hierarchical coherence metrics."""

    def test_within_layer_coherence(self):
        """Test within-layer coherence computation."""
        from simulations.field_dynamics import MultiLayerBioelectricSimulator

        sim = MultiLayerBioelectricSimulator(grid_size=(20, 20), n_layers=3)

        within = sim.compute_within_layer_coherence()

        assert len(within) == 3
        for coherence in within:
            assert 0 <= coherence <= 1

    def test_between_layer_coherence_uniform(self):
        """Test between-layer coherence for uniform layers."""
        from simulations.field_dynamics import MultiLayerBioelectricSimulator

        sim = MultiLayerBioelectricSimulator(grid_size=(20, 20), n_layers=3)

        # All layers uniform -> between-layer coherence at baseline
        between = sim.compute_between_layer_coherence()

        # Uniform layers should have at least baseline coherence
        # (correlation is 0 for uniform, which maps to 0.5)
        assert between >= 0.5

    def test_between_layer_coherence_different_patterns(self):
        """Test between-layer coherence when layers have different patterns."""
        from simulations.field_dynamics import MultiLayerBioelectricSimulator

        sim = MultiLayerBioelectricSimulator(grid_size=(20, 20), n_layers=2)

        # Give layers opposite patterns
        sim.layers[0].Vm[:, :10] = -30.0  # Left half depolarized
        sim.layers[1].Vm[:, 10:] = -30.0  # Right half depolarized

        between = sim.compute_between_layer_coherence()

        # Opposite patterns should have lower coherence
        assert between < 0.7

    def test_global_coherence(self):
        """Test global coherence computation."""
        from simulations.field_dynamics import MultiLayerBioelectricSimulator

        sim = MultiLayerBioelectricSimulator(grid_size=(20, 20), n_layers=3)

        global_coh = sim.compute_global_coherence()

        assert 0 <= global_coh <= 1

    def test_compute_all_coherence(self):
        """Test that compute_all_coherence returns all metrics."""
        from simulations.field_dynamics import MultiLayerBioelectricSimulator

        sim = MultiLayerBioelectricSimulator(grid_size=(20, 20), n_layers=3)

        result = sim.compute_all_coherence()

        assert 'within_layer' in result
        assert 'within_layer_mean' in result
        assert 'between_layer' in result
        assert 'global' in result


class TestMultiLayerSimulation:
    """Test simulation dynamics."""

    def test_step_updates_all_layers(self):
        """Test that stepping updates all layers."""
        from simulations.field_dynamics import MultiLayerBioelectricSimulator

        sim = MultiLayerBioelectricSimulator(grid_size=(20, 20), n_layers=3, dt=0.1)

        initial_time = sim.time
        sim.step(10)

        assert sim.time > initial_time
        assert np.isclose(sim.time, 1.0)

    def test_depolarize_column(self):
        """Test depolarizing through all layers."""
        from simulations.field_dynamics import MultiLayerBioelectricSimulator, V_REST

        sim = MultiLayerBioelectricSimulator(grid_size=(20, 20), n_layers=3)

        center = (10, 10)
        sim.depolarize_column(center, radius=2, target_Vm=-30.0)

        # All layers should be affected
        for layer in sim.layers:
            assert layer.Vm[center[0], center[1]] == -30.0


class TestMultiLayerPresets:
    """Test preset configurations."""

    def test_default_multilayer_preset(self):
        """Test default 3-layer preset."""
        from simulations.field_dynamics import create_default_multilayer

        sim = create_default_multilayer(grid_size=(20, 20))

        assert sim.n_layers == 3

    def test_epithelial_neural_pair(self):
        """Test 2-layer epithelial-neural preset."""
        from simulations.field_dynamics import create_epithelial_neural_pair, TissueType

        sim = create_epithelial_neural_pair(grid_size=(20, 20))

        assert sim.n_layers == 2
        assert sim.layer_types[0] == TissueType.EPITHELIAL
        assert sim.layer_types[1] == TissueType.NEURAL

    def test_decoupled_layers_preset(self):
        """Test preset with minimal vertical coupling."""
        from simulations.field_dynamics import create_decoupled_layers

        sim = create_decoupled_layers(grid_size=(20, 20))

        # Should have very low vertical coupling
        assert np.mean(sim.g_vertical) < 0.1

    def test_tightly_coupled_layers_preset(self):
        """Test preset with strong vertical coupling."""
        from simulations.field_dynamics import create_tightly_coupled_layers

        sim = create_tightly_coupled_layers(grid_size=(20, 20))

        # Should have high vertical coupling
        assert np.mean(sim.g_vertical) > 1.5

    def test_injured_multilayer_preset(self):
        """Test preset with through-injury."""
        from simulations.field_dynamics import create_injured_multilayer

        sim = create_injured_multilayer(grid_size=(20, 20))

        # Should have some zero conductance (injury)
        assert np.any(sim.g_vertical == 0.0)


class TestMultiLayerERMapping:
    """Test éR phase space mapping."""

    def test_er_mapping_returns_dict(self):
        """Test that éR mapping returns expected keys."""
        from simulations.field_dynamics import MultiLayerBioelectricSimulator

        sim = MultiLayerBioelectricSimulator(grid_size=(20, 20), n_layers=3)

        result = sim.map_to_er_space()

        assert 'energy_present' in result
        assert 'frequency' in result
        assert 'energy_resistance' in result
        assert 'global_coherence' in result
        assert 'vertical_connectivity' in result

    def test_er_formula(self):
        """Test éR = EP / f² relationship."""
        from simulations.field_dynamics import MultiLayerBioelectricSimulator

        sim = MultiLayerBioelectricSimulator(grid_size=(20, 20), n_layers=3)

        result = sim.map_to_er_space()

        er_computed = result['energy_present'] / (result['frequency'] ** 2)
        assert np.isclose(result['energy_resistance'], er_computed)


class TestVisualizerImport:
    """Test visualizer import."""

    def test_visualizer_import(self):
        """Test that MultiLayerVisualizer can be imported."""
        from simulations.field_dynamics import MultiLayerVisualizer
        assert MultiLayerVisualizer is not None
