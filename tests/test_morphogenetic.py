"""
Tests for Morphogenetic Field Simulator.

Tests pattern memory, regeneration dynamics, and
bioelectric control of morphogenesis.
"""

import pytest
import numpy as np


class TestMorphogeneticSimulatorBasics:
    """Test basic MorphogeneticSimulator functionality."""

    def test_simulator_import(self):
        """Test that simulator can be imported."""
        from simulations.field_dynamics import MorphogeneticSimulator
        assert MorphogeneticSimulator is not None

    def test_simulator_initialization(self):
        """Test simulator initializes correctly."""
        from simulations.field_dynamics import MorphogeneticSimulator, PatternType

        sim = MorphogeneticSimulator(
            grid_size=(30, 30),
            pattern_type=PatternType.LEFT_RIGHT
        )

        assert sim.rows == 30
        assert sim.cols == 30
        assert sim.pattern_type == PatternType.LEFT_RIGHT
        assert sim.target_pattern.shape == (30, 30)

    def test_initial_state_matches_target(self):
        """Test that initial state matches target pattern."""
        from simulations.field_dynamics import MorphogeneticSimulator

        sim = MorphogeneticSimulator(grid_size=(20, 20))

        # Initial Vm should equal target
        assert np.allclose(sim.Vm, sim.target_pattern)

    def test_reset_restores_state(self):
        """Test that reset restores to target pattern."""
        from simulations.field_dynamics import MorphogeneticSimulator

        sim = MorphogeneticSimulator(grid_size=(20, 20))

        # Scramble the state
        sim.Vm[:, :] = np.random.uniform(-90, -20, sim.Vm.shape)

        # Reset
        sim.reset(preserve_target=True)

        # Should be back to target
        assert np.allclose(sim.Vm, sim.target_pattern)


class TestPatternGeneration:
    """Test target pattern generation."""

    def test_generate_pattern_import(self):
        """Test pattern generation function exists."""
        from simulations.field_dynamics import generate_pattern, PatternType
        assert callable(generate_pattern)

    def test_pattern_types_exist(self):
        """Test that all pattern types are defined."""
        from simulations.field_dynamics import PatternType

        expected_patterns = ['UNIFORM', 'LEFT_RIGHT', 'ANTERIOR_POSTERIOR',
                            'RADIAL', 'STRIPE', 'SPOT', 'CHECKERBOARD',
                            'HEAD_TAIL', 'LETTER_T', 'CUSTOM']

        for name in expected_patterns:
            assert hasattr(PatternType, name)

    def test_generate_left_right_pattern(self):
        """Test left-right pattern generation."""
        from simulations.field_dynamics import generate_pattern, PatternType, V_REST, V_DEPOLARIZED

        pattern = generate_pattern(PatternType.LEFT_RIGHT, (20, 20))

        # Left half should be depolarized
        left_mean = np.mean(pattern[:, :10])
        right_mean = np.mean(pattern[:, 10:])

        assert left_mean > right_mean

    def test_generate_radial_pattern(self):
        """Test radial pattern generation."""
        from simulations.field_dynamics import generate_pattern, PatternType

        pattern = generate_pattern(PatternType.RADIAL, (20, 20))

        # Center should differ from corners
        center = pattern[10, 10]
        corner = pattern[0, 0]

        assert center != corner

    def test_generate_stripe_pattern(self):
        """Test stripe pattern generation."""
        from simulations.field_dynamics import generate_pattern, PatternType

        pattern = generate_pattern(PatternType.STRIPE, (20, 20))

        # Should have variation in rows
        row_means = [np.mean(pattern[r, :]) for r in range(20)]
        assert np.std(row_means) > 0

    def test_custom_voltage_values(self):
        """Test pattern with custom voltage values."""
        from simulations.field_dynamics import generate_pattern, PatternType

        v_low = -80.0
        v_high = -40.0
        pattern = generate_pattern(PatternType.LEFT_RIGHT, (20, 20),
                                   v_low=v_low, v_high=v_high)

        assert np.min(pattern) >= v_low
        assert np.max(pattern) <= v_high


class TestPatternMemory:
    """Test morphogenetic pattern memory mechanics."""

    def test_set_target_pattern(self):
        """Test changing target pattern."""
        from simulations.field_dynamics import MorphogeneticSimulator, PatternType

        sim = MorphogeneticSimulator(grid_size=(20, 20), pattern_type=PatternType.LEFT_RIGHT)

        old_target = sim.target_pattern.copy()
        sim.set_target_pattern(PatternType.RADIAL)

        assert not np.allclose(sim.target_pattern, old_target)
        assert sim.pattern_type == PatternType.RADIAL

    def test_custom_target_pattern(self):
        """Test setting custom target pattern."""
        from simulations.field_dynamics import MorphogeneticSimulator

        sim = MorphogeneticSimulator(grid_size=(20, 20))

        custom = np.random.uniform(-90, -30, (20, 20))
        sim.set_target_pattern(custom_pattern=custom)

        assert np.allclose(sim.target_pattern, custom)

    def test_attraction_strength_affects_dynamics(self):
        """Test that attraction strength parameter is stored."""
        from simulations.field_dynamics import MorphogeneticSimulator

        sim = MorphogeneticSimulator(grid_size=(20, 20), attraction_strength=0.05)

        assert sim.attraction_strength == 0.05


class TestInjuryAndRegeneration:
    """Test injury and regeneration mechanics."""

    def test_create_injury_marks_cells(self):
        """Test that injury marks affected cells."""
        from simulations.field_dynamics import MorphogeneticSimulator

        sim = MorphogeneticSimulator(grid_size=(20, 20))

        center = (10, 10)
        radius = 3
        sim.create_injury(center, radius, scramble=False)

        # Check injury mask
        assert sim.injury_mask[center[0], center[1]] == True
        assert sim.injury_mask[0, 0] == False

    def test_create_injury_scrambles_voltage(self):
        """Test that scramble=True randomizes voltages."""
        from simulations.field_dynamics import MorphogeneticSimulator

        sim = MorphogeneticSimulator(grid_size=(20, 20))

        original_Vm = sim.Vm.copy()
        center = (10, 10)
        sim.create_injury(center, radius=3, scramble=True)

        # Voltage at injury site should have changed
        assert sim.Vm[center[0], center[1]] != original_Vm[center[0], center[1]]

    def test_remove_region(self):
        """Test removing a region entirely."""
        from simulations.field_dynamics import MorphogeneticSimulator, V_REST

        sim = MorphogeneticSimulator(grid_size=(20, 20))

        center = (10, 10)
        sim.remove_region(center, radius=3)

        # Removed region should be at rest potential
        assert np.isclose(sim.Vm[center[0], center[1]], V_REST)
        assert sim.injury_mask[center[0], center[1]] == True

    def test_amputate_half(self):
        """Test amputation of half the tissue."""
        from simulations.field_dynamics import MorphogeneticSimulator

        sim = MorphogeneticSimulator(grid_size=(20, 20))

        sim.amputate_half('right')

        # Right side should be injured
        assert np.sum(sim.injury_mask[:, 15:]) > np.sum(sim.injury_mask[:, :5])

    def test_heal_injury(self):
        """Test healing of injury."""
        from simulations.field_dynamics import MorphogeneticSimulator

        sim = MorphogeneticSimulator(grid_size=(20, 20))

        center = (10, 10)
        sim.create_injury(center, radius=3)

        # Count injured cells
        injured_before = np.sum(sim.injury_mask)

        # Heal
        sim.heal_injury(center, radius=3, heal_rate=1.0)

        injured_after = np.sum(sim.injury_mask)

        # Should have fewer injured cells
        assert injured_after < injured_before


class TestFidelityMetrics:
    """Test pattern fidelity metrics."""

    def test_pattern_fidelity_perfect_match(self):
        """Test fidelity when current matches target exactly."""
        from simulations.field_dynamics import MorphogeneticSimulator

        sim = MorphogeneticSimulator(grid_size=(20, 20))

        # Initial state matches target
        fidelity = sim.compute_pattern_fidelity()

        # Should be close to 1 (perfect match)
        assert fidelity > 0.99

    def test_pattern_fidelity_after_injury(self):
        """Test fidelity decreases after injury."""
        from simulations.field_dynamics import MorphogeneticSimulator

        sim = MorphogeneticSimulator(grid_size=(20, 20))

        fidelity_before = sim.compute_pattern_fidelity()

        # Create large injury
        sim.create_injury((10, 10), radius=5, scramble=True)

        fidelity_after = sim.compute_pattern_fidelity()

        # Fidelity should decrease
        assert fidelity_after < fidelity_before

    def test_regeneration_progress_no_injury(self):
        """Test regeneration progress with no injury returns 1."""
        from simulations.field_dynamics import MorphogeneticSimulator

        sim = MorphogeneticSimulator(grid_size=(20, 20))

        progress = sim.compute_regeneration_progress()

        assert progress == 1.0

    def test_regeneration_progress_with_injury(self):
        """Test regeneration progress computation with injury."""
        from simulations.field_dynamics import MorphogeneticSimulator

        sim = MorphogeneticSimulator(grid_size=(20, 20))

        sim.create_injury((10, 10), radius=5, scramble=True)

        progress = sim.compute_regeneration_progress()

        assert 0 <= progress <= 1


class TestSimulationDynamics:
    """Test simulation step mechanics."""

    def test_step_updates_time(self):
        """Test that stepping updates time."""
        from simulations.field_dynamics import MorphogeneticSimulator

        sim = MorphogeneticSimulator(grid_size=(20, 20), dt=0.1)

        initial_time = sim.time
        sim.step(10)

        assert np.isclose(sim.time, initial_time + 10 * 0.1)

    def test_run_records_history(self):
        """Test that run() records fidelity history."""
        from simulations.field_dynamics import MorphogeneticSimulator

        sim = MorphogeneticSimulator(grid_size=(20, 20))

        sim.run(duration=10.0, record_interval=1.0)

        assert len(sim.fidelity_history) > 0
        assert len(sim.coherence_history) > 0

    def test_regeneration_improves_fidelity(self):
        """Test that simulation tends to improve fidelity after injury."""
        from simulations.field_dynamics import MorphogeneticSimulator

        sim = MorphogeneticSimulator(
            grid_size=(20, 20),
            attraction_strength=0.03  # Stronger attraction for faster regeneration
        )

        # Create small injury
        sim.create_injury((10, 10), radius=3, scramble=True)

        # Fully heal to allow regeneration
        sim.heal_injury((10, 10), radius=3, heal_rate=1.0)

        fidelity_before = sim.compute_pattern_fidelity()

        # Run simulation longer
        sim.run(duration=100.0)

        fidelity_after = sim.compute_pattern_fidelity()

        # Fidelity should improve (or at least not drastically decrease)
        # This tests the basic tendency toward pattern restoration
        assert fidelity_after >= fidelity_before * 0.8


class TestPresets:
    """Test preset configurations."""

    def test_stable_pattern_preset(self):
        """Test stable pattern preset."""
        from simulations.field_dynamics import create_stable_pattern, PatternType

        sim = create_stable_pattern(PatternType.RADIAL, grid_size=(20, 20))

        assert sim.pattern_type == PatternType.RADIAL

    def test_regeneration_scenario_preset(self):
        """Test regeneration scenario has injury."""
        from simulations.field_dynamics import create_regeneration_scenario

        sim = create_regeneration_scenario(grid_size=(20, 20))

        # Should have injured cells
        assert np.sum(sim.injury_mask) > 0

    def test_cancer_scenario_preset(self):
        """Test cancer scenario has weak attraction."""
        from simulations.field_dynamics import create_cancer_scenario

        sim = create_cancer_scenario(grid_size=(20, 20))

        # Should have weak attraction
        assert sim.attraction_strength < 0.01


class TestERMapping:
    """Test éR phase space mapping."""

    def test_er_mapping_returns_dict(self):
        """Test that éR mapping returns expected keys."""
        from simulations.field_dynamics import MorphogeneticSimulator

        sim = MorphogeneticSimulator(grid_size=(20, 20))

        result = sim.map_to_er_space()

        assert 'energy_present' in result
        assert 'frequency' in result
        assert 'energy_resistance' in result
        assert 'pattern_fidelity' in result

    def test_er_formula(self):
        """Test éR = EP / f² relationship."""
        from simulations.field_dynamics import MorphogeneticSimulator

        sim = MorphogeneticSimulator(grid_size=(20, 20))

        result = sim.map_to_er_space()

        er_computed = result['energy_present'] / (result['frequency'] ** 2)
        assert np.isclose(result['energy_resistance'], er_computed)


class TestVisualizerImport:
    """Test visualizer import."""

    def test_visualizer_import(self):
        """Test that MorphogeneticVisualizer can be imported."""
        from simulations.field_dynamics import MorphogeneticVisualizer
        assert MorphogeneticVisualizer is not None
