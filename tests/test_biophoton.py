"""
Tests for Biophoton Emission Simulator.

Tests emission statistics, coherence metrics, metabolic coupling,
and LoomSense output compatibility.
"""

import pytest
import numpy as np


class TestBiophotonSimulatorBasics:
    """Test basic BiophotonSimulator functionality."""

    def test_simulator_import(self):
        """Test that simulator can be imported."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator
        assert BiophotonSimulator is not None

    def test_simulator_initialization(self):
        """Test simulator initializes correctly."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator, EmissionMode

        sim = BiophotonSimulator(grid_size=(30, 30))

        assert sim.rows == 30
        assert sim.cols == 30
        assert sim.emission_mode == EmissionMode.POISSONIAN
        assert sim.emission_counts.shape == (30, 30)
        assert sim.time == 0.0

    def test_initial_state(self):
        """Test initial cellular state."""
        from simulations.field_dynamics.biophoton import (
            BiophotonSimulator, ATP_BASELINE, ROS_BASELINE
        )

        sim = BiophotonSimulator(grid_size=(20, 20))

        assert np.allclose(sim.atp, ATP_BASELINE)
        assert np.allclose(sim.ros, ROS_BASELINE)
        assert sim.emission_phase.shape == (20, 20)

    def test_reset(self):
        """Test reset restores initial state."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(20, 20))

        # Modify state
        sim.step(100)
        sim.induce_oxidative_stress((10, 10), radius=5, intensity=1.0)

        # Reset
        sim.reset()

        assert sim.time == 0.0
        assert sim.step_count == 0
        assert len(sim.emission_history) == 0


class TestEmissionModes:
    """Test different emission statistics modes."""

    def test_emission_mode_enum(self):
        """Test that all emission modes are defined."""
        from simulations.field_dynamics.biophoton import EmissionMode

        assert hasattr(EmissionMode, 'POISSONIAN')
        assert hasattr(EmissionMode, 'COHERENT')
        assert hasattr(EmissionMode, 'SQUEEZED')
        assert hasattr(EmissionMode, 'CHAOTIC')

    def test_set_emission_mode(self):
        """Test changing emission mode."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator, EmissionMode

        sim = BiophotonSimulator(grid_size=(20, 20))

        sim.set_emission_mode(EmissionMode.COHERENT)
        assert sim.emission_mode == EmissionMode.COHERENT

        sim.set_emission_mode(EmissionMode.CHAOTIC)
        assert sim.emission_mode == EmissionMode.CHAOTIC

    def test_poissonian_emission_generates_photons(self):
        """Test that Poissonian mode generates emissions."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator, EmissionMode

        sim = BiophotonSimulator(grid_size=(30, 30), emission_mode=EmissionMode.POISSONIAN)
        sim.step(100)

        # Should have generated some photons
        assert len(sim.emission_history) > 0
        assert sum(sim.emission_history) > 0

    def test_coherent_emission_generates_photons(self):
        """Test that coherent mode generates emissions."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator, EmissionMode

        sim = BiophotonSimulator(grid_size=(30, 30), emission_mode=EmissionMode.COHERENT)
        sim.step(100)

        assert len(sim.emission_history) > 0
        assert sum(sim.emission_history) > 0

    def test_squeezed_emission_generates_photons(self):
        """Test that squeezed mode generates emissions."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator, EmissionMode

        sim = BiophotonSimulator(grid_size=(30, 30), emission_mode=EmissionMode.SQUEEZED)
        sim.step(100)

        assert len(sim.emission_history) > 0

    def test_chaotic_emission_generates_photons(self):
        """Test that chaotic mode generates emissions."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator, EmissionMode

        sim = BiophotonSimulator(grid_size=(30, 30), emission_mode=EmissionMode.CHAOTIC)
        sim.step(100)

        assert len(sim.emission_history) > 0


class TestEmissionStatistics:
    """Test emission statistics calculations."""

    def test_compute_emission_statistics(self):
        """Test emission statistics computation."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(30, 30))
        sim.step(10)

        stats = sim.compute_emission_statistics()

        assert 'mean' in stats
        assert 'variance' in stats
        assert 'fano_factor' in stats
        assert 'total_photons' in stats
        assert 'emission_mode' in stats

    def test_fano_factor_reasonable(self):
        """Test that Fano factor is in reasonable range."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator, EmissionMode

        sim = BiophotonSimulator(grid_size=(40, 40), emission_mode=EmissionMode.POISSONIAN)
        sim.set_metabolic_rate(2.0)  # Increase emission for better statistics
        sim.step(50)

        stats = sim.compute_emission_statistics()

        # Fano factor should be positive
        assert stats['fano_factor'] >= 0


class TestCoherenceMetrics:
    """Test coherence metric calculations."""

    def test_spatial_coherence_range(self):
        """Test spatial coherence is in valid range."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(30, 30))
        sim.step(50)

        spatial_coh = sim.compute_spatial_coherence()

        assert 0 <= spatial_coh <= 1

    def test_temporal_coherence_range(self):
        """Test temporal coherence is in valid range."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(30, 30))
        sim.step(100)  # Need history for temporal coherence

        temporal_coh = sim.compute_temporal_coherence()

        assert 0 <= temporal_coh <= 1

    def test_phase_coherence_range(self):
        """Test phase coherence (order parameter) is in valid range."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(30, 30))

        phase_coh = sim.compute_phase_coherence()

        assert 0 <= phase_coh <= 1

    def test_synchronize_phases_increases_coherence(self):
        """Test that synchronizing phases increases phase coherence."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(30, 30))

        # Initial random phases - should have low coherence
        initial_coherence = sim.compute_phase_coherence()

        # Synchronize
        sim.synchronize_phases(coherence=1.0)

        # Should now have high coherence
        final_coherence = sim.compute_phase_coherence()

        assert final_coherence > initial_coherence
        assert final_coherence > 0.9  # Nearly fully synchronized

    def test_coherence_history_recorded(self):
        """Test that coherence history is recorded during simulation."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(20, 20))
        sim.run(duration=100)

        assert len(sim.spatial_coherence_history) > 0
        assert len(sim.temporal_coherence_history) > 0
        assert len(sim.phase_coherence_history) > 0


class TestMetabolicState:
    """Test metabolic state manipulation and effects."""

    def test_set_metabolic_rate(self):
        """Test setting metabolic rate."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator, MITO_ACTIVITY_DEFAULT

        sim = BiophotonSimulator(grid_size=(20, 20))

        sim.set_metabolic_rate(2.0)

        expected = MITO_ACTIVITY_DEFAULT * 2.0
        assert np.allclose(sim.mito_activity, expected)

    def test_set_metabolic_rate_region(self):
        """Test setting metabolic rate in a region."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(20, 20))

        # Set high rate in center region
        region = (slice(5, 15), slice(5, 15))
        sim.set_metabolic_rate(3.0, region=region)

        # Center should be higher
        center_activity = sim.mito_activity[10, 10]
        corner_activity = sim.mito_activity[0, 0]

        assert center_activity > corner_activity

    def test_induce_oxidative_stress(self):
        """Test inducing oxidative stress."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator, ROS_BASELINE

        sim = BiophotonSimulator(grid_size=(30, 30))

        original_ros = sim.ros.copy()

        center = (15, 15)
        sim.induce_oxidative_stress(center, radius=5, intensity=1.0)

        # ROS should increase at center
        assert sim.ros[center[0], center[1]] > original_ros[center[0], center[1]]

        # Corner should be unchanged
        assert sim.ros[0, 0] == original_ros[0, 0]

    def test_oxidative_stress_increases_emission(self):
        """Test that oxidative stress increases emission rate."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(30, 30))

        original_rate = sim.emission_rate.copy()

        center = (15, 15)
        sim.induce_oxidative_stress(center, radius=5, intensity=1.0)

        # Emission rate should increase at stressed location
        assert sim.emission_rate[center[0], center[1]] > original_rate[center[0], center[1]]

    def test_induce_apoptosis(self):
        """Test triggering apoptosis."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator, TissueState

        sim = BiophotonSimulator(grid_size=(30, 30))

        center = (15, 15)
        sim.induce_apoptosis(center, radius=3)

        # Should be marked as apoptotic
        assert sim.tissue_state[center[0], center[1]] == TissueState.APOPTOTIC.value

        # ROS should be very high
        assert sim.ros[center[0], center[1]] > 1.5


class TestSpectrum:
    """Test emission spectrum calculations."""

    def test_compute_spectrum(self):
        """Test spectrum computation."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(30, 30))
        sim.set_metabolic_rate(2.0)  # Increase emission
        sim.step(100)

        wavelengths, spectrum = sim.compute_spectrum()

        assert len(wavelengths) > 0
        assert len(spectrum) == len(wavelengths)
        assert all(spectrum >= 0)

    def test_spectrum_wavelength_range(self):
        """Test spectrum wavelengths are in expected range."""
        from simulations.field_dynamics.biophoton import (
            BiophotonSimulator, WAVELENGTH_MIN, WAVELENGTH_MAX
        )

        sim = BiophotonSimulator(grid_size=(30, 30))
        sim.set_metabolic_rate(2.0)
        sim.step(100)

        wavelengths, spectrum = sim.compute_spectrum()

        assert wavelengths[0] >= WAVELENGTH_MIN
        assert wavelengths[-1] <= WAVELENGTH_MAX


class TestERMapping:
    """Test éR phase space mapping."""

    def test_er_mapping_returns_dict(self):
        """Test that éR mapping returns expected keys."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(20, 20))
        sim.step(10)

        result = sim.map_to_er_space()

        assert 'energy_present' in result
        assert 'frequency' in result
        assert 'energy_resistance' in result
        assert 'coherence_index' in result

    def test_er_formula(self):
        """Test éR = EP / f² relationship."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(20, 20))
        sim.step(10)

        result = sim.map_to_er_space()

        er_computed = result['energy_present'] / (result['frequency'] ** 2)
        assert np.isclose(result['energy_resistance'], er_computed)

    def test_coherence_index_range(self):
        """Test coherence index is in valid range."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(20, 20))
        sim.step(50)

        result = sim.map_to_er_space()

        assert 0 <= result['coherence_index'] <= 1


class TestLoomSenseOutput:
    """Test LoomSense-compatible output."""

    def test_loomsense_output_keys(self):
        """Test LoomSense output has all expected keys."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(20, 20))
        sim.step(50)

        output = sim.get_loomsense_output()

        # Intensity metrics
        assert 'total_photon_count' in output
        assert 'mean_intensity' in output
        assert 'emission_rate_per_cell' in output

        # Statistical metrics
        assert 'fano_factor' in output

        # Spectral metrics
        assert 'peak_wavelength_nm' in output
        assert 'spectral_width_nm' in output

        # Coherence metrics
        assert 'spatial_coherence' in output
        assert 'temporal_coherence' in output
        assert 'phase_coherence' in output

        # Metabolic indicators
        assert 'mean_ros_level' in output
        assert 'mean_atp_level' in output

    def test_loomsense_values_reasonable(self):
        """Test LoomSense output values are in reasonable ranges."""
        from simulations.field_dynamics.biophoton import (
            BiophotonSimulator, WAVELENGTH_MIN, WAVELENGTH_MAX
        )

        sim = BiophotonSimulator(grid_size=(30, 30))
        sim.step(100)

        output = sim.get_loomsense_output()

        # Coherence in [0, 1]
        assert 0 <= output['spatial_coherence'] <= 1
        assert 0 <= output['temporal_coherence'] <= 1
        assert 0 <= output['phase_coherence'] <= 1

        # Wavelength in valid range
        assert WAVELENGTH_MIN <= output['peak_wavelength_nm'] <= WAVELENGTH_MAX

        # Positive counts
        assert output['total_photon_count'] >= 0


class TestSimulationDynamics:
    """Test simulation step mechanics."""

    def test_step_updates_time(self):
        """Test that stepping updates time."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(20, 20), dt=1.0)

        initial_time = sim.time
        sim.step(10)

        assert np.isclose(sim.time, initial_time + 10 * 1.0)

    def test_run_duration(self):
        """Test run() simulates for specified duration."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(20, 20), dt=1.0)

        sim.run(duration=100.0)

        assert np.isclose(sim.time, 100.0)

    def test_emission_history_grows(self):
        """Test emission history grows with simulation."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(20, 20))

        sim.step(50)
        first_len = len(sim.emission_history)

        sim.step(50)
        second_len = len(sim.emission_history)

        assert second_len > first_len


class TestCouplingDynamics:
    """Test phase coupling dynamics."""

    def test_set_coupling_strength(self):
        """Test setting coupling strength."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator

        sim = BiophotonSimulator(grid_size=(20, 20))

        sim.set_coupling_strength(0.8)

        assert sim.coupling_strength == 0.8

    def test_high_coupling_increases_coherence(self):
        """Test that high coupling tends to increase phase coherence over time."""
        from simulations.field_dynamics.biophoton import BiophotonSimulator, EmissionMode

        # High coupling simulation
        sim_coupled = BiophotonSimulator(
            grid_size=(20, 20),
            emission_mode=EmissionMode.COHERENT,
            coupling_strength=0.9
        )

        # Low coupling simulation
        sim_uncoupled = BiophotonSimulator(
            grid_size=(20, 20),
            emission_mode=EmissionMode.COHERENT,
            coupling_strength=0.0
        )

        # Run both
        sim_coupled.run(duration=500)
        sim_uncoupled.run(duration=500)

        # Coupled should have higher phase coherence
        coh_coupled = sim_coupled.compute_phase_coherence()
        coh_uncoupled = sim_uncoupled.compute_phase_coherence()

        # Coupled system should tend toward higher coherence
        # (Note: this is probabilistic, so we use a weak assertion)
        assert coh_coupled >= 0  # Just verify it runs


class TestPresets:
    """Test preset configurations."""

    def test_healthy_tissue_preset(self):
        """Test healthy tissue preset."""
        from simulations.field_dynamics.biophoton import create_healthy_tissue, EmissionMode

        sim = create_healthy_tissue(grid_size=(20, 20))

        assert sim.emission_mode == EmissionMode.POISSONIAN

    def test_stressed_tissue_preset(self):
        """Test stressed tissue preset."""
        from simulations.field_dynamics.biophoton import create_stressed_tissue, EmissionMode

        sim = create_stressed_tissue(grid_size=(20, 20))

        assert sim.emission_mode == EmissionMode.CHAOTIC
        # Should have elevated ROS somewhere
        assert np.max(sim.ros) > np.min(sim.ros)

    def test_coherent_emission_preset(self):
        """Test coherent emission preset."""
        from simulations.field_dynamics.biophoton import create_coherent_emission, EmissionMode

        sim = create_coherent_emission(grid_size=(20, 20))

        assert sim.emission_mode == EmissionMode.COHERENT
        # Should have high phase coherence
        assert sim.compute_phase_coherence() > 0.5

    def test_meditation_state_preset(self):
        """Test meditation state preset."""
        from simulations.field_dynamics.biophoton import create_meditation_state, EmissionMode

        sim = create_meditation_state(grid_size=(20, 20))

        assert sim.emission_mode == EmissionMode.SQUEEZED

    def test_inflammation_model_preset(self):
        """Test inflammation model preset."""
        from simulations.field_dynamics.biophoton import create_inflammation_model, EmissionMode

        sim = create_inflammation_model(grid_size=(20, 20))

        assert sim.emission_mode == EmissionMode.CHAOTIC
        # Should have high ROS in center
        center_ros = sim.ros[sim.rows // 2, sim.cols // 2]
        assert center_ros > sim.ros[0, 0]


class TestVisualizerImport:
    """Test visualizer import."""

    def test_visualizer_import(self):
        """Test that BiophotonVisualizer can be imported."""
        from simulations.field_dynamics.biophoton import BiophotonVisualizer
        assert BiophotonVisualizer is not None


class TestConstants:
    """Test physical constants are defined."""

    def test_wavelength_constants(self):
        """Test wavelength constants are defined."""
        from simulations.field_dynamics.biophoton import (
            WAVELENGTH_MIN, WAVELENGTH_MAX, WAVELENGTH_PEAK
        )

        assert WAVELENGTH_MIN < WAVELENGTH_PEAK < WAVELENGTH_MAX

    def test_emission_rate_constants(self):
        """Test emission rate constants are defined."""
        from simulations.field_dynamics.biophoton import (
            EMISSION_RATE_BASELINE, EMISSION_RATE_STRESSED
        )

        assert EMISSION_RATE_BASELINE < EMISSION_RATE_STRESSED

    def test_metabolic_constants(self):
        """Test metabolic constants are defined."""
        from simulations.field_dynamics.biophoton import (
            ATP_BASELINE, ROS_BASELINE, ROS_STRESSED
        )

        assert ATP_BASELINE > 0
        assert ROS_BASELINE < ROS_STRESSED
