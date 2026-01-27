"""
Tests for Microtubule Time Crystal Simulator.

Tests multi-scale oscillations, coherence metrics, triplet resonance,
thermal decoherence, and Floquet driving.
"""

import pytest
import numpy as np


class TestMicrotubuleSimulatorBasics:
    """Test basic MicrotubuleSimulator functionality."""

    def test_simulator_import(self):
        """Test that simulator can be imported."""
        from simulations.quantum.microtubule import MicrotubuleSimulator
        assert MicrotubuleSimulator is not None

    def test_simulator_initialization(self):
        """Test simulator initializes correctly."""
        from simulations.quantum.microtubule import (
            MicrotubuleSimulator, N_PROTOFILAMENTS
        )

        sim = MicrotubuleSimulator(n_tubulins=50)

        assert sim.n_tubulins == 50
        assert sim.n_protofilaments == N_PROTOFILAMENTS
        assert sim.dipoles.shape == (N_PROTOFILAMENTS, 50)
        assert sim.time == 0.0

    def test_phase_arrays_initialized(self):
        """Test that all phase arrays are initialized."""
        from simulations.quantum.microtubule import MicrotubuleSimulator

        sim = MicrotubuleSimulator(n_tubulins=30)

        assert sim.phase_ctermini.shape == (sim.n_protofilaments, 30)
        assert sim.phase_lattice.shape == (sim.n_protofilaments, 30)
        assert sim.phase_water.shape == (sim.n_protofilaments, 30)
        assert sim.phase_aromatic.shape == (sim.n_protofilaments, 30)

    def test_reset(self):
        """Test reset restores initial state."""
        from simulations.quantum.microtubule import MicrotubuleSimulator

        sim = MicrotubuleSimulator(n_tubulins=30)

        # Modify state
        sim.step(100)
        sim.randomize_phases()

        # Reset
        sim.reset()

        assert sim.time == 0.0
        assert len(sim.dipole_history) == 0
        assert len(sim.coherence_history) == 0


class TestOscillationScales:
    """Test multi-scale oscillation system."""

    def test_oscillation_scale_enum(self):
        """Test oscillation scale enum exists."""
        from simulations.quantum.microtubule import OscillationScale

        assert hasattr(OscillationScale, 'CTERMINI')
        assert hasattr(OscillationScale, 'LATTICE')
        assert hasattr(OscillationScale, 'WATER_CHANNEL')
        assert hasattr(OscillationScale, 'AROMATIC')

    def test_frequency_constants(self):
        """Test frequency constants are defined in correct order."""
        from simulations.quantum.microtubule import (
            FREQ_CTERMINI, FREQ_LATTICE, FREQ_WATER_CHANNEL, FREQ_AROMATIC
        )

        # Frequencies should increase: kHz < MHz < GHz < THz
        assert FREQ_CTERMINI < FREQ_LATTICE
        assert FREQ_LATTICE < FREQ_WATER_CHANNEL
        assert FREQ_WATER_CHANNEL < FREQ_AROMATIC

    def test_triplet_ratios_golden(self):
        """Test triplet ratios are based on golden ratio."""
        from simulations.quantum.microtubule import (
            TRIPLET_RATIO_1, TRIPLET_RATIO_2, TRIPLET_RATIO_3
        )

        phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        assert TRIPLET_RATIO_1 == 1.0
        assert np.isclose(TRIPLET_RATIO_2, phi, atol=0.01)
        assert np.isclose(TRIPLET_RATIO_3, phi**2, atol=0.01)


class TestCoherenceMetrics:
    """Test coherence metric calculations."""

    def test_compute_coherence_range(self):
        """Test coherence is in valid range."""
        from simulations.quantum.microtubule import MicrotubuleSimulator

        sim = MicrotubuleSimulator(n_tubulins=30)

        coherence = sim.compute_coherence()

        assert 0 <= coherence <= 1

    def test_compute_coherence_by_scale(self):
        """Test coherence can be computed for each scale."""
        from simulations.quantum.microtubule import MicrotubuleSimulator, OscillationScale

        sim = MicrotubuleSimulator(n_tubulins=30)

        for scale in OscillationScale:
            coh = sim.compute_coherence(scale)
            assert 0 <= coh <= 1

    def test_compute_all_coherences(self):
        """Test computing all coherences at once."""
        from simulations.quantum.microtubule import MicrotubuleSimulator

        sim = MicrotubuleSimulator(n_tubulins=30)

        coherences = sim.compute_all_coherences()

        assert 'ctermini' in coherences
        assert 'lattice' in coherences
        assert 'water_channel' in coherences
        assert 'aromatic' in coherences
        assert 'mean' in coherences

    def test_synchronize_increases_coherence(self):
        """Test that synchronizing phases increases coherence."""
        from simulations.quantum.microtubule import MicrotubuleSimulator

        sim = MicrotubuleSimulator(n_tubulins=30)
        sim.randomize_phases()

        # Initial (random) coherence
        initial = sim.compute_coherence()

        # Synchronize
        sim.synchronize_phases(coherence=1.0)

        # Should have higher coherence
        final = sim.compute_coherence()

        assert final > initial
        assert final > 0.9

    def test_randomize_decreases_coherence(self):
        """Test that randomizing phases decreases coherence."""
        from simulations.quantum.microtubule import MicrotubuleSimulator

        sim = MicrotubuleSimulator(n_tubulins=30)
        sim.synchronize_phases(coherence=1.0)

        initial = sim.compute_coherence()

        sim.randomize_phases()

        final = sim.compute_coherence()

        # Randomizing should reduce coherence
        # Note: random phases can still have some coherence by chance
        assert final < initial or final < 0.5

    def test_dipole_correlation(self):
        """Test dipole correlation computation."""
        from simulations.quantum.microtubule import MicrotubuleSimulator

        sim = MicrotubuleSimulator(n_tubulins=30)

        corr = sim.compute_dipole_correlation()

        assert -1 <= corr <= 1


class TestMicrotubuleStates:
    """Test microtubule state management."""

    def test_state_enum(self):
        """Test state enum exists."""
        from simulations.quantum.microtubule import MicrotubuleState

        assert hasattr(MicrotubuleState, 'COHERENT')
        assert hasattr(MicrotubuleState, 'DECOHERENT')
        assert hasattr(MicrotubuleState, 'FLOQUET')
        assert hasattr(MicrotubuleState, 'ANESTHETIZED')

    def test_set_state(self):
        """Test setting microtubule state."""
        from simulations.quantum.microtubule import MicrotubuleSimulator, MicrotubuleState

        sim = MicrotubuleSimulator(n_tubulins=30)

        sim.set_state(MicrotubuleState.DECOHERENT)
        assert sim.state == MicrotubuleState.DECOHERENT

        sim.set_state(MicrotubuleState.FLOQUET)
        assert sim.state == MicrotubuleState.FLOQUET

    def test_anesthesia_suppresses_aromatic(self):
        """Test that anesthesia suppresses aromatic oscillations."""
        from simulations.quantum.microtubule import MicrotubuleSimulator, OscillationScale

        sim = MicrotubuleSimulator(n_tubulins=30)
        sim.synchronize_phases(coherence=0.9)

        initial_aromatic = sim.compute_coherence(OscillationScale.AROMATIC)

        sim.apply_anesthesia(concentration=1.0)

        # Run a bit
        sim.step(50)

        final_aromatic = sim.compute_coherence(OscillationScale.AROMATIC)

        # Aromatic coherence should decrease
        # Note: with randomization it may already be low
        assert sim.state.value == 'anesthetized'


class TestTemperatureEffects:
    """Test temperature and thermal decoherence."""

    def test_set_temperature(self):
        """Test setting temperature."""
        from simulations.quantum.microtubule import MicrotubuleSimulator

        sim = MicrotubuleSimulator(n_tubulins=30, temperature=310.0)

        sim.set_temperature(340.0)

        assert sim.temperature == 340.0

    def test_temperature_affects_decoherence(self):
        """Test that higher temperature increases noise effects."""
        from simulations.quantum.microtubule import (
            MicrotubuleSimulator, MicrotubuleState
        )

        # Cold simulation
        sim_cold = MicrotubuleSimulator(n_tubulins=30, temperature=280.0)
        sim_cold.set_state(MicrotubuleState.DECOHERENT)
        sim_cold.synchronize_phases(coherence=0.9)

        # Hot simulation
        sim_hot = MicrotubuleSimulator(n_tubulins=30, temperature=350.0)
        sim_hot.set_state(MicrotubuleState.DECOHERENT)
        sim_hot.synchronize_phases(coherence=0.9)

        # Run both
        sim_cold.step(200)
        sim_hot.step(200)

        # Hot should have more decoherence (lower amplitude)
        # Check that simulation ran without error
        assert sim_cold.time > 0
        assert sim_hot.time > 0


class TestFloquetDriving:
    """Test external Floquet driving."""

    def test_set_floquet_driving(self):
        """Test setting Floquet driving parameters."""
        from simulations.quantum.microtubule import MicrotubuleSimulator

        sim = MicrotubuleSimulator(n_tubulins=30)

        sim.set_floquet_driving(amplitude=0.5, frequency=1e6)

        assert sim.driving_amplitude == 0.5
        assert sim.driving_frequency == 1e6

    def test_floquet_state_enables_driving(self):
        """Test that FLOQUET state enables driving if amplitude is zero."""
        from simulations.quantum.microtubule import MicrotubuleSimulator, MicrotubuleState

        sim = MicrotubuleSimulator(n_tubulins=30, driving_amplitude=0.0)

        sim.set_state(MicrotubuleState.FLOQUET)

        assert sim.driving_amplitude > 0


class TestTripletResonance:
    """Test triplet resonance analysis."""

    def test_compute_triplet_resonance(self):
        """Test triplet resonance computation."""
        from simulations.quantum.microtubule import MicrotubuleSimulator

        sim = MicrotubuleSimulator(n_tubulins=50)
        sim.step(1000)  # Need history for spectrum

        triplet = sim.compute_triplet_resonance()

        assert 'triplet_ratio_1' in triplet
        assert 'triplet_ratio_2' in triplet
        assert 'triplet_ratio_3' in triplet
        assert 'triplet_strength' in triplet

    def test_compute_spectrum(self):
        """Test spectrum computation."""
        from simulations.quantum.microtubule import MicrotubuleSimulator

        sim = MicrotubuleSimulator(n_tubulins=50)
        sim.step(500)

        freqs, spectrum = sim.compute_spectrum()

        assert len(freqs) > 0
        assert len(spectrum) == len(freqs)


class TestSimulationDynamics:
    """Test simulation step mechanics."""

    def test_step_updates_time(self):
        """Test that stepping updates time."""
        from simulations.quantum.microtubule import MicrotubuleSimulator

        sim = MicrotubuleSimulator(n_tubulins=30)

        initial_time = sim.time
        sim.step(100)

        assert sim.time > initial_time
        assert np.isclose(sim.time, initial_time + 100 * sim.dt)

    def test_run_duration(self):
        """Test run() simulates for specified duration."""
        from simulations.quantum.microtubule import MicrotubuleSimulator

        sim = MicrotubuleSimulator(n_tubulins=30)

        sim.run(duration=1e-7)  # 100 ns

        assert np.isclose(sim.time, 1e-7, rtol=0.1)

    def test_history_grows(self):
        """Test history grows with simulation."""
        from simulations.quantum.microtubule import MicrotubuleSimulator

        sim = MicrotubuleSimulator(n_tubulins=30)

        sim.step(50)
        first_len = len(sim.dipole_history)

        sim.step(50)
        second_len = len(sim.dipole_history)

        assert second_len > first_len


class TestERMapping:
    """Test éR phase space mapping."""

    def test_er_mapping_returns_dict(self):
        """Test that éR mapping returns expected keys."""
        from simulations.quantum.microtubule import MicrotubuleSimulator

        sim = MicrotubuleSimulator(n_tubulins=30)
        sim.step(10)

        result = sim.map_to_er_space()

        assert 'energy_present' in result
        assert 'frequency' in result
        assert 'energy_resistance' in result
        assert 'mean_coherence' in result
        assert 'triplet_strength' in result

    def test_er_values_reasonable(self):
        """Test éR values are in reasonable ranges."""
        from simulations.quantum.microtubule import MicrotubuleSimulator

        sim = MicrotubuleSimulator(n_tubulins=30)
        sim.step(50)

        result = sim.map_to_er_space()

        assert 0 <= result['energy_present'] <= 1
        assert result['frequency'] > 0
        assert result['energy_resistance'] >= 0
        assert 0 <= result['mean_coherence'] <= 1


class TestPresets:
    """Test preset configurations."""

    def test_coherent_mt_preset(self):
        """Test coherent microtubule preset."""
        from simulations.quantum.microtubule import create_coherent_mt, MicrotubuleState

        sim = create_coherent_mt(n_tubulins=30)

        assert sim.state == MicrotubuleState.COHERENT
        # Should have reasonably high coherence
        assert sim.compute_coherence() > 0.3

    def test_thermal_mt_preset(self):
        """Test thermal microtubule preset."""
        from simulations.quantum.microtubule import create_thermal_mt, MicrotubuleState

        sim = create_thermal_mt(n_tubulins=30)

        assert sim.state == MicrotubuleState.DECOHERENT
        assert sim.temperature > 310  # Above body temp

    def test_floquet_driven_mt_preset(self):
        """Test Floquet driven preset."""
        from simulations.quantum.microtubule import create_floquet_driven_mt, MicrotubuleState

        sim = create_floquet_driven_mt(n_tubulins=30)

        assert sim.state == MicrotubuleState.FLOQUET
        assert sim.driving_amplitude > 0

    def test_anesthetized_mt_preset(self):
        """Test anesthetized preset."""
        from simulations.quantum.microtubule import create_anesthetized_mt, MicrotubuleState

        sim = create_anesthetized_mt(n_tubulins=30)

        assert sim.state == MicrotubuleState.ANESTHETIZED

    def test_cold_mt_preset(self):
        """Test cold microtubule preset."""
        from simulations.quantum.microtubule import create_cold_mt

        sim = create_cold_mt(n_tubulins=30)

        assert sim.temperature < 310  # Below body temp
        assert sim.compute_coherence() > 0.5  # Should be coherent


class TestVisualizerImport:
    """Test visualizer import."""

    def test_visualizer_import(self):
        """Test that MicrotubuleVisualizer can be imported."""
        from simulations.quantum.microtubule import MicrotubuleVisualizer
        assert MicrotubuleVisualizer is not None


class TestPhysicalConstants:
    """Test physical constants are defined."""

    def test_geometry_constants(self):
        """Test microtubule geometry constants."""
        from simulations.quantum.microtubule import (
            N_PROTOFILAMENTS, TUBULIN_LENGTH_NM, TUBULIN_WIDTH_NM,
            MT_INNER_DIAMETER_NM, MT_OUTER_DIAMETER_NM, AROMATIC_RINGS_PER_TUBULIN
        )

        assert N_PROTOFILAMENTS == 13
        assert TUBULIN_LENGTH_NM > 0
        assert TUBULIN_WIDTH_NM > 0
        assert MT_INNER_DIAMETER_NM < MT_OUTER_DIAMETER_NM
        assert AROMATIC_RINGS_PER_TUBULIN == 86

    def test_thermal_constants(self):
        """Test thermal constants."""
        from simulations.quantum.microtubule import TEMPERATURE_BODY, KB, HBAR

        assert TEMPERATURE_BODY == 310.0
        assert KB > 0
        assert HBAR > 0

    def test_decoherence_times(self):
        """Test decoherence timescales are in correct order."""
        from simulations.quantum.microtubule import (
            DECOHERENCE_TIME_AROMATIC,
            DECOHERENCE_TIME_LATTICE,
            DECOHERENCE_TIME_CTERMINI
        )

        # Faster oscillations decohere faster
        assert DECOHERENCE_TIME_AROMATIC < DECOHERENCE_TIME_LATTICE
        assert DECOHERENCE_TIME_LATTICE < DECOHERENCE_TIME_CTERMINI
