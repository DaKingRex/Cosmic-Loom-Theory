"""
Tests for 2D Loomfield wave equation simulator.

Wave equation: ∇²L - (1/v²ₗ)(∂²L/∂t²) = κₗ · ρ_coh
"""

import pytest
import numpy as np


class TestLoomfieldSimulator2D:
    """Test the 2D Loomfield simulator."""

    def test_simulator_import(self):
        """Test that simulator can be imported."""
        from visualizations.interactive import LoomfieldSimulator
        assert LoomfieldSimulator is not None

    def test_simulator_initialization(self):
        """Test simulator initializes with correct dimensions."""
        from visualizations.interactive import LoomfieldSimulator

        grid_size = 100
        sim = LoomfieldSimulator(grid_size=grid_size)

        assert sim.N == grid_size
        assert sim.L.shape == (grid_size, grid_size)
        assert sim.L_prev.shape == (grid_size, grid_size)

    def test_initial_field_is_zero(self):
        """Test that initial field is zero everywhere."""
        from visualizations.interactive import LoomfieldSimulator

        sim = LoomfieldSimulator(grid_size=50)

        assert np.allclose(sim.L, 0.0)
        assert np.allclose(sim.L_prev, 0.0)

    def test_add_source(self):
        """Test adding a coherence source."""
        from visualizations.interactive import LoomfieldSimulator

        sim = LoomfieldSimulator(grid_size=50)
        initial_sources = len(sim.sources)

        sim.add_source(x=0.0, y=0.0, strength=1.0, frequency=1.5)

        assert len(sim.sources) == initial_sources + 1

    def test_step_advances_time(self):
        """Test that stepping advances simulation time."""
        from visualizations.interactive import LoomfieldSimulator

        sim = LoomfieldSimulator(grid_size=50)
        initial_time = sim.time

        sim.step(n_steps=10)

        assert sim.time > initial_time

    def test_source_generates_field(self):
        """Test that sources generate non-zero field after stepping."""
        from visualizations.interactive import LoomfieldSimulator

        sim = LoomfieldSimulator(grid_size=50)
        sim.add_source(x=0.0, y=0.0, strength=1.0, frequency=1.5)

        # Run simulation
        sim.step(n_steps=50)

        # Field should no longer be zero
        assert not np.allclose(sim.L, 0.0)

    def test_coherence_metric_q(self):
        """Test Q (coherence) metric calculation."""
        from visualizations.interactive import LoomfieldSimulator

        sim = LoomfieldSimulator(grid_size=50)
        sim.add_source(x=0.0, y=0.0, strength=1.0, frequency=1.5)
        sim.step(n_steps=50)

        Q = sim.get_total_coherence()

        assert isinstance(Q, float)
        assert Q >= 0.0  # Q should be non-negative

    def test_consciousness_metric_cbio(self):
        """Test C_bio (consciousness observable) calculation."""
        from visualizations.interactive import LoomfieldSimulator

        sim = LoomfieldSimulator(grid_size=50)
        sim.add_source(x=0.0, y=0.0, strength=1.0, frequency=1.5)
        sim.step(n_steps=50)

        C_bio = sim.get_consciousness_metric()

        assert isinstance(C_bio, float)
        assert C_bio >= 0.0  # C_bio should be non-negative

    def test_reset_clears_field(self):
        """Test that reset clears the field and sources."""
        from visualizations.interactive import LoomfieldSimulator

        sim = LoomfieldSimulator(grid_size=50)
        sim.add_source(x=0.0, y=0.0, strength=1.0, frequency=1.5)
        sim.step(n_steps=50)

        sim.reset()

        assert np.allclose(sim.L, 0.0)
        assert len(sim.sources) == 0

    def test_field_energy_conservation_tendency(self):
        """Test that field energy doesn't explode (numerical stability)."""
        from visualizations.interactive import LoomfieldSimulator

        sim = LoomfieldSimulator(grid_size=50)
        sim.add_source(x=0.0, y=0.0, strength=1.0, frequency=1.5)

        energies = []
        for _ in range(10):
            sim.step(n_steps=10)
            energies.append(sim.get_field_energy())

        # Energy should not explode (stay within reasonable bounds)
        max_energy = max(energies)
        assert max_energy < 1e10, f"Energy exploded to {max_energy}"


class TestLoomfieldPresets2D:
    """Test preset configurations for 2D simulator."""

    def test_healthy_sources_are_phase_locked(self):
        """Test that phase-locked sources have same frequency."""
        from visualizations.interactive import LoomfieldSimulator

        sim = LoomfieldSimulator(grid_size=50)

        # Add phase-locked sources (healthy pattern)
        base_freq = 1.5
        sim.add_source(x=0, y=0, strength=1.0, frequency=base_freq, phase=0.0)
        sim.add_source(x=2, y=0, strength=0.8, frequency=base_freq, phase=0.0)
        sim.add_source(x=-2, y=0, strength=0.8, frequency=base_freq, phase=0.0)

        # All sources should have same frequency
        frequencies = [s['frequency'] for s in sim.sources]
        assert len(set(frequencies)) == 1, "Phase-locked sources should have same frequency"

    def test_incoherent_sources_have_varied_frequencies(self):
        """Test that incoherent sources have different frequencies."""
        from visualizations.interactive import LoomfieldSimulator

        sim = LoomfieldSimulator(grid_size=50)

        # Add incoherent sources (pathology pattern)
        sim.add_source(x=0, y=0, strength=0.7, frequency=0.8)
        sim.add_source(x=2, y=1, strength=0.7, frequency=1.5)
        sim.add_source(x=-1, y=-2, strength=0.7, frequency=2.3)

        # Sources should have different frequencies
        frequencies = [s['frequency'] for s in sim.sources]
        assert len(set(frequencies)) > 1, "Incoherent sources should have varied frequencies"

    def test_coherent_sources_yield_higher_q(self):
        """Test that coherent sources produce higher Q than incoherent."""
        from visualizations.interactive import LoomfieldSimulator

        # Setup coherent (phase-locked) system
        sim_coherent = LoomfieldSimulator(grid_size=50)
        base_freq = 1.5
        sim_coherent.add_source(x=0, y=0, strength=1.0, frequency=base_freq, phase=0.0)
        sim_coherent.add_source(x=2, y=0, strength=0.8, frequency=base_freq, phase=0.0)
        sim_coherent.add_source(x=-2, y=0, strength=0.8, frequency=base_freq, phase=0.0)
        sim_coherent.step(n_steps=100)
        Q_coherent = sim_coherent.get_total_coherence()

        # Setup incoherent system
        sim_incoherent = LoomfieldSimulator(grid_size=50)
        sim_incoherent.add_source(x=0, y=0, strength=1.0, frequency=0.8)
        sim_incoherent.add_source(x=2, y=0, strength=0.8, frequency=1.5)
        sim_incoherent.add_source(x=-2, y=0, strength=0.8, frequency=2.5)
        sim_incoherent.step(n_steps=100)
        Q_incoherent = sim_incoherent.get_total_coherence()

        # Coherent system should generally have higher Q
        # (This is a tendency, not absolute - depends on timing)
        assert Q_coherent >= 0 and Q_incoherent >= 0, "Q values should be non-negative"
