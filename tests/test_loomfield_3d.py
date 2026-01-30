"""
Tests for 3D Loomfield wave equation simulator.

Wave equation (3D): ∇²L - (1/v²ₗ)(∂²L/∂t²) = κₗ · ρ_coh
"""

import pytest
import numpy as np


class TestLoomfieldSimulator3D:
    """Test the 3D Loomfield simulator."""

    def test_simulator_import(self):
        """Test that 3D simulator can be imported."""
        from visualizations.interactive import LoomfieldSimulator3D
        assert LoomfieldSimulator3D is not None

    def test_simulator_initialization(self):
        """Test simulator initializes with correct 3D dimensions."""
        from visualizations.interactive import LoomfieldSimulator3D

        grid_size = 32
        sim = LoomfieldSimulator3D(grid_size=grid_size)

        assert sim.N == grid_size
        assert sim.L.shape == (grid_size, grid_size, grid_size)
        assert sim.L_prev.shape == (grid_size, grid_size, grid_size)

    def test_initial_field_is_zero(self):
        """Test that initial 3D field is zero everywhere."""
        from visualizations.interactive import LoomfieldSimulator3D

        sim = LoomfieldSimulator3D(grid_size=24)

        assert np.allclose(sim.L, 0.0)
        assert np.allclose(sim.L_prev, 0.0)

    def test_add_source_3d(self):
        """Test adding a 3D coherence source."""
        from visualizations.interactive import LoomfieldSimulator3D

        sim = LoomfieldSimulator3D(grid_size=24)
        initial_sources = len(sim.sources)

        sim.add_source(x=0.0, y=0.0, z=0.0, strength=1.0, frequency=1.5)

        assert len(sim.sources) == initial_sources + 1
        assert 'z' in sim.sources[-1]  # Should have z coordinate

    def test_step_advances_time_3d(self):
        """Test that stepping advances simulation time."""
        from visualizations.interactive import LoomfieldSimulator3D

        sim = LoomfieldSimulator3D(grid_size=24)
        initial_time = sim.time

        sim.step(n_steps=5)

        assert sim.time > initial_time

    def test_source_generates_field_3d(self):
        """Test that 3D sources generate non-zero field."""
        from visualizations.interactive import LoomfieldSimulator3D

        sim = LoomfieldSimulator3D(grid_size=24)
        sim.add_source(x=0.0, y=0.0, z=0.0, strength=1.0, frequency=1.5)

        # Run simulation
        sim.step(n_steps=30)

        # Field should no longer be zero
        assert not np.allclose(sim.L, 0.0)

    def test_coherence_metric_q_3d(self):
        """Test Q metric calculation for 3D."""
        from visualizations.interactive import LoomfieldSimulator3D

        sim = LoomfieldSimulator3D(grid_size=24)
        sim.add_source(x=0.0, y=0.0, z=0.0, strength=1.0, frequency=1.5)
        sim.step(n_steps=30)

        Q = sim.get_total_coherence()

        assert isinstance(Q, float)
        assert Q >= 0.0

    def test_consciousness_metric_cbio_3d(self):
        """Test C_bio calculation for 3D (uses Q³ instead of Q²)."""
        from visualizations.interactive import LoomfieldSimulator3D

        sim = LoomfieldSimulator3D(grid_size=24)
        sim.add_source(x=0.0, y=0.0, z=0.0, strength=1.0, frequency=1.5)
        sim.step(n_steps=30)

        C_bio = sim.get_consciousness_metric()

        assert isinstance(C_bio, float)
        assert C_bio >= 0.0

    def test_3d_field_energy(self):
        """Test field energy calculation in 3D."""
        from visualizations.interactive import LoomfieldSimulator3D

        sim = LoomfieldSimulator3D(grid_size=24)
        sim.add_source(x=0.0, y=0.0, z=0.0, strength=1.0, frequency=1.5)
        sim.step(n_steps=30)

        energy = sim.get_field_energy()

        assert isinstance(energy, float)
        assert energy >= 0.0

    def test_reset_clears_3d_field(self):
        """Test that reset clears the 3D field."""
        from visualizations.interactive import LoomfieldSimulator3D

        sim = LoomfieldSimulator3D(grid_size=24)
        sim.add_source(x=0.0, y=0.0, z=0.0, strength=1.0, frequency=1.5)
        sim.step(n_steps=30)

        sim.reset()

        assert np.allclose(sim.L, 0.0)
        assert len(sim.sources) == 0

    def test_numerical_stability_3d(self):
        """Test that 3D simulation doesn't explode."""
        from visualizations.interactive import LoomfieldSimulator3D

        sim = LoomfieldSimulator3D(grid_size=24)
        sim.add_source(x=0.0, y=0.0, z=0.0, strength=1.0, frequency=1.5)

        # Run for many steps
        for _ in range(5):
            sim.step(n_steps=20)
            max_val = np.abs(sim.L).max()
            assert max_val < 1e6, f"Field exploded to {max_val}"


class TestLoomfieldPresets3D:
    """Test 3D preset configurations."""

    def test_healthy_preset_3d(self):
        """Test healthy preset for 3D simulator."""
        from visualizations.interactive import LoomfieldSimulator3D
        from visualizations.interactive.loomfield_3d import create_healthy_preset

        sim = LoomfieldSimulator3D(grid_size=24)
        create_healthy_preset(sim)

        # Should have multiple sources
        assert len(sim.sources) > 1

        # All sources should have same frequency
        frequencies = [s['frequency'] for s in sim.sources]
        assert len(set(frequencies)) == 1

    def test_pathology_preset_3d(self):
        """Test pathology preset for 3D simulator."""
        from visualizations.interactive import LoomfieldSimulator3D
        from visualizations.interactive.loomfield_3d import create_pathology_preset

        sim = LoomfieldSimulator3D(grid_size=24)
        create_pathology_preset(sim)

        # Should have multiple sources with varied frequencies
        assert len(sim.sources) > 1
        frequencies = [s['frequency'] for s in sim.sources]
        assert len(set(frequencies)) > 1

    def test_healing_preset_3d(self):
        """Test healing preset for 3D simulator."""
        from visualizations.interactive import LoomfieldSimulator3D
        from visualizations.interactive.loomfield_3d import create_healing_preset

        sim = LoomfieldSimulator3D(grid_size=24)
        create_healing_preset(sim)

        # Should have sources
        assert len(sim.sources) > 0


class TestLoomfieldVisualization3D:
    """Test 3D visualization functions."""

    def test_volumetric_figure_import(self):
        """Test volumetric figure function can be imported."""
        from visualizations.interactive import create_volumetric_figure
        assert callable(create_volumetric_figure)

    def test_slice_figure_import(self):
        """Test slice figure function can be imported."""
        from visualizations.interactive import create_slice_figure
        assert callable(create_slice_figure)
