"""
Tests for Energy Resistance (éR) calculations.

éR = EP / f² is the core relationship defining the viable window
for biological coherence in Cosmic Loom Theory.
"""

import pytest
import numpy as np


class TestEnergyResistanceCalculations:
    """Test the éR = EP/f² formula and regime classification."""

    def test_er_formula_basic(self):
        """Test basic éR calculation: éR = EP / f²"""
        EP = 100.0  # Energy Present
        f = 10.0    # Frequency

        eR = EP / (f ** 2)

        assert eR == 1.0, f"Expected éR=1.0, got {eR}"

    def test_er_increases_with_energy(self):
        """Higher energy at same frequency increases éR."""
        f = 5.0
        eR_low = 10.0 / (f ** 2)
        eR_high = 100.0 / (f ** 2)

        assert eR_high > eR_low

    def test_er_decreases_with_frequency(self):
        """Higher frequency at same energy decreases éR."""
        EP = 100.0
        eR_low_freq = EP / (2.0 ** 2)
        eR_high_freq = EP / (10.0 ** 2)

        assert eR_low_freq > eR_high_freq

    def test_viable_window_boundaries(self):
        """
        Test viable window concept:
        - Too low éR = chaos/decoherence
        - Viable range = coherent dynamics
        - Too high éR = rigidity/frozen
        """
        # Define viable window (example values)
        eR_min = 0.1   # Below this: chaos
        eR_max = 10.0  # Above this: rigidity

        # Test classification
        eR_chaos = 0.05
        eR_viable = 1.0
        eR_rigid = 15.0

        assert eR_chaos < eR_min, "Chaos regime should be below minimum"
        assert eR_min <= eR_viable <= eR_max, "Viable should be in window"
        assert eR_rigid > eR_max, "Rigidity regime should be above maximum"

    def test_er_never_negative(self):
        """éR should never be negative for physical values."""
        # Both EP and f² are positive quantities
        EP_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
        f_values = [0.1, 1.0, 5.0, 10.0, 50.0]

        for EP in EP_values:
            for f in f_values:
                eR = EP / (f ** 2)
                assert eR > 0, f"éR should be positive, got {eR} for EP={EP}, f={f}"

    def test_er_zero_frequency_undefined(self):
        """éR is undefined at f=0 (division by zero)."""
        EP = 100.0
        f = 0.0

        with pytest.raises(ZeroDivisionError):
            eR = EP / (f ** 2)


class TestEnergyResistanceVisualizer:
    """Test the EnergyResistanceVisualizer class."""

    def test_visualizer_import(self):
        """Test that visualizer can be imported."""
        from visualizations.interactive import EnergyResistanceVisualizer
        assert EnergyResistanceVisualizer is not None

    def test_calculate_system_er_import(self):
        """Test that calculate_system_er function exists."""
        from visualizations.interactive import calculate_system_er
        assert callable(calculate_system_er)

    def test_calculate_system_er_basic(self):
        """Test basic system éR calculation."""
        from visualizations.interactive import calculate_system_er

        # Calculate éR for a simple system
        result = calculate_system_er(energy_present=100.0, frequency=10.0)

        # Returns a dict with energy_resistance and other info
        assert isinstance(result, dict)
        assert 'energy_resistance' in result
        assert result['energy_resistance'] > 0
        assert result['energy_resistance'] == 1.0  # 100 / 10² = 1.0
