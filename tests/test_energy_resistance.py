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


class TestBiologicalParameterMapping:
    """Test biological parameter conversion functions."""

    def test_map_hrv_to_frequency_typical_values(self):
        """Test HRV to frequency mapping with typical values."""
        from visualizations.interactive import map_hrv_to_frequency

        # Low HRV (stressed) -> high frequency
        freq_stressed = map_hrv_to_frequency(20.0)
        assert 2.5 < freq_stressed < 4.5

        # Medium HRV (normal) -> medium frequency
        freq_normal = map_hrv_to_frequency(50.0)
        assert 2.0 < freq_normal < 4.0

        # High HRV (relaxed) -> lower frequency (but still moderate)
        freq_relaxed = map_hrv_to_frequency(100.0)
        assert 1.5 < freq_relaxed < 3.5

    def test_map_hrv_to_frequency_inverse_relationship(self):
        """Higher HRV should map to lower frequency (more coherent)."""
        from visualizations.interactive import map_hrv_to_frequency

        freq_low_hrv = map_hrv_to_frequency(30.0)
        freq_high_hrv = map_hrv_to_frequency(80.0)

        assert freq_low_hrv > freq_high_hrv, "Higher HRV should give lower frequency"

    def test_map_hrv_to_frequency_clamping(self):
        """Test that extreme HRV values are clamped to valid range."""
        from visualizations.interactive import map_hrv_to_frequency

        # Very low HRV (should be clamped)
        freq_extreme_low = map_hrv_to_frequency(1.0)
        assert 0.3 <= freq_extreme_low <= 4.5

        # Very high HRV (should be clamped)
        freq_extreme_high = map_hrv_to_frequency(500.0)
        assert 0.3 <= freq_extreme_high <= 4.5

    def test_map_metabolic_rate_to_energy_typical_values(self):
        """Test metabolic rate to energy mapping."""
        from visualizations.interactive import map_metabolic_rate_to_energy

        # Resting (low metabolic rate) -> low energy
        ep_resting = map_metabolic_rate_to_energy(70.0)
        assert 1.5 < ep_resting < 3.0

        # Active (medium metabolic rate) -> medium energy
        ep_active = map_metabolic_rate_to_energy(200.0)
        assert 2.5 < ep_active < 4.5

        # Exercise (high metabolic rate) -> high energy
        ep_exercise = map_metabolic_rate_to_energy(600.0)
        assert 5.0 < ep_exercise < 8.0

    def test_map_metabolic_rate_to_energy_direct_relationship(self):
        """Higher metabolic rate should map to higher energy."""
        from visualizations.interactive import map_metabolic_rate_to_energy

        ep_low = map_metabolic_rate_to_energy(80.0)
        ep_high = map_metabolic_rate_to_energy(400.0)

        assert ep_high > ep_low, "Higher metabolic rate should give higher energy"

    def test_map_metabolic_rate_to_energy_clamping(self):
        """Test that extreme metabolic values are clamped."""
        from visualizations.interactive import map_metabolic_rate_to_energy

        # Very low metabolic rate
        ep_extreme_low = map_metabolic_rate_to_energy(10.0)
        assert 0.5 <= ep_extreme_low <= 10.0

        # Very high metabolic rate
        ep_extreme_high = map_metabolic_rate_to_energy(2000.0)
        assert 0.5 <= ep_extreme_high <= 10.0

    def test_map_eeg_band_to_frequency(self):
        """Test EEG band to CLT frequency mapping."""
        from visualizations.interactive import map_eeg_band_to_frequency

        # Delta (deep sleep) -> lowest frequency
        freq_delta = map_eeg_band_to_frequency('delta')
        assert freq_delta < 1.0

        # Theta (drowsy/meditation)
        freq_theta = map_eeg_band_to_frequency('theta')
        assert 0.5 <= freq_theta < 1.5

        # Alpha (relaxed awake)
        freq_alpha = map_eeg_band_to_frequency('alpha')
        assert 0.8 <= freq_alpha < 2.0

        # Beta (active thinking)
        freq_beta = map_eeg_band_to_frequency('beta')
        assert 1.5 <= freq_beta < 3.5

        # Gamma (high cognition)
        freq_gamma = map_eeg_band_to_frequency('gamma')
        assert freq_gamma >= 2.5

    def test_map_eeg_band_order(self):
        """EEG bands should map to increasing frequency: delta < theta < alpha < beta < gamma."""
        from visualizations.interactive import map_eeg_band_to_frequency

        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        frequencies = [map_eeg_band_to_frequency(band) for band in bands]

        for i in range(len(frequencies) - 1):
            assert frequencies[i] < frequencies[i + 1], \
                f"{bands[i]} ({frequencies[i]}) should be < {bands[i+1]} ({frequencies[i+1]})"

    def test_biological_state_to_er(self):
        """Test conversion of biological state names to éR parameters."""
        from visualizations.interactive import biological_state_to_er

        # Test a known state
        result = biological_state_to_er('resting_awake')
        assert 'energy_present' in result
        assert 'frequency' in result
        assert 'energy_resistance' in result
        assert result['energy_present'] > 0
        assert result['frequency'] > 0
        assert result['energy_resistance'] > 0

    def test_biological_state_to_er_unknown_state(self):
        """Test that unknown states raise ValueError."""
        from visualizations.interactive import biological_state_to_er

        with pytest.raises(ValueError):
            biological_state_to_er('nonexistent_state')


class TestBiologicalStatesData:
    """Test the BIOLOGICAL_STATES reference data."""

    def test_biological_states_exists(self):
        """Test that BIOLOGICAL_STATES is properly defined."""
        from visualizations.interactive import BIOLOGICAL_STATES

        assert isinstance(BIOLOGICAL_STATES, dict)
        assert len(BIOLOGICAL_STATES) >= 6  # At least 6 states defined

    def test_biological_states_required_keys(self):
        """Test that each biological state has required keys."""
        from visualizations.interactive import BIOLOGICAL_STATES

        required_keys = ['ep', 'freq', 'label', 'description', 'color']

        for state_name, state_data in BIOLOGICAL_STATES.items():
            for key in required_keys:
                assert key in state_data, f"State '{state_name}' missing key '{key}'"

    def test_biological_states_valid_ranges(self):
        """Test that biological state values are in valid ranges."""
        from visualizations.interactive import BIOLOGICAL_STATES

        for state_name, state_data in BIOLOGICAL_STATES.items():
            assert 0 < state_data['ep'] <= 10.0, \
                f"State '{state_name}' EP out of range: {state_data['ep']}"
            assert 0 < state_data['freq'] <= 5.0, \
                f"State '{state_name}' freq out of range: {state_data['freq']}"


class TestPathologyZonesData:
    """Test the PATHOLOGY_ZONES reference data."""

    def test_pathology_zones_exists(self):
        """Test that PATHOLOGY_ZONES is properly defined."""
        from visualizations.interactive import PATHOLOGY_ZONES

        assert isinstance(PATHOLOGY_ZONES, dict)
        assert len(PATHOLOGY_ZONES) >= 6  # At least 6 pathologies defined

    def test_pathology_zones_required_keys(self):
        """Test that each pathology zone has required keys."""
        from visualizations.interactive import PATHOLOGY_ZONES

        required_keys = ['ep_center', 'freq_center', 'ep_spread', 'freq_spread',
                         'label', 'description', 'regime', 'color']

        for pathology_name, pathology_data in PATHOLOGY_ZONES.items():
            for key in required_keys:
                assert key in pathology_data, f"Pathology '{pathology_name}' missing key '{key}'"

    def test_pathology_zones_valid_regimes(self):
        """Test that pathology regimes are valid."""
        from visualizations.interactive import PATHOLOGY_ZONES

        valid_regimes = {'chaos', 'rigidity', 'boundary'}

        for pathology_name, pathology_data in PATHOLOGY_ZONES.items():
            assert pathology_data['regime'] in valid_regimes, \
                f"Pathology '{pathology_name}' has invalid regime: {pathology_data['regime']}"

    def test_pathology_zones_valid_ranges(self):
        """Test that pathology zone values are in valid ranges."""
        from visualizations.interactive import PATHOLOGY_ZONES

        for pathology_name, pathology_data in PATHOLOGY_ZONES.items():
            assert 0 < pathology_data['ep_center'] <= 10.0, \
                f"Pathology '{pathology_name}' EP center out of range"
            assert 0 < pathology_data['freq_center'] <= 5.0, \
                f"Pathology '{pathology_name}' freq center out of range"
            assert pathology_data['ep_spread'] > 0, \
                f"Pathology '{pathology_name}' EP spread must be positive"
            assert pathology_data['freq_spread'] > 0, \
                f"Pathology '{pathology_name}' freq spread must be positive"
