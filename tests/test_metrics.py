"""
Tests for the shared CLT metrics module (analysis/metrics).

Covers Energy Resistance, the ViableWindow, the Kuramoto order parameter,
signal complexity (Lempel-Ziv, spectral entropy), and critical-slowing-down
indicators.
"""

import numpy as np
import pytest


class TestEnergyResistance:
    """éR = EP / f² calculation."""

    def test_scalar_formula(self):
        from analysis.metrics import calculate_er
        assert np.isclose(calculate_er(2.0, 1.0), 2.0)
        assert np.isclose(calculate_er(8.0, 2.0), 2.0)

    def test_scalar_returns_float(self):
        from analysis.metrics import calculate_er
        assert isinstance(calculate_er(2.0, 1.0), float)

    def test_vectorized(self):
        from analysis.metrics import calculate_er
        er = calculate_er(np.array([2.0, 8.0]), np.array([1.0, 2.0]))
        assert np.allclose(er, [2.0, 2.0])

    def test_zero_frequency_guarded(self):
        from analysis.metrics import calculate_er
        # Should not raise or return inf thanks to the epsilon guard.
        assert np.isfinite(calculate_er(1.0, 0.0))


class TestKuramotoOrder:
    """Kuramoto order parameter R = |mean(e^{iθ})|."""

    def test_locked_is_one(self):
        from analysis.metrics import kuramoto_order
        assert kuramoto_order(np.zeros(100)) > 0.999

    def test_random_is_low(self):
        from analysis.metrics import kuramoto_order
        phases = np.random.uniform(0, 2 * np.pi, 2000)
        assert kuramoto_order(phases) < 0.15

    def test_bounds(self):
        from analysis.metrics import kuramoto_order
        for _ in range(5):
            r = kuramoto_order(np.random.uniform(0, 2 * np.pi, 50))
            assert 0.0 <= r <= 1.0

    def test_empty_is_zero(self):
        from analysis.metrics import kuramoto_order
        assert kuramoto_order(np.array([])) == 0.0


class TestViableWindow:
    """Dynamic viable window and regime classification."""

    def test_classify(self):
        from analysis.metrics import BASELINE_WINDOW
        assert BASELINE_WINDOW.classify(0.2) == "chaos"
        assert BASELINE_WINDOW.classify(2.0) == "viable"
        assert BASELINE_WINDOW.classify(9.0) == "rigidity"

    def test_contains(self):
        from analysis.metrics import ViableWindow
        w = ViableWindow(0.5, 5.0)
        assert w.contains(2.0)
        assert not w.contains(0.1)
        assert not w.contains(10.0)

    def test_width_and_center(self):
        from analysis.metrics import ViableWindow
        w = ViableWindow(1.0, 5.0)
        assert np.isclose(w.width, 4.0)
        assert np.isclose(w.center, 3.0)

    def test_scaled_contracts(self):
        from analysis.metrics import ViableWindow
        w = ViableWindow(0.5, 5.0)
        narrow = w.scaled(0.5)
        assert narrow.width < w.width
        # Geometric center (√(er_min·er_max)) is preserved.
        assert np.isclose(narrow.er_min * narrow.er_max, w.er_min * w.er_max)

    def test_scaled_expands(self):
        from analysis.metrics import ViableWindow
        w = ViableWindow(1.0, 5.0)
        wide = w.scaled(2.0)
        assert wide.width > w.width
        # Widening stays positive even for a wide window.
        assert wide.er_min > 0.0

    def test_shifted(self):
        from analysis.metrics import ViableWindow
        w = ViableWindow(0.5, 5.0)
        s = w.shifted(2.0)
        assert np.isclose(s.er_min, 1.0)
        assert np.isclose(s.er_max, 10.0)

    def test_invalid_bounds_raise(self):
        from analysis.metrics import ViableWindow
        with pytest.raises(ValueError):
            ViableWindow(5.0, 1.0)
        with pytest.raises(ValueError):
            ViableWindow(-1.0, 5.0)

    def test_classify_regime_helper(self):
        from analysis.metrics import classify_regime, ViableWindow
        assert classify_regime(2.0) == "viable"
        # A contracted window can reclassify the same éR.
        contracted = ViableWindow(1.5, 2.5)
        assert classify_regime(1.0, contracted) == "chaos"

    def test_baseline_values(self):
        from analysis.metrics import BASELINE_ER_MIN, BASELINE_ER_MAX
        assert BASELINE_ER_MIN == 0.5
        assert BASELINE_ER_MAX == 5.0


class TestComplexity:
    """Signal complexity / diversity metrics."""

    def test_lz_random_exceeds_periodic(self):
        from analysis.metrics import lz_complexity
        t = np.linspace(0, 20 * np.pi, 1000)
        periodic = np.sin(t)
        noise = np.random.randn(1000)
        assert lz_complexity(noise) > lz_complexity(periodic)

    def test_lz_constant_is_low(self):
        from analysis.metrics import lz_complexity
        # A constant signal binarizes to all-zeros — minimal complexity.
        assert lz_complexity(np.ones(500)) < 0.05

    def test_lz_nonnegative(self):
        from analysis.metrics import lz_complexity
        assert lz_complexity(np.random.randn(200)) >= 0.0

    def test_lz_short_signal(self):
        from analysis.metrics import lz_complexity
        assert lz_complexity([1.0]) == 0.0

    def test_spectral_entropy_random_exceeds_periodic(self):
        from analysis.metrics import spectral_entropy
        t = np.linspace(0, 20 * np.pi, 1000)
        assert spectral_entropy(np.random.randn(1000)) > spectral_entropy(np.sin(t))

    def test_spectral_entropy_bounds(self):
        from analysis.metrics import spectral_entropy
        se = spectral_entropy(np.random.randn(500))
        assert 0.0 <= se <= 1.0


class TestCriticalSlowingDown:
    """Early-warning-signal indicators."""

    def test_rolling_lengths(self):
        from analysis.metrics import rolling_autocorr, rolling_variance
        series = np.random.randn(100)
        assert len(rolling_autocorr(series, 20)) == 81
        assert len(rolling_variance(series, 20)) == 81

    def test_indicators_keys(self):
        from analysis.metrics import csd_indicators
        ind = csd_indicators(np.random.randn(100), 20)
        assert set(ind.keys()) == {"ac1", "variance"}

    def test_window_too_small_raises(self):
        from analysis.metrics import rolling_variance
        with pytest.raises(ValueError):
            rolling_variance(np.random.randn(100), 1)

    def test_window_longer_than_series_raises(self):
        from analysis.metrics import rolling_variance
        with pytest.raises(ValueError):
            rolling_variance(np.random.randn(10), 20)

    def test_kendall_tau_rising(self):
        from analysis.metrics import kendall_tau_trend
        assert kendall_tau_trend(np.arange(10)) == pytest.approx(1.0)

    def test_kendall_tau_falling(self):
        from analysis.metrics import kendall_tau_trend
        assert kendall_tau_trend(np.arange(10)[::-1]) == pytest.approx(-1.0)

    def test_kendall_tau_short(self):
        from analysis.metrics import kendall_tau_trend
        assert kendall_tau_trend(np.array([1.0])) == 0.0

    def test_rising_variance_gives_positive_tau(self):
        from analysis.metrics import csd_indicators, kendall_tau_trend
        # Increasing-amplitude segments → rising variance.
        series = np.concatenate([np.random.randn(200) * s
                                 for s in np.linspace(0.2, 3.0, 5)])
        tau = kendall_tau_trend(csd_indicators(series, 50)["variance"])
        assert tau > 0.3
