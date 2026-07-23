"""
Tests for the Phase 4 EEG observable core (analysis/eeg/observables.py).

Verified on synthetic signals with known properties (the "develop the pipeline on
simulated data first" discipline), so the measurement functions are correct before any
real recording is touched. Follows the repo conventions: class-grouped, imports-inside
-tests, and the autouse seed-42 fixture from conftest.
"""

import numpy as np

FS = 250.0
DUR = 4.0
N = int(FS * DUR)
_T = np.arange(N) / FS


def _sines(freq, n_channels=8, amp=1.0, phases=None):
    """Build an (n_channels, N) block of sinusoids at `freq` Hz."""
    if phases is None:
        phases = np.zeros(n_channels)
    return np.stack([amp * np.sin(2 * np.pi * freq * _T + p) for p in phases])


class TestPhaseCoherence:
    def test_in_phase_is_high_random_is_low(self):
        from analysis.eeg.observables import phase_coherence
        in_phase = _sines(10.0, n_channels=8)                       # identical phase
        random = _sines(10.0, n_channels=8,
                        phases=np.random.uniform(0, 2 * np.pi, 8))  # scattered phase
        c_in = phase_coherence(in_phase)
        c_rand = phase_coherence(random)
        assert c_in > 0.9
        assert c_rand < c_in
        assert 0.0 <= c_rand <= 1.0

    def test_single_channel_is_trivially_coherent(self):
        from analysis.eeg.observables import phase_coherence
        assert phase_coherence(_sines(10.0, n_channels=1)) == 1.0


class TestDominantFrequency:
    def test_recovers_sinusoid_frequency(self):
        from analysis.eeg.observables import dominant_frequency
        for freq in (6.0, 10.0, 20.0):
            f = dominant_frequency(_sines(freq, n_channels=4), fs=FS)
            assert abs(f - freq) < 2.0


class TestEnergyPresent:
    def test_scales_with_amplitude_squared(self):
        from analysis.eeg.observables import energy_present
        ep1 = energy_present(_sines(10.0, amp=1.0))
        ep2 = energy_present(_sines(10.0, amp=2.0))
        assert abs(ep2 / ep1 - 4.0) < 0.2   # doubling amplitude ~quadruples power


class TestErProxy:
    def test_slower_signal_has_higher_er(self):
        from analysis.eeg.observables import er_proxy
        # Same power, different dominant frequency: éR = EP / f² ⇒ slower ⇒ higher éR.
        slow = er_proxy(_sines(5.0), fs=FS)
        fast = er_proxy(_sines(20.0), fs=FS)
        assert slow["energy_resistance"] > fast["energy_resistance"]

    def test_returns_map_to_er_space_keys(self):
        from analysis.eeg.observables import er_proxy
        out = er_proxy(_sines(10.0), fs=FS)
        assert set(out) == {"energy_present", "frequency", "energy_resistance", "coherence"}


class TestComplexity:
    def test_random_more_complex_than_periodic(self):
        from analysis.eeg.observables import complexity
        periodic = _sines(10.0, n_channels=6)
        noise = np.random.standard_normal((6, N))
        assert complexity(noise, FS)["lz_complexity"] > complexity(periodic, FS)["lz_complexity"]


class TestEpochObservables:
    def test_bundle_has_all_keys(self):
        from analysis.eeg.observables import epoch_observables
        out = epoch_observables(_sines(10.0), fs=FS)
        assert set(out) == {
            "energy_present", "frequency", "energy_resistance",
            "coherence", "lz_complexity", "spectral_entropy",
        }

    def test_rigid_vs_flexible_regime_signatures(self):
        """A slow, phase-locked, periodic epoch (rigidity) should read as higher éR and
        lower complexity than a fast, incoherent, broadband epoch (toward chaos) — the
        CLT anesthesia-vs-arousal contrast, on synthetic data."""
        from analysis.eeg.observables import epoch_observables
        rigid = _sines(3.0, n_channels=8, amp=2.0)          # slow, high-power, locked
        flexible = np.random.standard_normal((8, N))         # broadband, incoherent
        r = epoch_observables(rigid, FS)
        f = epoch_observables(flexible, FS)
        assert r["energy_resistance"] > f["energy_resistance"]
        assert r["lz_complexity"] < f["lz_complexity"]
        assert r["coherence"] > f["coherence"]
