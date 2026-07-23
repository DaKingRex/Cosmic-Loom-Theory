"""
Tests for the Phase 4 analysis harness (analysis/eeg/analysis.py).

Window calibration and trajectory assembly are pure numpy and always tested; the
recording-level `analyze_recording` uses a synthetic mne file and `importorskip("mne")`
so CI stays light. Follows repo conventions (class-grouped, imports-inside-tests,
autouse seed).
"""

import numpy as np
import pytest


class TestCalibrateWindow:
    def test_window_spans_baseline_percentiles(self):
        from analysis.eeg.analysis import calibrate_window_from_baseline
        er = np.linspace(1.0, 3.0, 101)          # baseline éR distribution
        w = calibrate_window_from_baseline(er, lo_q=5.0, hi_q=95.0)
        assert w.er_min > 1.0 and w.er_max < 3.0   # trims the tails
        assert w.er_min < w.er_max

    def test_degenerate_baseline_falls_back(self):
        from analysis.eeg.analysis import calibrate_window_from_baseline
        w = calibrate_window_from_baseline(np.full(50, 2.0))   # zero spread
        assert w.er_min > 0 and w.er_min < w.er_max            # valid window, no crash


class TestStateTrajectory:
    def _fake(self, er_values):
        return {"summary": {"energy_resistance": float(np.mean(er_values)),
                            "frequency": 10.0, "coherence": 0.3, "lz_complexity": 0.5},
                "per_epoch": {"energy_resistance": np.asarray(er_values, float)}}

    def test_deepest_state_reads_as_rigidity(self):
        from analysis.eeg.analysis import state_regime_trajectory
        # Baseline éR ~1; a "deep" state with far higher éR should exceed the window.
        analyses = {
            "baseline": self._fake(np.random.normal(1.0, 0.1, 200)),
            "deep": self._fake(np.random.normal(5.0, 0.1, 200)),
            "recovery": self._fake(np.random.normal(1.1, 0.1, 200)),
        }
        out = state_regime_trajectory(analyses, baseline_key="baseline")
        assert out["states"]["baseline"]["regime"] == "viable"
        assert out["states"]["deep"]["regime"] == "rigidity"     # toward the rigid wall
        assert out["states"]["recovery"]["regime"] == "viable"   # rebounds
        # occupancy: the deep state spends far more epochs in rigidity than baseline
        assert out["states"]["deep"]["occupancy"]["rigidity"] > \
            out["states"]["baseline"]["occupancy"]["rigidity"]

    def test_missing_baseline_raises(self):
        from analysis.eeg.analysis import state_regime_trajectory
        with pytest.raises(KeyError):
            state_regime_trajectory({"deep": self._fake([2.0, 2.0])}, baseline_key="baseline")


class TestAnalyzeRecording:
    def test_analyze_synthetic_recording(self, tmp_path):
        pytest.importorskip("mne")
        import mne
        from analysis.eeg.analysis import analyze_recording
        fs, secs, n_ch = 200.0, 30, 6
        data = np.random.standard_normal((n_ch, int(fs * secs))) * 1e-5
        info = mne.create_info([f"CH{i}" for i in range(n_ch)], fs, ch_types="eeg")
        path = str(tmp_path / "sub-x_raw.fif")
        mne.io.RawArray(data, info, verbose="ERROR").save(path, overwrite=True, verbose="ERROR")

        out = analyze_recording(path, epoch_seconds=5.0)
        assert out["n_epochs"] == 6                       # 30 s / 5 s
        assert out["fs"] == 200.0
        for k in ("energy_resistance", "frequency", "coherence", "lz_complexity"):
            assert k in out["summary"] and k in out["per_epoch"]
            assert out["per_epoch"][k].shape == (6,)

    def test_resample_reduces_rate(self, tmp_path):
        pytest.importorskip("mne")
        import mne
        from analysis.eeg.analysis import analyze_recording
        fs, secs, n_ch = 250.0, 20, 4
        data = np.random.standard_normal((n_ch, int(fs * secs))) * 1e-5
        info = mne.create_info([f"CH{i}" for i in range(n_ch)], fs, ch_types="eeg")
        path = str(tmp_path / "sub-y_raw.fif")
        mne.io.RawArray(data, info, verbose="ERROR").save(path, overwrite=True, verbose="ERROR")

        out = analyze_recording(path, epoch_seconds=5.0, resample=125.0)
        assert out["fs"] == 125.0                          # decimated 250 -> 125
