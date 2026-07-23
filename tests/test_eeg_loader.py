"""
Tests for the Phase 4 EEG loader (analysis/eeg/loader.py).

`epoch_data` is pure numpy and always tested. The file-reading tests use a synthetic
recording that mne writes to a temp file, so they exercise the real read path without
needing any downloaded dataset; they `importorskip("mne")` so CI stays light (mne is not
in the CI install set, and the loader imports mne lazily so the package still imports).
"""

import numpy as np
import pytest


class TestEpochData:
    def test_shapes_and_count(self):
        from analysis.eeg.loader import epoch_data
        data = np.random.standard_normal((4, 1000))
        ep = epoch_data(data, fs=100.0, epoch_seconds=2.0)   # 200-sample windows
        assert ep.shape == (5, 4, 200)

    def test_overlap_increases_epoch_count(self):
        from analysis.eeg.loader import epoch_data
        data = np.random.standard_normal((3, 1000))
        no_ov = epoch_data(data, fs=100.0, epoch_seconds=2.0, overlap=0.0)
        half = epoch_data(data, fs=100.0, epoch_seconds=2.0, overlap=0.5)
        assert half.shape[0] > no_ov.shape[0]

    def test_too_short_returns_empty(self):
        from analysis.eeg.loader import epoch_data
        data = np.random.standard_normal((4, 100))
        ep = epoch_data(data, fs=100.0, epoch_seconds=2.0)   # window 200 > 100 samples
        assert ep.shape == (0, 4, 200)

    def test_invalid_overlap_raises(self):
        from analysis.eeg.loader import epoch_data
        with pytest.raises(ValueError):
            epoch_data(np.zeros((2, 100)), fs=100.0, epoch_seconds=1.0, overlap=1.0)


def _make_raw(tmp_path, n_ch=4, fs=100.0, secs=5, ch_types=None):
    """Write a synthetic FIF recording and return its path + expected params."""
    import mne
    if ch_types is None:
        ch_types = ["eeg"] * n_ch
    data = np.random.standard_normal((n_ch, int(fs * secs))) * 1e-5
    info = mne.create_info([f"CH{i}" for i in range(n_ch)], fs, ch_types=ch_types)
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    path = str(tmp_path / "sub-test_raw.fif")
    raw.save(path, overwrite=True, verbose="ERROR")
    return path


class TestLoadRecording:
    def test_roundtrip_shape_and_metadata(self, tmp_path):
        pytest.importorskip("mne")
        from analysis.eeg.loader import load_recording
        path = _make_raw(tmp_path, n_ch=4, fs=100.0, secs=5)
        data, info = load_recording(path)
        assert data.shape == (4, 500)
        assert info["fs"] == 100.0
        assert info["n_channels"] == 4
        assert info["duration"] == 5.0
        assert len(info["ch_names"]) == 4

    def test_picks_eeg_drops_other_channel_types(self, tmp_path):
        pytest.importorskip("mne")
        from analysis.eeg.loader import load_recording
        path = _make_raw(tmp_path, n_ch=4, ch_types=["eeg", "eeg", "eog", "misc"])
        data, info = load_recording(path, picks="eeg")
        assert info["n_channels"] == 2         # only the two EEG channels survive

    def test_unsupported_format_raises(self, tmp_path):
        pytest.importorskip("mne")
        from analysis.eeg.loader import load_recording
        bogus = tmp_path / "recording.xyz"
        bogus.write_text("not eeg")
        with pytest.raises(ValueError):
            load_recording(str(bogus))
