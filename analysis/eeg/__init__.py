# Cosmic Loom Theory - EEG analysis (Phase 4, deliverable B)
"""
Compute CLT observables from real EEG recordings and test the theory's published
predictions against open datasets.

This package adapts the canonical CLT metrics (``analysis.metrics``) to real
multi-channel EEG and adds the EEG-based éR proxy. It is the empirical-validation
engine for Phase 4: ingest open EEG → per-epoch CLT observables → state/time
trajectories, regime classification, critical-slowing-down trend, and hysteresis.

Modules:
- ``observables`` — per-epoch CLT observables (éR proxy, complexity, phase coherence).
- (future) ``loader`` — EDF/BrainVision/BIDS ingestion via mne.
- (future) ``analysis`` — trajectory/regime/CSD/hysteresis harness.
"""

from .observables import (
    phase_coherence,
    dominant_frequency,
    energy_present,
    er_proxy,
    complexity,
    epoch_observables,
)
from .loader import load_recording, epoch_data
from .analysis import (
    analyze_recording,
    calibrate_window_from_baseline,
    state_regime_trajectory,
)

__all__ = [
    # observables
    "phase_coherence",
    "dominant_frequency",
    "energy_present",
    "er_proxy",
    "complexity",
    "epoch_observables",
    # loader
    "load_recording",
    "epoch_data",
    # analysis harness
    "analyze_recording",
    "calibrate_window_from_baseline",
    "state_regime_trajectory",
]
