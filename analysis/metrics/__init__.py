# CLT Coherence Metrics
"""
Coherence observables and metrics from CLT v1.1 Section 7.

Key metrics:
- Q: Spatial coherence (phase-locking, not just energy)
- C_bio = ∫ ρ_coh · Λ dV (consciousness observable)
- éR = EP/f² (Energy Resistance)
- Coherence density ρ_coh

These metrics quantify Loomfield organization and can be
mapped to LoomSense experimental measurements.

Submodules:
- coherence: éR = EP/f², ViableWindow, regime classification, Kuramoto order
- complexity: Lempel-Ziv complexity, spectral entropy (signal diversity)
- csd: critical-slowing-down / early-warning indicators
"""

from .coherence import (
    calculate_er,
    kuramoto_order,
    ViableWindow,
    classify_regime,
    BASELINE_WINDOW,
    BASELINE_ER_MIN,
    BASELINE_ER_MAX,
)
from .complexity import (
    lz_complexity,
    spectral_entropy,
)
from .csd import (
    rolling_autocorr,
    rolling_variance,
    csd_indicators,
    kendall_tau_trend,
)

__all__ = [
    # coherence
    'calculate_er',
    'kuramoto_order',
    'ViableWindow',
    'classify_regime',
    'BASELINE_WINDOW',
    'BASELINE_ER_MIN',
    'BASELINE_ER_MAX',
    # complexity
    'lz_complexity',
    'spectral_entropy',
    # csd
    'rolling_autocorr',
    'rolling_variance',
    'csd_indicators',
    'kendall_tau_trend',
]
