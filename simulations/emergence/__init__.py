# Coherence Regime Simulations
"""
Simulations of coherence regime transitions in CLT.

CLT defines three regimes based on Energy Resistance (éR = EP/f²):
- Chaos: Low éR → decoherence, fragmentation
- Viable Window: Optimal éR → biological coherence, consciousness
- Rigidity: High éR → frozen dynamics, inability to adapt

This module will implement:
- Regime boundary detection and crossing dynamics
- Pathology as boundary collapse (seizure, depression, etc.)
- Healing as re-coupling toward viable window
- Critical slowing down near transitions
- Hysteresis effects in coherence dynamics

Implemented (Phase 3.1):
- regime_system: cusp/double-well stochastic order-parameter primitive
- kuramoto_network: coupled-oscillator synchronization primitive
- regime_transitions: named scenarios (threshold crossing, hysteresis, critical
  slowing down, sync transition) + éR-visualizer bridge
"""

from .regime_system import (
    RegimeSystem,
    RegimeSnapshot,
    fold_b,
    equilibria,
    is_stable,
    create_bistable_system,
    create_monostable_system,
    create_near_fold_system,
    DEFAULT_A,
    DEFAULT_B,
    DEFAULT_SIGMA,
    demo as regime_system_demo,
)
from .kuramoto_network import (
    KuramotoNetwork,
    critical_coupling,
    create_incoherent_network,
    create_partial_sync_network,
    create_hypersync_network,
    DEFAULT_N,
    DEFAULT_K,
    DEFAULT_GAMMA,
    demo as kuramoto_demo,
)
from .regime_transitions import (
    ScenarioResult,
    run_threshold_crossing,
    run_hysteresis_loop,
    run_critical_slowing_down,
    run_sync_transition,
    add_scenario_to_er_visualizer,
    SCENARIOS,
    demo as regime_transitions_demo,
)
from .scenario import (
    TimeCourse,
    smoothstep,
    make_engine,
    run_time_course,
)
from .pathology import (
    depression,
    anesthesia,
    seizure,
    PATHOLOGIES,
)

__all__ = [
    # regime_system
    'RegimeSystem',
    'RegimeSnapshot',
    'fold_b',
    'equilibria',
    'is_stable',
    'create_bistable_system',
    'create_monostable_system',
    'create_near_fold_system',
    'DEFAULT_A',
    'DEFAULT_B',
    'DEFAULT_SIGMA',
    'regime_system_demo',
    # kuramoto_network
    'KuramotoNetwork',
    'critical_coupling',
    'create_incoherent_network',
    'create_partial_sync_network',
    'create_hypersync_network',
    'DEFAULT_N',
    'DEFAULT_K',
    'DEFAULT_GAMMA',
    'kuramoto_demo',
    # regime_transitions
    'ScenarioResult',
    'run_threshold_crossing',
    'run_hysteresis_loop',
    'run_critical_slowing_down',
    'run_sync_transition',
    'add_scenario_to_er_visualizer',
    'SCENARIOS',
    'regime_transitions_demo',
    # scenario time-course driver (Phase 3.2/3.3)
    'TimeCourse',
    'smoothstep',
    'make_engine',
    'run_time_course',
    # pathology scenarios (Phase 3.2)
    'depression',
    'anesthesia',
    'seizure',
    'PATHOLOGIES',
]
