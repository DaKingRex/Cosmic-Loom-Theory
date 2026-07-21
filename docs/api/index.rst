API Reference
=============

This section documents the Python API for Cosmic Loom Theory simulations.

Simulators
----------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   visualizations.interactive.loomfield_wave.LoomfieldSimulator
   visualizations.interactive.loomfield_3d.LoomfieldSimulator3D
   visualizations.interactive.loomfield_3d_realtime.LoomfieldSimulator3DRealtime

Visualizers
-----------

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   visualizations.interactive.energy_resistance.EnergyResistanceVisualizer
   visualizations.interactive.loomfield_wave.LoomfieldVisualizer
   visualizations.interactive.loomfield_3d.LoomfieldVisualizer3D

Core Classes
------------

LoomfieldSimulator (2D)
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: visualizations.interactive.loomfield_wave.LoomfieldSimulator
   :members:
   :undoc-members:
   :show-inheritance:

LoomfieldSimulator3D
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: visualizations.interactive.loomfield_3d.LoomfieldSimulator3D
   :members:
   :undoc-members:
   :show-inheritance:

EnergyResistanceVisualizer
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: visualizations.interactive.energy_resistance.EnergyResistanceVisualizer
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

Energy Resistance
~~~~~~~~~~~~~~~~~

.. autofunction:: visualizations.interactive.energy_resistance.calculate_system_er

3D Visualization
~~~~~~~~~~~~~~~~

.. autofunction:: visualizations.interactive.loomfield_3d.create_volumetric_figure

.. autofunction:: visualizations.interactive.loomfield_3d.create_slice_figure

.. autofunction:: visualizations.interactive.loomfield_3d.create_animated_figure

Preset Functions
~~~~~~~~~~~~~~~~

.. autofunction:: visualizations.interactive.loomfield_3d.create_healthy_preset

.. autofunction:: visualizations.interactive.loomfield_3d.create_pathology_preset

.. autofunction:: visualizations.interactive.loomfield_3d.create_healing_preset

Phase 3.1: Coherence Regime Transitions
---------------------------------------

Shared coherence metrics (``analysis/metrics``) and the regime-transition engine
(``simulations/emergence``).

Shared Metrics
~~~~~~~~~~~~~~~

.. autofunction:: analysis.metrics.coherence.calculate_er

.. autoclass:: analysis.metrics.coherence.ViableWindow
   :members:

.. autofunction:: analysis.metrics.coherence.kuramoto_order

.. autofunction:: analysis.metrics.complexity.lz_complexity

.. autofunction:: analysis.metrics.complexity.spectral_entropy

.. autofunction:: analysis.metrics.csd.csd_indicators

.. autofunction:: analysis.metrics.csd.kendall_tau_trend

Regime Primitives
~~~~~~~~~~~~~~~~~~

.. autoclass:: simulations.emergence.regime_system.RegimeSystem
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: simulations.emergence.kuramoto_network.KuramotoNetwork
   :members:
   :undoc-members:
   :show-inheritance:

Scenarios and Visualizer
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: simulations.emergence.regime_transitions.run_threshold_crossing

.. autofunction:: simulations.emergence.regime_transitions.run_hysteresis_loop

.. autofunction:: simulations.emergence.regime_transitions.run_critical_slowing_down

.. autofunction:: simulations.emergence.regime_transitions.run_sync_transition

.. autoclass:: visualizations.interactive.regime_transitions.RegimeVisualizer
   :members:
   :undoc-members:
   :show-inheritance:

Module Reference
----------------

visualizations.interactive
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: visualizations.interactive
   :members:
   :undoc-members:
