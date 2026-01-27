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

Module Reference
----------------

visualizations.interactive
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: visualizations.interactive
   :members:
   :undoc-members:
