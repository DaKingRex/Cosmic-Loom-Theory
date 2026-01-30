Quick Start Guide
=================

This guide will get you running CLT visualizations in minutes.

Energy Resistance Visualizer
----------------------------

The éR phase space shows where biological coherence exists:

.. code-block:: bash

   python run_energy_resistance_demo.py

Or in Python:

.. code-block:: python

   from visualizations.interactive import EnergyResistanceVisualizer

   viz = EnergyResistanceVisualizer()
   viz.render(interactive=True, show_trajectories=True)

This displays the viable window where éR = EP/f² is neither too low (chaos)
nor too high (rigidity).

2D Loomfield Simulator
----------------------

Watch consciousness as propagating waves of coherence:

.. code-block:: bash

   python run_loomfield_demo.py

Or programmatically:

.. code-block:: python

   from visualizations.interactive import LoomfieldSimulator, LoomfieldVisualizer

   # Create simulator
   sim = LoomfieldSimulator(grid_size=200)

   # Add phase-locked sources (healthy pattern)
   sim.add_source(x=0, y=0, strength=1.5, frequency=1.5, phase=0.0)
   sim.add_source(x=3, y=0, strength=0.8, frequency=1.5, phase=0.0)

   # Run visualization
   viz = LoomfieldVisualizer(sim)
   viz.run()

**Controls:**

- Click to add source/perturbation
- Use preset buttons for different scenarios
- Watch Q (coherence) and C_bio (consciousness) metrics

3D Loomfield Visualizer
-----------------------

Generate web-compatible 3D visualizations:

.. code-block:: bash

   python run_loomfield_3d_demo.py

This creates HTML files you can open in any browser.

Programmatic usage:

.. code-block:: python

   from visualizations.interactive import (
       LoomfieldVisualizer3D,
       create_volumetric_figure,
       create_healthy_preset
   )

   # Create visualizer
   viz = LoomfieldVisualizer3D(grid_size=48)
   viz.load_preset('healthy')

   # Generate static figure
   fig = viz.create_static_figure(warm_up_steps=200)
   fig.write_html('loomfield_3d.html')

Real-Time 3D Simulator
----------------------

Interactive desktop application with live physics:

.. code-block:: bash

   # Requires vispy and PyQt5
   pip install vispy PyQt5

   python run_loomfield_3d_realtime.py

**Controls:**

- Drag to rotate
- Scroll to zoom
- Click to add source/perturbation
- 'V' key to toggle render mode (Slices/Isosurface)
- 'R' key to reset camera

Understanding the Metrics
-------------------------

**Q (Coherence)**

   Measures spatial organization of the field. High Q indicates well-organized
   wave patterns (like healthy brainwaves). Low Q indicates chaos.

**C_bio (Consciousness Observable)**

   Combines coherence with activity:

   - 2D: C_bio = Q² × activity
   - 3D: C_bio = Q³ × activity

   Consciousness requires both coherence AND activity.

**éR (Energy Resistance)**

   Determines which regime the system operates in:

   - Low éR → Chaos (too much freedom, no structure)
   - Viable éR → Coherent dynamics (life, consciousness)
   - High éR → Rigidity (frozen, no adaptation)

Preset Scenarios
----------------

**Healthy**
   Phase-locked sources create high coherence (high Q, stable C_bio).

**Pathology**
   Incoherent sources create fragmented patterns (low Q, erratic C_bio).

**Healing**
   Gradual re-coupling toward coherence (Q slowly increases).

Next Steps
----------

- Read the :doc:`theory/index` for the full theoretical framework
- Explore the :doc:`api/index` for programmatic usage
- Check out :doc:`tutorials/index` for detailed examples
