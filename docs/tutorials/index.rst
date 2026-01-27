Tutorials
=========

Step-by-step guides for working with Cosmic Loom Theory simulations.

Getting Started
---------------

.. toctree::
   :maxdepth: 1

   basic_simulation
   understanding_metrics
   creating_presets

Basic Simulation Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~

Create your first Loomfield simulation:

.. code-block:: python

   from visualizations.interactive import LoomfieldSimulator

   # Create a 2D simulator
   sim = LoomfieldSimulator(grid_size=100, domain_size=10.0)

   # Add a coherent source at the center
   sim.add_source(x=0.0, y=0.0, strength=1.5, frequency=1.5, phase=0.0)

   # Add peripheral sources (phase-locked for coherence)
   for angle in [0, 120, 240]:
       rad = angle * 3.14159 / 180
       x = 3.0 * np.cos(rad)
       y = 3.0 * np.sin(rad)
       sim.add_source(x=x, y=y, strength=0.8, frequency=1.5, phase=0.0)

   # Run simulation steps
   for step in range(100):
       sim.step(n_steps=5)
       Q = sim.get_total_coherence()
       C_bio = sim.get_consciousness_metric()
       print(f"Step {step}: Q={Q:.3f}, C_bio={C_bio:.3f}")

Understanding the Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~

**Q (Coherence Metric)**

Q measures how spatially organized the field is, independent of energy:

- Q ≈ 0: Random noise, no spatial structure
- Q ≈ 1: Moderate organization
- Q > 1.5: Well-organized wave patterns

.. code-block:: python

   Q = sim.get_total_coherence()

**C_bio (Consciousness Observable)**

C_bio combines coherence with activity:

.. code-block:: python

   C_bio = sim.get_consciousness_metric()

High C_bio requires:

1. High Q (organized patterns)
2. Temporal activity (changing field)
3. Source coupling (ρ_coh driving changes)

Creating Custom Presets
~~~~~~~~~~~~~~~~~~~~~~~

Define your own scenarios:

.. code-block:: python

   def create_meditation_preset(sim):
       """Meditation: Strong central source, calm periphery."""
       sim.reset()

       # Dominant central source
       sim.add_source(x=0, y=0, strength=2.0, frequency=0.8, phase=0.0)

       # Gentle peripheral support
       for r in [2, 4]:
           for angle in range(0, 360, 60):
               rad = angle * 3.14159 / 180
               x, y = r * np.cos(rad), r * np.sin(rad)
               sim.add_source(x=x, y=y, strength=0.3, frequency=0.8, phase=0.0)

   # Use it
   create_meditation_preset(sim)
   sim.step(200)

3D Simulations
--------------

Working with the 3D Loomfield:

.. code-block:: python

   from visualizations.interactive import LoomfieldSimulator3D

   # Create 3D simulator
   sim = LoomfieldSimulator3D(grid_size=32)

   # Add sources in 3D space
   sim.add_source(x=0, y=0, z=0, strength=1.5, frequency=1.2, phase=0.0)

   # Run and measure
   sim.step(50)
   Q = sim.get_total_coherence()
   C_bio = sim.get_consciousness_metric()  # Uses Q³ in 3D

Perturbations and Healing
-------------------------

Model disruption and recovery:

.. code-block:: python

   # Start with healthy system
   create_healthy_preset(sim)
   sim.step(100)
   Q_healthy = sim.get_total_coherence()

   # Add perturbation (trauma/disruption)
   sim.add_perturbation(x=0, y=0, strength=2.0, radius=1.5)
   sim.step(50)
   Q_disrupted = sim.get_total_coherence()

   # Watch healing (if sources maintain coherence)
   for _ in range(100):
       sim.step(10)
       Q = sim.get_total_coherence()
       print(f"Q recovering: {Q:.3f}")
