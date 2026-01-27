Cosmic Loom Theory Documentation
=================================

**Cosmic Loom Theory (CLT)** is a field-based framework proposing that consciousness
emerges from coherent system dynamics operating within constrained energetic regimes.

.. note::

   This documentation covers CLT v1.1, focusing on human biological consciousness.

Overview
--------

CLT introduces the **Loomfield** — an effective field description capturing how bioelectric
activity, biophotons, cytoskeletal structure, and genetic constraints work together to
maintain coherent organization in living systems.

The core relationship is **Energy Resistance**:

.. math::

   \acute{e}R = \frac{EP}{f^2}

Where:

- **EP** = Energy Present (metabolic/field energy)
- **f** = Frequency of system dynamics
- **éR** = Energy Resistance (determines regime)

Living systems operate in a "viable window" where éR balances between chaos
(decoherence) and rigidity (inability to adapt).

Quick Start
-----------

.. code-block:: bash

   # Clone and install
   git clone https://github.com/DaKingRex/Cosmic-Loom-Theory.git
   cd Cosmic-Loom-Theory
   pip install -r requirements.txt

   # Run the Energy Resistance demo
   python run_energy_resistance_demo.py

   # Run the Loomfield 2D wave simulator
   python run_loomfield_demo.py

   # Run the 3D Loomfield visualizer
   python run_loomfield_3d_demo.py

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory/index

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog


Key Concepts
------------

**Loomfield Wave Equation**

   .. math::

      \nabla^2 L - \frac{1}{v_L^2} \frac{\partial^2 L}{\partial t^2} = \kappa_L \cdot \rho_{coh}

   The Loomfield :math:`L` propagates as waves driven by coherence density :math:`\rho_{coh}`.

**Coherence Metric (Q)**

   Energy-independent measure of spatial organization. High Q = coherent patterns.

**Consciousness Observable (C_bio)**

   .. math::

      C_{bio} = Q^n \times \int |\rho_{coh}| \cdot |\partial L / \partial t| \, dV

   Where n=2 for 2D and n=3 for 3D simulations.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
