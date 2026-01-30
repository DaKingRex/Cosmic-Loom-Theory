Installation
============

This guide covers installing Cosmic Loom Theory and its dependencies.

Requirements
------------

- Python 3.9 or higher
- pip (Python package installer)

Core Dependencies
~~~~~~~~~~~~~~~~~

The following packages are required:

- **numpy** - Numerical computing
- **scipy** - Scientific computing
- **matplotlib** - 2D visualization
- **plotly** - Interactive 3D visualization

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

For the real-time 3D visualizer:

- **vispy** - OpenGL-based visualization
- **PyQt5** - GUI framework

For quantum coherence simulations:

- **qutip** - Quantum Toolbox in Python

Installation Methods
--------------------

From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/DaKingRex/Cosmic-Loom-Theory.git
   cd Cosmic-Loom-Theory

   # Create virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt

Minimal Installation
~~~~~~~~~~~~~~~~~~~~

For core functionality only:

.. code-block:: bash

   pip install numpy scipy matplotlib plotly

Real-Time 3D Visualizer
~~~~~~~~~~~~~~~~~~~~~~~

For the interactive desktop application:

.. code-block:: bash

   pip install vispy PyQt5

Verifying Installation
----------------------

Test that the installation works:

.. code-block:: python

   # Test imports
   from visualizations.interactive import EnergyResistanceVisualizer
   from visualizations.interactive import LoomfieldSimulator
   from visualizations.interactive import LoomfieldSimulator3D

   print("Installation successful!")

Or run the test suite:

.. code-block:: bash

   pytest tests/ -v

Troubleshooting
---------------

**ImportError: No module named 'visualizations'**

   Make sure you're running from the project root directory, or add it to your
   Python path:

   .. code-block:: python

      import sys
      sys.path.insert(0, '/path/to/Cosmic-Loom-Theory')

**vispy Volume rendering shows black screen (macOS)**

   This is a known issue with vispy's Volume visual on some Mac systems.
   The real-time 3D visualizer automatically falls back to slice or isosurface
   rendering modes.

**Qt platform plugin not found**

   Install PyQt5 with:

   .. code-block:: bash

      pip install PyQt5
