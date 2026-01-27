Contributing
============

Thank you for your interest in contributing to Cosmic Loom Theory!

Getting Started
---------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/YOUR-USERNAME/Cosmic-Loom-Theory.git
      cd Cosmic-Loom-Theory

3. Create a virtual environment and install dependencies:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate
      pip install -r requirements.txt

4. Create a branch for your changes:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

Development Workflow
--------------------

Running Tests
~~~~~~~~~~~~~

.. code-block:: bash

   pytest tests/ -v

All tests should pass before submitting a PR.

Code Style
~~~~~~~~~~

We follow PEP 8 with some relaxations:

- Line length up to 120 characters
- Use descriptive variable names (physics notation welcome)

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd docs
   pip install -r requirements.txt
   make html

Open ``_build/html/index.html`` in a browser.

Types of Contributions
----------------------

**Bug Fixes**
   Fix issues with existing code. Include tests when possible.

**New Visualizations**
   Add new ways to explore CLT concepts.

**Simulation Improvements**
   Optimize performance, add new physics features.

**Documentation**
   Improve explanations, add tutorials, fix typos.

**Theoretical Extensions**
   Propose extensions to the CLT framework (with justification).

Pull Request Guidelines
-----------------------

1. Update tests for any code changes
2. Update documentation for new features
3. Keep PRs focused on a single change
4. Write clear commit messages
5. Ensure CI passes before requesting review

Questions?
----------

- Open an issue on GitHub
- Tag it with ``question`` label
