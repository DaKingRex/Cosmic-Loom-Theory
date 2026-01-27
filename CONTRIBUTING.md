# Contributing to Cosmic Loom Theory

Thank you for your interest in contributing to the Cosmic Loom Theory computational research project! This document provides guidelines for contributing.

## How to Contribute

### Reporting Issues

- Use the [GitHub Issues](https://github.com/DaKingRex/Cosmic-Loom-Theory/issues) page to report bugs or suggest features
- Check existing issues before creating a new one
- Provide clear, detailed descriptions with steps to reproduce bugs
- Include your Python version and operating system

### Submitting Changes

1. **Fork the repository** and create a new branch for your changes
2. **Make your changes** following the coding standards below
3. **Write tests** for new functionality
4. **Run the test suite** to ensure nothing is broken:
   ```bash
   python -m pytest tests/ -v
   ```
5. **Submit a pull request** with a clear description of your changes

### Types of Contributions Welcome

- **Simulation improvements**: Performance optimizations, numerical accuracy
- **New visualizations**: Interactive tools for exploring CLT concepts
- **Biological substrate models**: New implementations of Phase 2+ components
- **Documentation**: Tutorials, docstring improvements, theoretical explanations
- **Bug fixes**: Corrections to existing code
- **Test coverage**: Additional unit tests

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable names that reflect CLT terminology
- Maximum line length: 100 characters
- Use type hints for function signatures

### Documentation

- All public functions and classes should have docstrings
- Use NumPy-style docstrings:
  ```python
  def compute_coherence(field: np.ndarray) -> float:
      """
      Compute spatial coherence of a Loomfield configuration.

      Parameters
      ----------
      field : np.ndarray
          2D or 3D array representing the Loomfield values

      Returns
      -------
      float
          Coherence metric Q in range [0, 1]
      """
  ```

### Testing

- Write tests for all new functionality
- Place tests in the `tests/` directory
- Use pytest fixtures for common setup
- Aim for descriptive test names: `test_coherence_increases_with_phase_locking`

### Commits

- Write clear, concise commit messages
- Use present tense ("Add feature" not "Added feature")
- Reference issues when applicable: "Fix #123: Correct boundary condition"

## Project Structure

```
Cosmic-Loom-Theory/
├── simulations/          # Core simulation modules
│   ├── field_dynamics/   # Bioelectric, biophoton, DNA modules
│   └── quantum/          # Microtubule and quantum coherence
├── visualizations/       # Visualization tools
├── analysis/             # Analysis and metrics
├── tests/                # Unit tests
├── docs/                 # Documentation
└── output/               # Generated outputs (not tracked)
```

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/DaKingRex/Cosmic-Loom-Theory.git
   cd Cosmic-Loom-Theory
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run tests to verify setup:
   ```bash
   python -m pytest tests/ -v
   ```

## Questions?

- Open an issue for questions about the codebase
- For theoretical questions about CLT, refer to `docs/theory/`
- For collaboration inquiries, contact via GitHub

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
