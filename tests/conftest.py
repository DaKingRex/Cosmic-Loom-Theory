"""
Pytest configuration and fixtures for Cosmic Loom Theory tests.
"""

import pytest
import numpy as np


@pytest.fixture
def small_grid_size():
    """Small grid size for fast tests."""
    return 24


@pytest.fixture
def medium_grid_size():
    """Medium grid size for moderate tests."""
    return 50


@pytest.fixture
def sample_source_params():
    """Sample source parameters for testing."""
    return {
        'x': 0.0,
        'y': 0.0,
        'z': 0.0,
        'strength': 1.0,
        'radius': 0.5,
        'frequency': 1.5,
        'phase': 0.0,
    }


@pytest.fixture
def viable_er_range():
    """Define the viable éR window for biological coherence."""
    return {
        'min': 0.1,   # Below: chaos/decoherence
        'max': 10.0,  # Above: rigidity/frozen
    }


# Configure numpy for reproducible tests
@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
