import numpy as np
import pytest
from ballot import core

@pytest.fixture
def balanced_2d_data():
    """
    Fixture providing a simple balanced dataset in 2D.
    
    This dataset contains 2 clusters with 4 points total, 2 points per cluster.
    The points are clustered around (0,0) and (10,10).
    """
    X = np.array([
        [0.0, 0.0],
        [0.1, 0.1],
        [10.0, 10.0],
        [10.1, 10.1]
    ])
    return X

@pytest.fixture
def unbalanced_input_error():
    """
    Fixture providing data that does not divide evenly by k=3.
    
    This is used to test input validation for unbalanced data.
    """
    return np.zeros((4, 2))

@pytest.fixture
def simple_centroids():
    """
    Fixture providing simple centroids for testing.
    """
    return np.array([[0.0, 0.0], [10.0, 10.0]])

@pytest.fixture
def cost_matrix(balanced_2d_data, simple_centroids):
    """
    Fixture providing a precomputed cost matrix for the 2D data.
    
    This cost matrix is derived from the balanced_2d_data and simple_centroids.
    """
    return core.compute_cost_matrix(balanced_2d_data, simple_centroids)