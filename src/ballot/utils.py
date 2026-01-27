import numpy as np
from scipy.optimize import linear_sum_assignment

def check_balance_constraints(n_samples: int, n_clusters: int) -> bool:
    if n_clusters <= 0:
        raise ValueError("Number of clusters must be positive.")
    if n_samples % n_clusters != 0:
        raise ValueError(
            f"Number of samples ({n_samples}) must be divisible by "
            f"number of clusters ({n_clusters}) for balanced clustering."
        )
    return True

def round_transport_matrix(soft_transport_matrix: np.ndarray, n_samples: int, n_clusters: int) -> np.ndarray:
    """
    Rounds a soft transport matrix to a hard balanced assignment (1/n per row, n/k rows per cluster).
    """
    m = n_samples // n_clusters
    # We want to maximize the probability weights, which is equivalent to minimizing negative weights
    cost = -soft_transport_matrix
    expanded_cost = np.repeat(cost, m, axis=1)
    _, col_ind = linear_sum_assignment(expanded_cost)
    cluster_assignments = col_ind // m
    
    F = np.zeros((n_samples, n_clusters))
    F[np.arange(n_samples), cluster_assignments] = 1.0 / n_samples
    return F