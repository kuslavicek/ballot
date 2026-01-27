import numpy as np
from scipy.optimize import linear_sum_assignment

def compute_cost_matrix(
    data_matrix: np.ndarray,
    centroids: np.ndarray
) -> np.ndarray:
    x_sq = np.sum(data_matrix**2, axis=1, keepdims=True)
    c_sq = np.sum(centroids**2, axis=1)
    xc = data_matrix @ centroids.T
    
    cost = x_sq + c_sq[None, :] - 2 * xc
    return np.maximum(cost, 0.0)

def solve_entropic_kantorovich(
    cost_matrix: np.ndarray,
    target_marginals_rows: np.ndarray,
    target_marginals_cols: np.ndarray,
    entropic_reg: float,
    sinkhorn_tolerance: float = 1e-2,
    max_iter: int = 100
) -> np.ndarray:
    n, k = cost_matrix.shape
    # Numerical stability: shift cost matrix
    C = cost_matrix - np.min(cost_matrix)
    K = np.exp(-C / entropic_reg)
    
    u = np.ones(n)
    v = np.ones(k)
    
    for _ in range(max_iter):
        Kv = K @ v
        u = target_marginals_rows / (Kv + 1e-16)
        
        KTu = K.T @ u
        v = target_marginals_cols / (KTu + 1e-16)
        
        # Check convergence if needed, though usually max_iter is fine for clustering inner loops
            
    return u[:, None] * K * v[None, :]

def solve_exact_kantorovich(
    cost_matrix: np.ndarray,
    target_marginals_rows: np.ndarray,
    target_marginals_cols: np.ndarray
) -> np.ndarray:
    n, k = cost_matrix.shape
    m = n // k  
    
    expanded_cost = np.repeat(cost_matrix, m, axis=1) 
    row_ind, col_ind = linear_sum_assignment(expanded_cost)
    cluster_assignments = col_ind // m
    
    F = np.zeros((n, k))
    F[np.arange(n), cluster_assignments] = 1.0 / n
    return F

def update_centroids(
    data_matrix: np.ndarray,
    transport_matrix: np.ndarray,
    n_clusters: int
) -> np.ndarray:
    # transport_matrix is (n, k) with values 1/n
    # Result should be (k, d)
    return n_clusters * (transport_matrix.T @ data_matrix)