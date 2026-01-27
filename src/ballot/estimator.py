import numpy as np
from typing import Optional, Literal
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_is_fitted

from .core import (
    compute_cost_matrix,
    solve_entropic_kantorovich,
    solve_exact_kantorovich,
    update_centroids
)
from .utils import check_balance_constraints, round_transport_matrix
from .initialization import (
    initialize_via_kmeans_pp, 
    initialize_via_diameter_sampling,
    initialize_via_clique_sampling
)

class BalancedKMeans(BaseEstimator, ClusterMixin):
    def __init__(
        self,
        n_clusters: int = 8,
        algorithm: Literal['sinkhorn', 'exact'] = 'exact',
        entropic_reg: float = 0.05,
        sinkhorn_tolerance: float = 1e-2,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
        init: Literal['k-means++', 'diameter', 'clique'] = 'k-means++',
        separation_param: float = 4.0
    ):
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.entropic_reg = entropic_reg
        self.sinkhorn_tolerance = sinkhorn_tolerance
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.init = init
        self.separation_param = separation_param

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'BalancedKMeans':
        n_samples, n_features = X.shape
        check_balance_constraints(n_samples, self.n_clusters)
        
        if self.init == 'diameter':
            self.cluster_centers_ = initialize_via_diameter_sampling(X, self.n_clusters)
        elif self.init == 'clique':
            self.cluster_centers_ = initialize_via_clique_sampling(
                X, self.n_clusters, self.separation_param
            )
        else:
            self.cluster_centers_ = initialize_via_kmeans_pp(
                X, self.n_clusters, random_state=self.random_state
            )
        
        r = np.full(n_samples, 1.0 / n_samples)
        c = np.full(self.n_clusters, 1.0 / self.n_clusters)
        
        for iteration in range(self.max_iter):
            old_centers = self.cluster_centers_.copy()
            cost_matrix = compute_cost_matrix(X, self.cluster_centers_)
            
            if self.algorithm == 'exact':
                F = solve_exact_kantorovich(cost_matrix, r, c)
            elif self.algorithm == 'sinkhorn':
                F_soft = solve_entropic_kantorovich(
                    cost_matrix, r, c,
                    entropic_reg=self.entropic_reg,
                    sinkhorn_tolerance=self.sinkhorn_tolerance
                )
                F = round_transport_matrix(F_soft, n_samples, self.n_clusters)
            else:
                raise ValueError(f"Unknown algorithm: {self.algorithm}")
                
            self.cluster_centers_ = update_centroids(X, F, self.n_clusters)
            
            center_shift = np.sum((self.cluster_centers_ - old_centers)**2)
            if center_shift < self.tol:
                break
                
        final_cost = compute_cost_matrix(X, self.cluster_centers_)
        F_final = solve_exact_kantorovich(final_cost, r, c)
            
        self.labels_ = np.argmax(F_final, axis=1)
        min_dists = np.min(final_cost, axis=1)
        self.inertia_ = np.sum(min_dists)
        
        return self

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).labels_

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, attributes=['cluster_centers_'])
        cost = compute_cost_matrix(X, self.cluster_centers_)
        return np.argmin(cost, axis=1)