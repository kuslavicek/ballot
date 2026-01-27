import numpy as np
import pytest
from ballot import core

class TestComputeCostMatrix:
    def test_compute_cost_matrix_correctness(self):
        X = np.array([[0.0], [10.0]])
        centroids = np.array([[0.0], [10.0]])
        C = core.compute_cost_matrix(X, centroids)
        expected = np.array([[0.0, 100.0], [100.0, 0.0]])
        np.testing.assert_allclose(C, expected)

    def test_compute_cost_matrix_shapes(self):
        X = np.random.rand(10, 5)
        centroids = np.random.rand(3, 5)
        C = core.compute_cost_matrix(X, centroids)
        assert C.shape == (10, 3)

    def test_compute_cost_matrix_values_non_negative(self):
        X = np.random.randn(20, 2)
        centroids = np.random.randn(4, 2)
        C = core.compute_cost_matrix(X, centroids)
        assert np.all(C >= 0)

class TestSolveEntropicKantorovich:
    def test_solve_entropic_kantorovich_converges(self, cost_matrix):
        n, k = cost_matrix.shape
        r, c = np.full(n, 1/n), np.full(k, 1/k)
        F = core.solve_entropic_kantorovich(cost_matrix, r, c, entropic_reg=0.1, max_iter=100)
        assert F.shape == (4, 2)

    def test_solve_entropic_kantorovich_marginals(self, cost_matrix):
        n, k = cost_matrix.shape
        r, c = np.full(n, 1/n), np.full(k, 1/k)
        F = core.solve_entropic_kantorovich(cost_matrix, r, c, entropic_reg=0.1, max_iter=500)
        np.testing.assert_allclose(F.sum(axis=1), r, atol=1e-4)
        np.testing.assert_allclose(F.sum(axis=0), c, atol=1e-4)

    def test_solve_entropic_kantorovich_low_reg_approximation(self, cost_matrix):
        n, k = cost_matrix.shape
        r, c = np.full(n, 1/n), np.full(k, 1/k)
        F = core.solve_entropic_kantorovich(cost_matrix, r, c, entropic_reg=1e-3, max_iter=1000)
        assert F[0, 0] > F[0, 1]
        assert F[2, 1] > F[2, 0]

class TestSolveExactKantorovich:
    def test_solve_exact_kantorovich_integrality(self, cost_matrix):
        n, k = cost_matrix.shape
        r, c = np.full(n, 1/n), np.full(k, 1/k)
        F = core.solve_exact_kantorovich(cost_matrix, r, c)
        target_val = 1.0 / n
        is_zero = np.isclose(F, 0)
        is_target = np.isclose(F, target_val)
        assert np.all(is_zero | is_target)

    def test_solve_exact_kantorovich_marginals(self, cost_matrix):
        n, k = cost_matrix.shape
        r, c = np.full(n, 1/n), np.full(k, 1/k)
        F = core.solve_exact_kantorovich(cost_matrix, r, c)
        np.testing.assert_allclose(F.sum(axis=1), r)
        np.testing.assert_allclose(F.sum(axis=0), c)

    def test_solve_exact_kantorovich_optimality(self, cost_matrix):
        n, k = cost_matrix.shape
        r, c = np.full(n, 1/n), np.full(k, 1/k)
        F_opt = core.solve_exact_kantorovich(cost_matrix, r, c)
        cost_opt = np.sum(F_opt * cost_matrix)
        
        F_sub = np.zeros_like(F_opt)
        F_sub[0, 0] = 1/n
        F_sub[1, 1] = 1/n
        F_sub[2, 0] = 1/n
        F_sub[3, 1] = 1/n
        
        cost_sub = np.sum(F_sub * cost_matrix)
        assert cost_opt <= cost_sub + 1e-9

class TestUpdateCentroids:
    def test_update_centroids_formula(self):
        X = np.array([[0.0], [2.0]])
        F = np.array([[0.5, 0.0], [0.0, 0.5]])
        mu = core.update_centroids(X, F, n_clusters=2)
        expected = np.array([[0.0], [2.0]])
        np.testing.assert_allclose(mu, expected)

    def test_update_centroids_shape(self):
        n, d, k = 10, 5, 2
        X = np.random.rand(n, d)
        F = np.zeros((n, k))
        F[:5, 0] = 1/n
        F[5:, 1] = 1/n
        mu = core.update_centroids(X, F, n_clusters=k)
        assert mu.shape == (k, d)