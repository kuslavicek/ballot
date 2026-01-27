import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from ballot.estimator import BalancedKMeans

@pytest.fixture
def blob_data():
    X, y = make_blobs(n_samples=90, n_features=2, centers=3, random_state=42, cluster_std=0.5)
    return X, y

@pytest.fixture
def unbalanced_size_data():
    return np.random.rand(10, 2)

class TestBalancedKMeans:
    def test_initialization(self):
        est = BalancedKMeans(n_clusters=3)
        assert est.n_clusters == 3
        assert est.algorithm == 'exact'

    def test_input_validation_divisibility(self, unbalanced_size_data):
        est = BalancedKMeans(n_clusters=4)
        msg = "must be divisible by number of clusters"
        with pytest.raises(ValueError, match=msg):
            est.fit(unbalanced_size_data)

    def test_fit_exact_attributes(self, blob_data):
        X, _ = blob_data
        k = 3
        est = BalancedKMeans(n_clusters=k, algorithm='exact', random_state=42)
        est.fit(X)
        check_is_fitted(est, attributes=['cluster_centers_', 'labels_', 'inertia_'])
        assert est.cluster_centers_.shape == (k, X.shape[1])
        assert est.labels_.shape == (X.shape[0],)

    def test_fit_sinkhorn_attributes(self, blob_data):
        X, _ = blob_data
        est = BalancedKMeans(n_clusters=3, algorithm='sinkhorn', entropic_reg=0.1)
        est.fit(X)
        check_is_fitted(est, attributes=['cluster_centers_'])

    def test_strict_balance_constraint(self, blob_data):
        X, _ = blob_data
        n, k = X.shape[0], 3
        expected_count = n // k

        est_exact = BalancedKMeans(n_clusters=k, algorithm='exact', random_state=42)
        est_exact.fit(X)
        _, counts_exact = np.unique(est_exact.labels_, return_counts=True)
        np.testing.assert_array_equal(counts_exact, [expected_count] * k)

        est_sink = BalancedKMeans(n_clusters=k, algorithm='sinkhorn', entropic_reg=0.1, random_state=42)
        est_sink.fit(X)
        _, counts_sink = np.unique(est_sink.labels_, return_counts=True)
        np.testing.assert_array_equal(counts_sink, [expected_count] * k)

    def test_fit_predict_equivalence(self, blob_data):
        X, _ = blob_data
        est = BalancedKMeans(n_clusters=3, random_state=0)
        labels = est.fit_predict(X)
        np.testing.assert_array_equal(labels, est.labels_)

    def test_predict_behavior(self, blob_data):
        X, _ = blob_data
        est = BalancedKMeans(n_clusters=3, random_state=42)
        est.fit(X)
        preds = est.predict(X[:5])
        assert preds.shape == (5,)

    def test_inertia_reduction(self, blob_data):
        X, _ = blob_data
        est = BalancedKMeans(n_clusters=3, random_state=42)
        est.fit(X)
        total_variance = np.sum((X - np.mean(X, axis=0)) ** 2)
        assert est.inertia_ < total_variance

    def test_comparison_sinkhorn_exact(self, blob_data):
        X, _ = blob_data
        k = 3
        est_exact = BalancedKMeans(n_clusters=k, algorithm='exact', random_state=1)
        est_exact.fit(X)
        est_sink = BalancedKMeans(n_clusters=k, algorithm='sinkhorn', entropic_reg=0.05, random_state=1)
        est_sink.fit(X)
        c_exact = np.sort(est_exact.cluster_centers_, axis=0)
        c_sink = np.sort(est_sink.cluster_centers_, axis=0)
        np.testing.assert_allclose(c_exact, c_sink, atol=0.8)

    def test_invalid_algorithm(self, blob_data):
        X, _ = blob_data
        est = BalancedKMeans(n_clusters=3, algorithm='invalid_algo')
        with pytest.raises(ValueError, match="algorithm"):
            est.fit(X)

    def test_not_fitted_error(self, blob_data):
        X, _ = blob_data
        est = BalancedKMeans(n_clusters=3)
        with pytest.raises(NotFittedError):
            est.predict(X)

    def test_zero_inertia_perfect_case(self):
        X = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0]])
        est = BalancedKMeans(n_clusters=2, algorithm='exact', random_state=42)
        est.fit(X)
        assert est.inertia_ == pytest.approx(0.0)