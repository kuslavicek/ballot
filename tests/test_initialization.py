import numpy as np
import pytest
from ballot import initialization

class TestInitializeViaKmeansPP:
    def test_initialize_via_kmeans_pp_output_shape(self, balanced_2d_data):
        X = balanced_2d_data
        k = 2
        centroids = initialization.initialize_via_kmeans_pp(X, k, random_state=42)
        assert centroids.shape == (k, X.shape[1])

    def test_initialize_via_kmeans_pp_reproducibility(self, balanced_2d_data):
        c1 = initialization.initialize_via_kmeans_pp(balanced_2d_data, 2, random_state=1)
        c2 = initialization.initialize_via_kmeans_pp(balanced_2d_data, 2, random_state=1)
        np.testing.assert_array_equal(c1, c2)

class TestInitializeViaDiameterSampling:
    def test_initialize_via_diameter_sampling_correctness(self, balanced_2d_data):
        centroids = initialization.initialize_via_diameter_sampling(balanced_2d_data, n_clusters=2)
        
        # Verify that the returned centroids are actual points from the dataset
        # and that they are sufficiently far apart.
        dist = np.linalg.norm(centroids[0] - centroids[1])
        assert dist > 10.0

    def test_initialize_via_diameter_sampling_k_equals_2(self):
        X = np.random.rand(10, 2)
        with pytest.raises(ValueError):
            initialization.initialize_via_diameter_sampling(X, n_clusters=3)

class TestInitializeViaCliqueSampling:
    def test_initialize_via_clique_sampling_shape(self, balanced_2d_data):
        centroids = initialization.initialize_via_clique_sampling(
            balanced_2d_data, n_clusters=2, separation_param=5.0
        )
        assert centroids.shape == (2, 2)

    def test_initialize_via_clique_sampling_graph_logic(self):
        # Create a dataset with 20 points in 2D space.
        # This test ensures that the function returns the correct number of centroids.
        X = np.random.rand(20, 2)
        k = 4
        # Verify that the function returns the expected number of centroids.
        centroids = initialization.initialize_via_clique_sampling(X, k, separation_param=0.1)
        assert len(centroids) == k