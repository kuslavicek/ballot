import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

def initialize_via_kmeans_pp(data_matrix, n_clusters, random_state=None):
    rng = np.random.default_rng(random_state)
    n_samples, n_features = data_matrix.shape
    centroids = np.empty((n_clusters, n_features))
    idx = rng.integers(n_samples)
    centroids[0] = data_matrix[idx]
    dist_sq = np.sum((data_matrix - centroids[0])**2, axis=1)
    for i in range(1, n_clusters):
        total_dist = np.sum(dist_sq)
        probs = dist_sq / total_dist if total_dist > 0 else np.ones(n_samples) / n_samples
        idx = rng.choice(n_samples, p=probs)
        centroids[i] = data_matrix[idx]
        new_dist_sq = np.sum((data_matrix - centroids[i])**2, axis=1)
        dist_sq = np.minimum(dist_sq, new_dist_sq)
    return centroids

def initialize_via_diameter_sampling(data_matrix, n_clusters):
    if n_clusters != 2:
        raise ValueError("Diameter sampling initialization is only defined for k=2.")
    D = squareform(pdist(data_matrix, metric='sqeuclidean'))
    i, j = np.unravel_index(np.argmax(D), D.shape)
    centroids = np.zeros((n_clusters, data_matrix.shape[1]))
    centroids[0] = data_matrix[i]
    centroids[1] = data_matrix[j]
    return centroids

def initialize_via_clique_sampling(data_matrix, n_clusters, separation_param):
    n_samples = data_matrix.shape[0]
    epsilon = 0.1
    M = int(np.ceil(n_clusters * np.log(2 * n_clusters / epsilon)))
    rng = np.random.default_rng()
    indices = rng.choice(n_samples, min(M, n_samples), replace=False)
    subset = data_matrix[indices]
    threshold = min(separation_param - 2.0, 2.0)
    dists = squareform(pdist(subset, metric='euclidean'))
    adj = dists <= threshold
    np.fill_diagonal(adj, False)
    n_components, labels = connected_components(csr_matrix(adj), directed=False)
    centroids = np.zeros((n_clusters, data_matrix.shape[1]))
    unique_labels = np.unique(labels)
    found_count = 0
    for lbl in unique_labels:
        if found_count >= n_clusters: break
        centroids[found_count] = data_matrix[indices[np.where(labels == lbl)[0][0]]]
        found_count += 1
    if found_count < n_clusters:
        centroids[found_count:] = data_matrix[rng.choice(n_samples, n_clusters - found_count)]
    return centroids