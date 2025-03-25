import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import List, Dict, Any, Tuple

def determine_optimal_clusters(embeddings: np.ndarray, max_clusters: int = 10, min_clusters: int = 2) -> int:
    """
    Determine the optimal number of clusters using the silhouette method.
    
    Args:
        embeddings: Numpy array of embeddings
        max_clusters: Maximum number of clusters to try
        min_clusters: Minimum number of clusters to try
    
    Returns:
        Optimal number of clusters
    """
    n_samples = embeddings.shape[0]
    
    # We need at least 2 samples and min_clusters should be at least 2
    if n_samples < 2:
        return 1
    
    # Adjust max_clusters based on sample size
    max_clusters = min(max_clusters, n_samples - 1)
    
    # For small datasets, don't try too many clusters
    max_clusters = min(max_clusters, n_samples // 2) if n_samples > 4 else 2
    
    # Ensure valid range
    min_clusters = max(2, min_clusters)
    max_clusters = max(min_clusters, max_clusters)
    
    # If we can't have at least min_clusters, return what we can
    if n_samples < min_clusters:
        return max(1, n_samples - 1)
    
    # Initialize variables
    optimal_clusters = min_clusters
    best_score = -1  # For silhouette score (higher is better)
    
    # Compute silhouette scores for different numbers of clusters
    for n_clusters in range(min_clusters, max_clusters + 1):
        if n_samples <= n_clusters:
            continue
        
        # Create KMeans instance and fit
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate silhouette score
        if len(set(cluster_labels)) > 1:  # Ensure we have more than one cluster
            try:
                score = silhouette_score(embeddings, cluster_labels)
                
                # Update optimal clusters if we found a better score
                if score > best_score:
                    best_score = score
                    optimal_clusters = n_clusters
            except Exception as e:
                # If silhouette score fails, continue with the loop
                print(f"Error calculating silhouette score: {e}")
                continue
    
    return optimal_clusters

def perform_clustering(embeddings: np.ndarray, n_clusters: int = None) -> Tuple[List[int], int]:
    """
    Perform clustering on the embeddings.
    
    Args:
        embeddings: Numpy array of embeddings
        n_clusters: Number of clusters to create (optional, will determine optimal if None)
    
    Returns:
        Tuple of (list of cluster assignments for each embedding, number of clusters used)
    """
    # Ensure we have at least one sample
    n_samples = embeddings.shape[0]
    if n_samples == 0:
        return [], 0
    
    # Default to automatic determination if n_clusters is not specified
    if n_clusters is None:
        n_clusters = determine_optimal_clusters(embeddings)
    else:
        # Ensure the n_clusters is valid if manually specified
        n_clusters = min(n_clusters, n_samples)
        n_clusters = max(1, n_clusters)
    
    # For a single sample, just return a single cluster
    if n_samples == 1:
        return [0], 1
    
    # Perform K-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    
    cluster_assignments = kmeans.fit_predict(embeddings)
    
    return cluster_assignments.tolist(), n_clusters
