import numpy as np
from sklearn.cluster import KMeans
from typing import List, Dict, Any

def perform_clustering(embeddings: np.ndarray, n_clusters: int) -> List[int]:
    """
    Perform clustering on the embeddings.
    
    Args:
        embeddings: Numpy array of embeddings
        n_clusters: Number of clusters to create
    
    Returns:
        List of cluster assignments for each embedding
    """
    # Ensure the n_clusters is valid
    n_samples = embeddings.shape[0]
    n_clusters = min(n_clusters, n_samples)
    
    # Perform K-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    
    cluster_assignments = kmeans.fit_predict(embeddings)
    
    return cluster_assignments.tolist()
