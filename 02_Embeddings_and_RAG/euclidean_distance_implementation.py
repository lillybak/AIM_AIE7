import numpy as np
from typing import List, Tuple

def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance (lower = more similar)
    """
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Manhattan distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Manhattan distance (lower = more similar)
    """
    return np.sum(np.abs(vec1 - vec2))

def dot_product_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate dot product similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Dot product (higher = more similar)
    """
    return np.dot(vec1, vec2)

# Example of how to modify your search function
def search_with_euclidean_distance(query_vector: np.ndarray, 
                                  database_vectors: List[Tuple[str, np.ndarray]], 
                                  k: int = 4) -> List[Tuple[str, float]]:
    """
    Search using Euclidean distance instead of cosine similarity.
    
    Args:
        query_vector: Query embedding
        database_vectors: List of (text, vector) pairs
        k: Number of results to return
        
    Returns:
        List of (text, distance) pairs, sorted by distance (lowest first)
    """
    distances = []
    
    for text, vector in database_vectors:
        distance = euclidean_distance(query_vector, vector)
        distances.append((text, distance))
    
    # Sort by distance (lower is better for Euclidean)
    distances.sort(key=lambda x: x[1])
    
    return distances[:k]

# Example usage
if __name__ == "__main__":
    # Example vectors (you'd get these from your embeddings)
    vec1 = np.array([0.1, 0.2, 0.3])
    vec2 = np.array([0.11, 0.21, 0.31])
    vec3 = np.array([0.9, 0.8, 0.7])
    
    print("Euclidean distances:")
    print(f"vec1 to vec2: {euclidean_distance(vec1, vec2):.4f}")
    print(f"vec1 to vec3: {euclidean_distance(vec1, vec3):.4f}")
    
    print("\nManhattan distances:")
    print(f"vec1 to vec2: {manhattan_distance(vec1, vec2):.4f}")
    print(f"vec1 to vec3: {manhattan_distance(vec1, vec3):.4f}")
    
    print("\nDot product similarities:")
    print(f"vec1 to vec2: {dot_product_similarity(vec1, vec2):.4f}")
    print(f"vec1 to vec3: {dot_product_similarity(vec1, vec3):.4f}") 