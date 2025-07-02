import numpy as np
from typing import List, Tuple, Callable

def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two vectors.
    Lower values indicate more similarity.
    """
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate Manhattan distance between two vectors.
    Lower values indicate more similarity.
    """
    return np.sum(np.abs(vec1 - vec2))

def dot_product_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate dot product similarity between two vectors.
    Higher values indicate more similarity.
    """
    return np.dot(vec1, vec2)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    Higher values indicate more similarity.
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

class CustomVectorDatabase:
    """
    Custom vector database with configurable distance metrics.
    """
    def __init__(self, distance_metric: str = "euclidean"):
        self.vectors = {}
        self.distance_metric = distance_metric
        self.distance_func = self._get_distance_function(distance_metric)
    
    def _get_distance_function(self, metric: str) -> Callable:
        """Get the appropriate distance function based on metric name."""
        metrics = {
            "euclidean": euclidean_distance,
            "manhattan": manhattan_distance,
            "dot_product": dot_product_similarity,
            "cosine": cosine_similarity
        }
        return metrics.get(metric, euclidean_distance)
    
    def insert(self, text: str, vector: np.ndarray):
        """Insert a text-vector pair into the database."""
        self.vectors[text] = vector
    
    def search(self, query_vector: np.ndarray, k: int = 4) -> List[Tuple[str, float]]:
        """
        Search for similar vectors using the configured distance metric.
        
        Args:
            query_vector: The query embedding
            k: Number of results to return
            
        Returns:
            List of (text, score) pairs, sorted by similarity
        """
        results = []
        
        for text, vector in self.vectors.items():
            score = self.distance_func(query_vector, vector)
            results.append((text, score))
        
        # Sort based on metric type
        if self.distance_metric in ["euclidean", "manhattan"]:
            # Lower is better for distance metrics
            results.sort(key=lambda x: x[1])
        else:
            # Higher is better for similarity metrics
            results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]

# Example usage and testing
if __name__ == "__main__":
    # Test vectors
    vec1 = np.array([0.1, 0.2, 0.3])
    vec2 = np.array([0.11, 0.21, 0.31])
    vec3 = np.array([0.9, 0.8, 0.7])
    
    print("Testing different distance metrics:")
    print(f"vec1: {vec1}")
    print(f"vec2: {vec2}")
    print(f"vec3: {vec3}")
    print()
    
    # Test each metric
    metrics = ["euclidean", "manhattan", "dot_product", "cosine"]
    
    for metric in metrics:
        print(f"{metric.upper()} METRIC:")
        db = CustomVectorDatabase(metric)
        db.insert("text1", vec1)
        db.insert("text2", vec2)
        db.insert("text3", vec3)
        
        results = db.search(vec1, k=3)
        for text, score in results:
            print(f"  {text}: {score:.4f}")
        print() 