import numpy as np
import asyncio
from typing import List, Tuple

def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors."""
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def manhattan_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate Manhattan distance between two vectors."""
    return np.sum(np.abs(vec1 - vec2))

def dot_product_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate dot product similarity between two vectors."""
    return np.dot(vec1, vec2)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

async def search_with_custom_metric(vector_db, query: str, distance_metric: str = "euclidean", k: int = 4):
    """
    Search using a custom distance metric with your existing vector database.
    
    Args:
        vector_db: Your existing VectorDatabase instance
        query: Query text
        distance_metric: One of "euclidean", "manhattan", "dot_product", "cosine"
        k: Number of results to return
        
    Returns:
        List of (text, score) pairs
    """
    # Get the distance function
    distance_functions = {
        "euclidean": euclidean_distance,
        "manhattan": manhattan_distance,
        "dot_product": dot_product_similarity,
        "cosine": cosine_similarity
    }
    
    distance_func = distance_functions.get(distance_metric, euclidean_distance)
    
    # Get query embedding
    query_embedding = await vector_db.embedding_model.async_get_embeddings([query])
    query_vector = np.array(query_embedding[0])
    
    # Calculate distances/similarities
    results = []
    for text, vector in vector_db.vectors.items():
        score = distance_func(query_vector, np.array(vector))
        results.append((text, score))
    
    # Sort based on metric type
    if distance_metric in ["euclidean", "manhattan"]:
        # Lower is better for distance metrics
        results.sort(key=lambda x: x[1])
    else:
        # Higher is better for similarity metrics
        results.sort(key=lambda x: x[1], reverse=True)
    
    return results[:k]

# Test function for your notebook
async def test_distance_metrics_on_existing_db(vector_db, test_query: str = "What are the limitations of reasoning models?"):
    """
    Test different distance metrics on your existing vector database.
    
    Args:
        vector_db: Your existing VectorDatabase instance
        test_query: Query to test
        
    Returns:
        Dictionary with results for each metric
    """
    metrics = ["euclidean", "manhattan", "dot_product", "cosine"]
    results = {}
    
    print(f"Testing different distance metrics with query: '{test_query}'")
    print("=" * 70)
    
    for metric in metrics:
        print(f"\n🔍 Testing {metric.upper()} metric:")
        print("-" * 40)
        
        try:
            metric_results = await search_with_custom_metric(vector_db, test_query, metric, k=3)
            results[metric] = metric_results
            
            print(f"Top 3 results:")
            for i, (text, score) in enumerate(metric_results, 1):
                print(f"  {i}. Score: {score:.4f}")
                print(f"     Text: {text[:100]}...")
                
        except Exception as e:
            print(f"Error with {metric}: {e}")
            results[metric] = []
    
    return results

# Example usage for your notebook
if __name__ == "__main__":
    print("Distance Metrics Testing Module")
    print("Available metrics: euclidean, manhattan, dot_product, cosine")
    print("\nTo use in your notebook:")
    print("1. Import: from simple_distance_metrics import test_distance_metrics_on_existing_db")
    print("2. Call: results = test_distance_metrics_on_existing_db(your_vector_db)") 
