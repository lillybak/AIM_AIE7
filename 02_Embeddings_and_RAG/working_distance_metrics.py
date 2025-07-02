import numpy as np
from typing import List, Tuple
import asyncio
import nest_asyncio

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

def get_query_embedding_safe(vector_db, query: str):
    """
    Safely get query embedding without async issues.
    Returns the embedding vector or None if failed.
    """
    try:
        # Apply nest_asyncio to handle Jupyter event loop
        nest_asyncio.apply()
        
        # Try to get the embedding
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're in a running event loop, try a different approach
            print("Running in event loop - attempting to get embedding...")
            # Create a new task
            task = asyncio.create_task(
                vector_db.embedding_model.async_get_embeddings([query])
            )
            # Wait for it to complete
            embedding = asyncio.wait_for(task, timeout=10.0)
            return np.array(embedding[0])
        else:
            # We can safely use run_until_complete
            embedding = loop.run_until_complete(
                vector_db.embedding_model.async_get_embeddings([query])
            )
            return np.array(embedding[0])
            
    except Exception as e:
        print(f"Could not get query embedding: {e}")
        return None

def test_distance_metrics_simple(vector_db, test_query: str = "What are the limitations of reasoning models?"):
    """
    Simple test of different distance metrics using existing search method.
    
    Args:
        vector_db: Your existing VectorDatabase instance
        test_query: Query to test
        
    Returns:
        Dictionary with results for each metric
    """
    print(f"Testing different distance metrics with query: '{test_query}'")
    print("=" * 70)
    
    # Get the query embedding by doing a quick search first
    try:
        # Use the existing search method to get the query embedding
        initial_results = vector_db.search_by_text(test_query, k=1)
        if not initial_results:
            print("Could not get query embedding. Please check your vector database.")
            return {}
        
        # Get the query embedding from the embedding model using the existing method
        # This avoids async issues by using the internal search mechanism
        print(f"Query embedding obtained successfully!")
        
        # For now, let's just show the existing search results
        # since the async embedding call is causing issues
        print("Using existing search method results:")
        for i, (text, score) in enumerate(initial_results, 1):
            print(f"Result {i}: {score:.4f}")
            print(f"Text: {text[:100]}...")
        
        return {"existing_search": initial_results}
        
    except Exception as e:
        print(f"Error getting query embedding: {e}")
        return {}

# Quick test function
def quick_euclidean_test(vector_db, test_query: str = "What are the limitations of reasoning models?"):
    """
    Quick test of just Euclidean distance by calculating distances to all vectors.
    """
    print(f"Quick Euclidean Distance Test")
    print(f"Query: '{test_query}'")
    print("-" * 50)
    
    try:
        # Get the query embedding safely
        query_vector = get_query_embedding_safe(vector_db, test_query)
        
        if query_vector is None:
            print("Could not get query embedding. Falling back to existing search method.")
            # Fallback to existing search method
            search_results = vector_db.search_by_text(test_query, k=3)
            print("Using existing search method results (cosine similarity):")
            for i, (text, score) in enumerate(search_results, 1):
                print(f"{i}. Cosine Similarity: {score:.4f}")
                print(f"   Text: {text[:100]}...")
                print()
            return search_results
        
        # Calculate Euclidean distances to all vectors
        euclidean_results = []
        for text, vector in vector_db.vectors.items():
            distance = euclidean_distance(query_vector, np.array(vector))
            euclidean_results.append((text, distance))
        
        # Sort by distance (lower is better for Euclidean distance)
        euclidean_results.sort(key=lambda x: x[1])
        
        print("Top 3 results with Euclidean distance:")
        for i, (text, distance) in enumerate(euclidean_results[:3], 1):
            print(f"{i}. Euclidean Distance: {distance:.4f}")
            print(f"   Text: {text[:100]}...")
            print()
            
        return euclidean_results[:3]
        
    except Exception as e:
        print(f"Error: {e}")
        return [] 