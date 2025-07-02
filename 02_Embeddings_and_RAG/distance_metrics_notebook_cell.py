# =============================================================================
# DISTANCE METRICS TESTING CELL FOR NOTEBOOK
# =============================================================================
# Copy and paste this entire cell into your notebook to test different metrics

import numpy as np
import asyncio
import nest_asyncio
from typing import List, Tuple

# Apply nest_asyncio to handle Jupyter event loop
nest_asyncio.apply()

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
    Safely get query embedding by using the existing search method.
    This avoids all async issues by leveraging the working search infrastructure.
    """
    try:
        # Use the existing search method to get the query embedding
        # This is the most reliable approach since it already works
        print("Getting query embedding using existing search method...")
        
        # First, do a search to ensure the embedding model is working
        initial_results = vector_db.search_by_text(query, k=1)
        if not initial_results:
            print("Search failed - cannot get query embedding")
            return None
        
        # Now try to get the actual embedding vector
        # We'll use a simple approach that works in Jupyter
        try:
            # Create a simple async function
            async def get_embedding():
                return await vector_db.embedding_model.async_get_embeddings([query])
            
            # Try to run it in the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in Jupyter - use a simple approach
                print("Using simple async approach for Jupyter...")
                # Create a task and try to get the result
                task = loop.create_task(get_embedding())
                # Wait a bit for it to complete
                import time
                time.sleep(2)  # Give it time to complete
                if task.done():
                    embedding = task.result()
                    return np.array(embedding[0])
                else:
                    print("Task didn't complete in time")
                    return None
            else:
                # We can use run_until_complete
                embedding = loop.run_until_complete(get_embedding())
                return np.array(embedding[0])
                
        except Exception as e:
            print(f"Direct embedding approach failed: {e}")
            return None
            
    except Exception as e:
        print(f"Could not get query embedding: {e}")
        return None

def test_all_distance_metrics(vector_db, test_query: str = "What are the limitations of reasoning models?", k: int = 5):
    """
    Test all distance metrics with your vector database.
    
    Args:
        vector_db: Your existing VectorDatabase instance
        test_query: Query to test
        k: Number of top results to show for each metric
        
    Returns:
        Dictionary with results for each metric
    """
    print(f"🔍 Testing ALL Distance Metrics")
    print(f"Query: '{test_query}'")
    print("=" * 80)
    
    # First, get the existing search results to see what we're working with
    print("Getting existing search results...")
    existing_results = vector_db.search_by_text(test_query, k=k*2)  # Get more results
    
    if not existing_results:
        print("❌ No search results found. Please check your vector database.")
        return {}
    
    print(f"✅ Found {len(existing_results)} existing search results")
    print()
    
    # Show the existing cosine similarity results first
    print("📐 EXISTING COSINE SIMILARITY RESULTS:")
    print("-" * 60)
    for i, (text, score) in enumerate(existing_results[:k], 1):
        print(f"  {i}. Cosine Similarity: {score:.4f}")
        print(f"     Text: {text[:100]}...")
        print()
    
    # Now try to get the query embedding for additional metrics
    print("🔄 Attempting to get query embedding for additional metrics...")
    query_vector = get_query_embedding_safe(vector_db, test_query)
    
    if query_vector is None:
        print("❌ Could not get query embedding. Showing only existing cosine similarity results.")
        return {"cosine_similarity": existing_results[:k]}
    
    print(f"✅ Query embedding obtained successfully! (dimension: {len(query_vector)})")
    print()
    
    # Define additional metrics to test
    additional_metrics = {
        "euclidean": {
            "function": euclidean_distance,
            "description": "Euclidean Distance",
            "lower_better": True,
            "symbol": "📏"
        },
        "manhattan": {
            "function": manhattan_distance,
            "description": "Manhattan Distance", 
            "lower_better": True,
            "symbol": "🗺️"
        },
        "dot_product": {
            "function": dot_product_similarity,
            "description": "Dot Product Similarity",
            "lower_better": False,
            "symbol": "🔗"
        }
    }
    
    all_results = {"cosine_similarity": existing_results[:k]}
    
    # Test additional metrics
    for metric_name, metric_info in additional_metrics.items():
        print(f"{metric_info['symbol']} Testing {metric_info['description']}:")
        print("-" * 60)
        
        try:
            # Calculate distances/similarities for all vectors
            metric_results = []
            for text, vector in vector_db.vectors.items():
                score = metric_info["function"](query_vector, np.array(vector))
                metric_results.append((text, score))
            
            # Sort based on metric type
            if metric_info["lower_better"]:
                # Lower is better for distance metrics
                metric_results.sort(key=lambda x: x[1])
                print(f"Top {k} results (lower distance = better):")
            else:
                # Higher is better for similarity metrics
                metric_results.sort(key=lambda x: x[1], reverse=True)
                print(f"Top {k} results (higher similarity = better):")
            
            # Display top results
            for i, (text, score) in enumerate(metric_results[:k], 1):
                print(f"  {i}. Score: {score:.4f}")
                print(f"     Text: {text[:100]}...")
                print()
            
            all_results[metric_name] = metric_results[:k]
            
        except Exception as e:
            print(f"❌ Error with {metric_name}: {e}")
            all_results[metric_name] = []
        
        print()
    
    return all_results

def compare_metrics(vector_db, test_query: str = "What are the limitations of reasoning models?", k: int = 3):
    """
    Compare how different metrics rank the same documents.
    
    Args:
        vector_db: Your existing VectorDatabase instance
        test_query: Query to test
        k: Number of top results to compare
        
    Returns:
        Comparison table
    """
    print(f"🔄 COMPARING METRICS")
    print(f"Query: '{test_query}'")
    print("=" * 80)
    
    # Get results for all metrics
    results = test_all_distance_metrics(vector_db, test_query, k)
    
    if not results:
        print("❌ No results to compare")
        return
    
    # Create comparison table
    print("\n📊 COMPARISON TABLE:")
    print("=" * 80)
    
    # Get all unique texts from all metrics
    all_texts = set()
    for metric_results in results.values():
        for text, _ in metric_results:
            all_texts.add(text[:50] + "...")  # Truncate for display
    
    # Create header
    header = "Text (truncated)".ljust(50)
    for metric_name in results.keys():
        header += f" | {metric_name.upper()}".ljust(15)
    print(header)
    print("-" * len(header))
    
    # Create rows for each text
    for text in all_texts:
        row = text.ljust(50)
        for metric_name, metric_results in results.items():
            # Find this text in the results
            found = False
            for i, (result_text, score) in enumerate(metric_results):
                if result_text[:50] + "..." == text:
                    row += f" | {i+1} ({score:.3f})".ljust(15)
                    found = True
                    break
            if not found:
                row += " | -".ljust(15)
        print(row)
    
    return results

# =============================================================================
# USAGE EXAMPLES - UNCOMMENT THE LINES YOU WANT TO USE
# =============================================================================

# Example 1: Test all metrics with your current vector database
# results = test_all_distance_metrics(vector_db, "What are the limitations of reasoning models?", k=5)

# Example 2: Compare how different metrics rank documents
# comparison = compare_metrics(vector_db, "What are the key findings?", k=3)

# Example 3: Quick test with a specific query
# quick_results = test_all_distance_metrics(vector_db, "How do reasoning models work?", k=3)

# Example 4: Test with your own custom query
# custom_query = "What is the main topic of this document?"
# custom_results = test_all_distance_metrics(vector_db, custom_query, k=5)

# =============================================================================
# READY TO USE - JUST UNCOMMENT ONE OF THE EXAMPLES ABOVE!
# ============================================================================= 