#!/usr/bin/env python3
"""
Simple Distance Metrics Test
Run this script to test different distance metrics with your vector database.
"""

import numpy as np
import asyncio
import nest_asyncio
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def euclidean_distance(vec1, vec2):
    """Calculate Euclidean distance between two vectors."""
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def manhattan_distance(vec1, vec2):
    """Calculate Manhattan distance between two vectors."""
    return np.sum(np.abs(vec1 - vec2))

def dot_product_similarity(vec1, vec2):
    """Calculate dot product similarity between two vectors."""
    return np.dot(vec1, vec2)

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

def test_distance_metrics(vector_db, query="What are the limitations of reasoning models?", k=5):
    """
    Test different distance metrics with your vector database.
    
    Args:
        vector_db: Your VectorDatabase instance
        query: Query to test
        k: Number of top results to show
    """
    print(f"🔍 Testing Distance Metrics")
    print(f"Query: '{query}'")
    print("=" * 60)
    
    # Step 1: Get existing search results (this always works)
    print("📊 Getting existing search results...")
    try:
        existing_results = vector_db.search_by_text(query, k=k)
        print(f"✅ Found {len(existing_results)} results")
        
        print("\n📐 EXISTING COSINE SIMILARITY RESULTS:")
        print("-" * 50)
        for i, (text, score) in enumerate(existing_results, 1):
            print(f"{i}. Score: {score:.4f}")
            print(f"   Text: {text[:80]}...")
            print()
            
    except Exception as e:
        print(f"❌ Error getting search results: {e}")
        return
    
    # Step 2: Try to get query embedding for additional metrics
    print("🔄 Attempting to get query embedding for additional metrics...")
    
    try:
        # Apply nest_asyncio
        nest_asyncio.apply()
        
        # Create async function
        async def get_embedding():
            return await vector_db.embedding_model.async_get_embeddings([query])
        
        # Try to get the embedding
        loop = asyncio.get_event_loop()
        if loop.is_running():
            print("⚠️  Running in event loop - using alternative approach")
            
            # Try to create a task and wait for it
            task = loop.create_task(get_embedding())
            
            # Wait a bit for it to complete
            import time
            time.sleep(3)
            
            if task.done():
                embedding = task.result()
                query_vector = np.array(embedding[0])
                print(f"✅ Got query embedding! (dimension: {len(query_vector)})")
                
                # Calculate additional metrics
                print("\n📏 ADDITIONAL METRICS FOR TOP RESULTS:")
                print("-" * 50)
                
                for i, (text, cosine_score) in enumerate(existing_results[:3], 1):
                    if text in vector_db.vectors:
                        vector = np.array(vector_db.vectors[text])
                        
                        euclidean_dist = euclidean_distance(query_vector, vector)
                        manhattan_dist = manhattan_distance(query_vector, vector)
                        dot_prod = dot_product_similarity(query_vector, vector)
                        
                        print(f"\nResult {i}:")
                        print(f"  Cosine Similarity: {cosine_score:.4f}")
                        print(f"  Euclidean Distance: {euclidean_dist:.4f}")
                        print(f"  Manhattan Distance: {manhattan_dist:.4f}")
                        print(f"  Dot Product: {dot_prod:.4f}")
                        print(f"  Text: {text[:60]}...")
            else:
                print("❌ Task didn't complete in time")
                print("📊 Showing only cosine similarity results")
                
        else:
            # We can use run_until_complete
            embedding = loop.run_until_complete(get_embedding())
            query_vector = np.array(embedding[0])
            print(f"✅ Got query embedding! (dimension: {len(query_vector)})")
            
            # Calculate additional metrics
            print("\n📏 ADDITIONAL METRICS FOR TOP RESULTS:")
            print("-" * 50)
            
            for i, (text, cosine_score) in enumerate(existing_results[:3], 1):
                if text in vector_db.vectors:
                    vector = np.array(vector_db.vectors[text])
                    
                    euclidean_dist = euclidean_distance(query_vector, vector)
                    manhattan_dist = manhattan_distance(query_vector, vector)
                    dot_prod = dot_product_similarity(query_vector, vector)
                    
                    print(f"\nResult {i}:")
                    print(f"  Cosine Similarity: {cosine_score:.4f}")
                    print(f"  Euclidean Distance: {euclidean_dist:.4f}")
                    print(f"  Manhattan Distance: {manhattan_dist:.4f}")
                    print(f"  Dot Product: {dot_prod:.4f}")
                    print(f"  Text: {text[:60]}...")
                    
    except Exception as e:
        print(f"❌ Could not get query embedding: {e}")
        print("📊 Showing only existing cosine similarity results")

def main():
    """Main function to run the distance metrics test."""
    print("🚀 Simple Distance Metrics Test")
    print("=" * 40)
    
    # Check if we're in a Jupyter environment
    try:
        # Try to import IPython to check if we're in a notebook
        import IPython
        print("📓 Running in Jupyter environment")
        print("Please run this in your notebook with:")
        print("exec(open('simple_distance_test.py').read())")
        return
    except ImportError:
        print("🐍 Running in Python environment")
    
    # If not in Jupyter, we need to create a mock vector database for testing
    print("Creating mock vector database for testing...")
    
    class MockVectorDB:
        def __init__(self):
            self.vectors = {
                "Sample text about AI limitations": np.array([0.1, 0.2, 0.3, 0.4]),
                "Sample text about machine learning": np.array([0.5, 0.6, 0.7, 0.8]),
                "Sample text about neural networks": np.array([0.9, 1.0, 1.1, 1.2])
            }
        
        def search_by_text(self, query, k=3):
            return [
                ("Sample text about AI limitations", 0.85),
                ("Sample text about machine learning", 0.72),
                ("Sample text about neural networks", 0.65)
            ]
        
        @property
        def embedding_model(self):
            class MockEmbeddingModel:
                async def async_get_embeddings(self, texts):
                    return [[0.3, 0.4, 0.5, 0.6]]
            return MockEmbeddingModel()
    
    mock_db = MockVectorDB()
    test_distance_metrics(mock_db, "What are the limitations?", k=3)

if __name__ == "__main__":
    main() 