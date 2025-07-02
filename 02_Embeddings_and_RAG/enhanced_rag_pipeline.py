import asyncio
import numpy as np
from typing import List, Tuple, Dict, Any
from aimakerspace.text_utils import PDFFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.prompts import UserRolePrompt, SystemRolePrompt
from aimakerspace.openai_utils.chatmodel import ChatOpenAI

# Import our custom distance metrics
from custom_distance_metrics import (
    euclidean_distance, manhattan_distance, dot_product_similarity, cosine_similarity
)

class EnhancedVectorDatabase:
    """
    Enhanced vector database that supports multiple distance metrics.
    """
    def __init__(self, embedding_model=None, distance_metric: str = "euclidean"):
        self.vectors = {}
        self.embedding_model = embedding_model
        self.distance_metric = distance_metric
        self.distance_func = self._get_distance_function(distance_metric)
    
    def _get_distance_function(self, metric: str):
        """Get the appropriate distance function."""
        metrics = {
            "euclidean": euclidean_distance,
            "manhattan": manhattan_distance,
            "dot_product": dot_product_similarity,
            "cosine": cosine_similarity
        }
        return metrics.get(metric, euclidean_distance)
    
    async def abuild_from_list(self, list_of_text: List[str]) -> "EnhancedVectorDatabase":
        """Build database from list of texts using embeddings."""
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding in zip(list_of_text, embeddings):
            self.vectors[text] = np.array(embedding)
        return self
    
    def search_by_text(self, query: str, k: int = 4) -> List[Tuple[str, float]]:
        """Search using text query."""
        # Get query embedding
        query_embedding = asyncio.run(self.embedding_model.async_get_embeddings([query]))[0]
        query_vector = np.array(query_embedding)
        
        return self.search_by_vector(query_vector, k)
    
    def search_by_vector(self, query_vector: np.ndarray, k: int = 4) -> List[Tuple[str, float]]:
        """Search using vector query."""
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

class EnhancedRAGPipeline:
    """
    Enhanced RAG pipeline with configurable distance metrics.
    """
    def __init__(self, llm: ChatOpenAI, vector_db: EnhancedVectorDatabase, 
                 response_style: str = "detailed", include_scores: bool = False):
        self.llm = llm
        self.vector_db = vector_db
        self.response_style = response_style
        self.include_scores = include_scores
        
        # Define RAG prompts
        self.RAG_SYSTEM_TEMPLATE = """You are a knowledgeable assistant that answers questions based strictly on provided context.

Instructions:
- Only answer questions using information from the provided context
- If the context doesn't contain relevant information, respond with "I don't know"
- Be accurate and cite specific parts of the context when possible
- Keep responses {response_style} and {response_length}
- Only use the provided context. Do not use external knowledge.
- Only provide answers when you are confident the context supports your response."""

        self.RAG_USER_TEMPLATE = """Context Information:
{context}

Number of relevant sources found: {context_count}
{similarity_scores}

Question: {user_query}

Please provide your answer based solely on the context above."""

        self.rag_system_prompt = SystemRolePrompt(self.RAG_SYSTEM_TEMPLATE)
        self.rag_user_prompt = UserRolePrompt(self.RAG_USER_TEMPLATE)

    def run_pipeline(self, user_query: str, k: int = 4, **system_kwargs) -> Dict[str, Any]:
        """Run the RAG pipeline with the configured distance metric."""
        # Retrieve relevant contexts
        context_list = self.vector_db.search_by_text(user_query, k=k)
        
        context_prompt = ""
        similarity_scores = []
        
        for i, (context, score) in enumerate(context_list, 1):
            context_prompt += f"[Source {i}]: {context}\n\n"
            similarity_scores.append(f"Source {i}: {score:.3f}")
        
        # Create system message with parameters
        system_params = {
            "response_style": self.response_style,
            "response_length": system_kwargs.get("response_length", "detailed")
        }
        
        formatted_system_prompt = self.rag_system_prompt.create_message(**system_params)
        
        user_params = {
            "user_query": user_query,
            "context": context_prompt.strip(),
            "context_count": len(context_list),
            "similarity_scores": f"Relevance scores ({self.vector_db.distance_metric}): {', '.join(similarity_scores)}" if self.include_scores else ""
        }
        
        formatted_user_prompt = self.rag_user_prompt.create_message(**user_params)

        return {
            "response": self.llm.run([formatted_system_prompt, formatted_user_prompt]),
            "context": context_list,
            "context_count": len(context_list),
            "similarity_scores": similarity_scores if self.include_scores else None,
            "distance_metric": self.vector_db.distance_metric,
            "prompts_used": {
                "system": formatted_system_prompt,
                "user": formatted_user_prompt
            }
        }

# Helper functions for your notebook
async def build_enhanced_vector_database(split_documents, embedding_model, distance_metric: str = "euclidean"):
    """Build enhanced vector database with custom distance metric."""
    print(f"Building vector database with {distance_metric} distance metric...")
    vector_db = EnhancedVectorDatabase(embedding_model=embedding_model, distance_metric=distance_metric)
    vector_db = await vector_db.abuild_from_list(split_documents)
    print(f"Vector database built successfully with {distance_metric} metric")
    return vector_db

def create_enhanced_rag_pipeline(vector_db: EnhancedVectorDatabase, llm: ChatOpenAI):
    """Create enhanced RAG pipeline."""
    return EnhancedRAGPipeline(vector_db=vector_db, llm=llm)

# Example usage
if __name__ == "__main__":
    print("Enhanced RAG Pipeline with Custom Distance Metrics")
    print("Available metrics: euclidean, manhattan, dot_product, cosine") 
