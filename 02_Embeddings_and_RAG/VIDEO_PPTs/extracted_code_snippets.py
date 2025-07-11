# Code snippets extracted from Pythonic_RAG_Assignment.ipynb
# Total snippets: 40

# --- Snippet 1 ---
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
import asyncio

================================================================================

# --- Snippet 2 ---
import nest_asyncio
nest_asyncio.apply()

================================================================================

# --- Snippet 3 ---
text_loader = TextFileLoader("data/PMarcaBlogs.txt")
documents = text_loader.load_documents()
len(documents)

================================================================================

# --- Snippet 4 ---
text_splitter = CharacterTextSplitter()
split_documents = text_splitter.split_texts(documents)
len(split_documents)

================================================================================

# --- Snippet 5 ---
import os
import openai
# from getpass import getpass
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = getpass("OpenAI API Key: ")
# os.environ["OPENAI_API_KEY"] = openai.api_key

================================================================================

# --- Snippet 6 ---
vector_db = VectorDatabase()
vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))

================================================================================

# --- Snippet 7 ---
vector_db.search_by_text("What is the Michael Eisner Memorial Weak Executive Problem?", k=3)

================================================================================

# --- Snippet 8 ---
from aimakerspace.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
    #AssistantRolePrompt, # not used
)

from aimakerspace.openai_utils.chatmodel import ChatOpenAI

chat_openai = ChatOpenAI()
user_prompt_template = "{content}"
user_role_prompt = UserRolePrompt(user_prompt_template)
system_prompt_template = (
    "You are an expert in {expertise}, you always answer in a kind way."
)
system_role_prompt = SystemRolePrompt(system_prompt_template)

messages = [
    system_role_prompt.create_message(expertise="Python"),
    user_role_prompt.create_message(
        content="What is the best way to write a loop?"
    ),
]

response = chat_openai.run(messages)

================================================================================

# --- Snippet 9 ---
RAG_SYSTEM_TEMPLATE = """You are a knowledgeable assistant that answers questions based strictly on provided context.

Instructions:
- Only answer questions using information from the provided context
- If the context doesn't contain relevant information, respond with "I don't know"
- Be accurate and cite specific parts of the context when possible
- Keep responses {response_style} and {response_length}
- Only use the provided context. Do not use external knowledge.
- Only provide answers when you are confident the context supports your response."""

RAG_USER_TEMPLATE = """Context Information:
{context}

Number of relevant sources found: {context_count}
{similarity_scores}

Question: {user_query}

Please provide your answer based solely on the context above."""

rag_system_prompt = SystemRolePrompt(
    RAG_SYSTEM_TEMPLATE,
    strict=True,
    defaults={
        "response_style": "concise",
        "response_length": "brief"
    }
)

rag_user_prompt = UserRolePrompt(
    RAG_USER_TEMPLATE,
    strict=True,
    defaults={
        "context_count": "",
        "similarity_scores": ""
    }
)

================================================================================

# --- Snippet 10 ---
class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI, vector_db_retriever: VectorDatabase, 
                 response_style: str = "detailed", include_scores: bool = False) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever
        self.response_style = response_style
        self.include_scores = include_scores

    def run_pipeline(self, user_query: str, k: int = 4, **system_kwargs) -> dict:
        # Retrieve relevant contexts
        context_list = self.vector_db_retriever.search_by_text(user_query, k=k)
        
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
        
        formatted_system_prompt = rag_system_prompt.create_message(**system_params)
        
        user_params = {
            "user_query": user_query,
            "context": context_prompt.strip(),
            "context_count": len(context_list),
            "similarity_scores": f"Relevance scores: {', '.join(similarity_scores)}" if self.include_scores else ""
        }
        
        formatted_user_prompt = rag_user_prompt.create_message(**user_params)

        return {
            "response": self.llm.run([formatted_system_prompt, formatted_user_prompt]), 
            "context": context_list,
            "context_count": len(context_list),
            "similarity_scores": similarity_scores if self.include_scores else None,
            "prompts_used": {
                "system": formatted_system_prompt,
                "user": formatted_user_prompt
            }
        }

================================================================================

# --- Snippet 11 ---
rag_pipeline = RetrievalAugmentedQAPipeline(
    vector_db_retriever=vector_db,
    llm=chat_openai,
    response_style="detailed",
    include_scores=True
)

result = rag_pipeline.run_pipeline(
    "What is the 'Michael Eisner Memorial Weak Executive Problem'?",
    k=3,
    response_length="comprehensive", 
    include_warnings=True,
    confidence_required=True
)

print(f"Response: {result['response']}")
print(f"\nContext Count: {result['context_count']}")
print(f"Similarity Scores: {result['similarity_scores']}")

================================================================================

# --- Snippet 12 ---
! pip install pdfplumber PyPDF2

================================================================================

# --- Snippet 13 ---
import os
import openai
# from getpass import getpass
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

================================================================================

# --- Snippet 14 ---
from aimakerspace.text_utils import PDFFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.prompts import UserRolePrompt, SystemRolePrompt
from aimakerspace.openai_utils.chatmodel import ChatOpenAI

================================================================================

# --- Snippet 15 ---
import pdfplumber
import PyPDF2
import asyncio

================================================================================

# --- Snippet 16 ---
def load_pdf_documents(pdf_file: str, use_pdfplumber: bool = True):
    """
    Load documents from a PDF file using PDFFileLoader.

    Args:
        pdf_path: Path to the PDF file
        use_pdfplumber: Whether to use pdfplumber (better) or PyPDF2 (faster)

    Returns:
        List of document texts
    """
    print(f"Loading PDF from: {pdf_file}")

    try:
        pdf_loader = PDFFileLoader(pdf_file, use_pdfplumber=use_pdfplumber)
        documents = pdf_loader.load_documents()
        print(f"Successfully loaded {len(documents)} document(s) from PDF")
        return documents
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages: pip install pdfplumber PyPDF2")
        return []
    except FileNotFoundError:
        print(f"PDF file not found: {pdf_file}")
        return []
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []

================================================================================

# --- Snippet 17 ---
def documents_splitter(documents, chunk_size: int = 1000, chunk_overlap: int = 200): 
    """
    Split documents into smaller chunks for processing.

    Args:
        documents: List of document texts
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_documents = text_splitter.split_texts(documents)
    print(f"Split documents into {len(split_documents)} chunks")
    return split_documents

================================================================================

# --- Snippet 18 ---
async def build_vector_database(split_documents):
    """
    Build vector database from document chunks.

    Args:
        split_documents: List of text chunks

    Returns:
        VectorDatabase instance
    """
    print("Building vector database...")
    vector_db = VectorDatabase()
    vector_db = await vector_db.abuild_from_list(split_documents)
    print("Vector database built successfully")
    return vector_db

================================================================================

# --- Snippet 19 ---
def create_rag_pipeline(vector_db, llm):
    """
    Create a RAG pipeline for question answering.

    Args:
        vector_db: VectorDatabase instance
        llm: ChatOpenAI instance

    Returns:
        RetrievalAugmentedQAPipeline instance
    """
    # Define RAG prompts
    RAG_SYSTEM_TEMPLATE = """You are a knowledgeable assistant that answers questions based strictly on provided context.

    Instructions:
    - Only answer questions using information from the provided context
    - If the context doesn't contain relevant information, respond with "I don't know"
    - Be accurate and cite specific parts of the context when possible
    - Keep responses {response_style} and {response_length}
    - Only use the provided context. Do not use external knowledge.
    - Only provide answers when you are confident the context supports your response.
    """

    RAG_USER_TEMPLATE = """Context Information:
    {context}

    Number of relevant sources found: {context_count}
    {similarity_scores}

    Question: {user_query}

    Please provide your answer based solely on the context above."""

    rag_system_prompt = SystemRolePrompt(RAG_SYSTEM_TEMPLATE)
    rag_user_prompt = UserRolePrompt(RAG_USER_TEMPLATE)

    class RetrievalAugmentedQAPipeline:
        def __init__(self, llm, vector_db_retriever,
                     response_style: str = "detailed", include_scores: bool = False):
            self.llm = llm
            self.vector_db_retriever = vector_db_retriever
            self.response_style = response_style
            self.include_scores = include_scores

        def run_pipeline(self, user_query: str, k: int = 4, **system_kwargs):
            # Retrieve relevant contexts
            context_list = self.vector_db_retriever.search_by_text(user_query, k=k)
                                                                                         
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

            formatted_system_prompt = rag_system_prompt.create_message(**system_params)

            user_params = {
                "user_query": user_query,
                "context": context_prompt.strip(),
                "context_count": len(context_list),
                "similarity_scores": f"Relevance scores: {', '.join(similarity_scores)}" if self.include_scores else ""
            }

            formatted_user_prompt = rag_user_prompt.create_message(**user_params)

            return {
                "response": self.llm.run([formatted_system_prompt, formatted_user_prompt]),
                "context": context_list,
                "context_count": len(context_list),
                "similarity_scores": similarity_scores if self.include_scores else None,
                "prompts_used": {
                    "system": formatted_system_prompt,
                    "user": formatted_user_prompt
                }
            }

    return RetrievalAugmentedQAPipeline(llm, vector_db)

================================================================================

# --- Snippet 20 ---
# 1. LOAD PDF DOCUMENT    
pdf_path = "/home/olb/AIE7-BC/AIM_AIE7/02_Embeddings_and_RAG/data/"
pdf_filename = "Apple_Illusion_of_thinking_2025.pdf"
documents = load_pdf_documents(pdf_path+pdf_filename)
if not documents:
    print("No documents loaded. Please check the PDF path and try again.")
else:
# Display sample of loaded content
    print(f"\nSample content (first 200 characters):")
    print(documents[0][:200] + ".../n")

================================================================================

# --- Snippet 21 ---
# 2. SPLIT DOCUMENTS INTO CHUNKS
split_documents = documents_splitter(documents)

================================================================================

# --- Snippet 22 ---
import tracemalloc
tracemalloc.start()

================================================================================

# --- Snippet 23 ---
async def main():
    vector_db = await build_vector_database(split_documents)
    return vector_db

# Run the async function
vector_db = await main()

================================================================================

# --- Snippet 24 ---
# SIMPLE METADATA SUPPORT - GENERIC VERSION
import datetime
import os

class SimpleMetadataDB:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.metadata = {}
    
    def add_basic_metadata(self, text, source_file=None, tags=None, chunk_index=None, file_dir="data"):
        """Add basic metadata to a document.
           text is each document chunk in the split_documents
           source_file is the name of the source file
           tags is a list of keywords to filter in the document
           chunk_index is the index of this chunk in the document
        """
        if text in self.vector_db.vectors:
            # Use defaults if not provided
            if source_file is None:
                source_file = "unknown_file"
            
            if tags is None:
                tags = ["general", "document", "text"]
            
            if chunk_index is None:
                chunk_index = 0
            
            # Get file info if possible
            try:
                # Use current working directory + data subfolder (generic)
                current_dir = os.getcwd()
                file_path = os.path.join(current_dir, file_dir, source_file)
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            except:
                file_path = os.path.join("data", source_file)  # fallback
                file_size = 0
            # Determine file type from extension
            file_type = "pdf"  # default
            if source_file:
                if source_file.lower().endswith('.pdf'):
                    file_type = "pdf"
                elif source_file.lower().endswith('.txt'):
                    file_type = "txt"
                elif source_file.lower().endswith('.docx'):
                    file_type = "docx"
                elif source_file.lower().endswith('.html'):
                    file_type = "html"
            
            self.metadata[text] = {
                "chunk_index": chunk_index,
                "source_file": source_file,
                "file_path": file_path,
                "file_size": file_size,
                "file_type": file_type,
                "file_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "file_keywords": tags,
                "file_language": "English",
                "file_encoding": "UTF-8",
                "file_creation_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "file_modification_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "file_last_modified": datetime.datetime.now().strftime("%Y-%m-%d")
            }     
            print(f"Added metadata for chunk {chunk_index} from: {source_file}")
        else:
            print(f"Document not found: {text[:50]}...")
    
    def add_metadata_to_all_documents(self, source_file, tags=None):
        """Add metadata to all documents in the vector database."""
        print(f"Adding metadata to all documents from: {source_file}")
        
        for i, text in enumerate(self.vector_db.vectors.keys()):
            self.add_basic_metadata(
                text=text,
                source_file=source_file,
                tags=tags,
                chunk_index=i
            )
        
        print(f"Added metadata to {len(self.vector_db.vectors)} documents")
    
    def search_with_source_filter(self, query, source_file=None):
        """Search and optionally filter by source file."""
        results = self.vector_db.search_by_text(query, k=10)
        
        filtered_results = []
        for text, score in results:
            metadata = self.metadata.get(text, {})
            
            # If source filter is specified, only include matching documents
            if source_file is None or metadata.get('source_file') == source_file:
                filtered_results.append((text, score, metadata))
        
        return filtered_results
    
    def search_by_tags(self, query, tags=None):
        """Search in documents tagged with tags and filter by tags."""
        results = self.vector_db.search_by_text(query, k=20)  # Get more results for filtering
        
        filtered_results = []
        for text, score in results:
            metadata = self.metadata.get(text, {})
            document_tags = metadata.get('file_keywords', [])
            
            # If tags filter is specified, check if any tag matches
            if tags is None or any(tag in document_tags for tag in tags):
                filtered_results.append((text, score, metadata))
        
        return filtered_results[:10]  # Return top 10
    
    def get_metadata_summary(self):
        """Get a summary of all metadata."""
        if not self.metadata:
            return {"total_documents": 0}
        
        summary = {
            "total_documents": len(self.metadata),
            "source_files": set(),
            "file_types": set(),
            "all_tags": set()
        }
        
        for metadata in self.metadata.values():
            if 'source_file' in metadata:
                summary["source_files"].add(metadata['source_file'])
            if 'file_type' in metadata:
                summary["file_types"].add(metadata['file_type'])
            if 'file_keywords' in metadata:
                summary["all_tags"].update(metadata['file_keywords'])
        
        # Convert sets to lists for easier handling
        summary["source_files"] = list(summary["source_files"])
        summary["file_types"] = list(summary["file_types"])
        summary["all_tags"] = list(summary["all_tags"])
        
        return summary

================================================================================

# --- Snippet 25 ---
# 1. Create the enhanced database
enhanced_db = SimpleMetadataDB(vector_db)

# 2. Add metadata to your documents
enhanced_db.add_metadata_to_all_documents(
    source_file=pdf_filename,
    tags=["Illusion of Thinking", "Large Reasoning Models", "LLMs", "RLMs", "Problem Complexity", "Reasoning Models", "AI", "Research"]
)

================================================================================

# --- Snippet 26 ---
# Test with more specific queries
test_queries = [
    "Large Reasoning Models",  # More specific
    "LLMs reasoning capabilities",  # More specific
    "thinking models",  # Different phrasing
    "reasoning limitations",  # Different aspect
    "problem complexity"  # Another topic from your tags
]

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    results = enhanced_db.search_with_source_filter(
        query=query,
        source_file="Apple_Illusion_of_thinking_2025.pdf"
    )
    
    if results:
        best_score = results[0][1]  # Get the highest score
        print(f"   Best score: {best_score:.4f}")
        print(f"   Best match: {results[0][0][:80]}...")
    else:
        print("   No results found")

================================================================================

# --- Snippet 27 ---
# 4. Initialize LLM
chat_openai = ChatOpenAI()

# 5. CREATE RAG PIPELINE
rag_pipeline = create_rag_pipeline(vector_db, chat_openai)

print("Vector database built successfully!")
print("RAG pipeline created successfully!")

================================================================================

# --- Snippet 28 ---
# 6. TEST RAG PIPELINE 
test_queries = [
    "What is the main topic of this document?",
    "What are the key points discussed?",
    "Can you summarize the content?"
]
for query in test_queries:
    print(f"\nQuestion: {query}")
    try:
        result = rag_pipeline.run_pipeline(
            query,
            k=3,
            include_scores=True,
            response_length="comprehensive"
        )    
    
        print(f"Response: {result['response']}")
        print(f"Context Count: {result['context_count']}")
        if result['similarity_scores']:
            print(f"Similarity Scores: {result['similarity_scores']}")
    except Exception as e:
        print(f"Error processing query: {e}")

================================================================================

# --- Snippet 29 ---
def cos_similarity(vector_db, test_query: str = "What are the limitations of reasoning models?"):
    """
    Quick test of cos similarity distance.
    """
    print(f"Quick cos similarity Distance Test")
    print(f"Query: '{test_query}'")
    print("-" * 50)
    
    try:
        # Use the existing search method which already handles embeddings internally
        # This avoids the async issues entirely
        search_results = vector_db.search_by_text(test_query, k=10)
        
        if not search_results:
            print("No results found. Please check your vector database.")
            return []
        
        print("Top 3 results from existing search method:")
        for i, (text, score) in enumerate(search_results[:3], 1):
            print(f"{i}. Similarity Score: {score:.4f}")
            print(f"   Text: {text[:100]}...")
            print()
            
        return search_results[:3]
        
    except Exception as e:
        print(f"Error: {e}")
        return []

================================================================================

# --- Snippet 30 ---
test_query = "What are the limitations of reasoning models?"
cos_similarity(vector_db, test_query = test_query)
search_results = vector_db.search_by_text(test_query, k=10)

================================================================================

# --- Snippet 31 ---
# 6. TEST RAG PIPELINE 
test_queries = [
    "What is the main topic of this document?",
    "What are the key points discussed?",
    "Can you summarize the content?"
]
for query in test_queries:
    print(f"\nQuestion: {query}")
    try:
        result = rag_pipeline.run_pipeline(
            query,
            k=3,
            include_scores=True,
            response_length="comprehensive"
        )    
    
        print(f"Response: {result['response']}")
        print(f"Context Count: {result['context_count']}")
        if result['similarity_scores']:
            print(f"Similarity Scores: {result['similarity_scores']}")
    except Exception as e:
        print(f"Error processing query: {e}")

================================================================================

# --- Snippet 32 ---
# CALCULATE METRICS BETWEEN EXISTING VECTORS
import numpy as np

def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))

def dot_product_similarity(vec1, vec2):
    return np.dot(vec1, vec2)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

def compare_metrics_between_vectors(vector_db, k=5):
    """Compare different metrics between existing vectors in the database."""
    
    print("�� COMPARING DISTANCE METRICS BETWEEN EXISTING VECTORS")
    print("=" * 60)
    
    # Get all vectors from the database
    all_texts = list(vector_db.vectors.keys())
    print(f"Found {len(all_texts)} vectors in database")
    
    if len(all_texts) < 2:
        print("Need at least 2 vectors to compare")
        return
    
    # Pick the first vector as reference
    reference_text = all_texts[0]
    reference_vector = np.array(vector_db.vectors[reference_text])
    
    print(f"Using as reference: {reference_text[:80]}...")
    print(f"Reference vector dimension: {len(reference_vector)}")
    print()
    
    # Calculate all metrics for all other vectors
    results = []
    
    for text in all_texts[1:]:  # Skip the reference vector
        vector = np.array(vector_db.vectors[text])
        
        # Calculate all metrics
        euclidean_dist = euclidean_distance(reference_vector, vector)
        manhattan_dist = manhattan_distance(reference_vector, vector)
        dot_prod = dot_product_similarity(reference_vector, vector)
        cosine_sim = cosine_similarity(reference_vector, vector)
        
        results.append({
            'text': text,
            'euclidean': euclidean_dist,
            'manhattan': manhattan_dist,
            'dot_product': dot_prod,
            'cosine': cosine_sim
        })
    
    # Sort by each metric
    print("📊 RANKING BY EACH METRIC:")
    print()
    
    # Euclidean (lower is better)
    euclidean_sorted = sorted(results, key=lambda x: x['euclidean'])
    print("�� EUCLIDEAN DISTANCE (lower = better):")
    for i, result in enumerate(euclidean_sorted[:k], 1):
        print(f"{i}. Distance: {result['euclidean']:.4f}")
        print(f"   Text: {result['text'][:80]}...")
    print()
    
    # Manhattan (lower is better)
    manhattan_sorted = sorted(results, key=lambda x: x['manhattan'])
    print("🗺️ MANHATTAN DISTANCE (lower = better):")
    for i, result in enumerate(manhattan_sorted[:k], 1):
        print(f"{i}. Distance: {result['manhattan']:.4f}")
        print(f"   Text: {result['text'][:80]}...")
    print()
    
    # Dot Product (higher is better)
    dot_sorted = sorted(results, key=lambda x: x['dot_product'], reverse=True)
    print("🔗 DOT PRODUCT SIMILARITY (higher = better):")
    for i, result in enumerate(dot_sorted[:k], 1):
        print(f"{i}. Score: {result['dot_product']:.4f}")
        print(f"   Text: {result['text'][:80]}...")
    print()
    
    # Cosine (higher is better)
    cosine_sorted = sorted(results, key=lambda x: x['cosine'], reverse=True)
    print("📐 COSINE SIMILARITY (higher = better):")
    for i, result in enumerate(cosine_sorted[:k], 1):
        print(f"{i}. Score: {result['cosine']:.4f}")
        print(f"   Text: {result['text'][:80]}...")
    print()
    
    return {
        'euclidean': euclidean_sorted[:k],
        'manhattan': manhattan_sorted[:k],
        'dot_product': dot_sorted[:k],
        'cosine': cosine_sorted[:k]
    }

# Run the comparison
results = compare_metrics_between_vectors(vector_db, k=5)

================================================================================

# --- Snippet 33 ---
# COMPARE MULTIPLE VECTORS
def compare_multiple_vectors(vector_db, num_vectors=3):
    """Compare multiple vectors against each other."""
    
    all_texts = list(vector_db.vectors.keys())
    if len(all_texts) < num_vectors:
        num_vectors = len(all_texts)
    
    print(f"🔍 COMPARING {num_vectors} VECTORS")
    print("=" * 50)
    
    # Pick first few vectors
    selected_texts = all_texts[:num_vectors]
    
    for i, text1 in enumerate(selected_texts):
        vector1 = np.array(vector_db.vectors[text1])
        print(f"\n�� Reference {i+1}: {text1[:60]}...")
        
        for j, text2 in enumerate(selected_texts):
            if i != j:  # Don't compare with itself
                vector2 = np.array(vector_db.vectors[text2])
                
                euclidean = euclidean_distance(vector1, vector2)
                manhattan = manhattan_distance(vector1, vector2)
                dot_prod = dot_product_similarity(vector1, vector2)
                cosine = cosine_similarity(vector1, vector2)
                
                print(f"  vs {j+1}: Euclidean={euclidean:.4f}, Manhattan={manhattan:.4f}, Dot={dot_prod:.4f}, Cosine={cosine:.4f}")

# Run the comparison
compare_multiple_vectors(vector_db, num_vectors=3)

================================================================================

# --- Snippet 34 ---
# import os
# import openai
# # from getpass import getpass
# from dotenv import load_dotenv

# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
# # openai.api_key = getpass("OpenAI API Key: ")
# # os.environ["OPENAI_API_KEY"] = openai.api_key

================================================================================

# --- Snippet 35 ---
def add_metadata_to_existing_chunks(split_documents, source_file, tags=None, file_dir="data"):
    """
    Add metadata directly to each text chunk.
    """
    enhanced_chunks = []
    
    for i, chunk in enumerate(split_documents):
        metadata_info = f"""
        CHUNK {i+1}
        Date: {datetime.datetime.now().strftime("%Y-%m-%d")}
        Source: {source_file}
        Path: {os.path.join(file_dir, source_file)}
        Type: {"pdf" if source_file.lower().endswith('.pdf') else "txt"}
        Keywords: {', '.join(tags or ["general", "document"])}
        Date: {datetime.datetime.now().strftime("%Y-%m-%d")}
        Total_chunks: {len(split_documents)}
        Language: English
        Encoding: UTF-8
        """
            
        enhanced_text = metadata_info + chunk
        enhanced_chunks.append(enhanced_text)
    
    print(f"Added metadata to {len(enhanced_chunks)} existing chunks")
    return enhanced_chunks

================================================================================

# --- Snippet 36 ---
documents_with_metadata = add_metadata_to_existing_chunks(
    split_documents,
    source_file=pdf_filename,  # Just the filename
    tags=["Illusion of Thinking", "Large Reasoning Models", "LLMs", "RLMs", "Problem Complexity", "Reasoning Models", "AI", "Research"],
    file_dir="data"  # Specify directory here
)

================================================================================

# --- Snippet 37 ---
def fix_reversed_text(text):
    """
    Fix common text reversal issues.
    """
    # Fix common reversed patterns
    corrections = {
        "5202": "2025",
        "viXra": "arXiv",
        "nuJ": "Jun",  # or "Jul" depending on context
        "]IA.sc[": "cs.AI",  # Common arXiv category
        "1v14960.6052": "2025.06052v1"  # arXiv ID format
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    return text

================================================================================

# --- Snippet 38 ---
if documents_with_metadata:
    documents_with_metadata[0] = fix_reversed_text(documents_with_metadata[0])
    print("Fixed text sample:")
    print(documents_with_metadata[0][400:500])

================================================================================

# --- Snippet 39 ---
vector_db = await build_vector_database(documents_with_metadata)

================================================================================

# --- Snippet 40 ---
# 4. Initialize LLM
chat_openai = ChatOpenAI()

# 5. CREATE RAG PIPELINE
rag_pipeline = create_rag_pipeline(vector_db, chat_openai)

print("Vector database built successfully!")
print("RAG pipeline created successfully!")

================================================================================

