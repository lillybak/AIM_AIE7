from aimakerspace.text_utils import PDFFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.prompts import UserRolePrompt, SystemRolePrompt
from aimakerspace.openai_utils.chatmodel import ChatOpenAI
import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def load_pdf_documents(pdf_path: str, pdf_filename: str, use_pdfplumber: bool = True):
    """
    Load documents from a PDF file using PDFFileLoader.

    Args:
        pdf_path: Path to the PDF file
        pdf_filename: Name of the PDF file
        use_pdfplumber: Whether to use pdfplumber (better) or PyPDF2 (faster)

    Returns:
        List of document texts
    """
    pdf_file = pdf_path + "/" + pdf_filename
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

def split_documents(documents, chunk_size: int = 1000, chunk_overlap: int = 200):
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
- Only provide answers when you are confident the context supports your response."""

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
    
    # Return an instance of the pipeline
    return RetrievalAugmentedQAPipeline(llm, vector_db)

async def main():
    """
    Main function demonstrating the complete PDF-based RAG pipeline.
    """
    print("=== PDF-based RAG Application ===")
    print()
    
    # Step 1: Load PDF documents
    # Replace these paths with your actual PDF file
    pdf_path = "data"  # Update this path
    pdf_filename = "your_document.pdf"  # Update this filename
    
    documents = load_pdf_documents(pdf_path, pdf_filename)
    if not documents:
        print("No documents loaded. Please check the PDF path and try again.")
        return
    
    # Display sample of loaded content
    print(f"\nSample content (first 200 characters):")
    print(documents[0][:200] + "...")
    print()
    
    # Step 2: Split documents into chunks
    split_docs = split_documents(documents, chunk_size=1000, chunk_overlap=200)
    
    # Step 3: Build vector database
    vector_db = await build_vector_database(split_docs)
    
    # Step 4: Initialize LLM
    chat_openai = ChatOpenAI()
    
    # Step 5: Create RAG pipeline
    rag_pipeline = create_rag_pipeline(vector_db, chat_openai)
    
    # Step 6: Test the pipeline
    print("\n=== Testing RAG Pipeline ===")
    
    # Example queries to test
    test_queries = [
        "What is the main topic of this document?",
        "What are the key points discussed?",
        "Can you summarize the content?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag_pipeline.run_pipeline(
                query,
                k=3,
                response_length="comprehensive",
                include_scores=True
            )
            
            print(f"Response: {result['response']}")
            print(f"Context Count: {result['context_count']}")
            if result['similarity_scores']:
                print(f"Similarity Scores: {result['similarity_scores']}")
                
        except Exception as e:
            print(f"Error processing query: {e}")
    
    print("\n=== PDF RAG Pipeline Complete ===")

if __name__ == "__main__":
    # Check for required dependencies
    try:
        import pdfplumber
        import PyPDF2
        print("✓ PDF dependencies available")
    except ImportError:
        print("✗ Missing PDF dependencies")
        print("Please install: pip install pdfplumber PyPDF2")
        exit(1)
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("✗ OPENAI_API_KEY not found in environment")
        print("Please set your OpenAI API key in a .env file or environment variable")
        exit(1)
    
    # Run the main function
    asyncio.run(main()) 