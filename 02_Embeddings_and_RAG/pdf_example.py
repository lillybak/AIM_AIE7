#!/usr/bin/env python3
"""
Example script demonstrating how to use PDFFileLoader instead of TextFileLoader
for reading PDF files in the RAG application.
"""

import asyncio
from aimakerspace.text_utils import PDFFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase

def main():
    """
    Example of how to use PDFFileLoader to process PDF documents
    instead of text files in the RAG pipeline.
    """
    
    # Example 1: Load a single PDF file
    print("=== Example 1: Loading a single PDF file ===")
    try:
        # Replace 'path/to/your/document.pdf' with actual PDF file path
        pdf_loader = PDFFileLoader("path/to/your/document.pdf", use_pdfplumber=True)
        documents = pdf_loader.load_documents()
        print(f"Loaded {len(documents)} document(s) from PDF")
        if documents:
            print(f"First 200 characters: {documents[0][:200]}...")
    except FileNotFoundError:
        print("PDF file not found. Please update the path to a valid PDF file.")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install required packages with: pip install pdfplumber PyPDF2")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Load multiple PDF files from a directory
    print("=== Example 2: Loading multiple PDF files from directory ===")
    try:
        # Replace 'path/to/pdf/directory' with actual directory path
        pdf_loader = PDFFileLoader("path/to/pdf/directory", use_pdfplumber=True)
        documents = pdf_loader.load_documents()
        print(f"Loaded {len(documents)} document(s) from directory")
    except FileNotFoundError:
        print("Directory not found. Please update the path to a valid directory.")
    except ImportError as e:
        print(f"Missing dependency: {e}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Complete RAG pipeline with PDF (if you have a PDF file)
    print("=== Example 3: Complete RAG pipeline with PDF ===")
    
    # Uncomment and modify the following code when you have a PDF file to test with:
    """
    # Load PDF documents
    pdf_loader = PDFFileLoader("your_document.pdf", use_pdfplumber=True)
    documents = pdf_loader.load_documents()
    
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_texts(documents)
    print(f"Split into {len(split_documents)} chunks")
    
    # Build vector database
    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))
    print("Vector database built successfully")
    
    # Test search
    query = "What is the main topic of this document?"
    results = vector_db.search_by_text(query, k=3)
    print(f"Search results for '{query}':")
    for i, (text, score) in enumerate(results, 1):
        print(f"Result {i} (score: {score:.3f}): {text[:100]}...")
    """

def install_dependencies():
    """Helper function to install required dependencies."""
    print("Installing required dependencies...")
    import subprocess
    import sys
    
    packages = ["pdfplumber", "PyPDF2"]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")

if __name__ == "__main__":
    print("PDF File Loader Example")
    print("This script demonstrates how to use PDFFileLoader instead of TextFileLoader")
    print("Make sure you have the required dependencies installed.")
    print()
    
    # Check if dependencies are available
    try:
        import pdfplumber
        import PyPDF2
        print("✓ All dependencies are available")
    except ImportError:
        print("✗ Missing dependencies detected")
        response = input("Would you like to install them now? (y/n): ")
        if response.lower() == 'y':
            install_dependencies()
        else:
            print("Please install manually: pip install pdfplumber PyPDF2")
    
    print()
    main() 