#!/usr/bin/env python3
"""
Comparison example showing how to use TextFileLoader vs PDFFileLoader
in the RAG application.
"""

import asyncio
from aimakerspace.text_utils import TextFileLoader, PDFFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase

def compare_loaders():
    """
    Compare TextFileLoader and PDFFileLoader usage patterns.
    """
    print("=== TextFileLoader vs PDFFileLoader Comparison ===\n")
    
    # Example 1: TextFileLoader (original)
    print("1. TEXT FILE LOADER (Original)")
    print("-" * 40)
    print("Usage:")
    print("text_loader = TextFileLoader('data/PMarcaBlogs.txt')")
    print("documents = text_loader.load_documents()")
    print()
    
    # Example 2: PDFFileLoader (new)
    print("2. PDF FILE LOADER (New)")
    print("-" * 40)
    print("Usage:")
    print("pdf_loader = PDFFileLoader('document.pdf', use_pdfplumber=True)")
    print("documents = pdf_loader.load_documents()")
    print()
    
    print("Key Differences:")
    print("• TextFileLoader: Works with .txt files")
    print("• PDFFileLoader: Works with .pdf files")
    print("• PDFFileLoader: Supports both pdfplumber (better) and PyPDF2 (faster)")
    print("• PDFFileLoader: Handles multi-page PDFs automatically")
    print("• Both: Same interface - drop-in replacement")
    print()

def show_migration_example():
    """
    Show how to migrate from TextFileLoader to PDFFileLoader.
    """
    print("=== Migration Example ===")
    print()
    
    print("BEFORE (TextFileLoader):")
    print("```python")
    print("from aimakerspace.text_utils import TextFileLoader")
    print("")
    print("text_loader = TextFileLoader('data/PMarcaBlogs.txt')")
    print("documents = text_loader.load_documents()")
    print("```")
    print()
    
    print("AFTER (PDFFileLoader):")
    print("```python")
    print("from aimakerspace.text_utils import PDFFileLoader")
    print("")
    print("pdf_loader = PDFFileLoader('document.pdf', use_pdfplumber=True)")
    print("documents = pdf_loader.load_documents()")
    print("```")
    print()
    
    print("The rest of your RAG pipeline remains exactly the same!")
    print()

def show_complete_pipeline_comparison():
    """
    Show complete pipeline comparison.
    """
    print("=== Complete Pipeline Comparison ===")
    print()
    
    print("TEXT FILE PIPELINE:")
    print("```python")
    print("# Load text documents")
    print("text_loader = TextFileLoader('data/PMarcaBlogs.txt')")
    print("documents = text_loader.load_documents()")
    print("")
    print("# Split into chunks")
    print("text_splitter = CharacterTextSplitter()")
    print("split_documents = text_splitter.split_texts(documents)")
    print("")
    print("# Build vector database")
    print("vector_db = VectorDatabase()")
    print("vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))")
    print("```")
    print()
    
    print("PDF FILE PIPELINE:")
    print("```python")
    print("# Load PDF documents")
    print("pdf_loader = PDFFileLoader('document.pdf', use_pdfplumber=True)")
    print("documents = pdf_loader.load_documents()")
    print("")
    print("# Split into chunks (SAME CODE)")
    print("text_splitter = CharacterTextSplitter()")
    print("split_documents = text_splitter.split_texts(documents)")
    print("")
    print("# Build vector database (SAME CODE)")
    print("vector_db = VectorDatabase()")
    print("vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))")
    print("```")
    print()
    
    print("Notice: Only the loader changes, everything else stays the same!")

def show_advanced_pdf_features():
    """
    Show advanced PDF features.
    """
    print("=== Advanced PDF Features ===")
    print()
    
    print("1. Multiple PDF Processing Options:")
    print("```python")
    print("# Better text extraction (recommended)")
    print("pdf_loader = PDFFileLoader('document.pdf', use_pdfplumber=True)")
    print("")
    print("# Faster processing (less accurate)")
    print("pdf_loader = PDFFileLoader('document.pdf', use_pdfplumber=False)")
    print("```")
    print()
    
    print("2. Directory Processing:")
    print("```python")
    print("# Process all PDFs in a directory")
    print("pdf_loader = PDFFileLoader('pdf_documents/')")
    print("documents = pdf_loader.load_documents()")
    print("```")
    print()
    
    print("3. Error Handling:")
    print("```python")
    print("try:")
    print("    pdf_loader = PDFFileLoader('document.pdf')")
    print("    documents = pdf_loader.load_documents()")
    print("except ImportError as e:")
    print("    print('Install dependencies: pip install pdfplumber PyPDF2')")
    print("except FileNotFoundError:")
    print("    print('PDF file not found')")
    print("```")
    print()

def main():
    """
    Main function to demonstrate the comparison.
    """
    compare_loaders()
    show_migration_example()
    show_complete_pipeline_comparison()
    show_advanced_pdf_features()
    
    print("=== Summary ===")
    print("• PDFFileLoader is a drop-in replacement for TextFileLoader")
    print("• Only the import and loader instantiation change")
    print("• All other RAG pipeline code remains identical")
    print("• PDFFileLoader supports both single files and directories")
    print("• Choose pdfplumber for better text extraction, PyPDF2 for speed")
    print()
    print("To get started:")
    print("1. Install dependencies: pip install pdfplumber PyPDF2")
    print("2. Replace TextFileLoader with PDFFileLoader")
    print("3. Update file paths to point to PDF files")
    print("4. Run your existing RAG pipeline!")

if __name__ == "__main__":
    main() 