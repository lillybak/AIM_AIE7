"""
PDF Reading Solutions and Fixes
==============================

This file contains various solutions for common PDF reading issues,
including text reversal, orientation problems, and extraction methods.

Common Issues:
- Text read backwards (e.g., "5202" instead of "2025")
- Text read upside down
- Poor text extraction quality
- Missing text or formatting issues
"""

import os
import datetime
from typing import List, Dict, Any

# ============================================================================
# SOLUTION 1: Enhanced PDF Loader with Multiple Methods
# ============================================================================

def load_pdf_documents_enhanced(pdf_file: str, use_pdfplumber: bool = True):
    """
    Enhanced PDF loader that tries multiple extraction methods.
    
    Args:
        pdf_file: Path to the PDF file
        use_pdfplumber: Whether to try pdfplumber first (better for complex layouts)
    
    Returns:
        List of document texts
    """
    print(f"Loading PDF from: {pdf_file}")

    try:
        # Method 1: Try pdfplumber with different settings
        if use_pdfplumber:
            import pdfplumber
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    # Try different extraction methods
                    page_text = page.extract_text()
                    if not page_text:
                        # Try with layout parameter
                        page_text = page.extract_text(layout=True)
                    if not page_text:
                        # Try extracting words and reconstructing
                        words = page.extract_words()
                        page_text = " ".join([word['text'] for word in words])
                    if not page_text:
                        # Try with different parameters
                        page_text = page.extract_text(keep_blank_chars=True)
                    text += page_text + "\n"
                return [text]
        
        # Method 2: Try PyPDF2
        else:
            import PyPDF2
            with open(pdf_file, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    page_text = page.extract_text()
                    text += page_text + "\n"
                return [text]
                
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

# ============================================================================
# SOLUTION 2: PyMuPDF Alternative (Often Better for Complex PDFs)
# ============================================================================

def load_pdf_with_pymupdf(pdf_file: str):
    """
    Load PDF using PyMuPDF (fitz) which often handles text orientation better.
    
    Args:
        pdf_file: Path to the PDF file
    
    Returns:
        List of document texts
    """
    try:
        import fitz
        doc = fitz.open(pdf_file)
        text = ""
        for page in doc:
            # Try different text extraction methods
            page_text = page.get_text()
            if not page_text:
                # Try with different parameters
                page_text = page.get_text("text")
            if not page_text:
                # Try extracting text blocks
                blocks = page.get_text("dict")
                page_text = ""
                for block in blocks["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                page_text += span["text"] + " "
            text += page_text + "\n"
        doc.close()
        return [text]
    except ImportError:
        print("PyMuPDF not installed. Install with: pip install pymupdf")
        return []
    except Exception as e:
        print(f"Error loading PDF with PyMuPDF: {e}")
        return []

# ============================================================================
# SOLUTION 3: Text Correction Functions
# ============================================================================

def fix_reversed_text(text: str) -> str:
    """
    Fix common text reversal issues found in PDFs.
    
    Args:
        text: The text to fix
    
    Returns:
        Corrected text
    """
    # Common reversed patterns and their corrections
    corrections = {
        # Year reversals
        "5202": "2025",
        "4202": "2024",
        "3202": "2023",
        
        # arXiv reversals
        "viXra": "arXiv",
        "]IA.sc[": "cs.AI",
        "]IA.ra[": "ar.AI",
        "]IA.lc[": "cl.AI",
        "]IA.ra[": "ar.AI",
        
        # Month reversals
        "nuJ": "Jun",
        "luJ": "Jul",
        "yaM": "May",
        "raA": "Apr",
        "raM": "Mar",
        "beF": "Feb",
        "naJ": "Jan",
        
        # Common arXiv ID patterns
        "1v14960.6052": "2025.06052v1",
        "1v14960.6051": "2025.06051v1",
        
        # Other common reversals
        "noitacinummoc": "communication",
        "noitamrofni": "information",
        "noitacilppa": "application",
    }
    
    # Apply corrections
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    return text

def fix_text_orientation(text: str) -> str:
    """
    Fix text that appears to be read in wrong orientation.
    
    Args:
        text: The text to fix
    
    Returns:
        Corrected text
    """
    # Split into lines and fix each line
    lines = text.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Check if line looks reversed
        if any(pattern in line for pattern in ["5202", "viXra", "]IA.sc["]):
            # Try to reverse the line
            fixed_line = line[::-1]  # Reverse the string
            fixed_lines.append(fixed_line)
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)

# ============================================================================
# SOLUTION 4: Comprehensive PDF Processing Pipeline
# ============================================================================

def process_pdf_comprehensive(pdf_file: str) -> List[str]:
    """
    Comprehensive PDF processing that tries multiple methods.
    
    Args:
        pdf_file: Path to the PDF file
    
    Returns:
        List of processed document texts
    """
    print(f"Processing PDF: {pdf_file}")
    
    # Try multiple PDF loaders
    documents = []
    
    # Method 1: Enhanced pdfplumber
    print("Trying pdfplumber...")
    documents = load_pdf_documents_enhanced(pdf_file, use_pdfplumber=True)
    
    # Method 2: PyPDF2
    if not documents:
        print("Trying PyPDF2...")
        documents = load_pdf_documents_enhanced(pdf_file, use_pdfplumber=False)
    
    # Method 3: PyMuPDF
    if not documents:
        print("Trying PyMuPDF...")
        documents = load_pdf_with_pymupdf(pdf_file)
    
    # Apply text corrections
    if documents:
        print("Applying text corrections...")
        for i, doc in enumerate(documents):
            # Fix reversed text
            doc = fix_reversed_text(doc)
            # Fix orientation issues
            doc = fix_text_orientation(doc)
            documents[i] = doc
        
        print(f"Successfully processed {len(documents)} document(s)")
        return documents
    else:
        print("Failed to load PDF with all methods")
        return []

# ============================================================================
# SOLUTION 5: Metadata Addition Functions
# ============================================================================

def add_metadata_to_existing_chunks(split_documents: List[str], 
                                   source_file: str, 
                                   tags: List[str] = None, 
                                   file_dir: str = "data") -> List[str]:
    """
    Add metadata directly to each text chunk.
    
    Args:
        split_documents: List of text chunks
        source_file: Name of the source file
        tags: List of tags/keywords
        file_dir: Directory where the file is located
    
    Returns:
        List of text chunks with metadata appended
    """
    enhanced_chunks = []
    
    for i, text in enumerate(split_documents):
        # Create metadata string
        metadata_info = f"""
CHUNK {i+1}
Source: {source_file}
Path: {os.path.join(file_dir, source_file)}
Type: {"pdf" if source_file.lower().endswith('.pdf') else "txt"}
Keywords: {', '.join(tags or ["general", "document"])}
Date: {datetime.datetime.now().strftime("%Y-%m-%d")}
Language: English
Encoding: UTF-8

"""
        
        # Append metadata to the text chunk
        enhanced_text = metadata_info + text
        enhanced_chunks.append(enhanced_text)
    
    print(f"Added metadata to {len(enhanced_chunks)} existing chunks")
    return enhanced_chunks

# ============================================================================
# SOLUTION 6: Document Class (Optional)
# ============================================================================

class DocumentWithMetadata:
    """
    Optional class for storing documents with metadata.
    Use this if you prefer object-oriented approach.
    """
    def __init__(self, text: str, metadata: Dict[str, Any] = None):
        self.text = text
        self.metadata = metadata or {}
    
    def __str__(self):
        return self.text
    
    def __repr__(self):
        return f"DocumentWithMetadata(text='{self.text[:50]}...', metadata={self.metadata})"

def create_documents_with_metadata(split_documents: List[str], 
                                  source_file: str, 
                                  tags: List[str] = None, 
                                  file_dir: str = "data") -> List[DocumentWithMetadata]:
    """
    Create DocumentWithMetadata objects from text chunks.
    
    Args:
        split_documents: List of text chunks
        source_file: Name of the source file
        tags: List of tags/keywords
        file_dir: Directory where the file is located
    
    Returns:
        List of DocumentWithMetadata objects
    """
    documents_with_metadata = []
    
    for i, text in enumerate(split_documents):
        metadata = {
            "chunk_index": i,
            "source_file": source_file,
            "file_path": os.path.join(file_dir, source_file),
            "file_type": "pdf" if source_file.lower().endswith('.pdf') else "txt",
            "file_keywords": tags or ["general", "document"],
            "file_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "file_language": "English",
            "file_encoding": "UTF-8"
        }
        
        doc = DocumentWithMetadata(text=text, metadata=metadata)
        documents_with_metadata.append(doc)
    
    print(f"Created {len(documents_with_metadata)} DocumentWithMetadata objects")
    return documents_with_metadata

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_usage():
    """
    Example usage of all the functions.
    """
    # Example 1: Basic PDF loading with corrections
    pdf_file = "data/example.pdf"
    documents = process_pdf_comprehensive(pdf_file)
    
    if documents:
        print("Sample of corrected text:")
        print(documents[0][:200])
    
    # Example 2: Add metadata to chunks
    if documents:
        split_documents = documents  # Assuming you have a splitter function
        enhanced_chunks = add_metadata_to_existing_chunks(
            split_documents,
            source_file="example.pdf",
            tags=["AI", "Research", "Machine Learning"],
            file_dir="data"
        )
        
        print("Sample enhanced chunk:")
        print(enhanced_chunks[0][:300])
    
    # Example 3: Create DocumentWithMetadata objects
    if documents:
        doc_objects = create_documents_with_metadata(
            split_documents,
            source_file="example.pdf",
            tags=["AI", "Research"],
            file_dir="data"
        )
        
        print("Sample document object:")
        print(doc_objects[0])

if __name__ == "__main__":
    example_usage() 