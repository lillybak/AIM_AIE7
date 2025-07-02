import os
from typing import List
import re

# Add PDF processing imports
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


class TextFileLoader:
    def __init__(self, path: str, encoding: str = "utf-8"):
        self.documents = []
        self.path = path
        self.encoding = encoding

    def load(self):
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".txt"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .txt file."
            )

    def load_file(self):
        with open(self.path, "r", encoding=self.encoding) as f:
            self.documents.append(f.read())

    def load_directory(self):
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".txt"):
                    with open(
                        os.path.join(root, file), "r", encoding=self.encoding
                    ) as f:
                        self.documents.append(f.read())

    def load_documents(self):
        self.load()
        return self.documents


class PDFFileLoader:
    def __init__(self, path: str, use_pdfplumber: bool = True):
        """
        Initialize PDFFileLoader to extract text from PDF files.
        
        Args:
            path: Path to PDF file or directory containing PDF files
            use_pdfplumber: If True, use pdfplumber (better text extraction), 
                          otherwise use PyPDF2 (faster but less accurate)
        """
        self.documents = []
        self.path = path
        self.use_pdfplumber = use_pdfplumber
        
        # Check if required libraries are available
        if use_pdfplumber and not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber is required for PDF processing. Install with: pip install pdfplumber")
        elif not use_pdfplumber and not PDF_AVAILABLE:
            raise ImportError("PyPDF2 is required for PDF processing. Install with: pip install PyPDF2")

    def load(self):
        """Load PDF file(s) and extract text content."""
        if os.path.isdir(self.path):
            self.load_directory()
        elif os.path.isfile(self.path) and self.path.endswith(".pdf"):
            self.load_file()
        else:
            raise ValueError(
                "Provided path is neither a valid directory nor a .pdf file."
            )

    def load_file(self):
        """Extract text from a single PDF file."""
        if self.use_pdfplumber:
            self._load_file_with_pdfplumber()
        else:
            self._load_file_with_pypdf2()

    def _load_file_with_pdfplumber(self):
        """Extract text using pdfplumber (better text extraction)."""
        with pdfplumber.open(self.path) as pdf:
            text_content = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
            
            # Join all pages with double newlines for better separation
            full_text = "\n\n".join(text_content)
            self.documents.append(full_text)

    def _load_file_with_pypdf2(self):
        """Extract text using PyPDF2 (faster but less accurate)."""
        with open(self.path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = []
            
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
            
            # Join all pages with double newlines for better separation
            full_text = "\n\n".join(text_content)
            self.documents.append(full_text)

    def load_directory(self):
        """Load all PDF files from a directory."""
        for root, _, files in os.walk(self.path):
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    try:
                        # Temporarily set path to current file
                        original_path = self.path
                        self.path = file_path
                        self.load_file()
                        self.path = original_path
                    except Exception as e:
                        print(f"Warning: Could not load PDF file {file_path}: {e}")

    def load_documents(self):
        """Load PDF documents and return extracted text."""
        self.load()
        return self.documents


class CharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        assert (
            chunk_size > chunk_overlap
        ), "Chunk size must be greater than chunk overlap"

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        chunks = []
        for text in texts:
            chunks.extend(self.split(text))
        return chunks


if __name__ == "__main__":
    loader = TextFileLoader("data/KingLear.txt")
    loader.load()
    splitter = CharacterTextSplitter()
    chunks = splitter.split_texts(loader.documents)
    print(len(chunks))
    print(chunks[0])
    print("--------")
    print(chunks[1])
    print("--------")
    print(chunks[-2])
    print("--------")
    print(chunks[-1])
