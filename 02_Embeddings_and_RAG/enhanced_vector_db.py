#!/usr/bin/env python3
"""
Enhanced Vector Database with Metadata Support
This extends the existing VectorDatabase to support metadata for each document.
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
import json
import datetime

class EnhancedVectorDatabase:
    """
    Enhanced vector database with metadata support.
    Extends the existing VectorDatabase functionality.
    """
    
    def __init__(self, embedding_model=None):
        """
        Initialize the enhanced vector database.
        
        Args:
            embedding_model: The embedding model to use (optional)
        """
        # Store vectors and their metadata
        self.vectors = defaultdict(np.array)
        self.metadata = {}  # Store metadata for each document
        self.embedding_model = embedding_model
        
        # Metadata schema for validation
        self.metadata_schema = {
            'source_file': str,
            'page_number': int,
            'chunk_index': int,
            'timestamp': str,
            'document_type': str,
            'tags': list,
            'confidence': float
        }
    
    def insert_with_metadata(self, text: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """
        Insert a document with its vector and metadata.
        
        Args:
            text: The document text
            vector: The document's embedding vector
            metadata: Dictionary containing metadata
        """
        # Validate metadata
        validated_metadata = self._validate_metadata(metadata)
        
        # Store the vector and metadata
        self.vectors[text] = vector
        self.metadata[text] = validated_metadata
    
    def _validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and normalize metadata.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Validated metadata dictionary
        """
        validated = {}
        
        # Add timestamp if not present
        if 'timestamp' not in metadata:
            validated['timestamp'] = datetime.datetime.now().isoformat()
        
        # Validate and add each metadata field
        for key, value in metadata.items():
            if key in self.metadata_schema:
                expected_type = self.metadata_schema[key]
                if isinstance(value, expected_type):
                    validated[key] = value
                else:
                    # Try to convert to expected type
                    try:
                        if expected_type == int:
                            validated[key] = int(value)
                        elif expected_type == float:
                            validated[key] = float(value)
                        elif expected_type == list:
                            validated[key] = list(value) if isinstance(value, (list, tuple)) else [value]
                        else:
                            validated[key] = str(value)
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert {key}={value} to {expected_type}")
                        validated[key] = value
            else:
                # Allow custom metadata fields
                validated[key] = value
        
        return validated
    
    def search_by_text_with_metadata(self, query: str, k: int = 5, 
                                   filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for documents by text and return results with metadata.
        
        Args:
            query: The search query
            k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of tuples: (text, similarity_score, metadata)
        """
        # Use the existing search method
        search_results = self.search_by_text(query, k=k*2)  # Get more results for filtering
        
        # Add metadata to results
        results_with_metadata = []
        for text, score in search_results:
            metadata = self.metadata.get(text, {})
            
            # Apply metadata filters if provided
            if filter_metadata and not self._matches_metadata_filter(metadata, filter_metadata):
                continue
                
            results_with_metadata.append((text, score, metadata))
        
        # Return top k results
        return results_with_metadata[:k]
    
    def _matches_metadata_filter(self, metadata: Dict[str, Any], filter_dict: Dict[str, Any]) -> bool:
        """
        Check if metadata matches the filter criteria.
        
        Args:
            metadata: Document metadata
            filter_dict: Filter criteria
            
        Returns:
            True if metadata matches filter, False otherwise
        """
        for key, filter_value in filter_dict.items():
            if key not in metadata:
                return False
            
            metadata_value = metadata[key]
            
            # Handle different filter types
            if isinstance(filter_value, (list, tuple)):
                # Check if metadata value is in the list
                if metadata_value not in filter_value:
                    return False
            elif isinstance(filter_value, dict):
                # Handle range filters (e.g., {'min': 1, 'max': 10})
                if 'min' in filter_value and metadata_value < filter_value['min']:
                    return False
                if 'max' in filter_value and metadata_value > filter_value['max']:
                    return False
            else:
                # Exact match
                if metadata_value != filter_value:
                    return False
        
        return True
    
    def get_documents_by_metadata(self, filter_dict: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get all documents that match specific metadata criteria.
        
        Args:
            filter_dict: Metadata filter criteria
            
        Returns:
            List of tuples: (text, metadata)
        """
        matching_documents = []
        
        for text, metadata in self.metadata.items():
            if self._matches_metadata_filter(metadata, filter_dict):
                matching_documents.append((text, metadata))
        
        return matching_documents
    
    def get_metadata_summary(self) -> Dict[str, Any]:
        """
        Get a summary of metadata across all documents.
        
        Returns:
            Dictionary with metadata statistics
        """
        if not self.metadata:
            return {"total_documents": 0}
        
        summary = {
            "total_documents": len(self.metadata),
            "metadata_fields": {},
            "source_files": set(),
            "document_types": set(),
            "tags": set()
        }
        
        for metadata in self.metadata.values():
            # Count metadata fields
            for key, value in metadata.items():
                if key not in summary["metadata_fields"]:
                    summary["metadata_fields"][key] = 0
                summary["metadata_fields"][key] += 1
            
            # Collect unique values
            if 'source_file' in metadata:
                summary["source_files"].add(metadata['source_file'])
            if 'document_type' in metadata:
                summary["document_types"].add(metadata['document_type'])
            if 'tags' in metadata:
                summary["tags"].update(metadata['tags'])
        
        # Convert sets to lists for JSON serialization
        summary["source_files"] = list(summary["source_files"])
        summary["document_types"] = list(summary["document_types"])
        summary["tags"] = list(summary["tags"])
        
        return summary
    
    def export_metadata(self, filename: str) -> None:
        """
        Export metadata to a JSON file.
        
        Args:
            filename: Output filename
        """
        export_data = {
            "metadata": self.metadata,
            "summary": self.get_metadata_summary(),
            "export_timestamp": datetime.datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"Metadata exported to {filename}")
    
    def import_metadata(self, filename: str) -> None:
        """
        Import metadata from a JSON file.
        
        Args:
            filename: Input filename
        """
        with open(filename, 'r') as f:
            import_data = json.load(f)
        
        if 'metadata' in import_data:
            self.metadata.update(import_data['metadata'])
            print(f"Imported metadata for {len(import_data['metadata'])} documents")
        else:
            print("No metadata found in import file")

# Helper functions for creating metadata
def create_document_metadata(source_file: str, page_number: int = None, 
                           chunk_index: int = None, document_type: str = "pdf",
                           tags: List[str] = None, confidence: float = 1.0) -> Dict[str, Any]:
    """
    Create standardized metadata for a document.
    
    Args:
        source_file: Name of the source file
        page_number: Page number in the document
        chunk_index: Index of the chunk within the document
        document_type: Type of document (pdf, txt, etc.)
        tags: List of tags for the document
        confidence: Confidence score for the chunk
        
    Returns:
        Metadata dictionary
    """
    metadata = {
        'source_file': source_file,
        'document_type': document_type,
        'timestamp': datetime.datetime.now().isoformat(),
        'confidence': confidence
    }
    
    if page_number is not None:
        metadata['page_number'] = page_number
    
    if chunk_index is not None:
        metadata['chunk_index'] = chunk_index
    
    if tags:
        metadata['tags'] = tags
    
    return metadata

def enhance_existing_vector_db(vector_db, source_file: str = "unknown", 
                             document_type: str = "pdf") -> EnhancedVectorDatabase:
    """
    Convert an existing VectorDatabase to EnhancedVectorDatabase with metadata.
    
    Args:
        vector_db: Existing VectorDatabase instance
        source_file: Source file name for metadata
        document_type: Type of document
        
    Returns:
        EnhancedVectorDatabase with metadata
    """
    enhanced_db = EnhancedVectorDatabase(embedding_model=vector_db.embedding_model)
    
    # Copy vectors and add metadata
    for i, (text, vector) in enumerate(vector_db.vectors.items()):
        metadata = create_document_metadata(
            source_file=source_file,
            chunk_index=i,
            document_type=document_type,
            tags=["auto_generated"]
        )
        enhanced_db.insert_with_metadata(text, vector, metadata)
    
    print(f"Enhanced vector database created with {len(enhanced_db.vectors)} documents")
    return enhanced_db 