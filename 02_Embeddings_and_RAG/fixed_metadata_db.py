# SIMPLE METADATA SUPPORT - GENERIC VERSION
import datetime
import os

class SimpleMetadataDB:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.metadata = {}
    
    def add_basic_metadata(self, text, source_file=None, tags=None, chunk_index=None):
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
                file_path = f"data/{source_file}"
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            except:
                file_path = f"data/{source_file}"
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
        """Search and filter by tags."""
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