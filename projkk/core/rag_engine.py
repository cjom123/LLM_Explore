"""
RAG (Retrieval-Augmented Generation) engine for Excel analysis.
Implements vector search and context retrieval for improved AI responses.
"""

import faiss
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import pickle
import json

from core.model_manager import ModelManager
from core.excel_processor import ExcelProcessor
from config.settings import (
    CHUNK_SIZE, 
    CHUNK_OVERLAP, 
    TOP_K, 
    VECTOR_DIMENSION,
    SIMILARITY_METRIC,
    INDEX_TYPE
)

logger = logging.getLogger(__name__)

class RAGEngine:
    """Implements RAG functionality for Excel data analysis."""
    
    def __init__(self, model_manager: ModelManager = None):
        self.model_manager = model_manager or ModelManager()
        self.excel_processor = ExcelProcessor()
        self.vector_index = None
        self.text_chunks = []
        self.chunk_metadata = []
        self.is_indexed = False
        
    def create_text_chunks(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Split text into overlapping chunks for better context retrieval."""
        if chunk_size is None:
            chunk_size = CHUNK_SIZE
        if overlap is None:
            overlap = CHUNK_OVERLAP
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk.strip())
            
            start = end - overlap
            if start >= len(text):
                break
        
        logger.info(f"Created {len(chunks)} text chunks from text of length {len(text)}")
        return chunks
    
    def process_excel_for_rag(self, file_path: str) -> Dict[str, Any]:
        """Process Excel file and prepare it for RAG indexing."""
        try:
            logger.info(f"Processing Excel file for RAG: {file_path}")
            
            # Process Excel file
            processed_data = self.excel_processor.process_excel_file(file_path)
            
            if "error" in processed_data:
                return processed_data
            
            # Extract text chunks from all sheets
            all_chunks = []
            chunk_metadata = []
            
            for sheet_name, sheet_data in processed_data["sheets"].items():
                sheet_text = sheet_data["text"]
                sheet_chunks = self.create_text_chunks(sheet_text)
                
                for i, chunk in enumerate(sheet_chunks):
                    all_chunks.append(chunk)
                    chunk_metadata.append({
                        "sheet_name": sheet_name,
                        "chunk_index": i,
                        "total_chunks": len(sheet_chunks),
                        "source": "excel_sheet"
                    })
            
            # Add combined text chunks
            combined_text = processed_data.get("combined_text", "")
            if combined_text:
                combined_chunks = self.create_text_chunks(combined_text)
                for i, chunk in enumerate(combined_chunks):
                    all_chunks.append(chunk)
                    chunk_metadata.append({
                        "sheet_name": "combined",
                        "chunk_index": i,
                        "total_chunks": len(combined_chunks),
                        "source": "combined_analysis"
                    })
            
            self.text_chunks = all_chunks
            self.chunk_metadata = chunk_metadata
            
            logger.info(f"Prepared {len(all_chunks)} text chunks for RAG indexing")
            
            return {
                "success": True,
                "total_chunks": len(all_chunks),
                "sheets_processed": len(processed_data["sheets"]),
                "chunks_per_sheet": {name: len([c for c in chunk_metadata if c["sheet_name"] == name]) 
                                   for name in processed_data["sheets"].keys()}
            }
            
        except Exception as e:
            logger.error(f"Error processing Excel for RAG: {str(e)}")
            return {"error": f"Failed to process Excel for RAG: {str(e)}"}
    
    def build_vector_index(self, chunks: List[str] = None) -> bool:
        """Build FAISS vector index from text chunks."""
        try:
            if chunks is None:
                chunks = self.text_chunks
            
            if not chunks:
                logger.error("No text chunks available for indexing")
                return False
            
            logger.info(f"Building vector index for {len(chunks)} chunks")
            
            # Generate embeddings
            embeddings = self.model_manager.generate_embeddings(chunks)
            embeddings_np = embeddings.cpu().numpy().astype('float32')
            
            # Create FAISS index
            dimension = embeddings_np.shape[1]
            
            if INDEX_TYPE == "Flat":
                self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            elif INDEX_TYPE == "IVFFlat":
                nlist = min(100, len(chunks) // 10)  # Number of clusters
                quantizer = faiss.IndexFlatIP(dimension)
                self.vector_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
                # Train the index
                self.vector_index.train(embeddings_np)
            else:
                self.vector_index = faiss.IndexFlatIP(dimension)
            
            # Add vectors to index
            self.vector_index.add(embeddings_np)
            
            self.is_indexed = True
            logger.info(f"Successfully built vector index with {self.vector_index.ntotal} vectors")
            return True
            
        except Exception as e:
            logger.error(f"Error building vector index: {str(e)}")
            return False
    
    def search_similar_chunks(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """Search for most similar text chunks to the query."""
        try:
            if not self.is_indexed or self.vector_index is None:
                logger.error("Vector index not built. Call build_vector_index() first.")
                return []
            
            if top_k is None:
                top_k = TOP_K
            
            # Generate query embedding
            query_embedding = self.model_manager.generate_embeddings([query])
            query_vector = query_embedding.cpu().numpy().astype('float32')
            
            # Search index
            scores, indices = self.vector_index.search(query_vector, min(top_k, len(self.text_chunks)))
            
            # Prepare results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.text_chunks):
                    result = {
                        "chunk": self.text_chunks[idx],
                        "score": float(score),
                        "metadata": self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {},
                        "chunk_index": int(idx)
                    }
                    results.append(result)
            
            logger.info(f"Retrieved {len(results)} similar chunks for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Error searching similar chunks: {str(e)}")
            return []
    
    def generate_rag_response(self, query: str, top_k: int = None, max_length: int = 512) -> Dict[str, Any]:
        """Generate a response using RAG approach."""
        try:
            logger.info(f"Generating RAG response for query: {query}")
            
            # Retrieve relevant context
            relevant_chunks = self.search_similar_chunks(query, top_k)
            
            if not relevant_chunks:
                logger.warning("No relevant chunks found for query")
                return {
                    "response": "I couldn't find relevant information in the Excel data to answer your question.",
                    "context_used": [],
                    "query": query
                }
            
            # Prepare context for generation
            context_text = "\n\n".join([chunk["chunk"] for chunk in relevant_chunks])
            
            # Create enhanced prompt
            enhanced_prompt = self._create_enhanced_prompt(query, context_text)
            
            # Generate response
            response = self.model_manager.generate_text(
                enhanced_prompt, 
                max_length=max_length,
                temperature=0.7,
                top_p=0.9
            )
            
            # Prepare result
            result = {
                "response": response,
                "context_used": relevant_chunks,
                "query": query,
                "prompt_length": len(enhanced_prompt),
                "response_length": len(response)
            }
            
            logger.info(f"Generated RAG response with length {len(response)}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {str(e)}")
            return {
                "error": f"Failed to generate response: {str(e)}",
                "query": query
            }
    
    def _create_enhanced_prompt(self, query: str, context: str) -> str:
        """Create an enhanced prompt combining query and retrieved context."""
        system_prompt = """You are an expert data analyst. Analyze the Excel data provided in the context and answer the user's question. 
        Always provide specific insights based on the data, use numbers and facts when available, and suggest relevant visualizations if appropriate.
        If the context doesn't contain enough information to answer the question, say so clearly."""
        
        enhanced_prompt = f"""{system_prompt}

Context from Excel data:
{context}

User Question: {query}

Analysis:"""
        
        return enhanced_prompt
    
    def save_index(self, file_path: str) -> bool:
        """Save the vector index and metadata to disk."""
        try:
            if not self.is_indexed or self.vector_index is None:
                logger.error("No index to save")
                return False
            
            file_path = Path(file_path)
            
            # Save FAISS index
            faiss.write_index(self.vector_index, str(file_path.with_suffix('.faiss')))
            
            # Save metadata
            metadata = {
                "text_chunks": self.text_chunks,
                "chunk_metadata": self.chunk_metadata,
                "index_type": INDEX_TYPE,
                "total_chunks": len(self.text_chunks)
            }
            
            with open(file_path.with_suffix('.json'), 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Saved index and metadata to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            return False
    
    def load_index(self, file_path: str) -> bool:
        """Load a previously saved vector index and metadata."""
        try:
            file_path = Path(file_path)
            
            # Load FAISS index
            self.vector_index = faiss.read_index(str(file_path.with_suffix('.faiss')))
            
            # Load metadata
            with open(file_path.with_suffix('.json'), 'r') as f:
                metadata = json.load(f)
            
            self.text_chunks = metadata["text_chunks"]
            self.chunk_metadata = metadata["chunk_metadata"]
            self.is_indexed = True
            
            logger.info(f"Loaded index with {len(self.text_chunks)} chunks from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading index: {str(e)}")
            return False
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get information about the current index."""
        if not self.is_indexed or self.vector_index is None:
            return {"status": "not_indexed"}
        
        return {
            "status": "indexed",
            "total_vectors": self.vector_index.ntotal,
            "total_chunks": len(self.text_chunks),
            "index_type": INDEX_TYPE,
            "chunks_per_sheet": self._get_chunks_per_sheet()
        }
    
    def _get_chunks_per_sheet(self) -> Dict[str, int]:
        """Get the number of chunks per sheet."""
        chunks_per_sheet = {}
        for metadata in self.chunk_metadata:
            sheet_name = metadata["sheet_name"]
            chunks_per_sheet[sheet_name] = chunks_per_sheet.get(sheet_name, 0) + 1
        return chunks_per_sheet
    
    def clear_index(self):
        """Clear the current index and free memory."""
        try:
            if self.vector_index is not None:
                del self.vector_index
                self.vector_index = None
            
            self.text_chunks = []
            self.chunk_metadata = []
            self.is_indexed = False
            
            logger.info("Index cleared and memory freed")
            
        except Exception as e:
            logger.error(f"Error clearing index: {str(e)}") 