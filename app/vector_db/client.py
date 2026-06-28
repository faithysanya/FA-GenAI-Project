"""ChromaVectorStore client for managing vector database operations."""

import logging
from typing import List, Dict, Any, Optional
import chromadb

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Vector store client using Chroma for persistence and retrieval."""
    
    def __init__(self, persistence_directory: str):
        """
        Initialize Chroma vector store with persistence.
        
        Args:
            persistence_directory: Path to store vector database files
            
        Raises:
            ValueError: If persistence directory is invalid
        """
        try:
            if not persistence_directory:
                raise ValueError("persistence_directory cannot be empty")
            
            self.persistence_directory = persistence_directory
            
            # Use the new Chroma API with persistent client
            self.client = chromadb.PersistentClient(path=persistence_directory)
            logger.info(f"ChromaVectorStore initialized with directory: {persistence_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaVectorStore: {str(e)}")
            raise
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[str],
        embeddings: List[List[float]],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Add documents with embeddings to a collection.
        
        Args:
            collection_name: Name of the collection
            documents: List of document texts
            embeddings: List of embedding vectors
            metadata: Optional list of metadata dicts for each document
            ids: Optional list of document IDs
            
        Returns:
            Dict with operation results
            
        Raises:
            ValueError: If inputs are invalid
            Exception: If Chroma operation fails
        """
        try:
            if not documents or not embeddings:
                raise ValueError("documents and embeddings cannot be empty")
            
            if len(documents) != len(embeddings):
                raise ValueError("documents and embeddings must have same length")
            
            # Default metadata and IDs
            if metadata is None:
                metadata = [{} for _ in documents]
            if ids is None:
                ids = [f"doc_{i}" for i in range(len(documents))]
            
            # Get or create collection
            collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Add documents with embeddings
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadata,
            )
            
            logger.info(f"Added {len(documents)} documents to collection '{collection_name}'")
            return {
                "success": True,
                "count": len(documents),
                "collection": collection_name,
            }
        except Exception as e:
            logger.error(f"Failed to add documents to collection '{collection_name}': {str(e)}")
            raise
    
    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to query embedding.
        
        Args:
            collection_name: Name of the collection
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            
        Returns:
            List of result dicts with keys: id, document, metadata, distance
            
        Raises:
            ValueError: If inputs are invalid
            Exception: If Chroma operation fails
        """
        try:
            if not query_embedding:
                raise ValueError("query_embedding cannot be empty")
            
            if top_k < 1:
                raise ValueError("top_k must be >= 1")
            
            collection = self.client.get_collection(name=collection_name)
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )
            
            # Format results
            formatted_results = []
            if results and results['ids'] and len(results['ids']) > 0:
                for i, doc_id in enumerate(results['ids'][0]):
                    formatted_results.append({
                        "id": doc_id,
                        "document": results['documents'][0][i],
                        "metadata": results['metadatas'][0][i],
                        "distance": results['distances'][0][i],
                    })
            
            logger.debug(f"Search in '{collection_name}' returned {len(formatted_results)} results")
            return formatted_results
        except Exception as e:
            logger.error(f"Failed to search collection '{collection_name}': {str(e)}")
            raise
    
    def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """
        Delete a collection from the vector store.
        
        Args:
            collection_name: Name of the collection to delete
            
        Returns:
            Dict with operation results
            
        Raises:
            Exception: If Chroma operation fails
        """
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Deleted collection '{collection_name}'")
            return {
                "success": True,
                "collection": collection_name,
                "action": "deleted",
            }
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {str(e)}")
            raise
    
    def get_collection_count(self, collection_name: str) -> int:
        """
        Get the number of documents in a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Number of documents in the collection
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            count = collection.count()
            logger.debug(f"Collection '{collection_name}' has {count} documents")
            return count
        except Exception as e:
            logger.error(f"Failed to get count for collection '{collection_name}': {str(e)}")
            raise
