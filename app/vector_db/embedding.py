"""Embedding provider for generating text embeddings."""

import logging
import time
from typing import List, Union
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """Generate embeddings for text using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding provider with specified model.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            
        Raises:
            Exception: If model cannot be loaded
        """
        try:
            self.model_name = model_name
            self.model = SentenceTransformer(model_name)
            logger.info(f"EmbeddingProvider initialized with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize EmbeddingProvider with model '{model_name}': {str(e)}")
            raise
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
            
        Raises:
            ValueError: If text is empty
            Exception: If embedding generation fails
        """
        try:
            if not text or not isinstance(text, str):
                raise ValueError("text must be a non-empty string")
            
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        except ValueError as e:
            logger.error(f"Invalid input for embedding: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {str(e)}")
            raise
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Size of batches for processing
            show_progress: Whether to show progress bar
            
        Returns:
            List of embedding vectors
            
        Raises:
            ValueError: If texts is empty or invalid
            Exception: If batch embedding generation fails
        """
        try:
            if not texts:
                raise ValueError("texts list cannot be empty")
            
            if not all(isinstance(t, str) for t in texts):
                raise ValueError("all items in texts must be strings")
            
            if batch_size < 1:
                raise ValueError("batch_size must be >= 1")
            
            logger.info(f"Generating embeddings for {len(texts)} texts (batch_size={batch_size})")
            
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_tensor=False,
            )
            
            # Convert to list of lists
            result = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
            logger.info(f"Successfully generated {len(result)} embeddings")
            return result
        except ValueError as e:
            logger.error(f"Invalid input for batch embedding: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this model.
        
        Returns:
            Dimension of embedding vectors
        """
        return self.model.get_embedding_dimension()
