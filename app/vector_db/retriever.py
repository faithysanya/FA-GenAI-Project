"""Document retrieval engine with re-ranking and confidence scoring."""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from app.vector_db.client import ChromaVectorStore
from app.vector_db.embedding import EmbeddingProvider
from app.models import RetrievalResult

logger = logging.getLogger(__name__)


class RankingStrategy(str, Enum):
    """Re-ranking strategies for search results."""
    NO_RERANK = "no_rerank"
    SEMANTIC_DIVERSITY = "semantic_diversity"
    MMR = "mmr"  # Maximal Marginal Relevance


@dataclass
class RetrieverConfig:
    """Configuration for retriever."""
    collection_name: str = "documents"
    top_k: int = 5
    ranking_strategy: RankingStrategy = RankingStrategy.MMR
    diversity_penalty: float = 0.3
    confidence_threshold: float = 0.0


class Retriever:
    """Retrieve relevant documents from vector store with re-ranking and scoring."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_provider: EmbeddingProvider,
        config: Optional[RetrieverConfig] = None,
    ):
        """
        Initialize retriever.

        Args:
            vector_store: ChromaVectorStore instance for document search
            embedding_provider: EmbeddingProvider for query embedding generation
            config: Optional configuration for retriever behavior
        """
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.config = config or RetrieverConfig()
        logger.info(
            f"Retriever initialized with strategy: {self.config.ranking_strategy}"
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Query text
            top_k: Number of results to return (uses config default if None)
            filters: Optional filters like document_ids constraint
            collection_name: Name of collection to search (uses config default if None)

        Returns:
            List of RetrievalResult objects sorted by relevance score

        Raises:
            ValueError: If query is empty or invalid
            Exception: If retrieval fails
        """
        try:
            if not query or not isinstance(query, str):
                raise ValueError("query must be a non-empty string")

            top_k = top_k or self.config.top_k
            collection_name = collection_name or self.config.collection_name

            # Generate query embedding
            logger.debug(f"Generating embedding for query: {query[:100]}...")
            query_embedding = self.embedding_provider.embed_text(query)

            # Search vector store (retrieve more than top_k for re-ranking)
            search_top_k = int(top_k * 1.5) + 5  # Retrieve extra for better re-ranking
            logger.debug(f"Searching vector store with search_top_k={search_top_k}")
            raw_results = self.vector_store.search(
                collection_name=collection_name,
                query_embedding=query_embedding,
                top_k=search_top_k,
            )

            # Apply filters if provided
            if filters and "document_ids" in filters:
                allowed_doc_ids = set(filters["document_ids"])
                raw_results = [
                    r
                    for r in raw_results
                    if r["metadata"].get("document_id") in allowed_doc_ids
                ]
                logger.debug(
                    f"Filtered results to {len(raw_results)} from "
                    f"{len(filters['document_ids'])} allowed documents"
                )

            if not raw_results:
                logger.warning(f"No results found for query: {query[:100]}")
                return []

            # Convert distance to similarity score (cosine distance -> cosine similarity)
            results_with_scores = self._convert_distances_to_scores(raw_results)

            # Apply re-ranking if configured
            if self.config.ranking_strategy != RankingStrategy.NO_RERANK:
                logger.debug(
                    f"Applying {self.config.ranking_strategy} re-ranking strategy"
                )
                results_with_scores = self._rerank_results(
                    results_with_scores, query_embedding, top_k
                )

            # Keep only top_k results
            results_with_scores = results_with_scores[:top_k]

            # Apply confidence threshold
            results_with_scores = [
                r
                for r in results_with_scores
                if r["confidence_score"] >= self.config.confidence_threshold
            ]

            logger.info(
                f"Retrieved {len(results_with_scores)} results for query "
                f"'{query[:50]}...' (top_k={top_k})"
            )

            # Convert to RetrievalResult objects
            retrieval_results = [
                RetrievalResult(
                    chunk_id=result["id"],
                    document_id=result["metadata"].get("document_id", "unknown"),
                    content=result["document"],
                    relevance_score=result["confidence_score"],
                    metadata=result["metadata"],
                )
                for result in results_with_scores
            ]

            return retrieval_results

        except ValueError as e:
            logger.error(f"Validation error in retrieval: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise

    def _convert_distances_to_scores(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Convert Chroma distances to similarity scores.

        Chroma uses cosine distance (0=identical, 2=opposite for normalized vectors).
        We convert to similarity score (0-1): similarity = 1 - (distance / 2)

        Args:
            results: Results from Chroma with 'distance' field

        Returns:
            Results with added 'confidence_score' field
        """
        for result in results:
            distance = result.get("distance", 0.0)
            # Normalize distance to 0-1 range and convert to similarity
            similarity = max(0.0, 1.0 - (distance / 2.0))
            result["confidence_score"] = similarity
        return results

    def _rerank_results(
        self,
        results: List[Dict[str, Any]],
        query_embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank results using specified strategy.

        Args:
            results: Initial ranked results with confidence scores
            query_embedding: Query embedding vector
            top_k: Number of results to keep after re-ranking

        Returns:
            Re-ranked results
        """
        if self.config.ranking_strategy == RankingStrategy.MMR:
            return self._mmr_rerank(results, query_embedding, top_k)
        elif self.config.ranking_strategy == RankingStrategy.SEMANTIC_DIVERSITY:
            return self._diversity_rerank(results, top_k)
        else:
            return results

    def _mmr_rerank(
        self,
        results: List[Dict[str, Any]],
        query_embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Maximal Marginal Relevance re-ranking.

        Balances relevance to query with diversity from already selected documents.

        Args:
            results: Initial results with confidence scores
            query_embedding: Query embedding vector
            top_k: Target number of results

        Returns:
            MMR re-ranked results
        """
        if len(results) <= top_k:
            return results

        query_embedding = np.array(query_embedding)
        selected = []
        remaining = list(results)

        # Select first result (highest relevance)
        selected.append(remaining.pop(0))

        while len(selected) < top_k and remaining:
            # Calculate diversity penalty for each remaining result
            best_idx = 0
            best_mmr_score = -float("inf")

            for i, result in enumerate(remaining):
                # Get result embedding if available, otherwise use confidence score
                result_embedding = np.array(
                    result.get("embedding_vector", [])
                )

                relevance = result["confidence_score"]

                # Calculate diversity (minimal overlap with selected)
                if len(result_embedding) > 0 and len(selected) > 0:
                    diversity = 0.0
                    for selected_result in selected:
                        selected_embedding = np.array(
                            selected_result.get("embedding_vector", [])
                        )
                        if len(selected_embedding) > 0:
                            # Cosine similarity
                            similarity = np.dot(
                                result_embedding, selected_embedding
                            ) / (
                                np.linalg.norm(result_embedding)
                                * np.linalg.norm(selected_embedding)
                                + 1e-10
                            )
                            diversity = max(diversity, similarity)
                    diversity = 1.0 - diversity
                else:
                    diversity = 1.0

                mmr_score = relevance - (
                    self.config.diversity_penalty * diversity
                )

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = i

            selected.append(remaining.pop(best_idx))

        return selected

    def _diversity_rerank(
        self, results: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Simple diversity-based re-ranking by adjusting scores.

        Args:
            results: Initial results with confidence scores
            top_k: Number of results to keep

        Returns:
            Diversity-adjusted results (top_k)
        """
        if len(results) <= top_k:
            return results

        # Apply diminishing returns to very similar results
        reranked = []
        for i, result in enumerate(results):
            adjusted_score = result["confidence_score"] * (
                1.0 - (i / len(results)) * self.config.diversity_penalty
            )
            result["confidence_score"] = adjusted_score
            reranked.append(result)

        # Sort by adjusted score and keep top_k
        reranked.sort(
            key=lambda x: x["confidence_score"], reverse=True
        )
        return reranked[:top_k]

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None,
        collection_name: Optional[str] = None,
    ) -> List[List[RetrievalResult]]:
        """
        Retrieve results for multiple queries.

        Args:
            queries: List of query texts
            top_k: Number of results per query
            collection_name: Name of collection to search

        Returns:
            List of result lists, one per query

        Raises:
            ValueError: If queries list is empty
        """
        if not queries:
            raise ValueError("queries list cannot be empty")

        logger.info(f"Batch retrieving results for {len(queries)} queries")
        results = []
        for query in queries:
            try:
                query_results = self.retrieve(query, top_k, None, collection_name)
                results.append(query_results)
            except Exception as e:
                logger.warning(f"Retrieval failed for query '{query[:50]}...': {e}")
                results.append([])

        return results


# Module-level convenience function using default instances
def retrieve(query: str, top_k: int = 5, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """Module-level retrieve wrapper using default vector store and embedding provider."""
    try:
        store = ChromaVectorStore()
        embedder = EmbeddingProvider()
        retriever = Retriever(store, embedder)
        results = retriever.retrieve(query, top_k=top_k)
        # Convert RetrievalResult objects to dicts for agent compatibility
        return [r.model_dump() if hasattr(r, 'model_dump') else vars(r) for r in results]
    except Exception as e:
        logger.warning(f"Module-level retrieve failed: {e}")
        return []
