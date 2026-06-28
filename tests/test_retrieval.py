"""Tests for document retrieval functionality."""

import pytest
import logging
import numpy as np
from unittest.mock import Mock

from app.vector_db.retriever import Retriever, RetrieverConfig, RankingStrategy
from app.vector_db.client import ChromaVectorStore
from app.vector_db.embedding import EmbeddingProvider
from app.models import RetrievalResult

logger = logging.getLogger(__name__)


class TestRetriever:
    """Test suite for Retriever class."""

    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = Mock(spec=ChromaVectorStore)
        return store

    @pytest.fixture
    def mock_embedding_provider(self):
        """Create a mock embedding provider."""
        provider = Mock(spec=EmbeddingProvider)
        # Return a 384-dimensional embedding (standard for all-MiniLM-L6-v2)
        provider.embed_text.return_value = np.random.randn(384).tolist()
        return provider

    @pytest.fixture
    def retriever(self, mock_vector_store, mock_embedding_provider):
        """Create a retriever instance with mocks."""
        config = RetrieverConfig(
            collection_name="test_collection",
            top_k=5,
            ranking_strategy=RankingStrategy.MMR,
        )
        return Retriever(mock_vector_store, mock_embedding_provider, config)

    def test_retriever_initialization(self, mock_vector_store, mock_embedding_provider):
        """Test retriever initialization."""
        config = RetrieverConfig()
        retriever = Retriever(mock_vector_store, mock_embedding_provider, config)

        assert retriever.vector_store == mock_vector_store
        assert retriever.embedding_provider == mock_embedding_provider
        assert retriever.config == config

    def test_retrieve_with_valid_query(self, retriever, mock_vector_store, mock_embedding_provider):
        """Test retrieve function with a valid query."""
        # Mock search results
        mock_results = [
            {
                "id": "chunk_001",
                "document": "This is the first relevant document.",
                "metadata": {"document_id": "doc_001", "page": 1},
                "distance": 0.1,
            },
            {
                "id": "chunk_002",
                "document": "This is the second relevant document.",
                "metadata": {"document_id": "doc_002", "page": 2},
                "distance": 0.3,
            },
        ]
        mock_vector_store.search.return_value = mock_results

        # Call retrieve
        results = retriever.retrieve("test query", top_k=2)

        # Assertions
        assert len(results) == 2
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert results[0].chunk_id == "chunk_001"
        assert results[0].content == "This is the first relevant document."
        assert 0.0 <= results[0].relevance_score <= 1.0
        assert results[0].document_id == "doc_001"

    def test_retrieve_empty_query_raises_error(self, retriever):
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            retriever.retrieve("")

    def test_retrieve_with_filters(self, retriever, mock_vector_store):
        """Test retrieve with document_ids filter."""
        mock_results = [
            {
                "id": "chunk_001",
                "document": "Document 1 content.",
                "metadata": {"document_id": "doc_001"},
                "distance": 0.1,
            },
            {
                "id": "chunk_002",
                "document": "Document 2 content.",
                "metadata": {"document_id": "doc_002"},
                "distance": 0.2,
            },
            {
                "id": "chunk_003",
                "document": "Document 3 content.",
                "metadata": {"document_id": "doc_003"},
                "distance": 0.15,
            },
        ]
        mock_vector_store.search.return_value = mock_results

        filters = {"document_ids": ["doc_001", "doc_003"]}
        results = retriever.retrieve("test query", filters=filters)

        # Should only return doc_001 and doc_003
        assert len(results) == 2
        assert results[0].document_id in ["doc_001", "doc_003"]
        assert results[1].document_id in ["doc_001", "doc_003"]

    def test_retrieve_no_results(self, retriever, mock_vector_store):
        """Test retrieve when no results found."""
        mock_vector_store.search.return_value = []

        results = retriever.retrieve("query with no results")

        assert results == []

    def test_convert_distances_to_scores(self, retriever):
        """Test conversion of Chroma distances to similarity scores."""
        results = [
            {"id": "1", "distance": 0.0, "document": "test"},
            {"id": "2", "distance": 0.5, "document": "test"},
            {"id": "3", "distance": 1.0, "document": "test"},
            {"id": "4", "distance": 2.0, "document": "test"},
        ]

        converted = retriever._convert_distances_to_scores(results)

        # distance 0.0 -> similarity 1.0 (identical)
        assert converted[0]["confidence_score"] == 1.0
        # distance 0.5 -> similarity 0.75
        assert abs(converted[1]["confidence_score"] - 0.75) < 0.01
        # distance 1.0 -> similarity 0.5
        assert abs(converted[2]["confidence_score"] - 0.5) < 0.01
        # distance 2.0 -> similarity 0.0 (opposite)
        assert converted[3]["confidence_score"] == 0.0

    def test_mmr_reranking(self, retriever):
        """Test MMR (Maximal Marginal Relevance) re-ranking."""
        # Create results with varying relevance
        results = [
            {
                "id": "chunk_1",
                "document": "First result",
                "metadata": {"document_id": "doc_1"},
                "confidence_score": 0.95,
                "embedding_vector": [0.9, 0.1, 0.0, 0.0],
            },
            {
                "id": "chunk_2",
                "document": "Second result similar to first",
                "metadata": {"document_id": "doc_1"},
                "confidence_score": 0.90,
                "embedding_vector": [0.85, 0.15, 0.0, 0.0],
            },
            {
                "id": "chunk_3",
                "document": "Third result different",
                "metadata": {"document_id": "doc_2"},
                "confidence_score": 0.80,
                "embedding_vector": [0.0, 0.0, 0.95, 0.05],
            },
        ]

        query_embedding = [0.9, 0.1, 0.0, 0.0]
        reranked = retriever._mmr_rerank(results, query_embedding, top_k=2)

        assert len(reranked) <= 2
        # First result should still be in top results (highest relevance)
        assert reranked[0]["id"] == "chunk_1"

    def test_diversity_reranking(self, retriever):
        """Test diversity-based re-ranking."""
        results = [
            {"id": "1", "confidence_score": 0.95, "document": "result 1"},
            {"id": "2", "confidence_score": 0.90, "document": "result 2"},
            {"id": "3", "confidence_score": 0.85, "document": "result 3"},
            {"id": "4", "confidence_score": 0.80, "document": "result 4"},
            {"id": "5", "confidence_score": 0.75, "document": "result 5"},
        ]

        reranked = retriever._diversity_rerank(results, top_k=3)

        assert len(reranked) == 3
        # Top result should remain top
        assert reranked[0]["id"] == "1"

    def test_batch_retrieve(self, retriever, mock_vector_store):
        """Test batch retrieval for multiple queries."""
        mock_results_1 = [
            {
                "id": "chunk_001",
                "document": "Result for query 1",
                "metadata": {"document_id": "doc_001"},
                "distance": 0.1,
            }
        ]
        mock_results_2 = [
            {
                "id": "chunk_002",
                "document": "Result for query 2",
                "metadata": {"document_id": "doc_002"},
                "distance": 0.2,
            }
        ]

        # Mock search to return different results for each call
        mock_vector_store.search.side_effect = [mock_results_1, mock_results_2]

        queries = ["query 1", "query 2"]
        results = retriever.batch_retrieve(queries)

        assert len(results) == 2
        assert len(results[0]) == 1
        assert len(results[1]) == 1
        assert results[0][0].chunk_id == "chunk_001"
        assert results[1][0].chunk_id == "chunk_002"

    def test_batch_retrieve_with_empty_list_raises_error(self, retriever):
        """Test that batch_retrieve with empty queries raises error."""
        with pytest.raises(ValueError, match="queries list cannot be empty"):
            retriever.batch_retrieve([])

    def test_confidence_scoring_applied(self, retriever, mock_vector_store):
        """Test that confidence scores are properly applied."""
        mock_results = [
            {
                "id": "chunk_001",
                "document": "High confidence result.",
                "metadata": {"document_id": "doc_001"},
                "distance": 0.05,
            },
            {
                "id": "chunk_002",
                "document": "Low confidence result.",
                "metadata": {"document_id": "doc_002"},
                "distance": 0.5,
            },
        ]
        mock_vector_store.search.return_value = mock_results

        results = retriever.retrieve("test query")

        # First result should have higher score than second
        assert results[0].relevance_score > results[1].relevance_score

    def test_confidence_threshold_filtering(self, retriever, mock_vector_store):
        """Test that low confidence results are filtered out."""
        mock_results = [
            {
                "id": "chunk_001",
                "document": "Good result.",
                "metadata": {"document_id": "doc_001"},
                "distance": 0.1,
            },
            {
                "id": "chunk_002",
                "document": "Poor result.",
                "metadata": {"document_id": "doc_002"},
                "distance": 1.5,  # Very low similarity
            },
        ]
        mock_vector_store.search.return_value = mock_results

        config = RetrieverConfig(confidence_threshold=0.7)
        retriever_with_threshold = Retriever(
            mock_vector_store, retriever.embedding_provider, config
        )

        results = retriever_with_threshold.retrieve("test query")

        # Only good result should pass threshold
        assert len(results) <= 2
        # All returned results should be above threshold
        assert all(r.relevance_score >= 0.7 for r in results)

    def test_retrieve_respects_top_k_parameter(self, retriever, mock_vector_store):
        """Test that retrieve respects the top_k parameter."""
        mock_results = [
            {
                "id": f"chunk_{i:03d}",
                "document": f"Document {i}",
                "metadata": {"document_id": f"doc_{i:03d}"},
                "distance": 0.1 * i,
            }
            for i in range(10)
        ]
        mock_vector_store.search.return_value = mock_results

        # Request only top 3
        results = retriever.retrieve("test query", top_k=3)

        assert len(results) <= 3


class TestRetrievalIntegration:
    """Integration tests for retrieval system."""

    @pytest.fixture
    def test_config(self):
        """Get test configuration."""
        return {
            "persistence_dir": "./test_vector_db",
            "collection_name": "test_docs",
        }

    def test_retrieve_real_embeddings_mock_store(self, test_config):
        """Test retrieval with real embeddings but mocked vector store."""
        # Use real embedding provider
        embedding_provider = Mock(spec=EmbeddingProvider)
        embedding_provider.embed_text.return_value = [0.1] * 384

        # Mock vector store
        vector_store = Mock(spec=ChromaVectorStore)
        mock_results = [
            {
                "id": "chunk_test",
                "document": "Test document content for retrieval.",
                "metadata": {
                    "document_id": "doc_test",
                    "source": "test.txt",
                },
                "distance": 0.15,
            }
        ]
        vector_store.search.return_value = mock_results

        config = RetrieverConfig(collection_name="test_docs")
        retriever = Retriever(vector_store, embedding_provider, config)

        results = retriever.retrieve("test query")

        assert len(results) > 0
        assert results[0].document_id == "doc_test"
        assert results[0].content == "Test document content for retrieval."
        assert results[0].relevance_score > 0

    def test_retriever_config_defaults(self):
        """Test RetrieverConfig default values."""
        config = RetrieverConfig()

        assert config.collection_name == "documents"
        assert config.top_k == 5
        assert config.ranking_strategy == RankingStrategy.MMR
        assert config.diversity_penalty == 0.3
        assert config.confidence_threshold == 0.0


class TestRetrievalScoring:
    """Tests for scoring and ranking logic."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components."""
        vector_store = Mock(spec=ChromaVectorStore)
        embedding_provider = Mock(spec=EmbeddingProvider)
        embedding_provider.embed_text.return_value = np.random.randn(384).tolist()
        return vector_store, embedding_provider

    def test_score_normalization(self, mock_components):
        """Test that scores are normalized to 0-1 range."""
        vector_store, embedding_provider = mock_components
        retriever = Retriever(vector_store, embedding_provider)

        results = [
            {
                "id": "1",
                "document": "test",
                "metadata": {},
                "distance": d,
            }
            for d in [0.0, 0.5, 1.0, 1.5, 2.0]
        ]

        normalized = retriever._convert_distances_to_scores(results)

        # All scores should be in 0-1 range
        for result in normalized:
            assert 0.0 <= result["confidence_score"] <= 1.0

    def test_score_ordering_preserved(self, mock_components):
        """Test that lower distances result in higher scores."""
        vector_store, embedding_provider = mock_components
        retriever = Retriever(vector_store, embedding_provider)

        results = [
            {
                "id": "1",
                "document": "test",
                "metadata": {},
                "distance": 0.1,
            },
            {
                "id": "2",
                "document": "test",
                "metadata": {},
                "distance": 0.5,
            },
            {
                "id": "3",
                "document": "test",
                "metadata": {},
                "distance": 1.0,
            },
        ]

        normalized = retriever._convert_distances_to_scores(results)

        # Scores should be in descending order
        scores = [r["confidence_score"] for r in normalized]
        assert scores[0] > scores[1] > scores[2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
