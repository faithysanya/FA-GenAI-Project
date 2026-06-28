"""End-to-end tests for the complete retrieval pipeline."""

import pytest
import logging
import tempfile
import shutil
from pathlib import Path

from app.vector_db.client import ChromaVectorStore
from app.vector_db.embedding import EmbeddingProvider
from app.vector_db.retriever import Retriever, RetrieverConfig, RankingStrategy
from app.models import RetrievalResult

logger = logging.getLogger(__name__)


class TestRetrieverEndToEnd:
    """End-to-end tests for retrieval system."""

    @pytest.fixture
    def temp_vector_db(self):
        """Create temporary vector database directory."""
        temp_dir = tempfile.mkdtemp(prefix="test_vector_db_")
        yield temp_dir
        # Cleanup with error handling for Windows file locks
        try:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Could not cleanup temp directory {temp_dir}: {e}")

    @pytest.fixture
    def vector_store(self, temp_vector_db):
        """Create vector store instance."""
        return ChromaVectorStore(persistence_directory=temp_vector_db)

    @pytest.fixture
    def embedding_provider(self):
        """Create embedding provider instance."""
        return EmbeddingProvider(model_name="all-MiniLM-L6-v2")

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return {
            "doc_001": [
                {
                    "chunk_id": "chunk_001",
                    "content": "Machine learning is a branch of artificial intelligence that focuses on building applications that learn from data.",
                    "metadata": {"document_id": "doc_001", "page": 1, "source": "ml_basics.txt"},
                },
                {
                    "chunk_id": "chunk_002",
                    "content": "Deep learning uses neural networks with multiple layers to process complex patterns in data.",
                    "metadata": {"document_id": "doc_001", "page": 2, "source": "ml_basics.txt"},
                },
            ],
            "doc_002": [
                {
                    "chunk_id": "chunk_003",
                    "content": "Natural language processing is used to analyze and understand human language text.",
                    "metadata": {"document_id": "doc_002", "page": 1, "source": "nlp_intro.txt"},
                },
                {
                    "chunk_id": "chunk_004",
                    "content": "Transformer models have revolutionized NLP by enabling better context understanding.",
                    "metadata": {"document_id": "doc_002", "page": 2, "source": "nlp_intro.txt"},
                },
            ],
            "doc_003": [
                {
                    "chunk_id": "chunk_005",
                    "content": "Data science combines statistics, programming, and domain expertise to extract insights from data.",
                    "metadata": {"document_id": "doc_003", "page": 1, "source": "data_science.txt"},
                },
                {
                    "chunk_id": "chunk_006",
                    "content": "Exploratory data analysis helps identify patterns and anomalies in datasets.",
                    "metadata": {"document_id": "doc_003", "page": 2, "source": "data_science.txt"},
                },
            ],
        }

    def test_end_to_end_retrieval_pipeline(
        self, vector_store, embedding_provider, sample_documents
    ):
        """Test complete retrieval pipeline from indexing to search."""
        collection_name = "ai_concepts"

        # Step 1: Index documents
        all_chunks = []
        all_embeddings = []
        all_metadata = []
        all_ids = []

        for doc_id, chunks in sample_documents.items():
            for chunk in chunks:
                all_chunks.append(chunk["content"])
                all_metadata.append(chunk["metadata"])
                all_ids.append(chunk["chunk_id"])

        # Generate embeddings
        all_embeddings = embedding_provider.embed_batch(all_chunks)

        # Add to vector store
        result = vector_store.add_documents(
            collection_name=collection_name,
            documents=all_chunks,
            embeddings=all_embeddings,
            metadata=all_metadata,
            ids=all_ids,
        )

        assert result["success"] is True
        assert result["count"] == 6

        # Step 2: Create retriever
        config = RetrieverConfig(
            collection_name=collection_name,
            top_k=3,
            ranking_strategy=RankingStrategy.MMR,
        )
        retriever = Retriever(vector_store, embedding_provider, config)

        # Step 3: Test various queries
        test_queries = [
            ("What is machine learning?", ["machine learning", "learning", "artificial intelligence"]),
            ("Tell me about transformers", ["transformer", "NLP", "neural"]),
            ("Data analysis methods", ["data", "analysis", "exploratory"]),
        ]

        for query, expected_keywords in test_queries:
            results = retriever.retrieve(query, top_k=3)

            assert len(results) > 0, f"No results for query: {query}"
            assert all(isinstance(r, RetrievalResult) for r in results)
            assert all(0.0 <= r.relevance_score <= 1.0 for r in results)

            # Check that at least one result contains expected keywords
            results_text = " ".join([r.content.lower() for r in results])
            assert any(
                keyword.lower() in results_text for keyword in expected_keywords
            ), f"Expected keywords not found in results for query: {query}"

            logger.info(
                f"Query '{query}': Retrieved {len(results)} results with scores "
                f"{[r.relevance_score for r in results]}"
            )

    def test_end_to_end_with_filtering(
        self, vector_store, embedding_provider, sample_documents
    ):
        """Test retrieval with document filtering."""
        collection_name = "test_filter"

        # Index all documents
        all_chunks = []
        all_embeddings = []
        all_metadata = []
        all_ids = []

        for doc_id, chunks in sample_documents.items():
            for chunk in chunks:
                all_chunks.append(chunk["content"])
                all_metadata.append(chunk["metadata"])
                all_ids.append(chunk["chunk_id"])

        all_embeddings = embedding_provider.embed_batch(all_chunks)

        vector_store.add_documents(
            collection_name=collection_name,
            documents=all_chunks,
            embeddings=all_embeddings,
            metadata=all_metadata,
            ids=all_ids,
        )

        # Create retriever
        config = RetrieverConfig(collection_name=collection_name, top_k=10)
        retriever = Retriever(vector_store, embedding_provider, config)

        # Query all documents (no filter)
        all_results = retriever.retrieve(
            "machine learning and data science", top_k=10
        )
        assert len(all_results) > 0

        # Query with filter to only doc_001 and doc_002
        filtered_results = retriever.retrieve(
            "machine learning and data science",
            top_k=10,
            filters={"document_ids": ["doc_001", "doc_002"]},
        )

        # All filtered results should be from allowed documents
        for result in filtered_results:
            assert result.document_id in ["doc_001", "doc_002"]

        logger.info(
            f"Unfiltered results: {len(all_results)}, Filtered results: {len(filtered_results)}"
        )

    def test_end_to_end_reranking_strategies(
        self, vector_store, embedding_provider, sample_documents
    ):
        """Test different re-ranking strategies."""
        collection_name = "test_rerank"

        # Index documents
        all_chunks = []
        all_embeddings = []
        all_metadata = []
        all_ids = []

        for doc_id, chunks in sample_documents.items():
            for chunk in chunks:
                all_chunks.append(chunk["content"])
                all_metadata.append(chunk["metadata"])
                all_ids.append(chunk["chunk_id"])

        all_embeddings = embedding_provider.embed_batch(all_chunks)

        vector_store.add_documents(
            collection_name=collection_name,
            documents=all_chunks,
            embeddings=all_embeddings,
            metadata=all_metadata,
            ids=all_ids,
        )

        query = "neural networks and learning"

        # Test MMR re-ranking
        mmr_config = RetrieverConfig(
            collection_name=collection_name,
            ranking_strategy=RankingStrategy.MMR,
            top_k=3,
        )
        mmr_retriever = Retriever(vector_store, embedding_provider, mmr_config)
        mmr_results = mmr_retriever.retrieve(query)

        # Test no re-ranking
        no_rerank_config = RetrieverConfig(
            collection_name=collection_name,
            ranking_strategy=RankingStrategy.NO_RERANK,
            top_k=3,
        )
        no_rerank_retriever = Retriever(
            vector_store, embedding_provider, no_rerank_config
        )
        no_rerank_results = no_rerank_retriever.retrieve(query)

        # Both should return results
        assert len(mmr_results) > 0
        assert len(no_rerank_results) > 0

        # Results may be in different order due to different strategies
        logger.info(
            f"MMR order: {[r.chunk_id for r in mmr_results]}"
        )
        logger.info(
            f"No-rerank order: {[r.chunk_id for r in no_rerank_results]}"
        )

    def test_end_to_end_scoring_consistency(
        self, vector_store, embedding_provider, sample_documents
    ):
        """Test that scoring is consistent across queries."""
        collection_name = "test_scoring"

        # Index documents
        all_chunks = []
        all_embeddings = []
        all_metadata = []
        all_ids = []

        for doc_id, chunks in sample_documents.items():
            for chunk in chunks:
                all_chunks.append(chunk["content"])
                all_metadata.append(chunk["metadata"])
                all_ids.append(chunk["chunk_id"])

        all_embeddings = embedding_provider.embed_batch(all_chunks)

        vector_store.add_documents(
            collection_name=collection_name,
            documents=all_chunks,
            embeddings=all_embeddings,
            metadata=all_metadata,
            ids=all_ids,
        )

        config = RetrieverConfig(collection_name=collection_name, top_k=5)
        retriever = Retriever(vector_store, embedding_provider, config)

        # Run same query multiple times
        query = "learning"
        results1 = retriever.retrieve(query)
        results2 = retriever.retrieve(query)

        # Scores should be consistent
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1.chunk_id == r2.chunk_id
            assert abs(r1.relevance_score - r2.relevance_score) < 0.001

        logger.info(f"Consistency check passed for {len(results1)} results")

    def test_end_to_end_batch_retrieval(
        self, vector_store, embedding_provider, sample_documents
    ):
        """Test batch retrieval for multiple queries."""
        collection_name = "test_batch"

        # Index documents
        all_chunks = []
        all_embeddings = []
        all_metadata = []
        all_ids = []

        for doc_id, chunks in sample_documents.items():
            for chunk in chunks:
                all_chunks.append(chunk["content"])
                all_metadata.append(chunk["metadata"])
                all_ids.append(chunk["chunk_id"])

        all_embeddings = embedding_provider.embed_batch(all_chunks)

        vector_store.add_documents(
            collection_name=collection_name,
            documents=all_chunks,
            embeddings=all_embeddings,
            metadata=all_metadata,
            ids=all_ids,
        )

        config = RetrieverConfig(collection_name=collection_name, top_k=2)
        retriever = Retriever(vector_store, embedding_provider, config)

        # Batch retrieve
        queries = [
            "What is machine learning?",
            "Explain natural language processing",
            "Data science applications",
        ]

        batch_results = retriever.batch_retrieve(queries)

        assert len(batch_results) == 3
        assert all(isinstance(results, list) for results in batch_results)
        assert all(
            all(isinstance(r, RetrievalResult) for r in results)
            for results in batch_results
        )

        for i, results in enumerate(batch_results):
            logger.info(
                f"Query {i+1}: Retrieved {len(results)} results"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
