"""API endpoint tests for retrieval routes."""

import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from app.main import app
from app.models import QueryRequest, RetrievalResult
from app.routes.retrieval import initialize_retriever
from app.vector_db.client import ChromaVectorStore
from app.vector_db.embedding import EmbeddingProvider


@pytest.fixture
def client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_retriever_setup():
    """Setup mock retriever for API tests."""
    vector_store = Mock(spec=ChromaVectorStore)
    embedding_provider = Mock(spec=EmbeddingProvider)
    
    # Mock retrieval results
    mock_results = [
        {
            "id": "chunk_001",
            "document": "Machine learning enables computers to learn from data.",
            "metadata": {"document_id": "doc_001", "page": 1},
            "distance": 0.1,
        }
    ]
    vector_store.search.return_value = mock_results
    embedding_provider.embed_text.return_value = [0.1] * 384
    
    # Initialize the retriever
    initialize_retriever(vector_store, embedding_provider)
    
    return vector_store, embedding_provider


class TestRetrievalAPI:
    """Test retrieval API endpoints."""

    def test_retrieve_endpoint_success(self, client, mock_retriever_setup):
        """Test successful retrieval via API."""
        request_body = {
            "query": "What is machine learning?",
            "top_k": 5
        }
        
        response = client.post("/retrieve", json=request_body)
        
        assert response.status_code == 200
        results = response.json()
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Verify result structure
        result = results[0]
        assert "chunk_id" in result
        assert "document_id" in result
        assert "content" in result
        assert "relevance_score" in result
        assert "metadata" in result
        
        # Verify score is normalized
        assert 0.0 <= result["relevance_score"] <= 1.0

    def test_retrieve_with_document_filter(self, client, mock_retriever_setup):
        """Test retrieval with document_ids filter."""
        request_body = {
            "query": "test query",
            "document_ids": ["doc_001", "doc_002"],
            "top_k": 5
        }
        
        response = client.post("/retrieve", json=request_body)
        
        assert response.status_code == 200
        results = response.json()
        assert isinstance(results, list)

    def test_retrieve_empty_query_error(self, client, mock_retriever_setup):
        """Test that empty query returns error."""
        request_body = {
            "query": "",
            "top_k": 5
        }
        
        response = client.post("/retrieve", json=request_body)
        
        # Should return 422 validation error (empty string fails min_length)
        assert response.status_code == 422

    def test_retrieve_missing_query_error(self, client, mock_retriever_setup):
        """Test that missing query field returns error."""
        request_body = {
            "top_k": 5
        }
        
        response = client.post("/retrieve", json=request_body)
        
        # Should return 422 validation error
        assert response.status_code == 422

    def test_retrieve_with_custom_top_k(self, client, mock_retriever_setup):
        """Test retrieval with custom top_k parameter."""
        request_body = {
            "query": "test query",
            "top_k": 10
        }
        
        response = client.post("/retrieve", json=request_body)
        
        assert response.status_code == 200

    def test_retrieve_invalid_top_k(self, client, mock_retriever_setup):
        """Test that invalid top_k returns error."""
        request_body = {
            "query": "test",
            "top_k": 100  # Max is 50
        }
        
        response = client.post("/retrieve", json=request_body)
        
        # Should return 422 validation error (too large)
        assert response.status_code == 422

    def test_retrieve_health_endpoint(self, client, mock_retriever_setup):
        """Test health check endpoint."""
        response = client.get("/retrieve/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "retrieval"
        assert data["retriever_initialized"] is True

    def test_retrieve_health_not_initialized(self, client):
        """Test health endpoint when retriever not initialized."""
        # This test would need a fresh app instance without retriever initialized
        # which is hard to do with the current setup. Skip for now.
        pytest.skip("Requires fresh app instance without initialization")

    def test_retrieve_response_model(self, client, mock_retriever_setup):
        """Test that response conforms to RetrievalResult model."""
        request_body = {
            "query": "test query",
            "top_k": 5
        }
        
        response = client.post("/retrieve", json=request_body)
        
        assert response.status_code == 200
        results = response.json()
        
        # Should be able to parse as list of RetrievalResult
        for result in results:
            retrieved = RetrievalResult(**result)
            assert isinstance(retrieved.chunk_id, str)
            assert isinstance(retrieved.document_id, str)
            assert isinstance(retrieved.content, str)
            assert isinstance(retrieved.relevance_score, float)
            assert 0.0 <= retrieved.relevance_score <= 1.0

    def test_retrieve_with_request_id_header(self, client, mock_retriever_setup):
        """Test that request ID header is handled."""
        request_body = {
            "query": "test query",
            "top_k": 5
        }
        headers = {
            "X-Request-ID": "test-request-123"
        }
        
        response = client.post("/retrieve", json=request_body, headers=headers)
        
        assert response.status_code == 200

    def test_retrieve_query_validation_max_length(self, client, mock_retriever_setup):
        """Test that overly long query returns error."""
        # Create query longer than 5000 characters
        long_query = "a" * 5001
        request_body = {
            "query": long_query,
            "top_k": 5
        }
        
        response = client.post("/retrieve", json=request_body)
        
        # Should return 422 validation error
        assert response.status_code == 422


class TestRetrievalAPIIntegration:
    """Integration tests for retrieval API."""

    def test_retrieve_full_workflow(self, client, mock_retriever_setup):
        """Test complete retrieval workflow."""
        # Test basic query
        response1 = client.post(
            "/retrieve",
            json={"query": "machine learning", "top_k": 3}
        )
        assert response1.status_code == 200
        results1 = response1.json()
        assert len(results1) > 0
        
        # Test filtered query
        response2 = client.post(
            "/retrieve",
            json={
                "query": "machine learning",
                "document_ids": ["doc_001"],
                "top_k": 3
            }
        )
        assert response2.status_code == 200
        results2 = response2.json()
        assert len(results2) > 0
        
        # Test health
        response3 = client.get("/retrieve/health")
        assert response3.status_code == 200
        
        health = response3.json()
        assert health["status"] == "healthy"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
