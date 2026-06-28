"""Retrieval API routes for document search and retrieval."""

import logging
import time
import uuid
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Header

from app.models import QueryRequest, RetrievalResult
from app.vector_db.retriever import Retriever, RetrieverConfig
from app.vector_db.client import ChromaVectorStore
from app.vector_db.embedding import EmbeddingProvider

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/retrieve", tags=["retrieval"])

# Module-level instances (will be initialized by main.py)
_retriever: Optional[Retriever] = None


def initialize_retriever(
    vector_store: ChromaVectorStore,
    embedding_provider: EmbeddingProvider,
    config: Optional[RetrieverConfig] = None,
) -> None:
    """
    Initialize the retriever module with dependencies.

    Args:
        vector_store: ChromaVectorStore instance
        embedding_provider: EmbeddingProvider instance
        config: Optional RetrieverConfig
    """
    global _retriever
    _retriever = Retriever(vector_store, embedding_provider, config)
    logger.info("Retrieval module initialized")


def get_retriever() -> Retriever:
    """Get the configured retriever instance."""
    if _retriever is None:
        raise RuntimeError(
            "Retriever not initialized. Call initialize_retriever() first."
        )
    return _retriever


class RetrievalResponse(RetrievalResult):
    """Enhanced retrieval result with additional metadata."""

    request_id: Optional[str] = None
    execution_time_ms: float = 0.0


@router.post(
    "",
    response_model=List[RetrievalResult],
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Documents retrieved successfully"},
        400: {"description": "Invalid query request"},
        404: {"description": "No results found"},
        500: {"description": "Retrieval service error"},
    },
)
async def retrieve_documents(
    request: QueryRequest,
    x_request_id: Optional[str] = Header(None, description="Request tracking ID"),
) -> List[RetrievalResult]:
    """
    Retrieve relevant documents for a query.

    This endpoint accepts a query and returns the most relevant documents
    from the vector store, with optional filtering by document IDs.

    **Request Body:**
    - `query` (string, required): The search query (1-5000 characters)
    - `document_ids` (array[string], optional): Limit search to specific documents
    - `top_k` (integer, optional): Number of results to return (1-50, default: 5)

    **Response:**
    Array of RetrievalResult objects containing:
    - `chunk_id`: ID of the retrieved chunk
    - `document_id`: ID of the parent document
    - `content`: The chunk text content
    - `relevance_score`: Similarity score (0-1)
    - `metadata`: Additional chunk metadata

    **Query Logging:**
    - Each query is logged with request ID for tracking and analysis
    - Includes execution time and result count
    - Failed queries are logged as warnings

    Example:
    ```json
    POST /retrieve
    {
        "query": "What are the main findings?",
        "document_ids": ["doc_123"],
        "top_k": 5
    }

    Response:
    [
        {
            "chunk_id": "chunk_001",
            "document_id": "doc_123",
            "content": "The main findings show...",
            "relevance_score": 0.92,
            "metadata": {"source": "page 1"}
        }
    ]
    ```
    """
    request_id = x_request_id or str(uuid.uuid4())
    start_time = time.time()

    try:
        retriever = get_retriever()

        # Log query
        logger.info(
            f"[{request_id}] Query received: '{request.query[:100]}...' "
            f"(top_k={request.top_k}, documents_filter={len(request.document_ids) if request.document_ids else 'None'})"
        )

        # Prepare filters
        filters = None
        if request.document_ids:
            filters = {"document_ids": request.document_ids}

        # Retrieve documents
        results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            filters=filters,
        )

        # Calculate execution time
        execution_time = (time.time() - start_time) * 1000  # Convert to ms

        # Log success
        logger.info(
            f"[{request_id}] Retrieved {len(results)} results in {execution_time:.2f}ms"
        )

        # If no results found, log as info (not an error)
        if not results:
            logger.info(f"[{request_id}] No results found for query: {request.query[:100]}")

        return results

    except ValueError as e:
        execution_time = (time.time() - start_time) * 1000
        logger.warning(f"[{request_id}] Validation error in retrieval: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid query: {str(e)}",
        )
    except RuntimeError as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] Retriever not initialized: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval service unavailable",
        )
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        logger.error(f"[{request_id}] Retrieval failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve documents",
        )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Retrieval service is healthy"},
        500: {"description": "Retrieval service error"},
    },
)
async def retrieval_health() -> dict:
    """Check if retrieval service is healthy and initialized."""
    try:
        retriever = get_retriever()
        return {
            "status": "healthy",
            "service": "retrieval",
            "retriever_initialized": retriever is not None,
        }
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval service not initialized",
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed",
        )
