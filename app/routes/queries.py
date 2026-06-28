"""Query processing routes"""

from fastapi import APIRouter, HTTPException, status, Header, Query
from typing import Optional, List
import logging
import uuid
import time

from app.models import (
    QueryRequest,
    QueryResponse,
    RetrievalResult,
    Reasoning,
    ValidationResult,
    ValidationLevel,
)
from app.utils.exceptions import (
    RetrievalError,
    ValidationError,
    LLMError,
    NoResultsFoundError,
)
from app.agents.orchestrator import run_agent_pipeline
from app.utils.validators import validate_query
from app.utils.guardrails import check_safety

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/queries", tags=["queries"])


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Query processed successfully"},
        400: {"description": "Invalid query request"},
        503: {"description": "LLM service unavailable"},
        504: {"description": "Query timeout"},
    },
)
async def process_query(
    request: QueryRequest,
    x_request_id: Optional[str] = Header(None, description="Request tracking ID"),
) -> QueryResponse:
    """
    Process a user query against uploaded documents.

    This endpoint retrieves relevant documents based on the query, generates
    an AI response, performs optional validation, and returns comprehensive
    results including source references and reasoning steps.

    **Request Body:**
    - query: User question or query (required, max 5000 chars)
    - document_ids: Specific documents to search (optional, defaults to all)
    - top_k: Number of results to return (1-50, default: 5)
    - include_reasoning: Include detailed reasoning steps (default: false)
    - validation_level: Validation strictness - strict/moderate/lenient (default: moderate)

    **Response:**
    - query: Original query text
    - response: Generated AI response
    - sources: Retrieved document chunks with relevance scores
    - confidence: Overall confidence score (0-1)
    - reasoning_steps: Detailed reasoning if requested
    - validation_result: Validation results if applicable
    - execution_time_ms: Query execution time

    **Status Codes:**
    - 200: Query processed successfully
    - 400: Invalid query request (malformed, empty, etc.)
    - 503: LLM service unavailable
    - 504: Query timeout

    **Example Request:**
    ```json
    {
        "query": "What are the main findings?",
        "document_ids": ["doc_123"],
        "top_k": 5,
        "include_reasoning": true,
        "validation_level": "moderate"
    }
    ```

    **Example Response:**
    ```json
    {
        "query": "What are the main findings?",
        "response": "The main findings include...",
        "sources": [
            {
                "chunk_id": "chunk_001",
                "document_id": "doc_123",
                "content": "Key finding 1...",
                "relevance_score": 0.92,
                "metadata": {"page": 1}
            }
        ],
        "confidence": 0.87,
        "reasoning_steps": [...],
        "validation_result": {...},
        "execution_time_ms": 1250.5
    }
    ```
    """
    request_id = x_request_id or str(uuid.uuid4())
    start_time = time.time()

    try:
        logger.info(
            f"[{request_id}] Processing query",
            extra={
                "query": request.query[:100],
                "document_ids": request.document_ids,
                "top_k": request.top_k,
                "validation_level": request.validation_level,
            },
        )

        # Validate and sanitize query
        clean_query = validate_query(request.query)

        # Safety check
        safety = check_safety(clean_query)
        if not safety["is_safe"]:
            raise HTTPException(status_code=400, detail=f"Unsafe input: {safety['flag']}")

        # Run the full agent pipeline (Plan → Retrieve → Reason → Validate)
        agent_result = run_agent_pipeline(
            query=clean_query,
            document_ids=request.document_ids,
            top_k=request.top_k,
        )

        response_text = agent_result["response"]
        sources: List[RetrievalResult] = []

        # Build reasoning steps if requested
        reasoning_steps = None
        if request.include_reasoning:
            steps = agent_result.get("reasoning_steps", [])
            reasoning_steps = [
                Reasoning(
                    step_number=i + 1,
                    action="agent_step",
                    input_data={"query": clean_query},
                    output_data={"step": s},
                    confidence=agent_result.get("confidence", 0.75),
                    reasoning_text=s,
                )
                for i, s in enumerate(steps)
            ]

        # Build validation result
        validation_result = None
        agent_val = agent_result.get("validation", {})
        if request.validation_level != ValidationLevel.LENIENT:
            validation_result = ValidationResult(
                is_valid=agent_val.get("is_grounded", True),
                validation_level=request.validation_level,
                errors=[],
                warnings=agent_val.get("issues", []),
                checks_performed=["grounding_check", "safety_check", "agent_validation"],
                confidence_score=agent_result.get("confidence", 0.75),
            )

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        logger.info(
            f"[{request_id}] Query processing completed",
            extra={
                "execution_time_ms": execution_time_ms,
                "sources_count": len(sources),
                "validation_performed": validation_result is not None,
            },
        )

        return QueryResponse(
            query=request.query,
            response=response_text,
            sources=sources,
            confidence=agent_result.get("confidence", 0.75),
            reasoning_steps=reasoning_steps,
            validation_result=validation_result,
            execution_time_ms=execution_time_ms,
        )

    except ValueError as e:
        logger.warning(
            f"[{request_id}] Validation error: {str(e)}",
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_type": "ValidationError",
                "message": str(e),
                "request_id": request_id,
            },
        )

    except (RetrievalError, NoResultsFoundError) as e:
        logger.warning(
            f"[{request_id}] Retrieval error: {str(e)}",
            extra={"request_id": request_id},
        )
        exc = e.to_http_exception()
        raise exc

    except LLMError as e:
        logger.error(
            f"[{request_id}] LLM error: {str(e)}",
            extra={"request_id": request_id},
        )
        exc = e.to_http_exception()
        raise exc

    except Exception as e:
        logger.error(
            f"[{request_id}] Unexpected error during query processing: {str(e)}",
            exc_info=True,
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": "QueryProcessingError",
                "message": "Failed to process query",
                "request_id": request_id,
            },
        )


@router.get(
    "/history",
    response_model=List[QueryResponse],
    responses={200: {"description": "Query history"}},
)
async def get_query_history(
    document_id: Optional[str] = Query(None, description="Filter by document ID"),
    limit: int = Query(10, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Result offset"),
    x_request_id: Optional[str] = Header(None),
) -> List[QueryResponse]:
    """
    Retrieve query history.

    **Query Parameters:**
    - document_id: Filter queries by specific document (optional)
    - limit: Number of queries to return (default: 10, max: 100)
    - offset: Number of queries to skip (default: 0)

    **Response:**
    List of previous QueryResponse objects.

    **Status Codes:**
    - 200: History retrieved successfully
    """
    request_id = x_request_id or str(uuid.uuid4())
    logger.info(
        f"[{request_id}] Retrieving query history",
        extra={"document_id": document_id, "limit": limit, "offset": offset},
    )

    # TODO: Implement query history retrieval from database
    # For now, return empty list
    return []


@router.get(
    "/{query_id}",
    response_model=QueryResponse,
    responses={
        200: {"description": "Query found"},
        404: {"description": "Query not found"},
    },
)
async def get_query(
    query_id: str,
    x_request_id: Optional[str] = Header(None),
) -> QueryResponse:
    """
    Retrieve a specific query result by ID.

    **Parameters:**
    - query_id: Query identifier

    **Response:**
    Full QueryResponse with all details.

    **Status Codes:**
    - 200: Query found
    - 404: Query not found
    """
    request_id = x_request_id or str(uuid.uuid4())
    logger.info(f"[{request_id}] Retrieving query: {query_id}")

    # TODO: Implement query lookup from database
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"message": "Query not found", "query_id": query_id},
    )


@router.delete(
    "/{query_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Query deleted"},
        404: {"description": "Query not found"},
    },
)
async def delete_query(
    query_id: str,
    x_request_id: Optional[str] = Header(None),
) -> None:
    """
    Delete a query result.

    **Parameters:**
    - query_id: Query identifier

    **Status Codes:**
    - 204: Query deleted successfully
    - 404: Query not found
    """
    request_id = x_request_id or str(uuid.uuid4())
    logger.info(f"[{request_id}] Deleting query: {query_id}")

    # TODO: Implement query deletion
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"message": "Query not found", "query_id": query_id},
    )
