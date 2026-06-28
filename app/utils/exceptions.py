"""Custom exceptions for the AI Knowledge Support System"""

from typing import Optional, List, Any
from pydantic import BaseModel, Field
from fastapi import HTTPException, status


# ============================================================================
# Error Response Schemas
# ============================================================================

class ErrorDetail(BaseModel):
    """Detailed error information"""
    code: str = Field(description="Error code")
    message: str = Field(description="Human-readable error message")
    field: Optional[str] = Field(default=None, description="Field that caused error (if applicable)")
    value: Optional[Any] = Field(default=None, description="Value that caused error")

    class Config:
        json_schema_extra = {
            "example": {
                "code": "INVALID_FILE_TYPE",
                "message": "File type not supported",
                "field": "file_type",
                "value": ".exe"
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response schema"""
    error_type: str = Field(description="Type of error")
    status_code: int = Field(description="HTTP status code")
    message: str = Field(description="Main error message")
    details: List[ErrorDetail] = Field(default_factory=list, description="Detailed error information")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")

    class Config:
        json_schema_extra = {
            "example": {
                "error_type": "DocumentProcessingError",
                "status_code": 400,
                "message": "Failed to process document",
                "details": [
                    {
                        "code": "FILE_TOO_LARGE",
                        "message": "File exceeds maximum size of 100MB",
                        "field": "file_size",
                        "value": 150000000
                    }
                ]
            }
        }


# ============================================================================
# Custom Exception Classes
# ============================================================================

class AIKnowledgeException(Exception):
    """Base exception for AI Knowledge Support System"""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[List[ErrorDetail]] = None,
        request_id: Optional[str] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or []
        self.request_id = request_id
        super().__init__(message)

    def to_response(self) -> ErrorResponse:
        """Convert exception to error response"""
        return ErrorResponse(
            error_type=self.__class__.__name__,
            status_code=self.status_code,
            message=self.message,
            details=self.details,
            request_id=self.request_id,
        )

    def to_http_exception(self) -> HTTPException:
        """Convert exception to FastAPI HTTPException"""
        return HTTPException(
            status_code=self.status_code,
            detail=self.to_response().model_dump(),
        )


class DocumentProcessingError(AIKnowledgeException):
    """Raised when document processing fails"""

    def __init__(
        self,
        message: str = "Document processing failed",
        error_code: str = "DOCUMENT_PROCESSING_ERROR",
        details: Optional[List[ErrorDetail]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details,
            request_id=request_id,
        )


class InvalidFileError(DocumentProcessingError):
    """Raised when uploaded file is invalid"""

    def __init__(
        self,
        message: str = "Invalid file",
        reason: Optional[str] = None,
        details: Optional[List[ErrorDetail]] = None,
        request_id: Optional[str] = None,
    ):
        if reason:
            message = f"{message}: {reason}"
        super().__init__(
            message=message,
            error_code="INVALID_FILE",
            details=details,
            request_id=request_id,
        )


class FileSizeError(DocumentProcessingError):
    """Raised when file exceeds size limit"""

    def __init__(
        self,
        file_size: int,
        max_size: int,
        request_id: Optional[str] = None,
    ):
        message = f"File size {file_size} bytes exceeds maximum {max_size} bytes"
        detail = ErrorDetail(
            code="FILE_TOO_LARGE",
            message=message,
            field="file_size",
            value=file_size,
        )
        super().__init__(
            message=message,
            error_code="FILE_SIZE_EXCEEDED",
            details=[detail],
            request_id=request_id,
        )


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when file type is not supported"""

    def __init__(
        self,
        file_type: str,
        supported_types: List[str],
        request_id: Optional[str] = None,
    ):
        message = f"File type '{file_type}' not supported. Supported types: {', '.join(supported_types)}"
        detail = ErrorDetail(
            code="UNSUPPORTED_FILE_TYPE",
            message=message,
            field="file_type",
            value=file_type,
        )
        super().__init__(
            message=message,
            error_code="UNSUPPORTED_FILE_TYPE",
            details=[detail],
            request_id=request_id,
        )


class RetrievalError(AIKnowledgeException):
    """Raised when document retrieval fails"""

    def __init__(
        self,
        message: str = "Document retrieval failed",
        error_code: str = "RETRIEVAL_ERROR",
        details: Optional[List[ErrorDetail]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
            request_id=request_id,
        )


class NoResultsFoundError(RetrievalError):
    """Raised when no relevant documents found"""

    def __init__(
        self,
        query: str,
        request_id: Optional[str] = None,
    ):
        message = f"No relevant documents found for query: '{query}'"
        super().__init__(
            message=message,
            error_code="NO_RESULTS_FOUND",
            details=[
                ErrorDetail(
                    code="NO_MATCHING_DOCUMENTS",
                    message=message,
                    field="query",
                    value=query,
                )
            ],
            request_id=request_id,
        )


class ValidationError(AIKnowledgeException):
    """Raised when validation fails"""

    def __init__(
        self,
        message: str = "Validation failed",
        error_code: str = "VALIDATION_ERROR",
        details: Optional[List[ErrorDetail]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details=details,
            request_id=request_id,
        )


class FactValidationError(ValidationError):
    """Raised when fact validation fails"""

    def __init__(
        self,
        message: str = "Fact validation failed",
        facts: Optional[List[str]] = None,
        request_id: Optional[str] = None,
    ):
        details = [
            ErrorDetail(
                code="FACT_VALIDATION_FAILED",
                message=f"Fact: {fact}",
                field="response",
            )
            for fact in (facts or [])
        ]
        super().__init__(
            message=message,
            error_code="FACT_VALIDATION_ERROR",
            details=details or [
                ErrorDetail(
                    code="FACT_VALIDATION_FAILED",
                    message=message,
                )
            ],
            request_id=request_id,
        )


class LLMError(AIKnowledgeException):
    """Raised when LLM processing fails"""

    def __init__(
        self,
        message: str = "LLM processing failed",
        error_code: str = "LLM_ERROR",
        provider: Optional[str] = None,
        details: Optional[List[ErrorDetail]] = None,
        request_id: Optional[str] = None,
    ):
        if provider:
            message = f"{message} (Provider: {provider})"
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=details,
            request_id=request_id,
        )


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out"""

    def __init__(
        self,
        timeout_seconds: int,
        provider: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        message = f"LLM request timed out after {timeout_seconds} seconds"
        super().__init__(
            message=message,
            error_code="LLM_TIMEOUT",
            provider=provider,
            details=[
                ErrorDetail(
                    code="REQUEST_TIMEOUT",
                    message=message,
                    field="timeout_seconds",
                    value=timeout_seconds,
                )
            ],
            request_id=request_id,
        )


class LLMRateLimitError(LLMError):
    """Raised when LLM rate limit is exceeded"""

    def __init__(
        self,
        message: str = "LLM rate limit exceeded",
        provider: Optional[str] = None,
        retry_after_seconds: Optional[int] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            error_code="LLM_RATE_LIMIT",
            provider=provider,
            details=[
                ErrorDetail(
                    code="RATE_LIMIT_EXCEEDED",
                    message=message,
                    field="retry_after",
                    value=retry_after_seconds,
                )
            ] if retry_after_seconds else [],
            request_id=request_id,
        )


class DatabaseError(AIKnowledgeException):
    """Raised when database operations fail"""

    def __init__(
        self,
        message: str = "Database operation failed",
        error_code: str = "DATABASE_ERROR",
        details: Optional[List[ErrorDetail]] = None,
        request_id: Optional[str] = None,
    ):
        super().__init__(
            message=message,
            error_code=error_code,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details=details,
            request_id=request_id,
        )
