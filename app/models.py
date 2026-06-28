from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# Enums
# ============================================================================

class DocumentStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ChunkStatus(str, Enum):
    """Chunk processing status"""
    PENDING = "pending"
    EMBEDDED = "embedded"
    INDEXED = "indexed"
    FAILED = "failed"


class ValidationLevel(str, Enum):
    """Validation strictness level"""
    STRICT = "strict"
    MODERATE = "moderate"
    LENIENT = "lenient"


# ============================================================================
# Document Models
# ============================================================================

class DocumentMetadata(BaseModel):
    """Metadata for a document"""
    source_url: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list, description="Tags for document categorization")
    custom_fields: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")

    class Config:
        json_schema_extra = {
            "example": {
                "source_url": "https://example.com/doc",
                "author": "John Doe",
                "tags": ["urgent", "finance"],
                "custom_fields": {"department": "sales"}
            }
        }


class Chunk(BaseModel):
    """A chunk of text from a document"""
    chunk_id: str = Field(description="Unique identifier for the chunk")
    document_id: str = Field(description="Parent document ID")
    content: str = Field(description="Chunk text content")
    sequence_number: int = Field(description="Order of chunk within document")
    token_count: int = Field(description="Number of tokens in chunk")
    status: ChunkStatus = Field(default=ChunkStatus.PENDING, description="Processing status")
    embedding: Optional[List[float]] = Field(default=None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk-specific metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_001",
                "document_id": "doc_123",
                "content": "This is the first chunk of text...",
                "sequence_number": 1,
                "token_count": 150,
                "status": "embedded"
            }
        }


class Document(BaseModel):
    """Document model with comprehensive metadata"""
    document_id: str = Field(description="Unique document identifier")
    filename: str = Field(description="Original filename")
    file_type: str = Field(description="File type (pdf, txt, docx, etc.)")
    file_size: int = Field(description="File size in bytes")
    content_preview: str = Field(description="Preview of document content")
    status: DocumentStatus = Field(default=DocumentStatus.PENDING, description="Processing status")
    total_chunks: int = Field(default=0, description="Total number of chunks")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata, description="Document metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    error_message: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123",
                "filename": "report.pdf",
                "file_type": "pdf",
                "file_size": 250000,
                "content_preview": "First 200 chars of content...",
                "status": "completed",
                "total_chunks": 10
            }
        }


# ============================================================================
# Health & Upload Response Models
# ============================================================================

class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    environment: str = Field(description="Deployment environment")

    class Config:
        json_schema_extra = {
            "example": {"status": "healthy", "version": "0.1.0", "environment": "development"}
        }


class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    document_id: str = Field(description="Unique document identifier")
    filename: str = Field(description="Uploaded filename")
    file_type: str = Field(description="File type")
    status: str = Field(description="Upload status")
    message: Optional[str] = None
    total_chunks: int = Field(default=0, description="Number of chunks created")

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123",
                "filename": "report.pdf",
                "file_type": "pdf",
                "status": "processing",
                "total_chunks": 10
            }
        }


# ============================================================================
# Agent Models
# ============================================================================

class Plan(BaseModel):
    """Agent planning model"""
    plan_id: str = Field(description="Unique plan identifier")
    query: str = Field(description="User query")
    steps: List[str] = Field(description="Planned steps to answer query")
    estimated_complexity: str = Field(description="Query complexity: low, medium, high")
    reasoning: str = Field(description="Rationale for the plan")

    class Config:
        json_schema_extra = {
            "example": {
                "plan_id": "plan_001",
                "query": "What are the key financial metrics?",
                "steps": ["Extract financial section", "Analyze metrics", "Summarize"],
                "estimated_complexity": "medium",
                "reasoning": "Need to identify and analyze financial data"
            }
        }


class RetrievalResult(BaseModel):
    """Result from document retrieval"""
    chunk_id: str = Field(description="Retrieved chunk ID")
    document_id: str = Field(description="Parent document ID")
    content: str = Field(description="Chunk content")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance score 0-1")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_001",
                "document_id": "doc_123",
                "content": "Relevant text content...",
                "relevance_score": 0.92,
                "metadata": {"source": "page 5"}
            }
        }


class Reasoning(BaseModel):
    """Reasoning step in agent workflow"""
    step_number: int = Field(description="Step sequence number")
    action: str = Field(description="Action taken (retrieve, analyze, validate, etc.)")
    input_data: Dict[str, Any] = Field(description="Input for this step")
    output_data: Dict[str, Any] = Field(description="Output from this step")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score")
    reasoning_text: str = Field(description="Explanation of reasoning")

    class Config:
        json_schema_extra = {
            "example": {
                "step_number": 1,
                "action": "retrieve",
                "confidence": 0.85,
                "reasoning_text": "Retrieved top 5 relevant chunks"
            }
        }


class ValidationResult(BaseModel):
    """Validation result model"""
    is_valid: bool = Field(description="Whether validation passed")
    validation_level: ValidationLevel = Field(description="Strictness level used")
    errors: List[str] = Field(default_factory=list, description="Validation errors if any")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    checks_performed: List[str] = Field(description="List of checks performed")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Confidence in validation")

    class Config:
        json_schema_extra = {
            "example": {
                "is_valid": True,
                "validation_level": "moderate",
                "checks_performed": ["fact_check", "format_check"],
                "confidence_score": 0.89
            }
        }


# ============================================================================
# Query Models
# ============================================================================

class QueryRequest(BaseModel):
    """Request model for user query"""
    query: str = Field(description="User query text", min_length=1, max_length=5000)
    document_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific documents to search (None = all)"
    )
    top_k: int = Field(default=5, ge=1, le=50, description="Number of top results to return")
    include_reasoning: bool = Field(
        default=False,
        description="Include reasoning steps in response"
    )
    validation_level: ValidationLevel = Field(
        default=ValidationLevel.MODERATE,
        description="How strict validation should be"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the main findings?",
                "document_ids": ["doc_123"],
                "top_k": 5,
                "include_reasoning": True,
                "validation_level": "moderate"
            }
        }


class QueryResponse(BaseModel):
    """Response model for query"""
    query: str = Field(description="Original query")
    response: str = Field(description="Generated response")
    sources: List[RetrievalResult] = Field(description="Retrieved source chunks")
    confidence: float = Field(ge=0.0, le=1.0, description="Response confidence score")
    reasoning_steps: Optional[List[Reasoning]] = Field(
        default=None,
        description="Detailed reasoning steps if requested"
    )
    validation_result: Optional[ValidationResult] = Field(
        default=None,
        description="Validation result if performed"
    )
    execution_time_ms: float = Field(description="Query execution time in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the main findings?",
                "response": "The main findings are...",
                "sources": [],
                "confidence": 0.87,
                "execution_time_ms": 1250.5
            }
        }
