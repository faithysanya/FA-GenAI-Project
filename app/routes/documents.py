"""Document management routes"""

from fastapi import APIRouter, File, UploadFile, HTTPException, status, Header
from typing import Optional, List
import logging
import os
import uuid

from app.models import DocumentUploadResponse, Document, DocumentStatus
from app.utils.exceptions import (
    InvalidFileError,
    FileSizeError,
    UnsupportedFileTypeError,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

# Configuration
SUPPORTED_FILE_TYPES = [".pdf", ".txt", ".docx", ".doc", ".json"]
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def validate_file_type(filename: str) -> str:
    """Validate file type by extension"""
    file_extension = os.path.splitext(filename)[1].lower()
    if file_extension not in SUPPORTED_FILE_TYPES:
        raise UnsupportedFileTypeError(
            file_type=file_extension,
            supported_types=SUPPORTED_FILE_TYPES,
        )
    return file_extension


def validate_file_size(file_size: int) -> None:
    """Validate file size"""
    if file_size > MAX_FILE_SIZE_BYTES:
        raise FileSizeError(
            file_size=file_size,
            max_size=MAX_FILE_SIZE_BYTES,
        )


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"description": "Document accepted for processing"},
        400: {"description": "Invalid file"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported file type"},
    },
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
    tags: Optional[str] = None,
    x_request_id: Optional[str] = Header(None, description="Request tracking ID"),
) -> DocumentUploadResponse:
    """
    Upload a document for processing.

    Accepts PDF, TXT, DOCX, DOC, and JSON files.
    Maximum file size: 100MB.

    **Request:**
    - file: Document file (multipart/form-data)
    - tags: Optional comma-separated tags for categorization
    - x-request-id: Optional request tracking ID (header)

    **Response:**
    - document_id: Unique identifier for the uploaded document
    - filename: Original filename
    - file_type: Document type
    - status: Current processing status
    - total_chunks: Number of chunks the document was split into

    **Status Codes:**
    - 202: Document accepted and queued for processing
    - 400: Invalid file (wrong type, corrupted, etc.)
    - 413: File exceeds 100MB limit
    - 415: Unsupported file type
    """
    request_id = x_request_id or str(uuid.uuid4())

    try:
        # Validate filename
        if not file.filename:
            raise InvalidFileError("Filename is required")

        logger.info(
            f"[{request_id}] Processing upload: {file.filename}",
            extra={"filename": file.filename, "request_id": request_id},
        )

        # Validate file type
        file_extension = validate_file_type(file.filename)

        # Read file content to check size
        content = await file.read()
        file_size = len(content)

        # Validate file size
        validate_file_size(file_size)

        # Check if file is empty
        if file_size == 0:
            raise InvalidFileError("File is empty")

        # Generate document ID
        document_id = f"doc_{uuid.uuid4().hex[:12]}"

        logger.info(
            f"[{request_id}] Document validated successfully",
            extra={
                "document_id": document_id,
                "file_size": file_size,
                "file_type": file_extension,
            },
        )

        # Return acceptance response with 202 status
        response = DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            file_type=file_extension,
            status=DocumentStatus.PROCESSING.value,
            message="Document accepted for processing",
            total_chunks=0,
        )

        logger.info(
            f"[{request_id}] Upload response prepared",
            extra={"document_id": document_id, "response": response.model_dump()},
        )

        return response

    except (InvalidFileError, FileSizeError, UnsupportedFileTypeError) as e:
        logger.warning(
            f"[{request_id}] File validation error: {str(e)}",
            extra={"request_id": request_id},
        )
        exc = e.to_http_exception()
        raise exc
    except Exception as e:
        logger.error(
            f"[{request_id}] Unexpected error during upload: {str(e)}",
            exc_info=True,
            extra={"request_id": request_id},
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_type": "DocumentProcessingError",
                "message": "Failed to process document",
                "request_id": request_id,
            },
        )


@router.get(
    "/{document_id}",
    response_model=Document,
    responses={
        200: {"description": "Document details"},
        404: {"description": "Document not found"},
    },
)
async def get_document(
    document_id: str,
    x_request_id: Optional[str] = Header(None),
) -> Document:
    """
    Retrieve document details and processing status.

    **Parameters:**
    - document_id: Document identifier

    **Response:**
    Document object with current status, metadata, and chunk information.

    **Status Codes:**
    - 200: Document found
    - 404: Document not found
    """
    request_id = x_request_id or str(uuid.uuid4())
    logger.info(f"[{request_id}] Retrieving document: {document_id}")

    # TODO: Implement database lookup
    # For now, return a placeholder
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"message": "Document not found", "document_id": document_id},
    )


@router.get(
    "",
    response_model=List[Document],
    responses={200: {"description": "List of documents"}},
)
async def list_documents(
    skip: int = 0,
    limit: int = 10,
    status_filter: Optional[str] = None,
    x_request_id: Optional[str] = Header(None),
) -> List[Document]:
    """
    List all uploaded documents with pagination.

    **Query Parameters:**
    - skip: Number of documents to skip (default: 0)
    - limit: Number of documents to return (default: 10, max: 100)
    - status_filter: Filter by status (pending, processing, completed, failed)

    **Response:**
    List of Document objects.
    """
    request_id = x_request_id or str(uuid.uuid4())
    logger.info(
        f"[{request_id}] Listing documents",
        extra={"skip": skip, "limit": limit, "status": status_filter},
    )

    # TODO: Implement database query with pagination and filtering
    # For now, return empty list
    return []


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={
        204: {"description": "Document deleted"},
        404: {"description": "Document not found"},
    },
)
async def delete_document(
    document_id: str,
    x_request_id: Optional[str] = Header(None),
) -> None:
    """
    Delete a document and its associated chunks.

    **Parameters:**
    - document_id: Document identifier

    **Status Codes:**
    - 204: Document deleted successfully
    - 404: Document not found
    """
    request_id = x_request_id or str(uuid.uuid4())
    logger.info(f"[{request_id}] Deleting document: {document_id}")

    # TODO: Implement document and chunk deletion
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"message": "Document not found", "document_id": document_id},
    )
