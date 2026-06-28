"""Input validation and sanitization utilities."""
import re
import logging
from app.utils.exceptions import InvalidFileError, FileSizeError

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE_MB = 50
MAX_QUERY_LENGTH = 5000
BLOCKED_PATTERNS = [
    r"<script.*?>.*?</script>",
    r"javascript:",
    r"data:text/html",
]


def validate_document_upload(filename: str, file_size_bytes: int) -> None:
    """Raise descriptive errors for invalid uploads."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise InvalidFileError(
            f"File type '{ext}' not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise FileSizeError(
            f"File size {file_size_bytes / 1024 / 1024:.1f}MB exceeds {MAX_FILE_SIZE_MB}MB limit"
        )


def validate_query(query: str) -> str:
    """Validate and sanitize a user query. Returns cleaned query."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters")
    return sanitize_input(query)


def sanitize_input(text: str) -> str:
    """Remove potentially harmful patterns from text."""
    for pattern in BLOCKED_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()
