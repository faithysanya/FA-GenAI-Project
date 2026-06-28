"""
Document processing module for parsing and extracting metadata from documents.
"""

from .parsers import (
    parse_pdf,
    parse_txt,
    parse_csv,
    parse_excel,
    parse_document,
)

from .metadata import (
    extract_metadata,
    DocumentMetadata,
)

from .chunking import (
    chunk_text,
    chunk_document,
    Chunk,
    ChunkingConfig,
)

__all__ = [
    'parse_pdf',
    'parse_txt',
    'parse_csv',
    'parse_excel',
    'parse_document',
    'extract_metadata',
    'DocumentMetadata',
    'chunk_text',
    'chunk_document',
    'Chunk',
    'ChunkingConfig',
]
