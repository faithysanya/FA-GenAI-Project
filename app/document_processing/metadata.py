"""
Metadata extraction module for documents.
Extracts file information such as name, size, modification date, and type.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import NamedTuple


class DocumentMetadata(NamedTuple):
    """Container for document metadata."""
    filename: str
    file_size: int
    file_size_mb: float
    file_type: str
    modified_date: str
    full_path: str


def extract_metadata(file_path: str, file_type: str) -> DocumentMetadata:
    """
    Extract metadata from a document file.
    
    Args:
        file_path: Path to the document file
        file_type: Type of the document (pdf, txt, csv, excel, etc.)
        
    Returns:
        DocumentMetadata object containing file information
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    path_obj = Path(file_path)
    
    # Get file stats
    stat_info = os.stat(file_path)
    file_size_bytes = stat_info.st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Get modification date
    mod_timestamp = stat_info.st_mtime
    mod_date = datetime.fromtimestamp(mod_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    
    # Get absolute path
    abs_path = str(path_obj.absolute())
    
    return DocumentMetadata(
        filename=path_obj.name,
        file_size=file_size_bytes,
        file_size_mb=round(file_size_mb, 4),
        file_type=file_type.lower(),
        modified_date=mod_date,
        full_path=abs_path
    )
