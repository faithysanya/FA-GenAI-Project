"""
Document chunking module for splitting text into semantic or fixed-size chunks.
Supports multiple strategies with overlap and metadata tracking.
"""

import re
from typing import List, NamedTuple, Literal
from dataclasses import dataclass


class Chunk(NamedTuple):
    """Container for a text chunk with metadata."""
    text: str
    start_idx: int
    end_idx: int
    chunk_num: int


@dataclass
class ChunkingConfig:
    """Configuration for chunking behavior."""
    strategy: Literal['semantic', 'fixed'] = 'semantic'
    max_chunk_size: int = 1024
    overlap: int = 100
    min_chunk_size: int = 50


def _is_table(text: str) -> bool:
    """Detect if text block is a table."""
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False
    
    table_indicators = [
        all('|' in line or line.strip() == '' for line in lines),
        all('\t' in line or line.strip() == '' for line in lines),
        all(',' in line for line in lines if line.strip()),
    ]
    return any(table_indicators)


def _is_code_block(text: str) -> bool:
    """Detect if text block is code."""
    patterns = [
        r'```',
        r'^(def |class |function |const |let |var |import |from )',
        r'[{}()\[\];]',
    ]
    code_chars = sum(1 for line in text.split('\n') for char in line if char in '{}()[];:')
    return any(re.search(pattern, text) for pattern in patterns) or code_chars > len(text) * 0.1


def _is_list_block(text: str) -> bool:
    """Detect if text block is a list."""
    lines = text.strip().split('\n')
    if len(lines) < 2:
        return False
    
    list_markers = ['- ', '* ', '• ', '+ ']
    list_lines = sum(1 for line in lines if any(line.strip().startswith(m) for m in list_markers))
    return list_lines / len(lines) > 0.7


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using improved heuristics.
    Handles common abbreviations and edge cases.
    """
    abbreviations = {
        'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Sr.', 'Jr.',
        'Ph.D.', 'M.D.', 'B.A.', 'M.A.', 'i.e.', 'e.g.',
        'etc.', 'vs.', 'vol.', 'pp.'
    }
    
    text = re.sub(r'(\d)\.(\d)', r'\1<DOT>\2', text)
    
    for abbr in abbreviations:
        text = text.replace(abbr, abbr.replace('.', '<ABBR>'))
    
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    sentences = [s.replace('<ABBR>', '.').replace('<DOT>', '.') for s in sentences]
    
    return [s.strip() for s in sentences if s.strip()]


def _get_tokens(text: str) -> List[str]:
    """Tokenize text into words."""
    return text.split()


def _semantic_chunk(text: str, max_chunk_size: int, overlap: int) -> List[Chunk]:
    """
    Split text into semantic chunks using sentence boundaries.
    """
    sentences = _split_sentences(text)
    
    if not sentences:
        if text.strip():
            return [Chunk(text=text.strip(), start_idx=0, end_idx=len(text), chunk_num=0)]
        return []
    
    chunks = []
    current_chunk = []
    current_size = 0
    chunk_num = 0
    start_idx = 0
    
    for sentence in sentences:
        sentence_tokens = _get_tokens(sentence)
        sentence_size = len(sentence_tokens)
        
        if current_size + sentence_size <= max_chunk_size:
            current_chunk.append(sentence)
            current_size += sentence_size
        else:
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                end_idx = start_idx + len(chunk_text)
                chunks.append(Chunk(
                    text=chunk_text,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    chunk_num=chunk_num
                ))
                chunk_num += 1
                
                overlap_sentences = []
                overlap_size = 0
                for sent in reversed(current_chunk):
                    sent_size = len(_get_tokens(sent))
                    if overlap_size + sent_size <= overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_size += sent_size
                    else:
                        break
                
                current_chunk = overlap_sentences + [sentence]
                current_size = overlap_size + sentence_size
                start_idx = end_idx - len(' '.join(overlap_sentences))
            else:
                current_chunk = [sentence]
                current_size = sentence_size
    
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        end_idx = start_idx + len(chunk_text)
        chunks.append(Chunk(
            text=chunk_text,
            start_idx=start_idx,
            end_idx=end_idx,
            chunk_num=chunk_num
        ))
    
    return chunks


def _fixed_chunk(text: str, max_chunk_size: int, overlap: int) -> List[Chunk]:
    """
    Split text into fixed-size chunks by token count.
    """
    tokens = _get_tokens(text)
    
    if not tokens:
        return []
    
    chunks = []
    chunk_num = 0
    i = 0
    
    while i < len(tokens):
        end = min(i + max_chunk_size, len(tokens))
        chunk_tokens = tokens[i:end]
        chunk_text = ' '.join(chunk_tokens)
        
        start_idx = sum(len(token) + 1 for token in tokens[:i]) - 1 if i > 0 else 0
        end_idx = start_idx + len(chunk_text)
        
        chunks.append(Chunk(
            text=chunk_text,
            start_idx=start_idx,
            end_idx=end_idx,
            chunk_num=chunk_num
        ))
        chunk_num += 1
        
        # Move forward: use overlap if specified, but ensure we always progress
        if end >= len(tokens):
            break
        
        next_i = end - overlap if overlap > 0 else end
        # Ensure minimum progress of at least 1
        i = max(next_i, i + 1)
    
    return chunks


def chunk_text(
    text: str,
    strategy: Literal['semantic', 'fixed'] = 'semantic',
    max_chunk_size: int = 1024,
    overlap: int = 100,
) -> List[Chunk]:
    """
    Split text into chunks using specified strategy.
    
    Args:
        text: Input text to chunk
        strategy: 'semantic' (sentence boundaries) or 'fixed' (token count)
        max_chunk_size: Maximum tokens per chunk
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of Chunk objects with metadata
        
    Raises:
        ValueError: If strategy is invalid or parameters are invalid
    """
    if not text or not text.strip():
        return []
    
    if strategy not in ('semantic', 'fixed'):
        raise ValueError(f"Invalid strategy: {strategy}. Must be 'semantic' or 'fixed'")
    
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be positive")
    
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    
    if overlap >= max_chunk_size:
        raise ValueError("overlap must be less than max_chunk_size")
    
    text = text.strip()
    
    if strategy == 'semantic':
        chunks = _semantic_chunk(text, max_chunk_size, overlap)
    else:
        chunks = _fixed_chunk(text, max_chunk_size, overlap)
    
    return chunks


def chunk_document(
    text: str,
    strategy: Literal['semantic', 'fixed'] = 'semantic',
    max_chunk_size: int = 1024,
    overlap: int = 100,
    handle_edge_cases: bool = True,
) -> List[Chunk]:
    """
    Chunk document text with optional edge case handling.
    
    Args:
        text: Input document text
        strategy: Chunking strategy
        max_chunk_size: Maximum chunk size in tokens
        overlap: Token overlap between chunks
        handle_edge_cases: Whether to detect and preserve tables/code/lists
        
    Returns:
        List of Chunk objects
    """
    if handle_edge_cases:
        sections = _separate_edge_cases(text)
        all_chunks = []
        current_idx = 0
        chunk_num = 0
        
        for section_type, section_text in sections:
            if section_type == 'edge_case':
                chunk = Chunk(
                    text=section_text,
                    start_idx=current_idx,
                    end_idx=current_idx + len(section_text),
                    chunk_num=chunk_num
                )
                all_chunks.append(chunk)
                chunk_num += 1
            else:
                chunks = chunk_text(section_text, strategy, max_chunk_size, overlap)
                for i, chunk in enumerate(chunks):
                    new_chunk = Chunk(
                        text=chunk.text,
                        start_idx=current_idx + chunk.start_idx,
                        end_idx=current_idx + chunk.end_idx,
                        chunk_num=chunk_num
                    )
                    all_chunks.append(new_chunk)
                    chunk_num += 1
            
            current_idx += len(section_text)
        
        return all_chunks
    else:
        return chunk_text(text, strategy, max_chunk_size, overlap)


def _separate_edge_cases(text: str) -> List[tuple]:
    """
    Separate text into regular and edge case sections.
    Returns list of (type, content) tuples where type is 'edge_case' or 'text'.
    """
    sections = []
    current_pos = 0
    
    code_pattern = r'```.*?```'
    for match in re.finditer(code_pattern, text, re.DOTALL):
        if match.start() > current_pos:
            sections.append(('text', text[current_pos:match.start()]))
        sections.append(('edge_case', match.group()))
        current_pos = match.end()
    
    if current_pos < len(text):
        sections.append(('text', text[current_pos:]))
    
    if not sections:
        sections = [('text', text)]
    
    return sections
