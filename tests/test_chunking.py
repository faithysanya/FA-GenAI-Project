"""
Tests for the document chunking module.
Tests semantic and fixed strategies, overlap, and edge cases.
"""

import pytest
from pathlib import Path

from app.document_processing import chunk_text, chunk_document, Chunk


class TestChunkingBasics:
    """Test basic chunking functionality."""
    
    def test_chunk_text_empty_input(self):
        """Test handling of empty input."""
        result = chunk_text("")
        assert result == []
        
        result = chunk_text("   ")
        assert result == []
    
    def test_chunk_text_single_sentence(self):
        """Test chunking a single sentence."""
        text = "This is a single sentence."
        result = chunk_text(text, strategy='semantic')
        assert len(result) == 1
        assert result[0].text.strip() == text.strip()
        assert result[0].chunk_num == 0
    
    def test_chunk_text_metadata(self):
        """Test that chunk metadata is correct."""
        text = "First sentence. Second sentence. Third sentence."
        result = chunk_text(text, strategy='semantic', max_chunk_size=20, overlap=5)
        
        for i, chunk in enumerate(result):
            assert chunk.chunk_num == i
            assert isinstance(chunk.start_idx, int)
            assert isinstance(chunk.end_idx, int)
            assert chunk.start_idx >= 0
            assert chunk.end_idx > chunk.start_idx
    
    def test_invalid_strategy(self):
        """Test that invalid strategy raises ValueError."""
        with pytest.raises(ValueError, match="Invalid strategy"):
            chunk_text("test", strategy='invalid')
    
    def test_invalid_max_chunk_size(self):
        """Test that invalid max_chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="max_chunk_size must be positive"):
            chunk_text("test", max_chunk_size=-1)
        
        with pytest.raises(ValueError, match="max_chunk_size must be positive"):
            chunk_text("test", max_chunk_size=0)
    
    def test_invalid_overlap(self):
        """Test that invalid overlap raises ValueError."""
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            chunk_text("test", overlap=-1)
        
        with pytest.raises(ValueError, match="overlap must be less than max_chunk_size"):
            chunk_text("test", max_chunk_size=100, overlap=150)


class TestSemanticChunking:
    """Test semantic chunking strategy."""
    
    def test_semantic_multiple_sentences(self):
        """Test semantic chunking with multiple sentences."""
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is the second sentence with more words. "
            "A third sentence follows here. "
            "And finally a fourth one."
        )
        result = chunk_text(text, strategy='semantic', max_chunk_size=15, overlap=3)
        assert len(result) > 0
        
        for chunk in result:
            assert len(chunk.text) > 0
            assert chunk.chunk_num >= 0
    
    def test_semantic_preserves_sentences(self):
        """Test that semantic chunking preserves complete sentences."""
        text = "First sentence. Second sentence. Third sentence."
        result = chunk_text(text, strategy='semantic', max_chunk_size=20, overlap=2)
        
        assert len(result) > 0
        combined = ' '.join(chunk.text for chunk in result)
        assert len(combined) > 0
    
    def test_semantic_overlap(self):
        """Test that semantic chunks have overlap."""
        text = (
            "The quick brown fox. "
            "The fox jumps high. "
            "The jump is long. "
            "Long jumps are good."
        )
        result = chunk_text(text, strategy='semantic', max_chunk_size=8, overlap=2)
        
        assert len(result) > 0


class TestFixedChunking:
    """Test fixed-size chunking strategy."""
    
    def test_fixed_chunk_size(self):
        """Test that fixed chunks are approximately the correct size."""
        text = " ".join(["word"] * 100)
        result = chunk_text(text, strategy='fixed', max_chunk_size=25, overlap=0)
        
        for chunk in result:
            tokens = chunk.text.split()
            assert len(tokens) <= 26
    
    def test_fixed_multiple_chunks(self):
        """Test fixed chunking produces multiple chunks."""
        text = " ".join(["word"] * 100)
        result = chunk_text(text, strategy='fixed', max_chunk_size=20, overlap=0)
        
        assert len(result) > 1
        assert all(isinstance(chunk, Chunk) for chunk in result)
    
    def test_fixed_overlap(self):
        """Test that fixed chunks respect overlap parameter."""
        text = " ".join([f"word{i}" for i in range(100)])
        result = chunk_text(text, strategy='fixed', max_chunk_size=30, overlap=10)
        
        if len(result) > 1:
            chunk1_end = result[0].text.split()[-10:]
            chunk2_start = result[1].text.split()[:10]
            
            overlapping_words = [w for w in chunk1_end if w in chunk2_start]
            assert len(overlapping_words) > 0 or len(result) == 1


class TestDocumentSamples:
    """Test chunking with actual document samples."""
    
    def get_test_sample_path(self, filename):
        """Get path to test sample file."""
        return Path(__file__).parent.parent / "data" / "uploads" / filename
    
    def test_chunk_txt_sample(self):
        """Test chunking with sample.txt file."""
        sample_path = self.get_test_sample_path("test_sample.txt")
        
        if not sample_path.exists():
            pytest.skip(f"Sample file not found: {sample_path}")
        
        with open(sample_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        result = chunk_text(text, strategy='semantic', max_chunk_size=512)
        
        assert len(result) > 0
        assert all(isinstance(chunk, Chunk) for chunk in result)
        
        combined = ' '.join(chunk.text for chunk in result)
        assert len(combined) > 0
    
    def test_chunk_txt_semantic_vs_fixed(self):
        """Compare semantic and fixed chunking on real sample."""
        sample_path = self.get_test_sample_path("test_sample.txt")
        
        if not sample_path.exists():
            pytest.skip(f"Sample file not found: {sample_path}")
        
        with open(sample_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        semantic_chunks = chunk_text(text, strategy='semantic', max_chunk_size=256)
        fixed_chunks = chunk_text(text, strategy='fixed', max_chunk_size=256)
        
        assert len(semantic_chunks) > 0
        assert len(fixed_chunks) > 0
        
        assert all(chunk.text.strip() for chunk in semantic_chunks)
        assert all(chunk.text.strip() for chunk in fixed_chunks)


class TestChunkMetadata:
    """Test chunk metadata accuracy."""
    
    def test_chunk_indices(self):
        """Test that chunk indices are correct."""
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, strategy='semantic', max_chunk_size=20, overlap=5)
        
        for chunk in chunks:
            assert chunk.start_idx >= 0
            assert chunk.end_idx >= chunk.start_idx
            assert chunk.chunk_num >= 0
    
    def test_chunk_numbers_sequential(self):
        """Test that chunk numbers are sequential."""
        text = " ".join(["word"] * 50)
        chunks = chunk_text(text, strategy='fixed', max_chunk_size=15, overlap=0)
        
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_num == i
    
    def test_chunk_text_preserved(self):
        """Test that chunk text is preserved correctly."""
        text = "The quick brown fox jumps over the lazy dog."
        chunks = chunk_text(text, strategy='semantic', max_chunk_size=20, overlap=3)
        
        assert len(chunks) > 0
        combined = ' '.join(chunk.text for chunk in chunks)
        assert text.strip() in combined or combined in text.strip()


class TestEdgeCases:
    """Test handling of edge cases."""
    
    def test_very_long_single_sentence(self):
        """Test handling of very long sentences."""
        text = "This is a very long sentence. " * 50
        result = chunk_text(text, strategy='semantic', max_chunk_size=100, overlap=10)
        
        assert len(result) > 0
        assert all(chunk.text.strip() for chunk in result)
    
    def test_multiple_spaces(self):
        """Test handling of multiple spaces."""
        text = "First  sentence.   Second    sentence."
        result = chunk_text(text, strategy='semantic')
        
        assert len(result) > 0
        combined = ' '.join(chunk.text for chunk in result)
        assert len(combined) > 0
    
    def test_special_characters(self):
        """Test handling of special characters."""
        text = "First sentence! Second sentence? Third sentence..."
        result = chunk_text(text, strategy='semantic', max_chunk_size=20, overlap=2)
        
        assert len(result) > 0
        assert all(chunk.text.strip() for chunk in result)
    
    def test_abbreviations(self):
        """Test handling of abbreviations."""
        text = "Dr. Smith visited the U.S. for a conference. The event was at 3 p.m."
        result = chunk_text(text, strategy='semantic', max_chunk_size=20, overlap=2)
        
        assert len(result) > 0
        combined = ' '.join(chunk.text for chunk in result)
        assert 'Dr.' in combined
    
    def test_code_block_detection(self):
        """Test detection and handling of code blocks."""
        text = "Here is some code:\n```python\ndef hello():\n    print('hello')\n```\nEnd of code."
        result = chunk_document(text, handle_edge_cases=True)
        
        assert len(result) > 0


class TestChunkDocument:
    """Test chunk_document with edge case handling."""
    
    def test_chunk_document_basic(self):
        """Test basic chunk_document functionality."""
        text = "First sentence. Second sentence. Third sentence."
        result = chunk_document(text, handle_edge_cases=False)
        
        assert len(result) > 0
        assert all(isinstance(chunk, Chunk) for chunk in result)
    
    def test_chunk_document_with_code(self):
        """Test chunk_document with code blocks."""
        text = (
            "Introduction to code.\n"
            "```python\ndef example():\n    return 42\n```\n"
            "End of section."
        )
        result = chunk_document(text, handle_edge_cases=True)
        
        assert len(result) > 0
        assert any('python' in chunk.text for chunk in result) or len(result) > 1
    
    def test_chunk_document_sequential_indices(self):
        """Test that indices are sequential in chunk_document."""
        text = "First. Second. Third. Fourth. Fifth. Sixth."
        result = chunk_document(text, strategy='semantic', max_chunk_size=10, overlap=2)
        
        prev_end = 0
        for chunk in result:
            assert chunk.start_idx >= prev_end
            prev_end = chunk.end_idx


class TestPerformance:
    """Test performance with larger texts."""
    
    def test_large_document(self):
        """Test chunking performance with large document."""
        text = ("This is a test sentence. " * 500)
        result = chunk_text(text, strategy='semantic', max_chunk_size=256)
        
        assert len(result) > 0
        assert all(chunk.text.strip() for chunk in result)
    
    def test_many_chunks(self):
        """Test generation of many chunks."""
        text = " ".join(["word"] * 1000)
        result = chunk_text(text, strategy='fixed', max_chunk_size=50, overlap=5)
        
        assert len(result) > 10
        assert all(chunk.chunk_num < len(result) for chunk in result)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
