# Document Parser Module - Quick Reference

## Module Location
`app/document_processing/`

## Main Functions

### parse_document()
```python
from app.document_processing import parse_document

# Auto-detect file type
text = parse_document('document.pdf')
text = parse_document('data.csv')
text = parse_document('spreadsheet.xlsx')

# Explicit type specification
text = parse_document('file.txt', 'txt')
text = parse_document('file.csv', 'csv')
text = parse_document('file.xlsx', 'excel')
```

### extract_metadata()
```python
from app.document_processing import extract_metadata

metadata = extract_metadata('document.pdf', 'pdf')
print(metadata.filename)       # 'document.pdf'
print(metadata.file_size)      # 15234 (bytes)
print(metadata.file_size_mb)   # 0.0145 (MB)
print(metadata.file_type)      # 'pdf'
print(metadata.modified_date)  # '2026-06-21 12:14:00'
print(metadata.full_path)      # Full absolute path
```

## Supported File Types
| Type | Extension | Parser Function |
|------|-----------|-----------------|
| PDF | .pdf | parse_pdf() |
| Text | .txt | parse_txt() |
| CSV | .csv | parse_csv() |
| Excel | .xlsx, .xls | parse_excel() |

## Features
- ✓ Automatic encoding detection (UTF-8, Latin-1)
- ✓ Multi-page PDF support
- ✓ Formatted CSV output with metadata
- ✓ Multi-sheet Excel extraction
- ✓ Comprehensive error handling
- ✓ Type hints throughout
- ✓ Production-ready code

## Test Files
Located in `data/uploads/`:
- `test_sample.txt` - Sample text document
- `test_sample.csv` - Employee data
- `test_sample.xlsx` - Multi-sheet workbook

## Test Suite
Run: `python tests/test_parsers.py`
Result: 5/5 tests passing ✓

## Integration Ready
- Export parsers.py functions ready for FastAPI routes
- Metadata available for document tracking/versioning
- Compatible with async/await patterns
- Ready for Step 4 vector database integration
