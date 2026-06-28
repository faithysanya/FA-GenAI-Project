"""
Test script for document parsing module.
Tests all parser functions and metadata extraction.
"""

from app.document_processing import (
    parse_document,
    parse_txt,
    parse_csv,
    extract_metadata,
)


def test_parse_txt():
    """Test TXT file parsing."""
    print("=" * 70)
    print("TEST 1: TXT File Parsing")
    print("=" * 70)
    
    file_path = "data/uploads/test_sample.txt"
    
    try:
        print(f"\nTesting: {file_path}")
        text = parse_txt(file_path)
        
        print("\n✓ Successfully parsed TXT file")
        print(f"  Text length: {len(text)} characters")
        print(f"  First 200 characters:\n{text[:200]}...")
        
        # Also test via parse_document
        text2 = parse_document(file_path)
        assert text == text2, "parse_document() result differs from parse_txt()"
        print("✓ parse_document() with automatic type detection works")
        
        return True
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_parse_csv():
    """Test CSV file parsing."""
    print("\n" + "=" * 70)
    print("TEST 2: CSV File Parsing")
    print("=" * 70)
    
    file_path = "data/uploads/test_sample.csv"
    
    try:
        print(f"\nTesting: {file_path}")
        text = parse_csv(file_path)
        
        print("\n✓ Successfully parsed CSV file")
        print(f"  Text length: {len(text)} characters")
        print(f"\nFormatted output:\n{text[:500]}...")
        
        # Also test via parse_document
        text2 = parse_document(file_path)
        assert text == text2, "parse_document() result differs from parse_csv()"
        print("\n✓ parse_document() with automatic type detection works")
        
        return True
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_parse_excel():
    """Test Excel file parsing (if available)."""
    print("\n" + "=" * 70)
    print("TEST 3: Excel File Parsing")
    print("=" * 70)
    
    # Create a simple Excel file for testing
    try:
        from openpyxl import Workbook
        
        excel_path = "data/uploads/test_sample.xlsx"
        
        print(f"\nCreating test Excel file: {excel_path}")
        
        # Create workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Employees"
        
        # Add headers
        headers = ["Name", "Department", "Salary", "Start_Date"]
        ws.append(headers)
        
        # Add sample data
        data = [
            ["Alice Wong", "Engineering", 100000, "2023-01-15"],
            ["Bob Chen", "Marketing", 80000, "2023-02-01"],
            ["Carol Lee", "Sales", 90000, "2022-12-20"],
        ]
        for row in data:
            ws.append(row)
        
        # Add second sheet
        ws2 = wb.create_sheet("Summary")
        ws2.append(["Total Employees", len(data)])
        ws2.append(["Average Salary", sum(row[2] for row in data) / len(data)])
        
        wb.save(excel_path)
        print("✓ Created test Excel file")
        
        # Now test parsing
        from app.document_processing import parse_excel
        
        print(f"\nParsing Excel file: {excel_path}")
        text = parse_excel(excel_path)
        
        print("✓ Successfully parsed Excel file")
        print(f"  Text length: {len(text)} characters")
        print(f"\nFormatted output (first 600 chars):\n{text[:600]}...")
        
        # Also test via parse_document
        text2 = parse_document(excel_path)
        assert text == text2, "parse_document() result differs from parse_excel()"
        print("\n✓ parse_document() with automatic type detection works")
        
        return True
    except ImportError:
        print("⊘ openpyxl not available, skipping Excel test")
        return True
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_metadata_extraction():
    """Test metadata extraction."""
    print("\n" + "=" * 70)
    print("TEST 4: Metadata Extraction")
    print("=" * 70)
    
    try:
        txt_file = "data/uploads/test_sample.txt"
        csv_file = "data/uploads/test_sample.csv"
        
        print("\nExtracting metadata for TXT file:")
        meta_txt = extract_metadata(txt_file, "txt")
        print(f"  Filename: {meta_txt.filename}")
        print(f"  File Size: {meta_txt.file_size} bytes ({meta_txt.file_size_mb} MB)")
        print(f"  File Type: {meta_txt.file_type}")
        print(f"  Modified Date: {meta_txt.modified_date}")
        print(f"  Full Path: {meta_txt.full_path}")
        
        print("\nExtracting metadata for CSV file:")
        meta_csv = extract_metadata(csv_file, "csv")
        print(f"  Filename: {meta_csv.filename}")
        print(f"  File Size: {meta_csv.file_size} bytes ({meta_csv.file_size_mb} MB)")
        print(f"  File Type: {meta_csv.file_type}")
        print(f"  Modified Date: {meta_csv.modified_date}")
        
        print("\n✓ Metadata extraction successful")
        
        return True
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling."""
    print("\n" + "=" * 70)
    print("TEST 5: Error Handling")
    print("=" * 70)
    
    try:
        # Test non-existent file
        print("\nTesting non-existent file:")
        try:
            parse_document("data/uploads/nonexistent.txt")
            print("✗ Should have raised FileNotFoundError")
            return False
        except FileNotFoundError as e:
            print(f"✓ Correctly raised FileNotFoundError: {str(e)}")
        
        # Test unsupported file type
        print("\nTesting unsupported file type:")
        try:
            parse_document("data/uploads/test_sample.txt", "unsupported")
            print("✗ Should have raised ValueError")
            return False
        except ValueError as e:
            print(f"✓ Correctly raised ValueError: {str(e)}")
        
        print("\n✓ Error handling works correctly")
        return True
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("█" * 70)
    print("█ Document Parser Module - Comprehensive Test Suite")
    print("█" * 70)
    
    results = []
    
    results.append(("TXT Parsing", test_parse_txt()))
    results.append(("CSV Parsing", test_parse_csv()))
    results.append(("Excel Parsing", test_parse_excel()))
    results.append(("Metadata Extraction", test_metadata_extraction()))
    results.append(("Error Handling", test_error_handling()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<40} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed successfully!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
