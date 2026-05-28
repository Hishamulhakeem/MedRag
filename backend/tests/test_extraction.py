import pytest
from unittest.mock import patch, MagicMock
from backend.services.extraction import DocumentExtractor, ExtractionResult

def test_extraction_routing_pdf_text():
    extractor = DocumentExtractor()
    
    with patch.object(extractor, '_extract_pdf_text_fitz', return_value=("Some patient results " * 10, 1)) as mock_fitz:
        with patch('os.path.exists', return_value=True):
            res = extractor.extract("fake_report.pdf", "application/pdf")
            assert res.raw_text == "Some patient results " * 10
            assert res.page_count == 1
            assert res.method == "pdf_text"
            mock_fitz.assert_called_once_with("fake_report.pdf")

def test_extraction_routing_pdf_scanned_fallback():
    extractor = DocumentExtractor()
    
    # Return sparse text to trigger scanned PDF fallback
    with patch.object(extractor, '_extract_pdf_text_fitz', return_value=("  ", 2)):
        with patch.object(extractor, '_ocr_pdf_pages', return_value=("Fasting Glucose: 95 mg/dL", 2)) as mock_ocr:
            with patch('os.path.exists', return_value=True):
                res = extractor.extract("scanned_report.pdf", "application/pdf")
                assert res.raw_text == "Fasting Glucose: 95 mg/dL"
                assert res.page_count == 2
                assert res.method == "pdf_ocr"
                mock_ocr.assert_called_once_with("scanned_report.pdf")

def test_extraction_routing_image():
    extractor = DocumentExtractor()
    
    with patch.object(extractor, '_ocr_image', return_value="Cholesterol: 180 mg/dL") as mock_ocr_img:
        with patch('os.path.exists', return_value=True):
            res = extractor.extract("scan.png", "image/png")
            assert res.raw_text == "Cholesterol: 180 mg/dL"
            assert res.page_count == 1
            assert res.method == "image_ocr"
            mock_ocr_img.assert_called_once_with("scan.png")
