import os
import io
import cv2
import numpy as np
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber
import logging

logger = logging.getLogger(__name__)

# Configure pytesseract path if specified in environment or defaults
# On Windows, if Tesseract is in standard paths, we can try to auto-locate it
if os.name == "nt":
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break

class ExtractionResult:
    def __init__(self, raw_text: str, page_count: int, confidence: float, method: str):
        self.raw_text = raw_text
        self.page_count = page_count
        self.confidence = confidence
        self.method = method  # "pdf_text", "pdf_ocr", "image_ocr"

class DocumentExtractor:
    def extract(self, file_path: str, mime_type: str) -> ExtractionResult:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        if mime_type == "application/pdf":
            # 1. Try PyMuPDF text extraction
            text, page_count = self._extract_pdf_text_fitz(file_path)
            
            # If text is extremely short, it's likely a scanned PDF
            if len(text.strip()) < 100:
                logger.info(f"PDF text content too short ({len(text)} chars). Falling back to OCR.")
                text, page_count = self._ocr_pdf_pages(file_path)
                return ExtractionResult(raw_text=text, page_count=page_count, confidence=0.8, method="pdf_ocr")
            
            return ExtractionResult(raw_text=text, page_count=page_count, confidence=1.0, method="pdf_text")
        else:
            # Image OCR
            text = self._ocr_image(file_path)
            return ExtractionResult(raw_text=text, page_count=1, confidence=0.7, method="image_ocr")

    def _extract_pdf_text_fitz(self, file_path: str) -> tuple[str, int]:
        """Extract text layer using PyMuPDF (fitz) with a fallback to pdfplumber."""
        text = ""
        page_count = 0
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)
            for page in doc:
                text += page.get_text() + "\n"
            doc.close()
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {str(e)}. Trying pdfplumber fallback.")
            try:
                with pdfplumber.open(file_path) as pdf:
                    page_count = len(pdf.pages)
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n"
            except Exception as ex:
                logger.error(f"pdfplumber fallback also failed: {str(ex)}")
                raise Exception(f"Failed to parse PDF text: {str(ex)}")

        return text, page_count

    def _ocr_pdf_pages(self, file_path: str) -> tuple[str, int]:
        """Convert each PDF page to a high-DPI image in memory and run OCR."""
        text = ""
        page_count = 0
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                # Zoom matrix to scale page up (2x zoom = 144 DPI) for better OCR accuracy
                zoom = 2
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Check pixel dimensions and channels
                # Convert fitz pixmap to RGB numpy array
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:  # RGBA
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:  # RGB
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                page_text = self._ocr_cv2_image(img_array)
                text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
            doc.close()
        except Exception as e:
            logger.error(f"OCR PDF failed: {str(e)}")
            text = f"[OCR Failed: {str(e)}]"
        
        return text, page_count

    def _ocr_image(self, path: str) -> str:
        """Load image and run preprocessing + OCR."""
        img = cv2.imread(path)
        if img is None:
            raise Exception(f"Failed to read image at path: {path}")
        return self._ocr_cv2_image(img)

    def _ocr_cv2_image(self, img: np.ndarray) -> str:
        """Run OCR on preprocessed CV2 image."""
        try:
            # 1. Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 2. Thresholding (OTSU + Binary)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # 3. Denoising
            denoised = cv2.fastNlMeansDenoising(thresh)
            
            # Send to pytesseract
            config = '--psm 6 --oem 3'
            return pytesseract.image_to_string(denoised, config=config)
        except pytesseract.TesseractNotFoundError:
            msg = "[OCR Error: Tesseract-OCR is not installed or configured in system path. Returning raw image metadata.]"
            logger.warning(msg)
            return msg
        except Exception as e:
            logger.error(f"OpenCV / Pytesseract processing failed: {str(e)}")
            # Fallback to direct pytesseract on original image if cv2 pipeline fails
            try:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                return pytesseract.image_to_string(pil_img)
            except Exception as inner_ex:
                return f"[OCR Exception: {str(inner_ex)}]"
