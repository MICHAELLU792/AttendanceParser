# python
import io
from typing import List

# PDF text extraction
try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None

# PDF rendering (PyMuPDF)
try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

# OCR (pytesseract + PIL)
try:
    import pytesseract
    from PIL import Image
except Exception:  # pragma: no cover
    pytesseract = None
    Image = None


def pdf_to_jpegs(pdf_bytes: bytes, dpi: int = 220, max_pages: int = 5) -> List[bytes]:
    if fitz is None:
        raise RuntimeError("缺少 PyMuPDF 套件，請先安裝：pip install PyMuPDF")
    images: List[bytes] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for i in range(min(len(doc), max_pages)):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            images.append(pix.tobytes("jpeg"))
    finally:
        doc.close()
    return images


def extract_pdf_text(file_like: io.BytesIO) -> str:
    if pdfplumber is None:
        raise RuntimeError("缺少 pdfplumber 套件，請先安裝依賴：pip install pdfplumber")
    text_pages: List[str] = []
    with pdfplumber.open(file_like) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text(x_tolerance=1.5, y_tolerance=1.5) or ""
            text_pages.append(extracted)
    return "\n\n".join(t for t in text_pages if t)


def ocr_image_to_text(image_bytes: bytes, lang: str = "eng+chi_tra") -> str:
    if pytesseract is None or Image is None:
        raise RuntimeError("缺少 OCR 依賴，請先安裝：pip install pytesseract pillow，並安裝 Tesseract 執行檔")
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    return pytesseract.image_to_string(image, lang=lang)


