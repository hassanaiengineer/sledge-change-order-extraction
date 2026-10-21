"""Layout OCR service, separate from ``ocr.extract_pdf``.

Native text is read left column first when a wide blank gutter sits in the
middle of the page. If the whole document is shorter than
``min_document_chars``, that text is discarded and every page is denoised,
contrast-stretched, binarized, deskewed, and read with Tesseract. A page
image covering at least 20% of the page is read even when the document
stays on the native path, and the native text is kept.

Pages are rendered with PyMuPDF, so Poppler is not required.
"""

from __future__ import annotations

import pymupdf
from PIL import Image

from .ocr import (
    ExtractionResult,
    OCRDependencyError,
    OCRError,
    PDFInput,
    PageText,
    _MIN_DIGITAL_CHARS,
    _image_page_text,
    _large_image_rect,
    _open_pdf,
    _read_image,
    _require_tesseract,
)


def extract_pdf_layout(
    source: PDFInput,
    *,
    min_document_chars: int = 400,
    column_dpi: int = 200,
    ocr_dpi: int = 300,
    lang: str = "eng",
) -> ExtractionResult:
    """Extract PDF text with column order and a document-level OCR fallback.

    ``source`` is a filesystem path or raw PDF bytes, same as ``extract_pdf``.
    Native text is kept only when the document has at least
    ``min_document_chars`` characters. Otherwise every page is preprocessed
    and read with Tesseract. On the native path, an image covering at least
    20% of a page is still read. A long page keeps its digital text; a short
    page is rendered whole.

    Raises the same file and OCR errors as ``extract_pdf``. Also raises
    ``OCRDependencyError`` when OpenCV or NumPy is not installed.
    """
    if min_document_chars < 1:
        raise ValueError("min_document_chars must be at least 1.")
    if column_dpi < 72 or ocr_dpi < 72:
        raise ValueError("column_dpi and ocr_dpi must be at least 72.")
    if not lang.strip():
        raise ValueError("lang must be a Tesseract language code, such as 'eng'.")

    document = _open_pdf(source)
    try:
        if document.page_count == 0:
            return ExtractionResult(())

        _load_cv()
        pages = _read_native_pages(document, column_dpi)
        native_chars = sum(len(page.text) for page in pages)
        if native_chars >= min_document_chars:
            pages = _supplement_large_images(document, pages, ocr_dpi, lang)
            return ExtractionResult(tuple(pages))
        return ExtractionResult(tuple(_ocr_all_pages(document, ocr_dpi, lang)))
    finally:
        document.close()


def _read_native_pages(document: pymupdf.Document, dpi: int) -> list[PageText]:
    pages = []
    for index in range(document.page_count):
        text = _native_page_text(document[index], dpi)
        kind = "digital" if text else "empty"
        pages.append(PageText(index + 1, text, kind))
    return pages


def _supplement_large_images(
    document: pymupdf.Document,
    pages: list[PageText],
    dpi: int,
    lang: str,
) -> list[PageText]:
    """OCR a drawing-sized image on a page that otherwise stays digital."""
    engine = None
    updated: list[PageText] = []
    for page_text in pages:
        page = document[page_text.page_number - 1]
        rect = _large_image_rect(page)
        if rect is None:
            updated.append(page_text)
            continue
        if engine is None:
            engine = _require_tesseract()
        updated.append(PageText(
            page_text.page_number,
            _image_page_text(
                engine,
                page,
                page_text.text,
                image_rect=rect,
                page_number=page_text.page_number,
                dpi=dpi,
                lang=lang,
                min_digital_chars=_MIN_DIGITAL_CHARS,
            ),
            "ocr",
        ))
    return updated


def _ocr_all_pages(document: pymupdf.Document, dpi: int, lang: str) -> list[PageText]:
    engine = _require_tesseract()
    pages = []
    for index in range(document.page_count):
        page_number = index + 1
        image = document[index].get_pixmap(dpi=dpi).pil_image()
        processed = _preprocess_for_ocr(image, page_number)
        text = _read_image(engine, processed, page_number=page_number, lang=lang)
        pages.append(PageText(page_number, text, "ocr"))
    return pages


def _load_cv():
    cached = getattr(_load_cv, "modules", None)
    if cached is not None:
        return cached
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise OCRDependencyError(
            "The layout OCR service needs opencv-python-headless and numpy. "
            "Install this project's dependencies."
        ) from exc
    _load_cv.modules = (cv2, np)
    return _load_cv.modules


def _native_page_text(page: pymupdf.Page, dpi: int) -> str:
    split = _gutter_x(page, _page_gray(page, dpi))
    if split is None:
        return _region_text(page, page.rect)

    left = pymupdf.Rect(0, 0, split, page.rect.height)
    right = pymupdf.Rect(split, 0, page.rect.width, page.rect.height)
    left_text = _region_text(page, left)
    right_text = _region_text(page, right)
    if left_text and right_text:
        return f"{left_text}\n\n{right_text}"
    return left_text or right_text


def _page_gray(page: pymupdf.Page, dpi: int):
    _, np = _load_cv()
    image = page.get_pixmap(dpi=dpi).pil_image().convert("L")
    return np.array(image)


def _gutter_x(page: pymupdf.Page, gray) -> float | None:
    """Return a PDF x coordinate for a wide central gutter, if one exists."""
    _, np = _load_cv()
    ink = np.sum(gray < 200, axis=0)
    width = int(ink.shape[0])
    if width == 0 or float(ink.max()) == 0:
        return None

    threshold = float(ink.max()) * 0.15
    left = int(width * 0.30)
    right = int(width * 0.70)
    min_gap = max(8, int(width * 0.04))
    run = _longest_run(ink[left:right] < threshold)
    if run is None or run[1] - run[0] < min_gap:
        return None

    pixel_x = left + (run[0] + run[1]) / 2
    return pixel_x * float(page.rect.width) / width


def _longest_run(flags) -> tuple[int, int] | None:
    """Return the start and end of the longest True span, or None."""
    best_start = 0
    best_end = 0
    start = None
    for index, is_set in enumerate(flags):
        if is_set:
            if start is None:
                start = index
            continue
        if start is not None and index - start > best_end - best_start:
            best_start, best_end = start, index
        start = None
    length = len(flags)
    if start is not None and length - start > best_end - best_start:
        best_start, best_end = start, length
    if best_end == best_start:
        return None
    return best_start, best_end


def _region_text(page: pymupdf.Page, rect: pymupdf.Rect) -> str:
    blocks = page.get_text("blocks", clip=rect)
    rows = []
    for block in blocks:
        if len(block) < 5:
            continue
        text = block[4].strip()
        if text:
            rows.append((block[1], text))
    rows.sort(key=lambda row: row[0])
    return "\n".join(text for _, text in rows)


def _preprocess_for_ocr(img: Image.Image, page_number: int) -> Image.Image:
    """Denoise, raise contrast, binarize, and deskew one page image."""
    cv2, np = _load_cv()
    try:
        if img.mode == "L":
            gray = np.array(img)
        else:
            rgb = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        _, binary = cv2.threshold(
            clahe.apply(denoised),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return Image.fromarray(_deskew(binary))
    except Exception as exc:
        raise OCRError(f"OCR preprocessing failed on page {page_number}: {exc}") from exc


def _deskew(image):
    cv2, np = _load_cv()
    coords = np.column_stack(np.where(image < 128))
    if len(coords) == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    if abs(angle) < 0.5:
        return image

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
