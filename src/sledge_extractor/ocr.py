"""Direct PDF text extraction.

Embedded text comes from PyMuPDF. A page is sent to Tesseract when it contains
an image and its embedded text is shorter than ``min_digital_chars``. An image
that covers at least 20% of a page is also read, and the text layer is kept.
The rendered page is not preprocessed. ``ocr_layout.py`` is a separate service.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pymupdf

PDFInput = str | Path | bytes | bytearray
PageSource = Literal["digital", "ocr", "empty"]

# A drawing or scan, not a logo or a small photo. Measured from the image box.
_LARGE_IMAGE_FRACTION = 0.20
_MIN_DIGITAL_CHARS = 40

_WINDOWS_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
_TESSERACT_HELP = (
    "Tesseract OCR is not available. Install it, add it to PATH, "
    "or set TESSERACT_CMD to the full path of tesseract.exe. "
    r"On Windows the usual location is C:\Program Files\Tesseract-OCR\tesseract.exe."
)


class PDFExtractionError(Exception):
    """Base error for PDF text extraction."""


class InvalidPDFError(PDFExtractionError):
    """The input is empty, encrypted, or not a readable PDF."""


class OCRDependencyError(PDFExtractionError):
    """A page needs OCR, but Tesseract or pytesseract is unavailable."""


class OCRError(PDFExtractionError):
    """Tesseract failed while reading a page."""


@dataclass(frozen=True)
class PageText:
    """Text from one PDF page.

    ``page_number`` is 1-based and matches the page in the PDF.
    ``source`` is ``digital`` for embedded text, ``ocr`` for Tesseract,
    or ``empty`` when that page produced no text.
    """

    page_number: int
    text: str
    source: PageSource


@dataclass(frozen=True)
class ExtractionResult:
    """Page-ordered text from one PDF. Empty pages are kept."""

    pages: tuple[PageText, ...]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    def full_text(self) -> str:
        """Join pages in order, with a marker in front of each page."""
        blocks = []
        for page in self.pages:
            header = f"--- Page {page.page_number} ---"
            blocks.append(f"{header}\n{page.text}" if page.text else header)
        return "\n\n".join(blocks)


def extract_pdf(
    source: PDFInput,
    *,
    min_digital_chars: int = _MIN_DIGITAL_CHARS,
    dpi: int = 300,
    lang: str = "eng",
) -> ExtractionResult:
    """Extract text from a PDF path or from raw PDF bytes.

    Text-only pages keep their embedded text, however short. A page that
    contains an image uses embedded text when it has at least
    ``min_digital_chars`` characters. If one image covers at least 20% of
    the page, that region is read with Tesseract and new lines are added
    to the text layer. A smaller image is rendered at ``dpi`` only when the
    embedded text is shorter than ``min_digital_chars``. Tesseract is not
    loaded for documents that never need it.

    Raises:
        FileNotFoundError: ``source`` is a path that does not exist.
        InvalidPDFError: the input is not a readable, unencrypted PDF.
        OCRDependencyError: OCR is required and Tesseract is not installed.
        OCRError: Tesseract failed on a page. Earlier pages are not returned.
        ValueError: ``min_digital_chars``, ``dpi``, or ``lang`` is invalid.
    """
    if min_digital_chars < 1:
        raise ValueError("min_digital_chars must be at least 1.")
    if dpi < 72:
        raise ValueError("dpi must be at least 72.")
    if not lang.strip():
        raise ValueError("lang must be a Tesseract language code, such as 'eng'.")

    document = _open_pdf(source)
    try:
        pages: list[PageText] = []
        engine = None
        for index in range(document.page_count):
            page = document[index]
            page_number = index + 1
            embedded = page.get_text("text", sort=True).strip()
            image_rect = _large_image_rect(page)
            # A large drawing stays in the image even when the page has a text layer.
            if image_rect is None and (
                len(embedded) >= min_digital_chars or not page.get_images()
            ):
                kind: PageSource = "digital" if embedded else "empty"
                pages.append(PageText(page_number, embedded, kind))
                continue

            if engine is None:
                engine = _require_tesseract()
            pages.append(PageText(
                page_number,
                _image_page_text(
                    engine,
                    page,
                    embedded,
                    image_rect=image_rect,
                    page_number=page_number,
                    dpi=dpi,
                    lang=lang,
                    min_digital_chars=min_digital_chars,
                ),
                "ocr",
            ))
        return ExtractionResult(tuple(pages))
    finally:
        document.close()


def _open_pdf(source: PDFInput) -> pymupdf.Document:
    if isinstance(source, (bytes, bytearray)):
        payload = bytes(source)
        if not payload:
            raise InvalidPDFError("PDF input is empty.")
        try:
            document = pymupdf.open(stream=payload, filetype="pdf")
        except Exception as exc:
            raise InvalidPDFError(f"Could not open PDF: {exc}") from exc
    else:
        if isinstance(source, str) and source.lstrip().startswith("%PDF"):
            raise InvalidPDFError(
                "Strings are file paths. Pass raw PDF bytes, or a path to a PDF file."
            )
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise InvalidPDFError(f"Path is not a file: {path}")
        try:
            document = pymupdf.open(path, filetype="pdf")
        except Exception as exc:
            raise InvalidPDFError(f"Could not open '{path}': {exc}") from exc

    if document.needs_pass:
        document.close()
        raise InvalidPDFError(
            "PDF is encrypted and cannot be read without a password."
        )
    if not document.is_pdf:
        document.close()
        raise InvalidPDFError("Input is not a PDF.")
    return document


def _import_pytesseract():
    try:
        import pytesseract
    except ImportError as exc:
        raise OCRDependencyError(
            "pytesseract is not installed. "
            "Install this project and the Tesseract program."
        ) from exc
    _configure_tesseract(pytesseract)
    return pytesseract


def _configure_tesseract(pytesseract) -> None:
    configured = os.environ.get("TESSERACT_CMD")
    if configured:
        pytesseract.pytesseract.tesseract_cmd = configured
        return
    if shutil.which("tesseract"):
        return
    if _WINDOWS_TESSERACT.is_file():
        pytesseract.pytesseract.tesseract_cmd = str(_WINDOWS_TESSERACT)


def _require_tesseract():
    engine = _import_pytesseract()
    try:
        engine.get_tesseract_version()
    except Exception as exc:
        raise OCRDependencyError(_TESSERACT_HELP) from exc
    return engine


def _read_image(engine, image, *, page_number: int, lang: str) -> str:
    """Read one PIL image with Tesseract and return stripped text."""
    try:
        raw = engine.image_to_string(image, lang=lang)
    except ImportError as exc:
        raise OCRDependencyError(
            "Pillow is required to OCR scanned pages. "
            "Install this project's dependencies."
        ) from exc
    except Exception as exc:
        missing = getattr(engine, "TesseractNotFoundError", ())
        if isinstance(exc, missing):
            raise OCRDependencyError(_TESSERACT_HELP) from exc
        raise OCRError(f"OCR failed on page {page_number}: {exc}") from exc

    if not isinstance(raw, str):
        raise OCRError(
            f"OCR failed on page {page_number}: Tesseract did not return text."
        )
    return raw.strip()


def _ocr_page(engine, page: pymupdf.Page, *, page_number: int, dpi: int, lang: str) -> str:
    image = page.get_pixmap(dpi=dpi).pil_image()
    return _read_image(engine, image, page_number=page_number, lang=lang)


def _large_image_rect(page: pymupdf.Page) -> pymupdf.Rect | None:
    """Return the largest image box when it covers a substantial part of the page."""
    area = float(page.rect.width * page.rect.height)
    if area <= 0:
        return None

    largest = 0.0
    rect = None
    for info in page.get_image_info():
        bbox = info.get("bbox")
        if not bbox:
            continue
        box = pymupdf.Rect(bbox)
        fraction = (box.width * box.height) / area
        if fraction > largest:
            largest = fraction
            rect = box
    if rect is None or largest < _LARGE_IMAGE_FRACTION:
        return None
    return rect


def _image_page_text(
    engine,
    page: pymupdf.Page,
    embedded: str,
    *,
    image_rect: pymupdf.Rect | None,
    page_number: int,
    dpi: int,
    lang: str,
    min_digital_chars: int,
) -> str:
    """Read a drawing region when the page already has a text layer.

    A short page is rendered whole. Cropping it drops labels that sit
    outside the largest image box.
    """
    if image_rect is not None and len(embedded) >= min_digital_chars:
        recognized = _ocr_clip(
            engine,
            page,
            image_rect,
            page_number=page_number,
            dpi=dpi,
            lang=lang,
        )
        return _append_new_lines(embedded, recognized)
    return _ocr_page(engine, page, page_number=page_number, dpi=dpi, lang=lang)


def _ocr_clip(
    engine,
    page: pymupdf.Page,
    rect: pymupdf.Rect,
    *,
    page_number: int,
    dpi: int,
    lang: str,
) -> str:
    """OCR one image region instead of the whole page."""
    image = page.get_pixmap(dpi=dpi, clip=rect, alpha=False).pil_image()
    return _read_image(engine, image, page_number=page_number, lang=lang)


def _append_new_lines(embedded: str, recognized: str) -> str:
    """Keep the text layer and add OCR lines that are not already in it."""
    new_lines = [
        line.strip()
        for line in recognized.splitlines()
        if line.strip() and line.strip() not in embedded
    ]
    if not new_lines:
        return embedded
    if not embedded:
        return "\n".join(new_lines)
    return embedded + "\n" + "\n".join(new_lines)
