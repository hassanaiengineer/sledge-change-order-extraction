"""Run OCR, when needed, and then field extraction.

Uploaded bytes stay in memory for the request and are not written to disk.
"""

from __future__ import annotations

from .extract import ExtractionError, extract_fields
from .fields import validate_fields
from .ocr import (
    InvalidPDFError,
    OCRDependencyError,
    OCRError,
    PDFExtractionError,
    extract_pdf,
)

MAX_PDF_BYTES = 15 * 1024 * 1024
MAX_PAGES = 30


def extract_change_order(
    *,
    text: str | None = None,
    pdf: bytes | None = None,
    fields: object,
    client,
) -> dict:
    """Extract fields from pasted text or from PDF bytes.

    PDF text comes from ``extract_pdf``. The PDF file is not sent to the model.
    ``client.complete`` performs one model call.
    """
    validate_fields(fields)
    document, warnings = read_document(text=text, pdf=pdf)
    result = extract_fields(document, fields, client.complete, model=client.model)
    result["document_warnings"] = warnings + result["document_warnings"]
    return result


def read_document(*, text: str | None = None, pdf: bytes | None = None) -> tuple[str, list[str]]:
    """Return document text and warnings. A PDF is read with ``extract_pdf``."""
    if pdf is not None and text is not None:
        raise ExtractionError("Provide pasted text or a PDF, not both.")
    if pdf is None and text is None:
        raise ExtractionError("Provide pasted text or a PDF.")
    if pdf is not None:
        return _text_from_pdf(pdf)
    if not isinstance(text, str) or not text.strip():
        raise ExtractionError("Document text is empty.")
    return text, []


def _text_from_pdf(pdf: bytes) -> tuple[str, list[str]]:
    if not pdf:
        raise ExtractionError("PDF is empty.")
    if len(pdf) > MAX_PDF_BYTES:
        raise ExtractionError("PDF is larger than 15 MB.")
    try:
        extracted = extract_pdf(pdf)
    except (InvalidPDFError, OCRDependencyError, OCRError, PDFExtractionError) as exc:
        raise ExtractionError(str(exc)) from exc

    if extracted.page_count > MAX_PAGES:
        raise ExtractionError(
            f"PDF has {extracted.page_count} pages. The limit is {MAX_PAGES}."
        )
    document = extracted.full_text()
    if extracted.page_count == 0 or not document.strip():
        raise ExtractionError("The PDF produced no text.")

    warnings = []
    ocr_pages = [page.page_number for page in extracted.pages if page.source == "ocr"]
    if ocr_pages:
        numbers = ", ".join(str(number) for number in ocr_pages)
        warnings.append(
            f"Pages {numbers} were read with OCR. Scanned text can be wrong."
        )
    return document, warnings
