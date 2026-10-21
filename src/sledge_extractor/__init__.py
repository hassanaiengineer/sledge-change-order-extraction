"""Text extraction and validated change-order fields."""

from .extract import ConfigurationError, ExtractionError, ModelError, extract_fields
from .fields import FieldDefinitionError, validate_fields
from .ocr import (
    ExtractionResult,
    InvalidPDFError,
    OCRDependencyError,
    OCRError,
    PDFExtractionError,
    PageText,
    extract_pdf,
)
from .ocr_layout import extract_pdf_layout
from .pipeline import extract_change_order

extract_pdf_direct = extract_pdf

OCR_SERVICES = {
    "direct": extract_pdf_direct,
    "layout": extract_pdf_layout,
}

__all__ = [
    "ConfigurationError",
    "ExtractionError",
    "ExtractionResult",
    "FieldDefinitionError",
    "InvalidPDFError",
    "ModelError",
    "OCRDependencyError",
    "OCRError",
    "OCR_SERVICES",
    "PDFExtractionError",
    "PageText",
    "extract_change_order",
    "extract_fields",
    "extract_pdf",
    "extract_pdf_direct",
    "extract_pdf_layout",
    "validate_fields",
]
