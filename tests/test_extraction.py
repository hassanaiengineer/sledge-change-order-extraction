"""Checks for reading a document, validating fields, and extraction.

The model is replaced except for one test, which stays skipped unless
RUN_LIVE_GEMINI=1. Scanned-page checks are skipped when Tesseract is missing.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient

from sledge_extractor import (
    ExtractionError,
    FieldDefinitionError,
    InvalidPDFError,
    OCRDependencyError,
    OCRError,
    extract_change_order,
    extract_fields,
    extract_pdf,
    validate_fields,
)
from sledge_extractor import ocr as ocr_module
from sledge_extractor.api import app, get_client
from sledge_extractor.extract import ConfigurationError, ModelError
from sledge_extractor.fields import MAX_DESCRIPTION_LENGTH, MAX_FIELDS
from sledge_extractor.gemini import DEFAULT_MODEL, GeminiClient, _safe_model_error
from sledge_extractor.ocr import ExtractionResult, PageText
from sledge_extractor.pipeline import MAX_PAGES, MAX_PDF_BYTES

DOCUMENT = (
    "Change Order No: CO-104\n"
    "Issued: March 3, 2024\n"
    "Additional cost: USD 4500\n"
    "Status: Approved\n"
)
SCANNED_PHRASE = "NORTH WALL FOOTING"
FIELDS = [{
    "name": "change_order_number",
    "description": "The printed change order identifier.",
    "type": "string",
    "required": True,
}]


def _tesseract_installed() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
    except Exception:
        return False
    return True


requires_tesseract = pytest.mark.skipif(
    not _tesseract_installed(),
    reason="Tesseract OCR binary is not installed",
)


def field(**overrides):
    data = {
        "name": "change_order_number",
        "description": "The printed change order identifier.",
        "type": "string",
        "required": True,
    }
    data.update(overrides)
    return data


class Reply:
    """Stand-in for one model response."""

    def __init__(self, body):
        self.body = body
        self.calls = 0
        self.prompt = ""

    def complete(self, prompt, schema):
        self.calls += 1
        self.prompt = prompt
        if isinstance(self.body, Exception):
            raise self.body
        if isinstance(self.body, str):
            return self.body
        return json.dumps(self.body)


def model_field(name, value, evidence, ambiguous=False, currency=None):
    return {
        "name": name,
        "value": value,
        "evidence": evidence,
        "ambiguous": ambiguous,
        "currency": currency,
        "confidence": 0.99,
    }


def _save(doc: fitz.Document) -> bytes:
    try:
        return doc.tobytes()
    finally:
        doc.close()


def text_pdf(lines: list[str]) -> bytes:
    doc = fitz.open()
    for line in lines:
        page = doc.new_page()
        if line:
            page.insert_text((72, 72), line, fontsize=12, fontname="helv")
    return _save(doc)


def logo_pdf(text: str) -> bytes:
    stamp = fitz.open()
    try:
        stamp_page = stamp.new_page(width=40, height=40)
        stamp_page.draw_rect(stamp_page.rect, color=(0, 0, 0), fill=(0, 0, 0))
        png = stamp_page.get_pixmap(dpi=72).tobytes("png")
    finally:
        stamp.close()
    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(fitz.Rect(500, 40, 540, 80), stream=png)
    if text:
        page.insert_text((72, 140), text, fontsize=12, fontname="helv")
    return _save(doc)


def text_and_large_image_pdf(text: str, image_phrase: str) -> bytes:
    rendered = fitz.open()
    try:
        source = rendered.new_page(width=500, height=280)
        source.insert_text((24, 150), image_phrase, fontsize=28, fontname="helv")
        png = source.get_pixmap(dpi=200).tobytes("png")
    finally:
        rendered.close()
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(56, 220, 556, 640), stream=png)
    page.insert_text((72, 72), text, fontsize=12, fontname="helv")
    return _save(doc)


def scanned_pdf(phrase: str) -> bytes:
    rendered = fitz.open()
    try:
        source = rendered.new_page(width=612, height=200)
        source.insert_text((36, 120), phrase, fontsize=36, fontname="helv")
        png = source.get_pixmap(dpi=120).tobytes("png")
    finally:
        rendered.close()
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(fitz.Rect(56, 300, 556, 500), stream=png)
    return _save(doc)


class _FakeTesseract:
    def __init__(self, text: str = "SCANNED", available: bool = True, error: Exception | None = None):
        self.text = text
        self.available = available
        self.error = error
        self.calls = 0

    class TesseractNotFoundError(Exception):
        pass

    def get_tesseract_version(self) -> str:
        if not self.available:
            raise self.TesseractNotFoundError("tesseract is not installed")
        return "fake"

    def image_to_string(self, image, lang: str = "eng") -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.text


def test_digital_pdf_keeps_page_text():
    result = extract_pdf(text_pdf(["ALPHA CHANGE ORDER PAGE", "", "BETA CHANGE ORDER PAGE"]))

    assert [(page.page_number, page.source, page.text) for page in result.pages] == [
        (1, "digital", "ALPHA CHANGE ORDER PAGE"),
        (2, "empty", ""),
        (3, "digital", "BETA CHANGE ORDER PAGE"),
    ]
    assert "--- Page 1 ---" in result.full_text()


def test_text_only_page_does_not_call_tesseract(monkeypatch):
    monkeypatch.setattr(
        ocr_module,
        "_import_pytesseract",
        lambda: (_ for _ in ()).throw(AssertionError("Tesseract should not load")),
    )
    result = extract_pdf(text_pdf(["See note."]), min_digital_chars=10_000)

    assert result.pages[0].source == "digital"
    assert result.pages[0].text == "See note."


def test_bad_pdfs_are_rejected():
    empty_document = (
        b"%PDF-1.4\n"
        b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n"
        b"2 0 obj<< /Type /Pages /Count 0 /Kids [] >>endobj\n"
        b"trailer<< /Root 1 0 R >>\n"
        b"%%EOF"
    )
    assert extract_pdf(empty_document).page_count == 0
    with pytest.raises(InvalidPDFError, match="empty"):
        extract_pdf(b"")
    with pytest.raises(InvalidPDFError, match="Could not open PDF"):
        extract_pdf(b"this is not a pdf")


def test_encrypted_pdf_is_rejected():
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Secret change order", fontsize=12, fontname="helv")
    payload = doc.tobytes(
        encryption=fitz.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="user-secret",
    )
    doc.close()

    with pytest.raises(InvalidPDFError, match="encrypted"):
        extract_pdf(payload)


def test_small_image_with_enough_text_stays_digital(monkeypatch):
    text = "Change Order 17 replaces the north wall concrete footing."
    monkeypatch.setattr(
        ocr_module,
        "_import_pytesseract",
        lambda: (_ for _ in ()).throw(AssertionError("embedded text should be used")),
    )
    result = extract_pdf(logo_pdf(text))

    assert result.pages[0].source == "digital"
    assert result.pages[0].text == text


def test_short_image_page_uses_ocr(monkeypatch):
    fake = _FakeTesseract(text="  scanned line  \n")
    monkeypatch.setattr(ocr_module, "_import_pytesseract", lambda: fake)
    result = extract_pdf(logo_pdf("See note."), dpi=72)

    assert fake.calls == 1
    assert result.pages[0].source == "ocr"
    assert result.pages[0].text == "scanned line"


def test_missing_tesseract_is_reported(monkeypatch):
    monkeypatch.setattr(ocr_module, "_import_pytesseract", lambda: _FakeTesseract(available=False))

    with pytest.raises(OCRDependencyError, match="Tesseract OCR is not available"):
        extract_pdf(logo_pdf(""), dpi=72)


def test_ocr_failure_names_the_page(monkeypatch):
    monkeypatch.setattr(
        ocr_module,
        "_import_pytesseract",
        lambda: _FakeTesseract(error=RuntimeError("engine failed")),
    )
    doc = fitz.open(stream=text_pdf(["A digital page with enough words to stay embedded."]), filetype="pdf")
    image_doc = fitz.open(stream=logo_pdf(""), filetype="pdf")
    try:
        doc.insert_pdf(image_doc)
        mixed = doc.tobytes()
    finally:
        image_doc.close()
        doc.close()

    with pytest.raises(OCRError, match="OCR failed on page 2: engine failed"):
        extract_pdf(mixed, dpi=72)


@requires_tesseract
def test_large_image_keeps_the_text_layer():
    text = "Change Order 17 replaces the north wall concrete footing."
    result = extract_pdf(text_and_large_image_pdf(text, SCANNED_PHRASE))

    assert result.pages[0].source == "ocr"
    assert text in result.pages[0].text
    assert SCANNED_PHRASE in result.pages[0].text


def test_valid_field_is_accepted():
    specs = validate_fields([field()])

    assert specs[0].name == "change_order_number"
    assert specs[0].type == "string"
    assert specs[0].required is True


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("fields", "must be a list"),
        ([], "At least one"),
        ([field(name="1bad")], "JSON identifier"),
        ([field(name="change-order")], "JSON identifier"),
        ([field(name="data")], "reserved"),
        ([field(description="  ")], "needs a description"),
        ([field(description="x" * (MAX_DESCRIPTION_LENGTH + 1))], "longer than"),
        ([field(type="money")], "type must be one of"),
        ([field(required="yes")], "true or false"),
        ([field(), field()], "Duplicate field name"),
        ([{"description": "Missing name"}], "JSON identifier"),
    ],
)
def test_invalid_fields_are_rejected(payload, match):
    with pytest.raises(FieldDefinitionError, match=match):
        validate_fields(payload)


def test_too_many_fields_are_rejected():
    fields = [
        field(name=f"field_{index}", description=f"Field {index}.")
        for index in range(MAX_FIELDS + 1)
    ]

    with pytest.raises(FieldDefinitionError, match="At most"):
        validate_fields(fields)


def test_supported_string_uses_evidence_and_ignores_model_confidence():
    fake = Reply({"fields": [model_field("change_order_number", "CO-104", "Change Order No: CO-104")]})
    result = extract_fields(DOCUMENT, [field()], fake.complete, model="gemini-3.1-flash-lite")
    found = result["data"]["change_order_number"]

    assert found["value"] == "CO-104"
    assert found["evidence"] == "Change Order No: CO-104"
    assert found["confidence"] == 0.85
    assert result["validation_errors"] == []
    assert result["meta"]["confidence_method"] == "heuristic_uncalibrated"


def test_number_in_the_document_without_evidence_stays_low():
    fake = Reply({"fields": [model_field("additional_cost", 4500, None)]})
    result = extract_fields(
        DOCUMENT,
        [field(name="additional_cost", description="The additional cost.", type="number", required=False)],
        fake.complete,
    )

    assert result["data"]["additional_cost"]["value"] == 4500
    assert result["data"]["additional_cost"]["confidence"] == 0.4


def test_number_missing_from_the_document_is_not_kept():
    fake = Reply({"fields": [model_field("additional_cost", 9999, None)]})
    result = extract_fields(
        DOCUMENT,
        [field(name="additional_cost", description="The additional cost.", type="number", required=False)],
        fake.complete,
    )

    assert result["data"]["additional_cost"]["value"] is None
    assert result["validation_errors"][0]["message"] == "Value is not supported by the document."


def test_currency_code_in_the_quote_is_kept():
    fake = Reply({
        "fields": [model_field("additional_cost", "USD 4,500", "Additional cost: USD 4500", currency="usd")]
    })
    result = extract_fields(
        DOCUMENT,
        [field(name="additional_cost", description="The additional cost.", type="number", required=False)],
        fake.complete,
    )

    assert result["data"]["additional_cost"]["value"] == 4500
    assert result["data"]["additional_cost"]["currency"] == "USD"
    assert result["data"]["additional_cost"]["confidence"] == 0.85


def test_currency_symbol_without_a_code_is_a_warning():
    fake = Reply({
        "fields": [model_field("additional_cost", 4500, "Additional cost: $4,500", currency="USD")]
    })
    result = extract_fields(
        "Additional cost: $4,500\n",
        [field(name="additional_cost", description="The additional cost.", type="number", required=False)],
        fake.complete,
    )

    assert result["data"]["additional_cost"]["currency"] is None
    assert any("currency code" in warning for warning in result["document_warnings"])


def test_single_currency_code_in_the_document_is_kept():
    fake = Reply({"fields": [model_field("additional_cost", 4500, "4500", currency="USD")]})
    result = extract_fields(
        DOCUMENT,
        [field(name="additional_cost", description="The additional cost.", type="number", required=False)],
        fake.complete,
    )

    assert result["data"]["additional_cost"]["currency"] == "USD"


def test_missing_required_field_is_null():
    result = extract_fields(DOCUMENT, [field()], Reply({"fields": []}).complete)

    assert result["data"]["change_order_number"]["value"] is None
    assert result["validation_errors"] == [{
        "field": "change_order_number",
        "message": "Required field was not found in the document.",
    }]


def test_required_null_date_is_a_validation_error():
    result = extract_fields(
        DOCUMENT,
        [field(name="approval_date", description="Approval date.", type="date", required=True)],
        Reply({"fields": [model_field("approval_date", None, None)]}).complete,
    )

    assert result["data"]["approval_date"]["value"] is None
    assert result["validation_errors"][0]["message"] == "Required field was not found in the document."


def test_optional_null_date_has_no_validation_error():
    result = extract_fields(
        DOCUMENT,
        [field(name="approval_date", description="Approval date.", type="date", required=False)],
        Reply({"fields": [model_field("approval_date", None, None)]}).complete,
    )

    assert result["data"]["approval_date"]["value"] is None
    assert result["validation_errors"] == []


def test_ambiguous_date_is_rejected():
    result = extract_fields(
        "Issued: 01/02/2024\n",
        [field(name="issue_date", description="Issue date.", type="date", required=False)],
        Reply({"fields": [model_field("issue_date", "01/02/2024", "Issued: 01/02/2024")]}).complete,
    )

    assert result["data"]["issue_date"]["value"] is None
    assert result["validation_errors"][0]["message"] == "Date is ambiguous."


def test_iso_and_named_dates_are_accepted():
    iso = extract_fields(
        DOCUMENT,
        [field(name="issue_date", description="Issue date.", type="date", required=False)],
        Reply({"fields": [model_field("issue_date", "2024-03-03", "March 3, 2024")]}).complete,
    )
    named = extract_fields(
        DOCUMENT,
        [field(name="issue_date", description="Issue date.", type="date", required=False)],
        Reply({"fields": [model_field("issue_date", "March 3, 2024", "Issued: March 3, 2024")]}).complete,
    )

    assert iso["data"]["issue_date"]["value"] == "2024-03-03"
    assert named["data"]["issue_date"]["value"] == "2024-03-03"
    assert iso["validation_errors"] == []


def test_boolean_needs_evidence():
    kept = extract_fields(
        DOCUMENT,
        [field(name="approved", description="Whether it is approved.", type="boolean", required=False)],
        Reply({"fields": [model_field("approved", True, "Status: Approved")]}).complete,
    )
    dropped = extract_fields(
        DOCUMENT,
        [field(name="approved", description="Whether it is approved.", type="boolean", required=False)],
        Reply({"fields": [model_field("approved", True, None)]}).complete,
    )

    assert kept["data"]["approved"]["value"] is True
    assert kept["data"]["approved"]["confidence"] == 0.7
    assert dropped["data"]["approved"]["value"] is None


def test_conflicting_values_are_capped():
    result = extract_fields(
        DOCUMENT,
        [field(required=False)],
        Reply({
            "fields": [
                model_field("change_order_number", "CO-104", "Change Order No: CO-104"),
                model_field("change_order_number", "CO-105", "Change Order No: CO-104"),
            ]
        }).complete,
    )

    assert result["data"]["change_order_number"]["confidence"] == 0.45
    assert any("conflicting" in error["message"] for error in result["validation_errors"])


def test_invalid_type_and_bad_model_output_stay_in_schema():
    invalid = extract_fields(
        DOCUMENT,
        [field(name="additional_cost", description="The additional cost.", type="number", required=False)],
        Reply({"fields": [model_field("additional_cost", "several", "Additional cost: USD 4500")]}).complete,
    )
    unparsed = extract_fields(DOCUMENT, [field()], Reply("this is not json").complete)
    wrong_shape = extract_fields(DOCUMENT, [field(required=False)], Reply('{"answer": "CO-104"}').complete)

    assert invalid["validation_errors"][0]["message"] == "Value is not a number."
    assert unparsed["validation_errors"][0]["message"] == "The model response could not be parsed."
    assert "expected schema" in wrong_shape["validation_errors"][0]["message"]
    assert unparsed["data"]["change_order_number"]["value"] is None


def test_evidence_missing_from_the_document_is_dropped():
    result = extract_fields(
        DOCUMENT,
        [field(required=False)],
        Reply({"fields": [model_field("change_order_number", "CO-104", "Invented quote CO-104")]}).complete,
    )

    assert result["data"]["change_order_number"]["evidence"] is None
    assert result["data"]["change_order_number"]["confidence"] == 0.4


def test_empty_and_oversized_text_do_not_call_the_model():
    fake = Reply({"fields": []})
    with pytest.raises(ExtractionError, match="empty"):
        extract_fields("  ", [field()], fake.complete)
    with pytest.raises(ExtractionError, match="longer than"):
        extract_fields("a" * 80_001, [field()], fake.complete)

    assert fake.calls == 0


def test_recorded_live_response_keeps_the_synthetic_fields():
    recorded = {
        "fields": [
            {"name": "change_order_number", "value": "CO-104", "evidence": "CO-104", "ambiguous": False, "currency": None},
            {"name": "issue_date", "value": "2024-03-03", "evidence": "March 3, 2024", "ambiguous": False, "currency": None},
            {"name": "additional_cost", "value": 4500, "evidence": "4500", "ambiguous": False, "currency": "USD"},
            {"name": "approved", "value": True, "evidence": "Approved", "ambiguous": False, "currency": None},
        ]
    }
    result = extract_fields(
        DOCUMENT,
        [
            field(),
            field(name="issue_date", description="Issue date.", type="date", required=False),
            field(name="additional_cost", description="The additional cost.", type="number", required=False),
            field(name="approved", description="Whether it is approved.", type="boolean", required=False),
        ],
        Reply(recorded).complete,
        model="gemini-3.1-flash-lite",
    )

    assert result["data"]["change_order_number"]["value"] == "CO-104"
    assert result["data"]["issue_date"]["value"] == "2024-03-03"
    assert result["data"]["additional_cost"]["value"] == 4500
    assert result["data"]["additional_cost"]["currency"] == "USD"
    assert result["data"]["approved"]["value"] is True
    assert result["validation_errors"] == []


def test_pasted_text_does_not_run_ocr(monkeypatch):
    monkeypatch.setattr(
        "sledge_extractor.pipeline.extract_pdf",
        lambda source: (_ for _ in ()).throw(AssertionError("pasted text should not open a PDF")),
    )
    fake = Reply({"fields": [model_field("change_order_number", "CO-104", "Change Order No: CO-104")]})
    fake.model = "gemini-3.1-flash-lite"
    result = extract_change_order(text="Change Order No: CO-104", fields=FIELDS, client=fake)

    assert result["data"]["change_order_number"]["value"] == "CO-104"
    assert "--- Page" not in fake.prompt


def test_digital_pdf_sends_page_text_to_the_model():
    fake = Reply({"fields": [model_field("change_order_number", "CO-104", "Change Order No: CO-104")]})
    fake.model = "gemini-3.1-flash-lite"
    result = extract_change_order(pdf=text_pdf(["Change Order No: CO-104"]), fields=FIELDS, client=fake)

    assert "--- Page 1 ---" in fake.prompt
    assert result["data"]["change_order_number"]["value"] == "CO-104"
    assert result["document_warnings"] == []


@requires_tesseract
def test_scanned_pdf_ocr_text_reaches_the_model():
    fake = Reply({"fields": [model_field("change_order_number", "CO-104", "Change Order No: CO-104")]})
    fake.model = "gemini-3.1-flash-lite"
    result = extract_change_order(pdf=scanned_pdf(SCANNED_PHRASE), fields=FIELDS, client=fake)

    assert SCANNED_PHRASE in fake.prompt
    assert any("OCR" in warning for warning in result["document_warnings"])


def test_ocr_failure_does_not_call_the_model(monkeypatch):
    monkeypatch.setattr(
        "sledge_extractor.pipeline.extract_pdf",
        lambda source: (_ for _ in ()).throw(OCRError("OCR failed on page 1: engine failed")),
    )
    fake = Reply({"fields": []})
    fake.model = "gemini-3.1-flash-lite"

    with pytest.raises(ExtractionError, match="OCR failed on page 1"):
        extract_change_order(pdf=b"%PDF-1.4", fields=FIELDS, client=fake)

    assert fake.calls == 0


def test_limits_reject_large_or_mixed_input(monkeypatch):
    pages = tuple(PageText(index, "Note", "digital") for index in range(1, MAX_PAGES + 2))
    monkeypatch.setattr("sledge_extractor.pipeline.extract_pdf", lambda source: ExtractionResult(pages))
    with pytest.raises(ExtractionError, match=str(MAX_PAGES + 1)):
        extract_change_order(pdf=b"%PDF-1.4", fields=FIELDS, client=Reply({"fields": []}))
    with pytest.raises(ExtractionError, match="15 MB"):
        extract_change_order(pdf=b"x" * (MAX_PDF_BYTES + 1), fields=FIELDS, client=Reply({"fields": []}))
    with pytest.raises(ExtractionError, match="not both"):
        extract_change_order(text="hello", pdf=b"%PDF", fields=FIELDS, client=Reply({"fields": []}))


class _StatusError(Exception):
    def __init__(self, status):
        super().__init__("request failed with secret=SHOULD_NOT_LEAK")
        self.status = status


class _CodeError(Exception):
    def __init__(self, code, status):
        super().__init__("503 UNAVAILABLE. high demand secret=SHOULD_NOT_LEAK")
        self.code = code
        self.status = status


def test_model_errors_hide_provider_details():
    unavailable = _safe_model_error(_CodeError(503, "UNAVAILABLE"), "gemini-3.1-flash-lite")
    timeout = _safe_model_error(TimeoutError("timed out secret=SHOULD_NOT_LEAK"), "gemini-3.1-flash-lite")
    limited = _safe_model_error(_StatusError(429), "gemini-3.1-flash-lite")
    missing = _safe_model_error(_StatusError(404), "gemini-3.1-flash-lite")

    assert str(unavailable) == "The model service is unavailable."
    assert str(timeout) == "The model request timed out."
    assert "rate limit" in str(limited)
    assert "gemini-3.1-flash-lite" in str(missing)
    for error in (unavailable, timeout, limited, missing):
        assert "SHOULD_NOT_LEAK" not in str(error)


def test_missing_api_key_is_reported(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="GEMINI_API_KEY"):
        GeminiClient(api_key="", model="gemini-3.1-flash-lite").complete("prompt", {})


def test_page_does_not_contain_an_api_key():
    page = Path("static/index.html").read_text(encoding="utf-8")
    secret = ""
    env = Path(".env")
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("GEMINI_API_KEY="):
                secret = line.split("=", 1)[1].strip()
    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert response.text.index("How it works") < response.text.index("1. Document")
    assert response.text.index("1. Document") < response.text.index("2. Fields")
    assert response.text.index("2. Fields") < response.text.index("3. Extract")
    assert "GEMINI_API_KEY=" not in response.text
    if secret:
        assert secret not in response.text
        assert secret not in page


def test_read_returns_document_text_without_a_model():
    client = TestClient(app)
    pasted = client.post("/api/read", data={"text": "Change Order No: CO-104"})
    uploaded = client.post(
        "/api/read",
        files={"file": ("order.pdf", text_pdf(["Change Order No: CO-104"]), "application/pdf")},
    )

    assert pasted.json()["text"] == "Change Order No: CO-104"
    assert "Change Order No: CO-104" in uploaded.json()["text"]
    assert "--- Page 1 ---" in uploaded.json()["text"]


def test_extract_routes_for_text_and_pdf():
    class FakeClient:
        model = "gemini-3.1-flash-lite"

        def complete(self, prompt, schema):
            return json.dumps({
                "fields": [model_field("change_order_number", "CO-104", "Change Order No: CO-104")]
            })

    app.dependency_overrides[get_client] = lambda: FakeClient()
    client = TestClient(app)
    body = json.dumps(FIELDS)
    try:
        pasted = client.post("/api/extract", data={"fields": body, "text": "Change Order No: CO-104"})
        uploaded = client.post(
            "/api/extract",
            data={"fields": body},
            files={"file": ("order.pdf", text_pdf(["Change Order No: CO-104"]), "application/pdf")},
        )
    finally:
        app.dependency_overrides.clear()

    assert pasted.status_code == 200
    assert pasted.json()["data"]["change_order_number"]["value"] == "CO-104"
    assert uploaded.json()["data"]["change_order_number"]["confidence"] == 0.85


def test_invalid_fields_and_model_errors_are_safe():
    class Broken:
        model = "gemini-3.1-flash-lite"

        def complete(self, prompt, schema):
            raise ModelError("The model request timed out.")

    app.dependency_overrides[get_client] = lambda: Broken()
    client = TestClient(app)
    try:
        invalid = client.post("/api/extract", data={"fields": "[]", "text": "Change Order No: CO-104"})
        failed = client.post(
            "/api/extract",
            data={"fields": json.dumps(FIELDS), "text": "Change Order No: CO-104 secret=SHOULD_NOT_LEAK"},
        )
    finally:
        app.dependency_overrides.clear()

    assert invalid.status_code == 400
    assert failed.status_code == 502
    assert failed.json()["error"] == "The model request timed out."
    assert "SHOULD_NOT_LEAK" not in failed.text


def test_missing_api_key_is_a_configuration_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app.dependency_overrides.clear()
    response = TestClient(app).post(
        "/api/extract",
        data={"fields": json.dumps(FIELDS), "text": "Change Order No: CO-104"},
    )

    assert response.status_code == 503
    assert response.json()["error"] == "GEMINI_API_KEY is not configured."


@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY") or os.environ.get("RUN_LIVE_GEMINI") != "1",
    reason="Set GEMINI_API_KEY and RUN_LIVE_GEMINI=1 to call Gemini once.",
)
def test_live_model_reads_the_synthetic_change_order():
    result = extract_change_order(
        text=DOCUMENT,
        fields=[
            field(description="The change order identifier printed on the document."),
            field(name="issue_date", description="The date the change order was issued.", type="date", required=False),
            field(
                name="additional_cost",
                description="The additional cost written on the change order. Do not calculate a new total.",
                type="number",
                required=False,
            ),
            field(name="approved", description="Whether the change order is marked approved.", type="boolean", required=False),
        ],
        client=GeminiClient(model=DEFAULT_MODEL),
    )

    assert result["data"]["change_order_number"]["value"] == "CO-104"
    assert result["data"]["issue_date"]["value"] == "2024-03-03"
    assert result["data"]["additional_cost"]["value"] == 4500
    assert result["data"]["additional_cost"]["currency"] == "USD"
    assert result["data"]["approved"]["value"] is True
