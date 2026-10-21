"""Turn document text and field definitions into validated JSON.

Confidence is a heuristic. It is not a calibrated probability, and the
model's own confidence number is ignored.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from datetime import datetime

from .fields import FieldSpec, validate_fields

MAX_TEXT_CHARS = 80_000
MAX_EVIDENCE_CHARS = 500
_CONFIDENCE_CAP = 0.95

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {"type": "boolean"},
                            {"type": "null"},
                        ]
                    },
                    "evidence": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "ambiguous": {"type": "boolean"},
                    "currency": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["name", "value", "evidence", "ambiguous", "currency"],
            },
        }
    },
    "required": ["fields"],
}


class ExtractionError(Exception):
    """A request failed before a validated result could be built."""

    status_code = 400


class ConfigurationError(ExtractionError):
    """The server is missing a setting required to call the model."""

    status_code = 503


class ModelError(ExtractionError):
    """The model service failed. The message is safe to show a user."""

    status_code = 502


def extract_fields(
    text: str,
    fields: object,
    complete: Callable[[str, dict], str],
    *,
    model: str = "",
) -> dict:
    """Extract ``fields`` from ``text``.

    ``complete`` sends one prompt and a JSON schema to a model and returns
    the raw response text. It must not be retried by this function.
    """
    specs = validate_fields(fields)
    document = _prepare_text(text)
    raw = complete(build_prompt(document, specs), RESPONSE_SCHEMA)
    return interpret(raw, specs, document, model=model)


def build_prompt(document: str, specs: tuple[FieldSpec, ...]) -> str:
    """Instructions plus the document. The response shape is the JSON schema."""
    lines = [
        "Extract the listed fields from the document.",
        "Use only facts written in the document.",
        "If a value is missing, set it to null. Do not guess.",
        "Do not calculate totals or invent a currency.",
        "evidence must be a short exact quote from the document, or null.",
        "Set ambiguous to true when the document states conflicting values.",
        "currency is a 3-letter code only when that code is written in the document, otherwise null.",
        "Use YYYY-MM-DD for a date only when the month is unambiguous.",
        "Fields:",
    ]
    for spec in specs:
        need = "required" if spec.required else "optional"
        lines.append(f"- {spec.name} ({spec.type}, {need}): {spec.description}")
    lines.append("Document:")
    lines.append(document)
    return "\n".join(lines)


def interpret(raw: str, specs: tuple[FieldSpec, ...], document: str, *, model: str) -> dict:
    """Validate model JSON and score each field from the document."""
    errors: list[dict] = []
    warnings: list[str] = []
    grouped = {spec.name: [] for spec in specs}
    items, parse_error = _parse_items(raw)
    if parse_error:
        errors.append({"field": None, "message": parse_error})
    for item in items:
        name = item.get("name") if isinstance(item, dict) else None
        if isinstance(name, str) and name in grouped:
            grouped[name].append(item)

    data = {}
    for spec in specs:
        data[spec.name] = _read_field(spec, grouped[spec.name], document, errors, warnings)
    return {
        "data": data,
        "validation_errors": errors,
        "document_warnings": _unique(warnings),
        "meta": {
            "confidence_method": "heuristic_uncalibrated",
            "model": model,
        },
    }


def _prepare_text(text: object) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ExtractionError("Document text is empty.")
    if len(text) > MAX_TEXT_CHARS:
        raise ExtractionError(
            f"Document text is longer than {MAX_TEXT_CHARS} characters."
        )
    return text


def _parse_items(raw: object) -> tuple[list, str | None]:
    if not isinstance(raw, str) or not raw.strip():
        return [], "The model response could not be parsed."
    payload = raw.strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*", "", payload)
        payload = re.sub(r"\s*```$", "", payload)
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return [], "The model response could not be parsed."
    if not isinstance(parsed, dict) or not isinstance(parsed.get("fields"), list):
        return [], "The model response did not match the expected schema."
    return [item for item in parsed["fields"] if isinstance(item, dict)], None


def _read_field(spec, items, document, errors, warnings) -> dict:
    candidate = items[0] if items else None
    ambiguous = _conflicting(items, spec.type) or bool(
        candidate and candidate.get("ambiguous") is True
    )
    raw_value = candidate.get("value") if candidate else None
    value, type_error = _coerce(raw_value, spec.type)
    raw_evidence = candidate.get("evidence") if candidate else None
    evidence = _evidence(raw_evidence, document, warnings)
    currency = None
    if spec.type == "number" and value is not None:
        currency = _currency(
            candidate.get("currency") if candidate else None,
            evidence,
            document,
            warnings,
        )

    if type_error:
        value = None
        currency = None
        errors.append({"field": spec.name, "message": type_error})
    elif value is not None and not _supported(spec.type, value, evidence, document):
        value = None
        currency = None
        errors.append({
            "field": spec.name,
            "message": "Value is not supported by the document.",
        })
    elif value is None and spec.required is True:
        errors.append({
            "field": spec.name,
            "message": "Required field was not found in the document.",
        })

    if ambiguous and any(item.get("value") not in (None, "") for item in items):
        errors.append({
            "field": spec.name,
            "message": "The document contains conflicting information for this field.",
        })

    return {
        "value": value,
        "confidence": _confidence(spec.type, value, evidence, document, ambiguous),
        "evidence": evidence,
        "currency": currency,
    }


def _conflicting(items: list, field_type: str) -> bool:
    found = []
    for item in items:
        value, error = _coerce(item.get("value"), field_type)
        if error or value is None or value in found:
            continue
        found.append(value)
    return len(found) > 1


def _coerce(value, field_type: str) -> tuple[object, str | None]:
    if value is None:
        return None, None
    if isinstance(value, str) and not value.strip():
        return None, None
    try:
        if field_type == "string":
            return _coerce_string(value), None
        if field_type == "number":
            return _coerce_number(value), None
        if field_type == "date":
            return _coerce_date(value), None
        if field_type == "boolean":
            return _coerce_boolean(value), None
    except ValueError as exc:
        return None, str(exc)
    return None, f"Value is not a {field_type}."


def _coerce_string(value) -> str:
    if not isinstance(value, str):
        raise ValueError("Value is not a string.")
    return value.strip()


def _coerce_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError("Value is not a number.")
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        cleaned = re.sub(r"^[^\d-]+", "", cleaned)
        cleaned = re.sub(r"[^\d.]+$", "", cleaned)
        if not re.fullmatch(r"-?\d+(\.\d+)?", cleaned):
            raise ValueError("Value is not a number.")
        value = float(cleaned)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Value is not a number.")
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _coerce_date(value) -> str:
    if not isinstance(value, str):
        raise ValueError("Value is not a date.")
    text = value.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
        except ValueError:
            raise ValueError("Value is not a date.") from None
    if re.fullmatch(r"\d{1,4}[/-]\d{1,2}[/-]\d{1,4}", text):
        raise ValueError("Date is ambiguous.")
    for fmt in ("%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%d %b %Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError("Value is not a date.")


def _coerce_boolean(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().casefold() in {"true", "false"}:
        return value.strip().casefold() == "true"
    raise ValueError("Value is not a boolean.")


def _evidence(raw, document: str, warnings: list[str]) -> str | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    quote = raw.strip()
    if len(quote) > MAX_EVIDENCE_CHARS:
        warnings.append("An evidence quote was dropped because it was too long.")
        return None
    if _normalize(quote) not in _normalize(document):
        warnings.append("An evidence quote was dropped because it is not in the document.")
        return None
    return quote


# Codes we will keep when the document itself writes them. A symbol is not a code.
_CURRENCY_CODES = {
    "AED", "AUD", "CAD", "CHF", "CNY", "DKK", "EUR", "GBP", "INR",
    "JPY", "NOK", "NZD", "PKR", "SAR", "SEK", "USD",
}


def _currency(raw, evidence: str | None, document: str, warnings: list[str]) -> str | None:
    """Keep a currency code only when the document states that code."""
    code = raw.strip().upper() if isinstance(raw, str) else ""
    quote = evidence or ""
    stated = [
        item for item in sorted(_CURRENCY_CODES)
        if re.search(rf"\b{item}\b", document.upper())
    ]
    in_quote = bool(code in _CURRENCY_CODES and re.search(rf"\b{code}\b", quote.upper()))
    if in_quote or (code in stated and stated == [code]):
        return code
    if code:
        warnings.append(
            f"Currency '{code}' was ignored because the document does not state it clearly."
        )
    if re.search(r"[$€£¥]", quote) and code not in stated:
        warnings.append(
            "A currency symbol is present, but the document does not state a currency code."
        )
    if len(stated) > 1 and not in_quote:
        warnings.append(
            "More than one currency code is in the document, so currency was left empty."
        )
    return None


def _supported(field_type: str, value, evidence: str | None, document: str) -> bool:
    quoted = evidence is not None
    if field_type == "string":
        folded = _normalize(str(value))
        return folded in _normalize(evidence or "") or folded in _normalize(document)
    if field_type == "number":
        return _number_in_text(value, evidence or "") or _number_in_text(value, document)
    if field_type == "date":
        return quoted or _date_in_document(value, document)
    return quoted


def _confidence(field_type, value, evidence, document, ambiguous: bool) -> float:
    """Score evidence, a value found in the text, and ambiguity.

    A type-valid value with nothing to point at in the document scores 0.25.
    Evidence found in the document adds 0.45. A value that itself appears in
    the document adds 0.15. Boolean values do not get that second bonus.
    Ambiguity caps the score at 0.45. The score never exceeds 0.95.
    """
    if value is None:
        return 0.0
    score = 0.25
    if evidence:
        score += 0.45
    if field_type != "boolean" and _value_in_document(field_type, value, document):
        score += 0.15
    if ambiguous:
        score = min(score, 0.45)
    return round(min(score, _CONFIDENCE_CAP), 2)


def _value_in_document(field_type: str, value, document: str) -> bool:
    if field_type == "string":
        return _normalize(str(value)) in _normalize(document)
    if field_type == "number":
        return _number_in_text(value, document)
    if field_type == "date":
        return _date_in_document(value, document)
    return False


def _number_in_text(number, text: str) -> bool:
    compact = text.replace(",", "")
    for match in re.finditer(r"-?\d+(?:\.\d+)?", compact):
        try:
            if float(match.group()) == float(number):
                return True
        except ValueError:
            continue
    return False


def _date_in_document(iso: str, document: str) -> bool:
    date = datetime.strptime(iso, "%Y-%m-%d").date()
    month = date.strftime("%B")
    short = date.strftime("%b")
    candidates = (
        iso,
        f"{month} {date.day}, {date.year}",
        f"{date.day} {month} {date.year}",
        f"{short} {date.day}, {date.year}",
        f"{date.day} {short} {date.year}",
    )
    folded = document.casefold()
    return any(item.casefold() in folded for item in candidates)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).casefold().strip()


def _unique(items: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered
