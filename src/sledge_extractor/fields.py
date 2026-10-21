"""Validate user-defined change-order fields before they reach the model."""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_FIELDS = 20
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 400
FIELD_TYPES = ("string", "number", "date", "boolean")
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED = {"data", "validation_errors", "document_warnings", "meta"}


class FieldDefinitionError(ValueError):
    """The caller sent field definitions the extractor will not use."""


@dataclass(frozen=True)
class FieldSpec:
    """One field the model is allowed to fill."""

    name: str
    description: str
    type: str
    required: bool


def validate_fields(fields: object) -> tuple[FieldSpec, ...]:
    """Return field specs, or raise ``FieldDefinitionError``."""
    if not isinstance(fields, list):
        raise FieldDefinitionError("fields must be a list.")
    if not fields:
        raise FieldDefinitionError("At least one field is required.")
    if len(fields) > MAX_FIELDS:
        raise FieldDefinitionError(f"At most {MAX_FIELDS} fields are allowed.")

    specs: list[FieldSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(fields, start=1):
        spec = _one_field(raw, index)
        if spec.name in seen:
            raise FieldDefinitionError(f"Duplicate field name '{spec.name}'.")
        seen.add(spec.name)
        specs.append(spec)
    return tuple(specs)


def _one_field(raw: object, index: int) -> FieldSpec:
    if not isinstance(raw, dict):
        raise FieldDefinitionError(f"Field {index} must be an object.")

    name = raw.get("name")
    description = raw.get("description")
    field_type = raw.get("type")
    required = raw.get("required")

    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise FieldDefinitionError(
            f"Field {index} name must be a JSON identifier "
            "(letters, digits, and underscores, not starting with a digit)."
        )
    if len(name) > MAX_NAME_LENGTH:
        raise FieldDefinitionError(
            f"Field '{name}' is longer than {MAX_NAME_LENGTH} characters."
        )
    if name in _RESERVED:
        raise FieldDefinitionError(f"Field name '{name}' is reserved.")

    if not isinstance(description, str) or not description.strip():
        raise FieldDefinitionError(f"Field '{name}' needs a description.")
    description = description.strip()
    if len(description) > MAX_DESCRIPTION_LENGTH:
        raise FieldDefinitionError(
            f"Field '{name}' description is longer than "
            f"{MAX_DESCRIPTION_LENGTH} characters."
        )

    if field_type not in FIELD_TYPES:
        allowed = ", ".join(FIELD_TYPES)
        raise FieldDefinitionError(
            f"Field '{name}' type must be one of: {allowed}."
        )
    if not isinstance(required, bool):
        raise FieldDefinitionError(f"Field '{name}' required must be true or false.")

    return FieldSpec(name, description, field_type, required)
