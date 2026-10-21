"""Small HTTP API. The Gemini key is read here and is never sent to the browser."""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .extract import ExtractionError
from .fields import FieldDefinitionError
from .gemini import GeminiClient
from .pipeline import extract_change_order, read_document

_ROOT = Path(__file__).resolve().parents[2]
_PAGE = _ROOT / "static" / "index.html"
_MAX_FIELD_JSON = 50_000

app = FastAPI(title="Sledge change-order extraction")


def load_local_env() -> None:
    """Load a gitignored .env without overriding variables already set."""
    path = _ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_local_env()


def get_client() -> GeminiClient:
    return GeminiClient()


@app.get("/")
def index():
    if not _PAGE.is_file():
        return JSONResponse({"error": "Interface file is missing."}, status_code=500)
    return FileResponse(_PAGE)


@app.post("/api/read")
async def read_text(text: str = Form(""), file: UploadFile | None = File(None)):
    """Return the PDF text, or the pasted text, without calling the model."""
    pasted, pdf = await _uploaded_document(text, file)
    try:
        document, warnings = read_document(text=pasted, pdf=pdf)
    except ExtractionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=exc.status_code)
    except Exception:
        return JSONResponse({"error": "Could not read the document."}, status_code=500)
    return {"text": document, "document_warnings": warnings}


@app.post("/api/extract")
async def extract(
    fields: str = Form(...),
    text: str = Form(""),
    file: UploadFile | None = File(None),
    client: GeminiClient = Depends(get_client),
):
    if len(fields) > _MAX_FIELD_JSON:
        return JSONResponse({"error": "Field definitions are too large."}, status_code=400)
    try:
        parsed = json.loads(fields)
    except json.JSONDecodeError:
        return JSONResponse({"error": "fields must be JSON."}, status_code=400)

    pasted, pdf = await _uploaded_document(text, file)
    try:
        return extract_change_order(
            text=pasted,
            pdf=pdf,
            fields=parsed,
            client=client,
        )
    except FieldDefinitionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except ExtractionError as exc:
        return JSONResponse({"error": str(exc)}, status_code=exc.status_code)
    except Exception:
        return JSONResponse({"error": "Extraction failed."}, status_code=500)


async def _uploaded_document(text: str, file: UploadFile | None) -> tuple[str | None, bytes | None]:
    pdf = None
    if file is not None and file.filename:
        pdf = await file.read()
        if not pdf:
            pdf = None
    pasted = text if text.strip() else None
    if pdf is not None:
        return None, pdf
    return pasted, None
