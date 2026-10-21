# Sledge change-order extraction

Read a change-order PDF or pasted text, then fill the fields you define. The result is JSON with a value, a short evidence quote, and a heuristic confidence score.

## How it works

Upload a PDF or paste text, add the fields and descriptions you want, then extract.

1. The PDF text is read first. Scanned pages go through OCR. Pasted text is used as-is.
2. Gemini fills only the fields you defined.
3. The result is JSON: a value, a short evidence quote, and a heuristic confidence score.

## What it uses

- **Python / FastAPI** handles the request and runs the workflow.
- **PyMuPDF** reads embedded PDF text and renders pages.
- **Tesseract OCR** reads scanned pages.
- **Gemini 3.1 Flash Lite** extracts the requested fields. The key stays on the server.
- **Validation** checks each value, its type, and whether the evidence is in the document.
- **The HTML page** is where you define fields, upload a document, and read the result.

## Run

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env
```

Set `GEMINI_API_KEY` in `.env`. Leave `GEMINI_MODEL=gemini-3.1-flash-lite`. Do not commit `.env`.

```powershell
python app.py
```

Open http://127.0.0.1:8000.

1. Upload a PDF or paste text, then choose **Show extracted text**.
2. Add or edit fields.
3. Choose **Extract**.

## Tests

```powershell
python -m pytest
```

The live Gemini check stays skipped unless you set `RUN_LIVE_GEMINI=1`. Confidence scores are not probabilities. The API key is used only on the server.
