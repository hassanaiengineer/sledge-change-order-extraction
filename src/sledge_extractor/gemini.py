"""Gemini calls for change-order extraction.

The API key stays in this process. This module does not log the key or the
document. Requests are not retried.
"""

from __future__ import annotations

import os

from .extract import ConfigurationError, ModelError

DEFAULT_MODEL = "gemini-3.1-flash-lite"
TIMEOUT_MS = 30_000
_SYSTEM = (
    "You extract change-order fields from a document. "
    "Return only JSON that matches the schema. Do not guess."
)


class GeminiClient:
    """One model call through the Google Gen AI SDK."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_ms: int = TIMEOUT_MS,
    ):
        if api_key is None:
            api_key = os.environ.get("GEMINI_API_KEY", "")
        self.api_key = api_key.strip()
        chosen = model if model is not None else os.environ.get("GEMINI_MODEL", "")
        self.model = (chosen or DEFAULT_MODEL).strip()
        self.timeout_ms = timeout_ms

    def complete(self, prompt: str, schema: dict) -> str:
        """Return the model's raw JSON text."""
        if not self.api_key:
            raise ConfigurationError("GEMINI_API_KEY is not configured.")
        if not self.model:
            raise ConfigurationError("GEMINI_MODEL is empty.")

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ConfigurationError("The Google Gen AI SDK is not installed.") from exc

        client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(
                timeout=self.timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        try:
            response = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_SYSTEM,
                    temperature=0,
                    max_output_tokens=4096,
                    response_mime_type="application/json",
                    response_json_schema=schema,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.MINIMAL
                    ),
                    http_options=types.HttpOptions(
                        timeout=self.timeout_ms,
                        retry_options=types.HttpRetryOptions(attempts=1),
                    ),
                ),
            )
        except Exception as exc:
            raise _safe_model_error(exc, self.model) from exc

        text = getattr(response, "text", None)
        return text.strip() if isinstance(text, str) else ""


def _safe_model_error(exc: Exception, model: str) -> ModelError:
    """Map a provider exception to a message that contains no request data."""
    code = getattr(exc, "code", None)
    status = getattr(exc, "status", None)
    kind = type(exc).__name__.lower()
    if "timeout" in kind or status == "DEADLINE_EXCEEDED":
        return ModelError("The model request timed out.")
    if code == 429 or status in {429, "RESOURCE_EXHAUSTED"}:
        return ModelError("The model service rate limit was reached. Try again later.")
    if code == 404 or status in {404, "NOT_FOUND"}:
        return ModelError(f"The model '{model}' is not available.")
    if code in {408, 500, 502, 503, 504} or status in {
        408, 500, 502, 503, 504, "UNAVAILABLE", "INTERNAL",
    }:
        return ModelError("The model service is unavailable.")
    return ModelError("The model request failed.")
