"""Run the local extraction page.

The process reads GEMINI_API_KEY from the environment or from a gitignored .env.
The browser never receives that key.
"""

from sledge_extractor.api import app


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
