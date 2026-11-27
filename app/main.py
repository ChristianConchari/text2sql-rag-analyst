"""ASGI entrypoint: `uv run uvicorn app.main:app`.

Reads the LLM endpoint from configs/model/llm.yaml (LLM_BASE_URL overrides) and the database from DB_PATH.
Serves the web UI at "/" when it has been built (`cd frontend && npm ci && npm run build`).
"""
from pathlib import Path

from src.interfaces.fastapi_app.app import create_app
from src.runtime.pipelines.text2sql import build_default_pipeline

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist" / "frontend" / "browser"

app = create_app(build_default_pipeline(), static_dir=FRONTEND_DIST)
