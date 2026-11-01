"""ASGI entrypoint: `uv run uvicorn app.main:app`.

Reads the LLM endpoint from configs/model/llm.yaml (LLM_BASE_URL overrides) and the database from DB_PATH.
"""
from src.interfaces.fastapi_app.app import create_app
from src.runtime.pipelines.text2sql import build_default_pipeline

app = create_app(build_default_pipeline())
