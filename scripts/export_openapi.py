"""Write the /api OpenAPI schema to docs/openapi.json (the frontend types are generated from it: `npm run gen:api`).

No model, database or retriever is loaded: the schema only depends on the FastAPI routes and pydantic models.
tests/unit/test_api_contract.py fails if the committed file is stale.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interfaces.fastapi_app.app import create_app  # noqa: E402

OUT = ROOT / "docs" / "openapi.json"


def openapi_json() -> str:
    return json.dumps(create_app(None).openapi(), ensure_ascii=False, indent=2) + "\n"   # type: ignore[arg-type]


if __name__ == "__main__":
    OUT.write_text(openapi_json(), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
