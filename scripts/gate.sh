#!/usr/bin/env bash
# Repo gate: exit 0 = green. Runs the pytest suite with the frozen lock.
# Needs no Ollama and no data/ (the tests use FakeLLM); network only if uv has to download something.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Local settings must not change the result.
unset VIRTUAL_ENV UV_PROJECT_ENVIRONMENT LLM_BASE_URL LLM_MODEL DB_PATH PYTEST_ADDOPTS PYTHONPATH
# The repo pins no Python version; 3.12 unless GATE_PYTHON says otherwise.
uv run --frozen --python "${GATE_PYTHON:-3.12}" pytest -q
