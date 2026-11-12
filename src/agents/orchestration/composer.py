"""Composer: turns (already masked) rows into a short Spanish answer.

Only sees what the output guardrail let through. Shows at most MAX_ROWS_IN_PROMPT rows to the model and says so.
If the model fails it falls back to a plain-text table instead of raising.
"""
from __future__ import annotations

import json

from src.rag.prompts.sql_prompt import COMPOSER_SYSTEM
from src.runtime.providers.llm_client import ChatModel, LLMError

MAX_ROWS_IN_PROMPT = 20


def _table(columns: list[str], rows: list[tuple]) -> str:
    lines = [" | ".join(columns)] + [" | ".join(str(v) for v in r) for r in rows[:MAX_ROWS_IN_PROMPT]]
    return "\n".join(lines)


def compose(question: str, columns: list[str], rows: list[tuple], llm: ChatModel,
            static_prefix: str | None = None) -> str:
    """Write the answer for `question` from masked `rows`. `static_prefix`: see `route_question`."""
    system = COMPOSER_SYSTEM if static_prefix is None else static_prefix + "\n\n" + COMPOSER_SYSTEM
    shown = rows[:MAX_ROWS_IN_PROMPT]
    note = f"\n(Se muestran {len(shown)} de {len(rows)} filas.)" if len(rows) > len(shown) else ""
    payload = json.dumps({"columnas": columns, "filas": shown}, ensure_ascii=False, default=str)
    try:
        text = llm.chat([{"role": "system", "content": system},
                         {"role": "user", "content": f"Pregunta: {question}\nResultado: {payload}"}], max_tokens=400)
    except LLMError:
        text = "No pude redactar la respuesta; este es el resultado:\n" + _table(columns, rows)
    return text.strip() + note
