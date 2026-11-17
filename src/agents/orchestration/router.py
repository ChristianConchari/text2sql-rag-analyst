"""Router: asks the model which tool should answer a question (SQL, document retrieval), whether it is a harmless
question outside the domain (out_of_scope: no tool runs), or an unsafe request to refuse.

Never guesses: an unparseable or ambiguous reply, or a model failure, yields "unknown" so the caller can ask the
user to rephrase instead of running the wrong tool.
"""
from __future__ import annotations

import json
import re

from src.rag.prompts.sql_prompt import ROUTER_SYSTEM
from src.runtime.providers.llm_client import ChatModel, LLMError

ROUTES = ("sql", "docs", "out_of_scope", "refuse")


def route_question(question: str, llm: ChatModel, static_prefix: str | None = None,
                   instruction: str = ROUTER_SYSTEM) -> str:
    """Return 'sql' | 'docs' | 'out_of_scope' | 'refuse' | 'unknown' (unparseable or LLM failure: never guess).

    `static_prefix` (layout cache_friendly): text placed before the unchanged router instruction so the model
    server can reuse its prefix cache across agents. The instruction itself is not modified.
    `instruction`: the router prompt; overridden only by evaluation scripts that compare prompt versions.
    """
    system = instruction if static_prefix is None else static_prefix + "\n\n" + instruction
    try:
        raw = llm.chat([{"role": "system", "content": system}, {"role": "user", "content": question}],
                       max_tokens=32, temperature=0)
    except LLMError:
        return "unknown"
    m = re.search(r"\{.*?\}", raw, re.S)
    if m:
        try:
            r = str(json.loads(m.group(0)).get("route", "")).lower()
            return r if r in ROUTES else "unknown"
        except (json.JSONDecodeError, AttributeError):
            pass
    found = [r for r in ROUTES if re.search(rf"\b{r}\b", raw.lower())]
    return found[0] if len(found) == 1 else "unknown"
