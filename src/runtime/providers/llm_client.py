"""LLM access behind configs/model/llm.yaml: a `ChatModel` interface, an HTTP client and a deterministic fake.

`LLMClient` talks to any OpenAI-compatible endpoint (Ollama's /v1) with the standard library only. The provider
and endpoint live in the config, never in code; `LLM_BASE_URL` / `LLM_MODEL` override the YAML (e.g. a tunnel to
a remote Ollama). Only provider "local" is implemented. Network failures surface as `LLMError`, never as
silent empty answers. `FakeLLM` replays scripted replies so orchestration is testable without a model.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "model" / "llm.yaml"


class LLMError(RuntimeError):
    """The model endpoint is unreachable, returned an error, or answered with an unexpected shape."""


class ChatModel(Protocol):
    """What the orchestration needs from a model: one chat turn in, text out. `LLMClient` and `FakeLLM` fit."""

    def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float | None = None) -> str: ...


@dataclass(frozen=True)
class LLMConfig:
    """Resolved settings (YAML + environment overrides)."""

    provider: str
    base_url: str
    model: str
    temperature: float = 0.0
    timeout_s: float = 120.0


def load_config(path: str | Path = DEFAULT_CONFIG) -> LLMConfig:
    """Read the YAML and apply LLM_BASE_URL / LLM_MODEL. Raises LLMError for a provider that is not implemented."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    provider = cfg["provider"]
    if provider != "local":
        raise LLMError(f"provider '{provider}' is not implemented yet; only 'local' (OpenAI-compatible, e.g. Ollama)")
    loc = cfg["local"]
    return LLMConfig(
        provider=provider,
        base_url=(os.environ.get("LLM_BASE_URL") or loc["base_url"]).rstrip("/"),
        model=os.environ.get("LLM_MODEL") or loc["model"],
        temperature=float(loc.get("temperature", 0)),
        timeout_s=float(loc.get("timeout_s", 120)),
    )


class LLMClient:
    """Blocking client for an OpenAI-compatible chat endpoint."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @classmethod
    def from_config(cls, path: str | Path = DEFAULT_CONFIG) -> LLMClient:
        return cls(load_config(path))

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.config.base_url + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.config.timeout_s) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise LLMError(f"HTTP {e.code} from {req.full_url}: {e.read()[:200]!r}") from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise LLMError(f"cannot reach {req.full_url}: {e}") from e
        except json.JSONDecodeError as e:
            raise LLMError(f"non-JSON response from {req.full_url}") from e

    def list_models(self, timeout: float = 10) -> list[str]:
        """Ids of the models the endpoint serves."""
        return [m["id"] for m in self._request("GET", "/models", timeout=timeout).get("data", [])]

    def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float | None = None) -> str:
        """One non-streaming chat completion. `temperature=None` uses the configured one."""
        out = self._request("POST", "/chat/completions", {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": self.config.temperature if temperature is None else temperature,
            "stream": False,
        })
        try:
            return out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response shape: {str(out)[:200]}") from e

    def smoke(self) -> dict:
        """Reachability + model availability + one tiny completion. Returns timings; raises LLMError."""
        t0 = time.time()
        models = self.list_models()
        t_models = time.time() - t0
        if self.config.model not in models:
            raise LLMError(f"model '{self.config.model}' not served at {self.config.base_url}; available: {models}")
        t1 = time.time()
        reply = self.chat([{"role": "user", "content": "Responde solo con la palabra OK."}], max_tokens=8)
        return {"base_url": self.config.base_url, "model": self.config.model, "models_s": round(t_models, 2),
                "chat_s": round(time.time() - t1, 2), "reply": reply.strip()}


class FakeLLM:
    """Deterministic stand-in for the model, to test orchestration without a server.

    `responses`: replies consumed in order (one per call). `rules`: (substring, reply) pairs checked first
    against the full text of the messages; the first match wins. Every call is recorded in `calls`.
    Raises LLMError when it has nothing to say, so tests fail loudly instead of looping.
    """

    def __init__(self, responses: list[str] | None = None, rules: list[tuple[str, str]] | None = None):
        self._responses = list(responses or [])
        self._rules = list(rules or [])
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], max_tokens: int = 512, temperature: float | None = None) -> str:
        self.calls.append([dict(m) for m in messages])
        text = "\n".join(m["content"] for m in messages)
        for needle, reply in self._rules:
            if needle in text:
                return reply
        if self._responses:
            return self._responses.pop(0)
        raise LLMError("FakeLLM has no scripted reply left")
