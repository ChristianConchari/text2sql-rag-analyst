import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.runtime.providers.llm_client import LLMClient, LLMConfig, LLMError, load_config


def make_server(models=("qwen2.5-coder:14b",), reply="OK"):
    """Start a tiny OpenAI-compatible server on a free local port; returns (server, base_url)."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/v1/models":
                self._send({"data": [{"id": m} for m in models]})
            else:
                self._send({}, 404)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["stream"] is False and body["temperature"] == 0
            self._send({"choices": [{"message": {"content": reply}}]})
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}/v1"


def cfg(url): return LLMConfig("local", url, "qwen2.5-coder:14b", 0.0, 5)


def test_smoke_ok():
    srv, url = make_server()
    try:
        r = LLMClient(cfg(url)).smoke()
        assert r["reply"] == "OK" and r["model"] == "qwen2.5-coder:14b"
    finally:
        srv.shutdown()


def test_model_not_served():
    srv, url = make_server(models=("otro",))
    try:
        with pytest.raises(LLMError, match="not served"):
            LLMClient(cfg(url)).smoke()
    finally:
        srv.shutdown()


def test_unreachable_is_llmerror():
    with pytest.raises(LLMError, match="cannot reach"):
        LLMClient(cfg("http://127.0.0.1:1/v1")).list_models()


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://tunel:11434/v1/")
    c = load_config()
    assert c.base_url == "http://tunel:11434/v1" and c.provider == "local"


def test_unimplemented_provider(tmp_path):
    p = tmp_path / "l.yaml"
    p.write_text("provider: anthropic\nanthropic: {model: x}\n")
    with pytest.raises(LLMError, match="not implemented"):
        load_config(p)
