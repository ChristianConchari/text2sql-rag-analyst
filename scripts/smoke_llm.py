# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Smoke test for the LLM endpoint configured in configs/model/llm.yaml.

Usage: uv run scripts/smoke_llm.py [--base-url http://host:11434/v1] [--model qwen2.5-coder:14b]
Overrides also via LLM_BASE_URL / LLM_MODEL. Exit code 0 = reachable, model served, completion returned.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.runtime.providers.llm_client import LLMClient, LLMError  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base-url")
ap.add_argument("--model")
args = ap.parse_args()
if args.base_url:
    os.environ["LLM_BASE_URL"] = args.base_url
if args.model:
    os.environ["LLM_MODEL"] = args.model

try:
    client = LLMClient.from_config()
    print(f"-> {client.config.base_url}  model={client.config.model}")
    print("OK", client.smoke())
except LLMError as e:
    print("FAIL:", e)
    sys.exit(1)
