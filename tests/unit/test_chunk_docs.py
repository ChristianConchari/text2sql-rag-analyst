import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("chunk_docs", ROOT / "scripts" / "chunk_docs.py")
chunk_docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chunk_docs)


def test_article_number():
    assert chunk_docs.article_number("Artículo 8 bis") == "8 bis"
    assert chunk_docs.article_number("Articulo 12") == "12"
    assert chunk_docs.article_number("(sin etiqueta 3)") == "(sin etiqueta 3)"


def test_manifest_hashes_xml_and_pdf_only(tmp_path):
    (tmp_path / "a.xml").write_bytes(b"<x/>")
    (tmp_path / "b.pdf").write_bytes(b"%PDF")
    (tmp_path / "c.jsonl").write_text("{}")
    dest = tmp_path / "manifest.json"
    assert chunk_docs.write_manifest(tmp_path, dest) == 2
    files = json.loads(dest.read_text())["files"]
    assert [f["file"] for f in files] == ["a.xml", "b.pdf"]
    assert len(files[0]["sha256"]) == 64 and files[0]["bytes"] == 4
