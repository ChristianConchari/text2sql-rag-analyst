"""Build retrieval chunks from the parsed articles: one chunk per article, long articles split at
sentence / item boundaries, every chunk prefixed with its norm and article label.

Usage: python3 -I scripts/chunk_docs.py data/docs/ley_19886.jsonl data/docs/decreto_661.jsonl --out data/docs/chunks.jsonl
           [--manifest data/docs/manifest.json]

Each chunk carries: id, art_id, norma, label, article_number, version (norm), article_version, vigencia and the
source file name + SHA-256, so any citation can be traced to the exact downloaded file. `--manifest` also writes
the SHA-256, size and modification date of every .xml/.pdf next to `--out`. Standard library only.
"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

MAX_CHARS = 1200


def split_units(text: str) -> list[str]:
    return [u for u in re.split(r"(?<=[\.;:])\s+", text) if u]


def pack(units: list[str], limit: int) -> list[str]:
    chunks, cur = [], ""
    for u in units:
        if cur and len(cur) + len(u) + 1 > limit:
            chunks.append(cur)
            cur = u
        else:
            cur = f"{cur} {u}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def article_number(label: str) -> str:
    """"Artículo 8 bis" -> "8 bis"; labels without the prefix are returned unchanged."""
    return re.sub(r"^Art[íi]culo\s+", "", label)


def write_manifest(directory: Path, dest: Path) -> int:
    """Record sha256, size and modification date of every .xml/.pdf in `directory`. Returns the file count."""
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in (".xml", ".pdf"))
    entries = [{"file": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size,
                "modified": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).date().isoformat()} for p in files]
    dest.write_text(json.dumps({"files": entries}, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(entries)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, help="also write a hash manifest of the .xml/.pdf files next to --out")
    args = ap.parse_args()
    out = []
    for path in args.inputs:
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            r = json.loads(line)
            label = r["label"] if r["label"] and r["label"] != " (transitorio)" else f"(sin etiqueta {i})"
            art_id = f"{r['norma']}|{label}"
            parts = pack(split_units(r["text"]), MAX_CHARS) if len(r["text"]) > MAX_CHARS else [r["text"]]
            for j, part in enumerate(parts):
                out.append({"id": f"{art_id}#{j}", "art_id": art_id, "norma": r["norma"], "label": label,
                            "article_number": article_number(label), "version": r["version"],
                            "article_version": r.get("article_version"), "vigencia": r.get("vigencia"),
                            "source_file": r.get("source_file"), "source_sha256": r.get("source_sha256"), "text": f"{r['norma']}, {label}: {part}"})
    args.out.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in out), encoding="utf-8")
    lens = sorted(len(c["text"]) for c in out)
    print(f"{len(out)} chunks from {len({c['art_id'] for c in out})} articles; chars min/median/max = "
          f"{lens[0]}/{lens[len(lens)//2]}/{lens[-1]}")
    if args.manifest:
        print(f"manifest: {write_manifest(args.out.parent, args.manifest)} files -> {args.manifest}")


if __name__ == "__main__":
    main()
