"""Build retrieval chunks from the parsed articles: one chunk per article, long articles split at
sentence / item boundaries, every chunk prefixed with its norm and article label.

Usage: python3 -I scripts/chunk_docs.py data/docs/ley_19886.jsonl data/docs/decreto_661.jsonl --out data/docs/chunks.jsonl
"""
import argparse
import json
import re
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
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
                            "version": r["version"], "text": f"{r['norma']}, {label}: {part}"})
    args.out.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in out), encoding="utf-8")
    lens = sorted(len(c["text"]) for c in out)
    print(f"{len(out)} chunks from {len({c['art_id'] for c in out})} articles; chars min/median/max = "
          f"{lens[0]}/{lens[len(lens)//2]}/{lens[-1]}")


if __name__ == "__main__":
    main()
