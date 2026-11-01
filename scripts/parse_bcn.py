"""Split BCN LeyChile XML (obtxml?opt=7) into one chunk per article (JSONL).

Usage: python3 -I scripts/parse_bcn.py data/docs/bcn_213004.xml data/docs/ley_19886.jsonl
Each line: {norma, norma_id, version, article, title, text}. Standard library only.
"""
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

NS = {"l": "http://www.leychile.cl/esquemas"}


def text_of(el):
    return " ".join("".join(el.itertext()).split())


def main(src: Path, dst: Path) -> None:
    root = ET.parse(src).getroot()
    ident = root.find("l:Identificador/l:TiposNumeros/l:TipoNumero", NS)
    norma = f"{ident.findtext('l:Tipo', namespaces=NS)} {ident.findtext('l:Numero', namespaces=NS)}"
    title = text_of(root.find("l:Encabezado/l:Titulo", NS)) if root.find("l:Encabezado/l:Titulo", NS) is not None else ""
    out = []
    seen = set()
    for ef in root.iter(f"{{{NS['l']}}}EstructuraFuncional"):
        if ef.get("tipoParte") not in ("Artículo", "Articulo"):
            continue
        t = ef.find("l:Texto", NS)
        if t is None:
            continue
        txt = text_of(t)
        m = re.match(r"(Art[íi]culo\s+(?:\d+|[a-záéíóú]+)(?:\s+(?:bis|ter|quáter|quater|quinquies|sexies|septies|octies|nonies|decies|undecies|duodecies))?)", txt)
        label = m.group(1) if m else ""
        if label in seen:  # numbering restarts in the transitory provisions
            label += " (transitorio)"
        seen.add(label)
        out.append({"norma": norma, "label": label, "norma_id": root.get("normaId"), "version": root.get("fechaVersion"),
                    "nombre_parte": ef.findtext("l:NombreParte", namespaces=NS) or "", "text": txt})
    dst.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out), encoding="utf-8")
    lens = sorted(len(r["text"]) for r in out)
    print(f"{norma} ({root.get('fechaVersion')}): {len(out)} articles; chars min/median/max = "
          f"{lens[0]}/{lens[len(lens)//2]}/{lens[-1]}; title: {title[:90]}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
