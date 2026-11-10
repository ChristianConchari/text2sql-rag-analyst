"""Value lookup: find the real database values a question refers to ("Copec" -> 'COPEC S.A.').

The generator cannot see the data, so it either guesses spellings or declines ("no information about Copec").
This index holds the distinct values of a few configured columns and, for a question, returns the values whose
words match the question's words (accent- and case-insensitive). No LLM, no embeddings: deterministic and cheap.

Words that match too many values of a column ("municipalidad", "hospital") are ignored for that column, so the
specific word ("temuco") decides; they only break ties. Values of the masked column (supplier names) are indexed
only where the row flag says it is NOT a natural person, so the model never sees a natural person's name.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import yaml

STOPWORDS = frozenset("""
de del la las el los en y o a al por para con sin que se su sus un una uno unos unas es son fue fueron como mas
menos cual cuales cuanto cuanta cuantos cuantas donde quien quienes cada entre sobre total monto montos pesos
chilenos chile orden ordenes compra compras compro compraron vendio vendieron proveedor proveedores organismo
organismos enero 2026 mes año dame muestra lista top mayor mayores menor menores numero cantidad gasto gasta
gastaron gastar emitio emitieron emitidas recibio recibieron enviaron envio tiene tienen hay region sector
""".split())


@dataclass(frozen=True)
class ValueLookupConfig:
    """`columns` to index; masked column/flag come from the policy's masking section."""

    table: str
    columns: tuple[str, ...]
    max_per_column: int = 5
    max_total: int = 20
    max_values_per_word: int = 30     # a word matching more values than this is "generic" for that column...
    max_share_per_word: float = 0.2   # ...or more than this share of the column's values (small columns)
    min_relative_score: float = 0.6   # drop hits scoring below this fraction of the best hit
    min_word_len: int = 3
    mask_column: str | None = None
    flag_column: str | None = None


@dataclass(frozen=True)
class ValueHit:
    column: str
    value: str          # trimmed
    lines: int
    score: int


@dataclass
class _ColumnIndex:
    values: list[str] = field(default_factory=list)
    lines: list[int] = field(default_factory=list)
    words: dict[str, set[int]] = field(default_factory=lambda: defaultdict(set))   # word -> value ids


def normalize(text: str) -> str:
    """Lower case, no accents, only letters/digits separated by single spaces."""
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode().lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def load_value_lookup_config(path: str | Path) -> ValueLookupConfig | None:
    """Read `value_lookup` (and `masking`) from a domain YAML; None if the section is absent."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    vl = cfg.get("value_lookup")
    if not vl:
        return None
    masking = cfg.get("masking") or {}
    return ValueLookupConfig(
        table=vl.get("table") or cfg["allowed_tables"][0],
        columns=tuple(vl["columns"]),
        max_per_column=int(vl.get("max_per_column", 5)),
        max_total=int(vl.get("max_total", 20)),
        max_values_per_word=int(vl.get("max_values_per_word", 30)),
        max_share_per_word=float(vl.get("max_share_per_word", 0.2)),
        min_relative_score=float(vl.get("min_relative_score", 0.6)),
        mask_column=masking.get("column"),
        flag_column=masking.get("flag_column"),
    )


class ValueIndex:
    """In-memory index of distinct values per column. Build once with `from_duckdb`."""

    def __init__(self, config: ValueLookupConfig, columns: dict[str, _ColumnIndex]):
        self.config, self._columns = config, columns

    @classmethod
    def from_duckdb(cls, db_path: str | Path, config: ValueLookupConfig) -> ValueIndex:
        con = duckdb.connect(str(db_path), read_only=True, config={"enable_external_access": False})
        try:
            columns = {c: cls._index_column(con, config, c) for c in config.columns}
        finally:
            con.close()
        return cls(config, columns)

    @staticmethod
    def _index_column(con: duckdb.DuckDBPyConnection, config: ValueLookupConfig, column: str) -> _ColumnIndex:
        col = '"' + column.replace('"', '""') + '"'
        where = f"{col} IS NOT NULL AND trim({col}) <> ''"
        if config.mask_column and column.lower() == config.mask_column.lower():
            flag = '"' + config.flag_column.replace('"', '""') + '"'
            # only values that are NOT a natural person on every row; NULL flag counts as unknown -> excluded
            having = f"HAVING bool_and(coalesce({flag} = FALSE, FALSE))"
        else:
            having = ""
        rows = con.execute(f'SELECT trim({col}) v, count(*) n FROM "{config.table}" WHERE {where} '
                           f"GROUP BY 1 {having}").fetchall()
        idx = _ColumnIndex()
        for i, (value, n) in enumerate(rows):
            idx.values.append(value)
            idx.lines.append(int(n))
            for word in set(normalize(value).split()):
                idx.words[word].add(i)
        return idx

    def lookup(self, question: str) -> list[ValueHit]:
        """Values whose words match the question's specific words, best first, capped by the config."""
        cfg = self.config
        tokens = [t for t in dict.fromkeys(normalize(question).split())
                  if len(t) >= cfg.min_word_len and t not in STOPWORDS]
        hits: list[ValueHit] = []
        for column, idx in self._columns.items():
            limit = min(cfg.max_values_per_word, max(2, int(cfg.max_share_per_word * len(idx.values))))
            specific: dict[int, int] = defaultdict(int)
            generic: dict[int, int] = defaultdict(int)
            for token in tokens:
                ids = self._match(idx, token)
                target = specific if 0 < len(ids) <= limit else generic
                for i in ids:
                    target[i] += 1
            ranked = sorted(specific, key=lambda i: (-(specific[i] * 10 + generic.get(i, 0)), -idx.lines[i]))
            hits += [ValueHit(column, idx.values[i], idx.lines[i], specific[i] * 10 + generic.get(i, 0))
                     for i in ranked[:cfg.max_per_column]]
        hits.sort(key=lambda h: (-h.score, -h.lines))
        if hits:
            hits = [h for h in hits if h.score >= cfg.min_relative_score * hits[0].score]
        return hits[:cfg.max_total]

    @staticmethod
    def _match(idx: _ColumnIndex, token: str) -> set[int]:
        """Exact word match; tokens of 5+ letters also match words that differ only by a short ending
        (plural/singular: 'medicamento' ~ 'medicamentos'), never a longer word ('gasto' does not match 'gaston'
        unless the ending is at most 2 letters: that is the price of matching plurals)."""
        found = set(idx.words.get(token, ()))
        if len(token) >= 5:
            for word, ids in idx.words.items():
                if word == token or len(word) < 5:
                    continue
                longer, shorter = (word, token) if len(word) > len(token) else (token, word)
                if longer.startswith(shorter) and len(longer) - len(shorter) <= 2 and longer[len(shorter):] in ("s", "es"):
                    found |= ids
        return found


def format_hits(hits: list[ValueHit]) -> str:
    """Prompt text for the generator; empty string when nothing matched."""
    if not hits:
        return ""
    by_column: dict[str, list[ValueHit]] = defaultdict(list)
    for h in hits:
        by_column[h.column].append(h)
    lines = ["Valores reales de la base que coinciden con palabras de la pregunta (escríbelos tal cual; compara con "
             "trim(columna) = 'valor' o con ILIKE, porque algunos valores tienen espacios al final):"]
    for column, column_hits in by_column.items():
        quoted = f'"{column}"' if not column.isidentifier() else column
        shown = ", ".join("'{}' ({} líneas)".format(h.value.replace("'", "''"), f"{h.lines:,}".replace(",", "."))
                          for h in column_hits)
        lines.append(f"- {quoted}: {shown}")
    return "\n".join(lines)
