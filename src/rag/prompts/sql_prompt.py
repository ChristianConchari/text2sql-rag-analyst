"""Prompt texts for the router, SQL generator and composer, plus the schema context built from the glossary.

The prompt rules below still name ChileCompra columns (RutSucursal, NombreProveedor, es_persona_natural); moving
them into the domain config is pending. Columns in `hidden_columns` are never shown to the model.
"""
from __future__ import annotations

from pathlib import Path

import yaml

ROUTER_SYSTEM = """Eres el enrutador de un analista de compras públicas de Chile. Clasifica la pregunta del usuario.
Responde SOLO con JSON: {"route": "sql" | "docs" | "refuse"}
- sql: pregunta sobre datos de órdenes de compra (conteos, montos, rankings, filtros) de enero de 2026.
- docs: pregunta sobre la ley de compras públicas (Ley 19.886) o su reglamento (Decreto 661): qué dice, plazos,
  definiciones, artículos.
- refuse: pide modificar o borrar datos, leer archivos, datos personales, o no tiene relación con compras públicas."""

SQL_RULES = """Genera UNA sola consulta SQL en dialecto DuckDB sobre la tabla oc_lineas.
Reglas obligatorias:
- Solo un SELECT (o WITH ... SELECT). Nada de DDL/DML, ATTACH, COPY, PRAGMA, funciones de archivo o red.
- Solo la tabla oc_lineas. Sin comentarios SQL. No uses SELECT * (lista las columnas).
- No uses las columnas de datos personales RutSucursal ni Sucursal.
- Si seleccionas NombreProveedor, selecciona también es_persona_natural (si agrupas, agrúpala también).
- Una fila es una LÍNEA de orden: para contar o sumar por orden, deduplica por Codigo (ver reglas de negocio).
- Los nombres de columna con espacios o barras van entre comillas dobles.
- Responde SOLO con la consulta dentro de un bloque ```sql. Si la pregunta no se puede responder con estos datos
  (otro período, pagos efectuados, información inexistente), responde exactamente: NO_SQL: <motivo breve>."""

COMPOSER_SYSTEM = """Eres un analista de compras públicas. Redacta en español una respuesta breve y precisa a la pregunta
usando SOLO las filas entregadas. Indica supuestos relevantes (por ejemplo, que son órdenes de enero de 2026 y compromisos
de compra, no pagos). Si hay valores enmascarados, no intentes adivinarlos. No inventes cifras."""


CLIP_CHARS = 320            # per glossary field, to keep the prompt small
CLIP_CHARS_RULES = 400      # business rules matter more: allow a bit more


def _clip(text: str, n: int = CLIP_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def build_schema_context(glossary_path: str | Path, hidden_columns: set[str] | frozenset[str] = frozenset()) -> str:
    """Render the glossary (columns with desc/usar/ojo/regla_adoptada, then business rules) as prompt text."""
    g = yaml.safe_load(Path(glossary_path).read_text(encoding="utf-8"))
    hidden = {c.lower() for c in hidden_columns}
    lines = [f"Tabla {g['tabla']['nombre']}: {g['tabla']['grano']}.", "Columnas:"]
    for name, spec in g["columnas"].items():
        if name.lower() in hidden:
            continue
        quoted = f'"{name}"' if not name.isidentifier() else name
        parts = [_clip(spec.get("desc", ""))]
        for k in ("usar", "ojo", "regla_adoptada"):
            if spec.get(k):
                parts.append(f"{k}: {_clip(spec[k])}")
        lines.append(f"- {quoted}: " + " | ".join(parts))
    lines.append("Reglas de negocio:")
    lines += [f"- {r['tema']}: {_clip(r['regla'], CLIP_CHARS_RULES)}" for r in g.get("reglas_de_negocio", [])]
    return "\n".join(lines)


def build_sql_system(schema_context: str) -> str:
    """System prompt of the SQL generator: rules + schema context."""
    return SQL_RULES + "\n\n" + schema_context
