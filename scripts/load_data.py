# /// script
# requires-python = ">=3.11"
# dependencies = ["duckdb"]
# ///
"""Load ChileCompra purchase-order bulk files (monthly/semester zips) into DuckDB.

Usage:
    uv run scripts/load_data.py ~/Downloads/2026-1.zip [more.zip ...] [--db data/chilecompra.duckdb]

The source CSVs are cp1252, ';'-delimited, with 'NA' as null and quoted text that can
contain line breaks. One row = one purchase-order *line item* (order-level amounts are
repeated on every line of the order). Each zip is converted to UTF-8 under data/raw/
and appended to table `oc_lineas`; re-running a file replaces its rows.
"""
import argparse
import zipfile
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
DATE_COLS = ["FechaCreacion", "FechaEnvio", "FechaSolicitudCancelacion",
             "fechaUltimaModificacion", "FechaAceptacion", "FechaCancelacion"]
NUM_COLS = ["MontoTotalOC", "MontoTotalOC_PesosChilenos", "Impuestos", "Descuentos", "Cargos",
            "TotalNetoOC", "PromedioCalificacion", "CantidadEvaluacion", "PorcentajeIva",
            "cantidad", "precioNeto", "totalCargos", "totalDescuentos", "totalImpuestos",
            "totalLineaNeto"]


# Derived flag (assumption documented in domains/chilecompra/glossary.yaml, column RutSucursal):
# natural person <=> RUT number (no dots, no check digit) < 50,000,000. NULL when the RUT cannot be parsed,
# so the output guardrail masks NombreProveedor for true AND null.
PERSONA_NATURAL_EXPR = (
    "try_cast(replace(split_part(\"RutSucursal\", '-', 1), '.', '') as bigint) < 50000000"
)


def ensure_derived_columns(con: duckdb.DuckDBPyConnection) -> None:
    """Backfill es_persona_natural on a table created before the column existed (idempotent)."""
    has = con.execute(
        "select count(*) from information_schema.columns "
        "where table_name='oc_lineas' and column_name='es_persona_natural'"
    ).fetchall()[0][0]
    if not has:
        con.execute("alter table oc_lineas add column es_persona_natural boolean")
        con.execute(f"update oc_lineas set es_persona_natural = {PERSONA_NATURAL_EXPR}")
        print("es_persona_natural added and backfilled")


def to_utf8(src: Path, dst: Path) -> None:
    with src.open("rb") as fin, dst.open("w", encoding="utf-8", newline="") as fout:
        for chunk in iter(lambda: fin.read(1 << 24), b""):
            fout.write(chunk.decode("cp1252", errors="replace"))


def load(con: duckdb.DuckDBPyConnection, zip_path: Path) -> None:
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        member = z.namelist()[0]
        z.extract(member, raw)
    src = raw / member
    utf8 = raw / (src.stem + ".utf8.csv")
    to_utf8(src, utf8)
    src.unlink()

    replace = ", ".join(
        [f"try_cast(\"{c}\" as date) as \"{c}\"" for c in DATE_COLS]
        + [f"try_cast(\"{c}\" as double) as \"{c}\"" for c in NUM_COLS]
    )
    con.execute(f"""
        create or replace temp table _stage as
        select * replace ({replace}), '{zip_path.stem}' as archivo_origen,
               {PERSONA_NATURAL_EXPR} as es_persona_natural
        from read_csv('{utf8}', delim=';', quote='"', header=true, all_varchar=true,
                      nullstr='NA', ignore_errors=true, store_rejects=true)
    """)
    rejected = con.execute("select count(*) from reject_errors").fetchall()[0][0]
    exists = con.execute("select count(*) from information_schema.tables where table_name='oc_lineas'").fetchall()[0][0]
    if exists:
        con.execute("delete from oc_lineas where archivo_origen = ?", [zip_path.stem])
        con.execute("insert into oc_lineas select * from _stage")
    else:
        con.execute("create table oc_lineas as select * from _stage")
    n = con.execute("select count(*) from _stage").fetchall()[0][0]
    print(f"{zip_path.name}: {n:,} lines loaded, {rejected} rejected rows")
    utf8.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("zips", nargs="+", type=Path)
    ap.add_argument("--db", type=Path, default=ROOT / "data" / "chilecompra.duckdb")
    args = ap.parse_args()
    args.db.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(args.db))
    for z in args.zips:
        load(con, z.expanduser())
    ensure_derived_columns(con)
    print(con.sql("""
        select count(*) lines, count(distinct Codigo) orders, count(distinct CodigoProveedor) suppliers,
               count(distinct CodigoOrganismoPublico) public_bodies,
               min(FechaEnvio) first_sent, max(FechaEnvio) last_sent
        from oc_lineas""").fetchall())


if __name__ == "__main__":
    main()
