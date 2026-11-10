import duckdb
import pytest

from src.rag.retrievers.value_index import ValueIndex, ValueLookupConfig, format_hits, normalize


@pytest.fixture()
def index(tmp_path):
    db = tmp_path / "t.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE oc_lineas (NombreProveedor VARCHAR, es_persona_natural BOOLEAN, OrganismoPublico VARCHAR, "
                "RegionUnidadCompra VARCHAR)")
    rows = [("COPEC S.A.", False, "I MUNICIPALIDAD DE TEMUCO", "Región de la Araucanía "),
            ("COPEC S.A.", False, "HOSPITAL DE TEMUCO", "Región de la Araucanía "),
            ("JUAN PEREZ COPEC", True, "I MUNICIPALIDAD DE ARICA", "Región de Arica y Parinacota"),
            ("SIN RUT COPEC", None, "I MUNICIPALIDAD DE CALAMA", "Región de Antofagasta"),
            ("GASTON SPA", False, "I MUNICIPALIDAD DE IQUIQUE", "Región de Tarapacá"),
            ("MEDICAMENTOS DEL SUR", False, "I MUNICIPALIDAD DE OSORNO", "Región de los Lagos")]
    con.executemany("INSERT INTO oc_lineas VALUES (?, ?, ?, ?)", rows)
    con.close()
    cfg = ValueLookupConfig(table="oc_lineas", columns=("NombreProveedor", "OrganismoPublico", "RegionUnidadCompra"),
                            max_values_per_word=3, mask_column="NombreProveedor", flag_column="es_persona_natural")
    return ValueIndex.from_duckdb(db, cfg)


def test_normalize_drops_accents_case_and_punctuation():
    assert normalize("Sótero del Río, S.A.") == "sotero del rio s a"


def test_finds_supplier_by_partial_name(index):
    hits = index.lookup("¿Cuánto vendió Copec en pesos chilenos?")
    assert [(h.column, h.value) for h in hits] == [("NombreProveedor", "COPEC S.A.")]
    assert hits[0].lines == 2


def test_natural_persons_and_unknown_flags_are_never_indexed(index):
    values = {h.value for h in index.lookup("copec juan perez sin rut")}
    assert "JUAN PEREZ COPEC" not in values and "SIN RUT COPEC" not in values


def test_generic_word_only_breaks_ties(index):
    hits = index.lookup("¿Cuánto compró la Municipalidad de Temuco?")
    assert hits[0].value == "I MUNICIPALIDAD DE TEMUCO"
    assert all("temuco" in normalize(h.value) for h in hits)


def test_values_are_trimmed_and_small_columns_use_a_share_threshold(index):
    hits = index.lookup("órdenes de la región de la Araucanía")
    assert [h.value for h in hits] == ["Región de la Araucanía"]


def test_plural_matches_but_short_prefix_does_not(index):
    assert any(h.value == "MEDICAMENTOS DEL SUR" for h in index.lookup("gasto en medicamento"))
    assert not any(h.value == "GASTON SPA" for h in index.lookup("¿cuánto gasto hubo?"))


def test_format_hits_quotes_values_and_is_empty_without_hits(index):
    assert format_hits([]) == ""
    text = format_hits(index.lookup("copec"))
    assert "- NombreProveedor: 'COPEC S.A.' (2 líneas)" in text
