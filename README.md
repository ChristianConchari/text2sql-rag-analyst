# text2sql-rag-analyst

Final project for *Procesamiento de Lenguaje Natural III* (Master of AI, FIUBA / UBA). A conversational analyst over Chilean public procurement data with two tools behind a router: SQL written by one agent and verified by a second, and retrieval over the procurement law and its regulation (answers cite the article).

The architecture diagram is in `docs/arquitectura.html`. Scope, evaluation design and measurements are documented by the project authors outside this repository.

**Status:** data tooling, evaluation harness, the deterministic SQL verifier, the output guardrail and the orchestration skeleton (router, generator⇄verifier loop, composer, FastAPI) exist and are tested against a fake LLM. Document retrieval (RAG) is not wired into the pipeline yet, and the real model has not been run end to end.

## Quick start (fresh clone)

```bash
uv sync                                          # Python >= 3.11; creates .venv
uv run pytest -q                                 # the whole suite; needs no data and no model (it uses FakeLLM)
```

## Run the system

```bash
# Data (git-ignored, regenerate it): download the zip as in "Reproduce" step 1, then
uv run scripts/load_data.py ~/Downloads/2026-1.zip   # also computes the derived column es_persona_natural

# LLM: Ollama with the project model (any machine; no cost)
#   install Ollama from https://ollama.com, then:
ollama pull qwen2.5-coder:14b                    # ~9 GB, 4-bit; fits a 12 GB GPU
#   if Ollama runs on another host/tunnel: export LLM_BASE_URL=http://HOST:11434/v1   (or edit configs/model/llm.yaml)
uv run scripts/smoke_llm.py                      # reachability + model served + one tiny completion
#   IMPORTANT: the SQL prompt is ~17k characters (~5-6k tokens). Ollama's default context window is small and
#   truncates silently, so start the server with a larger one: OLLAMA_CONTEXT_LENGTH=8192 ollama serve
#   (or bake `PARAMETER num_ctx 8192` into a Modelfile). Not yet verified against the real model.

# API
uv run uvicorn app.main:app --reload             # POST /ask {"question": "..."}; DB path via DB_PATH
```

A database loaded before `es_persona_natural` existed is upgraded in place by running `load_data.py` once more
(or by `ensure_derived_columns`). Without that column the output guardrail masks every `NombreProveedor`.

## Reproduce the data and the evaluation

All commands run from this folder with `uv`.

```bash
# 1. Data: download the monthly purchase-order zip from https://datos-abiertos.chilecompra.cl/descargas
#    (section "Órdenes de compra y licitaciones", bulk download) into ~/Downloads, then:
uv run scripts/load_data.py ~/Downloads/2026-1.zip          # ~9 s, creates data/chilecompra.duckdb

# 2. Check the reference SQL and score a prediction file
uv run scripts/run_reference_queries.py --questions eval/candidate_questions_v2.yaml
uv run scripts/score_baseline.py eval/baseline_v2_full_schema.yaml --questions eval/candidate_questions_v2.yaml

# 3. Law corpus (BCN LeyChile XML; ids: Ley 19.886 = 213004, Decreto 661 = 1209290)
curl -L "https://www.bcn.cl/leychile/Consulta/obtxml?opt=7&idNorma=213004"  -o data/docs/bcn_213004.xml
curl -L "https://www.bcn.cl/leychile/Consulta/obtxml?opt=7&idNorma=1209290" -o data/docs/bcn_1209290.xml
python3 -I scripts/parse_bcn.py data/docs/bcn_213004.xml  data/docs/ley_19886.jsonl
python3 -I scripts/parse_bcn.py data/docs/bcn_1209290.xml data/docs/decreto_661.jsonl
python3 -I scripts/chunk_docs.py data/docs/ley_19886.jsonl data/docs/decreto_661.jsonl --out data/docs/chunks.jsonl

# 4. Retrieval comparison on CPU (first run downloads ~2 GB of models; embeddings are cached in data/cache/)
UV_TORCH_BACKEND=cpu uv run scripts/eval_retrieval.py --misses
```

Other source: monthly Compra Ágil quotations at `https://transparenciachc.blob.core.windows.net/trnspchc/COT_{year}-{month}.zip` (month zero-padded; ~70 MB zipped, ~1.1 GB extracted per month; joins with the orders by `CodigoOC`). Not loaded by any script yet.

## Layout

| Path | What |
|---|---|
| `scripts/load_data.py` | ChileCompra purchase-order zip → UTF-8 → typed DuckDB table `oc_lineas` (idempotent) |
| `scripts/run_reference_queries.py` | runs every reference SQL of a question file against the DB |
| `scripts/score_baseline.py` | execution-accuracy scorer (strict and lenient) for predicted SQL |
| `scripts/parse_bcn.py`, `scripts/chunk_docs.py` | BCN LeyChile XML → one chunk per article (`data/docs/chunks.jsonl`) |
| `scripts/eval_retrieval.py` | BM25 / dense / hybrid / re-ranker comparison on the document questions |
| `eval/candidate_questions.yaml`, `eval/candidate_questions_v2.yaml` | 21 + 25 SQL questions with reference SQL, kind and known traps |
| `eval/baseline_full_schema.yaml`, `eval/baseline_v2_full_schema.yaml` | raw baseline predictions (full schema, proxy model) |
| `eval/doc_questions.yaml` | 25 retrieval questions with gold articles (`gold`, `gold_extra`) |
| `src/policies/sql_verifier.py` | deterministic verifier (sqlglot): one read-only SELECT, allowed tables, LIMIT 100/1000, no comments, personal-data columns |
| `src/policies/policy.py` | `Policy` and `load_policy`: the domain rules the verifier enforces, read from `configs/domain/*.yaml` |
| `src/policies/masking.py` | finds the output columns to mask and appends the `es_persona_natural` flag to the query when safe |
| `src/policies/output_guardrail.py` | masks `NombreProveedor` unless the row has `es_persona_natural = FALSE` (fail-closed) |
| `src/agents/orchestration/` | router, generator⇄verifier loop (up to 3 retries), composer |
| `src/runtime/pipelines/text2sql.py`, `src/interfaces/fastapi_app/`, `app/main.py` | pipeline and API (`POST /ask`, `GET /health`) |
| `src/toolkits/db/executor.py` | read-only DuckDB execution with external access disabled |
| `src/rag/prompts/sql_prompt.py` | router / generator / composer prompts and the schema context built from the glossary |
| `src/runtime/providers/llm_client.py` | LLM client behind `configs/model/llm.yaml` + deterministic `FakeLLM` |
| `configs/domain/chilecompra.yaml`, `domains/chilecompra/glossary.yaml` | domain policy (tables, limits, blocked columns) and column glossary with evidence status |
| `eval/attack_cases.yaml`, `tests/unit/` | attack cases for the verifier and the pytest suite |
| `docs/arquitectura.html` | architecture diagram page |
| `src/rag/{retrievers,ingestion,evaluators}`, `src/vectorstore/chroma`, `src/common/telemetry` | empty on purpose: reserved for the document-retrieval (RAG) work and telemetry |
| `data/` | **git-ignored**: DuckDB file, law XML/PDF, chunks, embedding cache, logs |

## Pending

- Move the ChileCompra column names out of the SQL prompt (`RutSucursal`, `NombreProveedor`, `es_persona_natural` in `src/rag/prompts/sql_prompt.py`) into the domain config. Deliberately postponed until the real model has been tried, because it changes what the model sees.
- Wire document retrieval (RAG) into the pipeline; today the router's `docs` route answers `not_implemented`.
- Run the pipeline against the real model (Ollama) and check that the prompt fits the context window.

## Notes and cautions

- **The local model has not been run yet** (Ollama + `qwen2.5-coder:14b` is the chosen setup; see Setup). The orchestration is tested with `FakeLLM` only. The SQL baselines were run with Claude subagents as a proxy: one prompt per batch, full schema, no tools. Replace with the real model before the final measurements.
- **Personal-data assumption:** a supplier is a natural person when the number of `RutSucursal` is < 50,000,000 (documented, not verifiable in the data). `NombreProveedor` is masked for those, for unparseable RUTs, and whenever the flag is missing from a result.
- **Personal data:** the quotation files contain contact names and, in a free-text field, e-mails and phone numbers of supplier representatives. Drop or mask them when loading, never copy them into a repo, and keep extracted copies out of `data/` once used.
- The reference SQL and the gold articles were written by the assistant and are **not validated** by a domain or legal expert.
- `data/`, `.env` and `.venv/` are git-ignored; nothing in `data/` is needed to run the tests.

## License

MIT, see `LICENSE`. It covers the code only: the law texts and the ChileCompra data are not redistributed here and keep their own terms.
