# Project file reference

This document describes **what each file in the repository is for**. Paths
are relative to the project root (`mt_mvp/`).

---

## Root directory

| File | Purpose |
|------|---------|
| **`README_ENG.md`** | English-language project documentation: setup, configuration, API overview, pipeline, evaluation results, how to run with Docker. |
| **`README_RUS.md`** | Russian translation of `README_ENG.md` (kept in sync). |
| **`PROJECT_FILES.md`** | This file — a per-file map of the codebase and assets. |
| **`requirements.txt`** | Runtime Python deps: FastAPI, Transformers, CTranslate2, PyMuPDF, Gradio, pytest. Torch is installed separately or via `requirements-gpu.txt`. |
| **`requirements-gpu.txt`** | Adds the **CUDA 12.4** PyTorch wheels (`torch==2.6.0+cu124`) on top of `requirements.txt`. Pinned to ≥ 2.6 because `transformers` 4.46+ refuses to load `.bin` checkpoints on older torch (CVE-2025-32434). |
| **`requirements-eval.txt`** | Evaluation-only stack: `unbabel-comet`, `bert-score`, `matplotlib`, `fasttext-wheel` (kept out of the runtime image because it is heavy). |
| **`Dockerfile`** | Multi-stage build with **four** targets: `base` / `base-gpu` install Python deps into `/opt/venv` (CPU torch on `python:3.11-slim`; cu124 torch on `nvidia/cuda:12.4.1-runtime-ubuntu22.04`), and `runtime` / `runtime-gpu` copy the matching venv + app sources, drop to UID-10001, expose `:8000`, run Uvicorn. |
| **`docker-compose.yml`** | Four service profiles: `default` (CPU API + `/gradio`), `gpu` (builds the `runtime-gpu` target and reserves NVIDIA devices), `demo` (Gradio with `--share` for client links), `prewarm` (one-shot job that primes the `hf_cache` volume via `scripts/prewarm_hf_cache.py`). |
| **`.dockerignore`** | Keeps the build context lean (excludes `models/`, `data/csv_final/`, `translated_data_final/`, `results_final/`, `.venv/`, etc.). |
| **`.env.example`** | Template for local `MT_MVP_*` env-var overrides; copy to `.env` and edit. |
| **`.gitignore`** | Ignores venvs, caches, `.env`, local weights under `models/opus-mt-en-es-ct2/`, etc. |
| **`pytest.ini`** | `pythonpath = .`, `testpaths = tests`. |
| **`raw_glossary.xlsx`** | Source glossary (English / Spanish, no header). Compiled into `glossary/en_es_shotblasting.json` by `scripts/compile_glossary_from_xlsx.py`. |

---

## `app/` — Python application package

| File | Purpose |
|------|---------|
| **`app/__init__.py`** | Marks `app` as a package. |
| **`app/__main__.py`** | `python -m app` boots Uvicorn with reload (dev entry point). |
| **`app/main.py`** | FastAPI factory: CORS, mounts `/api/v1` router and `/static`, serves `/` as `frontend/index.html`, mounts the Gradio UI at `/gradio` when `MT_MVP_ENABLE_GRADIO=1`. |
| **`app/gradio_app.py`** | Gradio demo UI (used by `/gradio` mount and the standalone client-demo entrypoint `python -m app.gradio_app --share`). Calls the same `run_translate` function as the REST API, so behaviour is identical. |

### `app/core/`

| File | Purpose |
|------|---------|
| **`app/core/__init__.py`** | Marks `app.core` as a subpackage. |
| **`app/core/config.py`** | Central settings via Pydantic `BaseSettings`. Reads `MT_MVP_*` env vars and `.env`. Covers glossary path, post-edit prompt, MT engine + model name (`ctranslate2` / `marian_hf` / `nllb`), CT2 directory + compute type, NLLB language codes + beams + dtype, device, Qwen flags and token caps. |

### `app/api/`

| File | Purpose |
|------|---------|
| **`app/api/__init__.py`** | Marks `app.api` as the HTTP subpackage. |
| **`app/api/schemas.py`** | Pydantic `TranslateRequest` / `TranslateResponse`. |
| **`app/api/deps.py`** | FastAPI dependencies: cached singletons for MT engine (CT2, Marian or NLLB-200), Glossary (by path), Qwen post-edit service, PostEditor. Includes graceful CT2 → Marian fallback if `model.bin` is missing or fails to load, so the API stays functional even without converted weights. |
| **`app/api/routes.py`** | HTTP routes: `POST /translate`, `GET /health`, `POST /translate/cache/clear`. |

### `app/services/`

| File | Purpose |
|------|---------|
| **`app/services/__init__.py`** | Marks `app.services` as the business-logic subpackage. |
| **`app/services/translation.py`** | Core translation orchestration (no HTTP). Wires the ALL-CAPS preprocessor (sentence-case before MT) and UPPER postprocessor around glossary + MT + post-edit. Builds the optional `debug` payload when `include_debug=true`. |
| **`app/services/text_case.py`** | ALL-CAPS detector + sentence-case preprocessor used by `translation.py` and `scripts/translate_csv.py`. Threshold 60 % uppercase letters with an 8-alpha-char floor; restores upper case on the output after post-edit. |
| **`app/services/translate_cache.py`** | In-memory LRU+TTL response cache plus stable key builder (text + flags + MT mode + glossary/prompt mtimes + Qwen knobs). |
| **`app/services/glossary.py`** | Glossary engine: load JSON, sort entries by source length, **protect** (word-bounded placeholders), **enforce placeholders** (tolerant of `__GLS{i}_` truncation), **enforce phrases in target** (English-leak fix), **reassert targets after edit** (accent/case-insensitive + fallback-to-reference when Qwen drops a canonical term), **spacing around multi-word targets**. |
| **`app/services/ct2_engine.py`** | CTranslate2 backend (default): loads `model.bin`, MarianTokenizer from HF, `translate_batch`. |
| **`app/services/mt_engine.py`** | Hugging Face Marian backend (pure PyTorch fallback). |
| **`app/services/nllb_engine.py`** | HuggingFace NLLB-200 backend (alternative). `AutoTokenizer` with `src_lang=eng_Latn`, `forced_bos_token_id` for `spa_Latn`. Supports `distilled-600M`, `distilled-1.3B`, `3.3B`; bf16 on CUDA, fp32 on CPU. |
| **`app/services/qwen_postedit.py`** | Loads Qwen2.5 Instruct with `device_map={"": 0}`, runs `generate` on a chat template (system = post-edit markdown, user = EN + draft ES). |
| **`app/services/postedit.py`** | `PostEditor`: optional Qwen pass → glossary reassert → English-leak fix → spacing fix → whitespace normalise. |

---

## `frontend/` — Static web UI

| File | Purpose |
|------|---------|
| **`frontend/index.html`** | Single-page UI: hero copy, English textarea, sample chips, glossary/post-edit/debug toggles, Spanish output, cache-hint pill, link to `/docs` and `/gradio`. |
| **`frontend/app.js`** | Fetches `POST /api/v1/translate`, handles errors, fills examples from chip clicks, surfaces the `from_cache` flag. |
| **`frontend/styles.css`** | Industrial / safety theme (dark + amber). |

---

## `glossary/`

| File | Purpose |
|------|---------|
| **`glossary/en_es_shotblasting.json`** | Runtime glossary, 101 entries (78 hand-curated + 23 verbatim brand and standards supplements: Spartan, Panblast, Titan, Apollo, Galaxy, NPT, BSP, OSHA, NIOSH, ISO, etc.). The API reads its path from settings. |

---

## `prompts/`

| File | Purpose |
|------|---------|
| **`prompts/postedit_en_es.md`** | System-style instructions for Qwen post-editing (and for human QC reviewers). Not term pairs — those stay in `glossary/`. |

---

## `models/`

| File | Purpose |
|------|---------|
| **`models/.gitkeep`** | Keeps the empty directory in Git. |
| **`models/opus-mt-en-es-ct2/`** | Converted CTranslate2 Marian weights (`model.bin`, `config.json`, `shared_vocabulary.json`). Produced by `scripts/convert_marian_to_ct2.py`. Mount-only inside Docker. |
| **`models/lid.176.bin`** | fastText `lid.176.bin` (131 MB) — language-identification model used by `pdfs_to_csv.py` and `filter_english.py`. Downloaded once with `Invoke-WebRequest` (URL in `scripts/filter_english.py`). |
| **`models/wmt20-comet-qe-da/`** | COMET-QE checkpoint (~2.28 GB) for reference-free quality estimation. Staged via `scripts/download_comet_qe.py`. |

---

## `scripts/`

| File | Purpose |
|------|---------|
| **`scripts/compile_glossary_from_xlsx.py`** | Excel → `glossary/en_es_shotblasting.json`. Dedupes duplicate English rows (keeps longer Spanish). |
| **`scripts/convert_marian_to_ct2.py`** | Wraps `python -m ctranslate2.converters.transformers` to convert HF Marian → CT2. Supports `--force` and `--quantization`. |
| **`scripts/download_panblast_manuals.py`** | Step 1 of the corpus pipeline: scrape `panblast.com/manuals.acv`, download 18 PDFs (~57 MB), write `data/manifest.json`. Idempotent. |
| **`scripts/pdfs_to_csv.py`** | Step 2: PDF → CSV via PyMuPDF. Drops **CAPS** parts-list dumps, **Title-Case** parts-list dumps, and **non-English** segments via fastText `lid.176.bin` (lowercased before LID predict, default `--lang-min-confidence 0.55`). Writes to `data/csv_final/`. |
| **`scripts/translate_csv.py`** | Step 3: batch translate `data/csv_final/all_segments.csv` into `translated_data_final/`. Resumable, atomic flushes every 25 rows, with a cross-row memo cache so duplicate EN paragraphs translate once (~25 % corpus dedup saving). Supports `--engine ctranslate2|marian_hf|nllb` and `--mt-model <hf_id>` for the model sweep used to produce `results_final/report.md`. Applies the v4 ALL-CAPS preprocessor + UPPER restoration via `app.services.text_case`. |
| **`scripts/refresh_per_pdf.py`** | Rebuilds per-PDF mirrored CSVs from the live `translated_data_final/all_segments.csv` mid-run (lets you inspect partial progress). |
| **`scripts/filter_english.py`** | Standalone fastText LID filter over `translated_data_final/all_segments.csv`. The same LID logic is **embedded inline** in `pdfs_to_csv.py`; this script remains for post-hoc audits / debugging. |
| **`scripts/evaluate_glossary.py`** | Glossary application audit: per-row, per-term, per-PDF CSVs + stdout summary. Mirrors the protect-source matcher in `Glossary.protect_source`. |
| **`scripts/evaluate_quality.py`** | LaBSE + BERTScore + COMET: **auto** Kiwi (wmt22) if HF token present, else wmt20 QE (local ckpt preferred; same family as `run_comet_on_worst`). |
| **`scripts/download_comet_qe.py`** | One-time COMET-QE checkpoint stage. Uses `huggingface_hub.snapshot_download` with `local_dir_use_symlinks=False` to bypass Windows symlink privileges. |
| **`scripts/run_comet_on_worst.py`** | COMET-QE on the N worst rows (or `--top N` for full corpus). Writes `comet_qe_worst.csv`. |
| **`scripts/make_eval_plots.py`** | PNG plots for the eval reports: score distributions, per-PDF buckets, glossary heatmap, score-vs-length scatter. |
| **`scripts/run_full_eval.py`** | One-shot driver: glossary + `evaluate_quality` (LaBSE, BERTScore, COMET auto Kiwi/wmt20) + `run_comet_on_worst` (wmt20 slice) + plots → `results_final/` by default. |
| **`scripts/compare_versions.py`** | Generic 2-way side-by-side diff between evaluation runs. Reach for it only if you iterate on the pipeline and want to compare against an archived `results_final_*` snapshot. |
| **`scripts/sweep_summary.py`** | N-way engine sweep summary: reads multiple result directories produced by `run_full_eval.py` and ranks the engines on `0.5·LaBSE + 0.3·BERTScore + 0.2·σ(COMET-QE)`, emitting a Markdown table + winner recommendation. Used to produce the *Engine sweep* block in the READMEs. |
| **`scripts/build_resulting_files.py`** | Builds the client-facing two-column deliverable in `resulting_files/`: per-PDF CSVs with only `source_en` and `target_es`. Reads any `translated_data_*/all_segments.csv`. |
| **`scripts/prefetch_nllb.py`** | Pre-downloads NLLB-200 weights via `huggingface_hub.snapshot_download` so the first run of the NLLB engine doesn't spend several minutes on network I/O. Bypasses `from_pretrained` (which would also try to load weights into RAM). |
| **`scripts/prewarm_hf_cache.py`** | Pre-downloads Marian + Qwen weights (~5.5 GB) into `$HF_HOME` so the first `/translate` request doesn't stall on a multi-minute download. Used by the `prewarm` Compose profile but also runs standalone (`python scripts/prewarm_hf_cache.py [--no-qwen]`). |

---

## `data/`, `translated_data_final/`, `resulting_files/`, `results_final/`

| Path | Purpose |
|------|---------|
| **`data/manifest.json`** | `product_code → pdf` mapping (one PDF often covers dozens of SKUs). |
| **`data/<category>/*.pdf`** | Downloaded PanBlast manuals (18 PDFs across 5 categories). |
| **`data/csv_final/`** | Final extraction: 1,334 segments after CAPS + Title-Case parts-list filters and fastText LID at `--lang-min-confidence 0.55` with lowercased prediction. |
| **`translated_data_final/`** | Final translations (pipeline-internal layout): `all_segments.csv` and per-PDF mirrors. |
| **`resulting_files/`** | Client-facing deliverable (per-PDF + consolidated): only `source_en` and `target_es` columns. Regenerated by `scripts/build_resulting_files.py`. |
| **`results_final/`** | Final evaluation snapshot. Layout: `glossary/`, `quality/`, `plots/`, `report.md`. |

---

## `tests/`

| File | Purpose |
|------|---------|
| **`tests/conftest.py`** | Fixtures: temp glossary JSON + post-edit markdown, `Settings(_env_file=None)`, `app.dependency_overrides` to stub MT engine / Qwen out of integration tests. |
| **`tests/test_glossary_protect_restore.py`** | Glossary unit tests: protect_source, enforce_placeholders (incl. tolerant truncation), reassert_targets_after_edit (accent drift, multiple drifts, canonical+drifted mix, fallback-to-reference when Qwen drops a term, multi-entry round trip), word-bounded protection (Spartan not matching Spartans). |
| **`tests/test_pdf_extraction_filters.py`** | `is_caps_parts_dump` and the v3 `is_titlecase_parts_dump` heuristics: catches glued ALL-CAPS and Title-Case parts lists; keeps legitimate uppercase warnings and instructional sentences. |
| **`tests/test_text_case.py`** | v4 ALL-CAPS preprocessor: threshold + min-alpha floor for `is_mostly_uppercase`, sentence-case round-trip for `to_sentence_case`, glossary-placeholder safety, and the full `preprocess_for_mt` / `postprocess_after_mt` cycle on shouted safety warnings. |
| **`tests/test_api_integration.py`** | FastAPI `TestClient` end-to-end: full translate flow with stubbed MT, second-request cache hit, debug bypass, 422 on empty text, `GET /health`, cache-clear route. |
| **`.github/workflows/tests.yml`** | GitHub Actions CI: runs the full 53-test pytest suite on push + PR against `main`/`master`, on a Python 3.10 + 3.11 matrix, with CPU-only torch and `FakeMTEngine` so no real weights are pulled on the runner. |

---

## Files not in the repo but relevant

| Location | Purpose |
|----------|---------|
| **`.venv/` or `venv/`** | Local virtual environment (ignored by Git). |
| **`__pycache__/`** | Bytecode caches. |
| **`.env`** | Optional local secrets / `MT_MVP_*` overrides. |
| **`hf_cache` Docker volume** | HuggingFace cache shared between container restarts so Qwen weights aren't re-downloaded. |

---

## Quick "where do I change X?"

| Goal | File(s) |
|------|---------|
| Add an API route | `app/api/routes.py`, mount in `app/main.py` |
| Change runtime defaults (models, paths) | `app/core/config.py` or `.env` |
| Change translation steps | `app/services/translation.py`, `postedit.py`, `glossary.py` |
| Change glossary terms | `glossary/en_es_shotblasting.json` (or regenerate from Excel) |
| Change post-edit instructions | `prompts/postedit_en_es.md` |
| Change UI copy or layout | `frontend/index.html` + `styles.css` + `app.js`; for the demo UI: `app/gradio_app.py` |
| Tighten the corpus extraction filters | `scripts/pdfs_to_csv.py` (`is_caps_parts_dump`, `is_titlecase_parts_dump`, `LanguageFilter`) |
| Rebuild CT2 weights | `scripts/convert_marian_to_ct2.py` |
| Refresh glossary from sheet | `scripts/compile_glossary_from_xlsx.py` |
| Spin up a public client demo URL | `python -m app.gradio_app --share` (or `docker compose --profile demo up`) |
