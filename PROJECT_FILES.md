# Project file reference

This document describes **what each file in the repository is for**. Paths are relative to the project root (`mt_mvp/`).

---

## Root directory

| File | Purpose |
|------|---------|
| **README.md** | Main project documentation: setup, configuration, API overview, pipeline, and how to run the app. |
| **PROJECT_FILES.md** | This file — a per-file map of the codebase and assets. |
| **requirements.txt** | Python dependencies for the API, MT stack (Transformers, CTranslate2), glossary tooling (pandas/openpyxl), etc. Torch is installed separately or via `requirements-gpu.txt`. |
| **requirements-gpu.txt** | Installs PyTorch **CUDA 12.4** wheels (`torch==2.5.1+cu124`) plus everything from `requirements.txt`. Use when you want GPU-accelerated PyTorch. |
| **.gitignore** | Tells Git to ignore virtualenvs, caches, `.env`, local model weights under `models/opus-mt-en-es-ct2/`, and similar generated or secret files. |
| **raw_glossary.xlsx** | **Source** glossary: two columns (English, Spanish), no header row. Compiled into `glossary/en_es_shotblasting.json` by the script in `scripts/`. |

---

## `app/` — Python application package

| File | Purpose |
|------|---------|
| **`app/__init__.py`** | Marks `app` as a package; may hold a short package docstring. |
| **`app/__main__.py`** | Allows `python -m app` to start **Uvicorn** with reload (development entry point). |
| **`app/main.py`** | Creates the **FastAPI** application: CORS, mounts **`/api/v1`** router, serves **`/`** as `frontend/index.html`, mounts **`/static`** for frontend assets. Sets OpenAPI title/description. |

### `app/core/`

| File | Purpose |
|------|---------|
| **`app/core/__init__.py`** | Marks `app.core` as a subpackage. |
| **`app/core/config.py`** | **Central settings** via Pydantic `BaseSettings`: paths to glossary and post-edit prompt, MT engine (`ctranslate2` vs `marian_hf`), model names, CT2 directory and compute type, device, Qwen post-edit flags and token limits. Values load from environment variables prefixed with **`MT_MVP_`** and optional `.env`. |

### `app/api/`

| File | Purpose |
|------|---------|
| **`app/api/__init__.py`** | Marks `app.api` as the HTTP API subpackage. |
| **`app/api/schemas.py`** | **Pydantic models** for JSON bodies: `TranslateRequest` (`text`, `apply_glossary`, `apply_postedit`, `include_debug`) and `TranslateResponse` (`translation`, flags, `from_cache`, optional `debug`). |
| **`app/api/deps.py`** | **FastAPI dependencies**: `get_settings`, cached **singletons** for the MT engine (CT2 or Marian), **Glossary** (by path), **Qwen** post-edit service (when enabled), and **PostEditor**. Ensures heavy models are not reloaded on every request when settings match. |
| **`app/api/routes.py`** | **HTTP routes**: `POST /translate` (cache check → `run_translate` → cache set), `GET /health` (GPU/cache/pipeline hints), `POST /translate/cache/clear` (empty LRU cache). |

### `app/services/`

| File | Purpose |
|------|---------|
| **`app/services/__init__.py`** | Marks `app.services` as the business-logic / infrastructure subpackage. |
| **`app/services/translation.py`** | **Core translation orchestration** (no HTTP): glossary protect → MT → glossary restore → post-edit; builds the **`debug`** dict when `include_debug` is true. Imported by `routes.py`. |
| **`app/services/translate_cache.py`** | **In-memory LRU + TTL cache** for translation JSON responses; **`build_translate_cache_key()`** builds a stable tuple key from text, flags, MT mode, glossary/prompt mtimes, and Qwen settings. |
| **`app/services/glossary.py`** | Loads **`glossary/*.json`**, sorts entries by English phrase length, implements **protect** / **enforce placeholders** / **English leak fix** / **reassert Spanish targets after Qwen** / **spacing around multi-word targets**. |
| **`app/services/ct2_engine.py`** | **CTranslate2** backend: loads `model.bin` from disk, **MarianTokenizer** from Hugging Face for the same subwords as Marian, runs **`translate_batch`**. |
| **`app/services/mt_engine.py`** | **Hugging Face Marian** backend: `MarianMTModel` + `generate()` on PyTorch (optional path when `MT_MVP_MT_ENGINE=marian_hf`). |
| **`app/services/qwen_postedit.py`** | Loads **Qwen2.5 Instruct** with GPU-friendly **`device_map`**, runs **`generate`** on a chat template built from system (post-edit markdown) + user (English + draft Spanish). |
| **`app/services/postedit.py`** | **PostEditor**: loads post-edit instructions from disk (**`@lru_cache`**), optionally calls Qwen, then glossary **reassert** / English cleanup / spacing / whitespace normalize. |

---

## `frontend/` — Static web UI

| File | Purpose |
|------|---------|
| **`frontend/index.html`** | Single-page UI: hero copy for blast/PPE domain, English textarea, sample **chips**, toggles for glossary/post-edit/debug, Spanish output, cache notice, link to **`/docs`**. |
| **`frontend/app.js`** | Fetches **`POST /api/v1/translate`**, handles errors, shows **debug JSON** when requested, fills examples from chip clicks, shows **from cache** hint. |
| **`frontend/styles.css`** | Layout and **industrial / safety** styling (dark theme, amber accent, badges, chips, panels). |

---

## `glossary/`

| File | Purpose |
|------|---------|
| **`glossary/en_es_shotblasting.json`** | **Runtime glossary**: JSON with `entries` (`source`, `target`, `notes`). Built from **`raw_glossary.xlsx`** via `scripts/compile_glossary_from_xlsx.py`. The API reads this path from settings (default under `glossary/`). |

---

## `prompts/`

| File | Purpose |
|------|---------|
| **`prompts/postedit_en_es.md`** | **System-style instructions** for Qwen post-editing (and human QC): facts, standards, Spanish register, glossary boundaries. **Not** term pairs — those stay in `glossary/`. Loaded by `PostEditor` (cached by path). |

---

## `models/`

| File | Purpose |
|------|---------|
| **`models/.gitkeep`** | Keeps the `models/` directory in Git when weight files are gitignored. |
| **`models/opus-mt-en-es-ct2/config.json`** | CTranslate2 **model metadata** produced by the Marian → CT2 converter (architecture, vocab references). |
| **`models/opus-mt-en-es-ct2/model.bin`** | **CTranslate2 binary weights** — the actual Marian model converted for fast inference. Large file; typically not committed. |
| **`models/opus-mt-en-es-ct2/shared_vocabulary.json`** | **Shared vocabulary** artifact from conversion; used by the CT2 runtime with `model.bin`. |

---

## `scripts/`

| File | Purpose |
|------|---------|
| **`scripts/compile_glossary_from_xlsx.py`** | Reads **`raw_glossary.xlsx`** (two columns, no header), dedupes duplicate English rows (keeps longer Spanish), writes **`glossary/en_es_shotblasting.json`**. |
| **`scripts/convert_marian_to_ct2.py`** | Runs **`python -m ctranslate2.converters.transformers`** to convert a Hugging Face **Marian** checkpoint into **`models/.../model.bin`** (and sidecar JSON). Supports **`--force`** and optional **`--quantization`**. |

---

## Files not in the repo but relevant

| Location | Purpose |
|----------|---------|
| **`.venv/` or `venv/`** | Local virtual environment (ignored by Git). |
| **`__pycache__/`** | Bytecode caches (ignored). |
| **`.env`** | Optional local secrets / overrides for `MT_MVP_*` (ignored by Git). |

---

## Quick “where do I change X?”

| Goal | File(s) |
|------|---------|
| Change API URL or add routes | `app/api/routes.py`, mount in `app/main.py` |
| Change defaults (models, paths) | `app/core/config.py` or `.env` |
| Change translation steps | `app/services/translation.py`, `postedit.py`, `glossary.py` |
| Change glossary terms | `glossary/en_es_shotblasting.json` (or regenerate from Excel) |
| Change post-edit behavior text | `prompts/postedit_en_es.md` |
| Change UI copy or layout | `frontend/index.html`, `styles.css`, `app.js` |
| Rebuild CT2 weights | `scripts/convert_marian_to_ct2.py` |
| Refresh glossary from sheet | `scripts/compile_glossary_from_xlsx.py` |
