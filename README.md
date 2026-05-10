# Blast equipment EN→ES translator (MVP)

Machine translation stack for **English → Spanish** in **shotblasting, abrasive blast, and operator PPE** copy (datasheets, SOPs, catalog lines). It combines **CTranslate2 Marian** (or optional **PyTorch Marian**), a **JSON glossary**, **Qwen2.5** post-editing, **LRU/TTL response caching**, and a **domain-styled web UI** plus **OpenAPI (Swagger)**.

A **per-file map** of the repository lives in **[PROJECT_FILES.md](PROJECT_FILES.md)**.

---

## Features

- **FastAPI + Uvicorn** REST API
- **Glossary-aware MT**: protect English terms → translate → restore Spanish targets; post-edit **re-asserts** those targets so the LLM cannot drift catalog wording
- **Backends**: **CTranslate2** (default) or **Hugging Face Marian**
- **Post-edit**: optional **Qwen2.5 Instruct** (defaults to 3B + short generation caps for latency)
- **Cache**: in-memory LRU (**1024** entries), **24h** TTL; `include_debug: true` bypasses cache
- **Frontend**: blast-room themed page at `/` with sample chips and link to `/docs`
- **Tooling**: `raw_glossary.xlsx` → `glossary/en_es_shotblasting.json`; Marian → CT2 conversion script

---

## How the pipeline works

For **`POST /api/v1/translate`** when the response is **not** cached:

1. **Cache lookup** (skipped if `include_debug` is true)
2. **Glossary protect** — longest-match English phrases → `__GLS0__` placeholders
3. **Machine translate** — Marian EN→ES
4. **Glossary restore** — placeholders → canonical Spanish (regex tolerates missing spaces around tokens)
5. **Post-edit** — Qwen (optional) → **reassert** glossary targets from pre–Qwen Spanish → English leak fix → spacing fix around multi-word targets

`GET /api/v1/health` — CUDA hints, MT mode, cache size, pipeline summary.

`POST /api/v1/translate/cache/clear` — empty the translation cache (no auth; do not expose publicly without a proxy).

---

## Speed & quality (why it feels fast)

| Layer | Choice |
|-------|--------|
| MT | **CTranslate2** for efficient `model.bin` inference |
| GPU | CUDA auto-selection; Qwen pinned with **`device_map={"": 0}`** |
| Post-edit | **3B** Qwen, **256** max new tokens, **2048** input truncation |
| Quality | Glossary pipeline + **reassert** after Qwen + spacing repair |
| Repeat calls | **LRU + TTL** cache returns identical payloads without MT/Qwen |

Disable Qwen entirely: `MT_MVP_POSTEDIT_USE_QWEN=false`.

---

## Requirements

- **Python 3.11+**
- **PyTorch** (CPU or CUDA — see `requirements-gpu.txt` for cu124 example)
- Optional: **NVIDIA GPU** for MT + Qwen

---

## Installation

```powershell
cd C:\Users\Alexandra\Desktop\mt_mvp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-gpu.txt   # when using CUDA wheels
```

**Convert Marian to CTranslate2** (creates `models\opus-mt-en-es-ct2\model.bin`):

```powershell
python scripts\convert_marian_to_ct2.py --model Helsinki-NLP\opus-mt-en-es --output-dir models\opus-mt-en-es-ct2 --force
```

Use a current **`transformers`** (see `requirements.txt`; 5.x avoids some Marian converter issues).

**Rebuild glossary from Excel**:

```powershell
python scripts\compile_glossary_from_xlsx.py
```

---

## Configuration (`MT_MVP_*` or `.env`)

| Variable | Role |
|----------|------|
| `MT_MVP_MT_ENGINE` | `ctranslate2` (default) or `marian_hf` |
| `MT_MVP_MT_MODEL_NAME` | HF id for tokenizer / Marian weights |
| `MT_MVP_CT2_MODEL_DIR` | Directory with `model.bin` |
| `MT_MVP_CT2_COMPUTE_TYPE` | e.g. `int8`, `float16`, `default` |
| `MT_MVP_DEVICE` | `cuda`, `cpu`, or unset (auto) |
| `MT_MVP_POSTEDIT_USE_QWEN` | `true` / `false` |
| `MT_MVP_POSTEDIT_QWEN_MODEL` | e.g. `Qwen/Qwen2.5-3B-Instruct` |
| `MT_MVP_POSTEDIT_MAX_NEW_TOKENS` | Qwen decode cap |
| `MT_MVP_POSTEDIT_QWEN_MAX_INPUT_TOKENS` | Prompt truncation |
| `MT_MVP_GLOSSARY_PATH` | Glossary JSON path |
| `MT_MVP_POSTEDIT_PROMPT_PATH` | Post-edit markdown path |

---

## Run

```powershell
cd C:\Users\Alexandra\Desktop\mt_mvp
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- **UI**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Swagger**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## Code layout (refactored)

```
mt_mvp/
├── app/
│   ├── main.py                 # FastAPI factory, CORS, static files
│   ├── core/config.py          # Settings
│   ├── api/
│   │   ├── routes.py           # HTTP: cache wrap, health, cache clear
│   │   ├── deps.py             # Singletons: MT engine, glossary, Qwen, PostEditor
│   │   └── schemas.py          # Pydantic request/response models
│   └── services/
│       ├── translation.py      # Core translate path (glossary + MT + post-edit)
│       ├── translate_cache.py  # LRU/TTL cache + cache key builder
│       ├── glossary.py
│       ├── ct2_engine.py / mt_engine.py
│       ├── qwen_postedit.py / postedit.py
│       └── …
├── frontend/                   # Domain-themed static UI
├── glossary/   prompts/   models/   scripts/
├── requirements.txt
├── requirements-gpu.txt
└── README.md
```

**Refactor notes:** HTTP concerns stay in `app/api/routes.py`; pure translation logic lives in **`app/services/translation.py`**. Cache keys live next to the cache in **`translate_cache.py`**. The tiny **`prompts.py`** helper was folded into **`postedit.py`**. The **`TranslationEngine` protocol** was replaced by an explicit **`Ctranslate2Engine | MarianEngine`** union in types.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/translate` | Body: `text`, `apply_glossary`, `apply_postedit`, `include_debug` |
| `GET` | `/api/v1/health` | Status, GPU, cache entries, pipeline |
| `POST` | `/api/v1/translate/cache/clear` | Clear translation cache |

---

## License & contributing

Add a `LICENSE` before public distribution.

After changing **`glossary/*.json`** or **`prompts/*.md`**, call **`POST /api/v1/translate/cache/clear`** or restart Uvicorn so cached translations refresh. Post-edit prompt text is **cached in-process** (`lru_cache` on the prompt path); restart if you edit the markdown without changing the path.
