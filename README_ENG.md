# Blast equipment EN→ES translator (MVP)

Machine-translation stack for **English → Spanish** in **shotblasting,
abrasive blast, and operator-PPE** copy (datasheets, SOPs, catalogue lines).
It combines **CTranslate2 Marian** (or optional **PyTorch Marian**), a
**JSON glossary**, **Qwen 2.5** post-editing, **LRU/TTL response caching**,
a **vanilla web UI**, a **Gradio demo** with shareable client links, and a
production-shaped **Docker** image.

A **per-file map** lives in **[`PROJECT_FILES.md`](PROJECT_FILES.md)**. The
Russian README is in **[`README_RUS.md`](README_RUS.md)**.

---

## Final pipeline at a glance

The full PanBlast corpus (18 PDFs, ~57 MB) is extracted to 1,334 confidently
English sentences, translated through the production pipeline, and scored
on four independent reference-free metrics:

| Metric | **Final** (n = 1,334) |
|---|---:|
| LaBSE cosine (mean ± 95 % CI) | **0.895 ± 0.003** |
| BERTScore F1 (mean) | **0.911** |
| COMET-QE (mean, DA z-score) | **−0.047** |
| COMET-QE "broken" rows (< −1.0) | **16 (1.2 %)** |
| Glossary term-level hit rate | **96.4 %** (1,029 / 1,067) |
| Rows with all glossary terms applied | **95.5 %** (514 / 538) |
| Translation wall-clock on RTX 3060 | **34 m 03 s** (with 23 % cross-row cache hits) |

The full numerical breakdown — per-term hit rates, per-PDF means, COMET-QE
buckets, best/worst translation gallery, and remaining failure classes —
lives in [`results_final/report.md`](results_final/report.md) and is
reproduced below.

### How we got here

The pipeline went through **three evaluation iterations** before landing
on this final shape. The first pass translated the entire multilingual
PDF corpus (4,201 rows; LaBSE 0.887, glossary hit rate 84.9 %, 4 h 41 m on
GPU). Most failures clustered into three causes: (1) glued ALL-CAPS
parts-list dumps the Marian model could not parse, (2) two-thirds of rows
were not English yet went through MT anyway, and (3) Qwen post-edits
silently dropped canonical Spanish terms or stripped accents. The second
pass added (1) a CAPS parts-list filter at extraction, (2) inline fastText
`lid.176.bin` language identification, and (3) accent- and case-tolerant
glossary re-assertion with fallback to the MT-only output when Qwen
deletes a term — together lifting glossary to 97.0 % and shrinking the
runtime to 44 m. The final pass tightened all three: a Title-Case
parts-list detector caught what CAPS missed, the LID predict is now
lowercased and confidence-thresholded at 0.55 (rejecting 2,450 non-English
segments across 10 languages), `protect_source` became word-bounded so 23
new verbatim brand/standards entries (Spartan, Titan, NPT, OSHA, …) can
safely be added without matching inside longer words, and a cross-row
memo cache reused the 23 % of EN sentences that repeat across PDFs.
**Net effect:** COMET-QE mean moved from −0.066 → −0.047 (+0.019), the
broken-row count dropped 38 % (26 → 16), `Remote control valve →
Válvula neumática` went from 100 % missed to 100 % applied (37/37), and
the full translate finished in 34 minutes instead of nearly 5 hours.

---

## Features

- **FastAPI + Uvicorn** REST API at `/api/v1/*`
- **Glossary-aware MT**: protect EN terms → translate → restore ES targets;
  Qwen post-edit re-asserts canonical targets and falls back to MT-only
  when Qwen paraphrases a term away
- **Backends**: **CTranslate2** (default) or **HuggingFace Marian**
- **Post-edit**: optional **Qwen 2.5 Instruct** (3B default, 7B available)
- **Cache**: in-memory LRU (1,024 entries), 24 h TTL
- **Two UIs**:
  - the production-styled vanilla page at `/`
  - a **Gradio** demo at `/gradio` (and `python -m app.gradio_app --share`
    publishes a public `https://*.gradio.live` URL for client demos)
- **Docker**: `docker compose up` runs the API + Gradio together; `--profile
  gpu` uses NVIDIA Container Toolkit
- **Tooling**: PDF → CSV → translate → evaluate pipeline with a one-shot
  driver (`scripts/run_full_eval.py`)

---

## How the pipeline works

For **`POST /api/v1/translate`** on cache miss:

1. **Cache lookup** (bypassed when `include_debug=true`)
2. **Glossary protect** — longest-match, word-bounded EN phrases → `__GLS{i}__`
   placeholders
3. **Machine translate** — Marian EN→ES via CTranslate2 (or pure PyTorch)
4. **Glossary restore** — placeholders → canonical Spanish (regex tolerant
   of MT-induced truncation of `__GLS{i}__` → `__GLS{i}_`)
5. **Post-edit** — Qwen 2.5 Instruct refines fluency, then **reassert**
   glossary targets with an accent- / case-insensitive matcher, fall back
   to the pre-Qwen MT output if Qwen deletes a canonical term, then fix
   English leaks and spacing

`GET /api/v1/health` — CUDA hints, MT engine, cache size, pipeline summary.
`POST /api/v1/translate/cache/clear` — empty the in-memory cache.

---

## Speed & quality

| Layer | Choice |
|-------|--------|
| MT | **CTranslate2** for efficient `model.bin` inference |
| GPU | CUDA auto-select; Qwen pinned with `device_map={"": 0}` |
| Post-edit | Qwen 2.5 **3B**, 256 max-new-tokens, 2,048 input truncation |
| Quality | Glossary protect / restore / reassert + Qwen + spacing repair |
| Repeat calls | LRU + TTL cache returns identical payloads without MT/Qwen |
| Batch dedup | Cross-row memo cache in `scripts/translate_csv.py` (23 % hit rate) |

Disable Qwen entirely: `MT_MVP_POSTEDIT_USE_QWEN=false`.

---

## Requirements

- **Python 3.11+**
- **PyTorch** (CPU or CUDA; CUDA 12.4 wheels in `requirements-gpu.txt`)
- Optional: NVIDIA GPU for MT + Qwen
- Optional: Docker 24+ for the containerised path

---

## Installation (host)

```powershell
cd mt_mvp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-gpu.txt   # CUDA wheels (optional)
python -m pip install -r requirements-eval.txt  # only needed to reproduce the eval
```

**Convert Marian to CTranslate2** (creates `models/opus-mt-en-es-ct2/model.bin`):

```powershell
python scripts\convert_marian_to_ct2.py --model Helsinki-NLP/opus-mt-en-es --output-dir models\opus-mt-en-es-ct2 --force
```

**Rebuild glossary from Excel**:

```powershell
python scripts\compile_glossary_from_xlsx.py
```

---

## Installation (Docker)

```bash
# default profile: API + Gradio demo on http://localhost:8000  (CPU image,
# Dockerfile target `runtime`, slim Python base ~1.5 GB)
docker compose up --build

# GPU profile: builds Dockerfile target `runtime-gpu` (nvidia/cuda:12.4.1 base
# + cu124 PyTorch wheels, ~6 GB image). Host needs the NVIDIA Container
# Toolkit and a CUDA-12.x driver.
docker compose --profile gpu up --build

# Public client-demo profile: prints an https://*.gradio.live URL on startup
docker compose --profile demo up --build

# Optional one-shot: pre-download Marian + Qwen (~5.5 GB) into the shared
# `hf_cache` volume so the first /translate request is instant instead of
# stalling on the download. Run once after the first build.
docker compose --profile prewarm run --rm prewarm
```

The Dockerfile has four targets — `base` / `base-gpu` install the venv
(CPU torch or cu124 torch respectively), and `runtime` / `runtime-gpu`
copy it plus the app sources, drop to UID 10001, and expose `:8000`.
Models are **not** baked into either image — mount them at runtime via
`./models:/app/models`. If `model.bin` is missing, the API falls back to
the HuggingFace Marian engine automatically (see `app/api/deps.py`).

`MT_MVP_*` env vars (full list in `app/core/config.py` and `.env.example`)
override every default; the most useful one is
`MT_MVP_POSTEDIT_USE_QWEN=false` on memory-constrained boxes.

### Verify the container

After `docker compose up -d --build`, walk through this checklist to
confirm the stack is healthy end-to-end. All commands assume PowerShell
on Windows (replace `Invoke-RestMethod` with `curl` on Linux/macOS).

**1. Status & healthcheck**

```powershell
docker compose ps                       # expect: api  Up X seconds (healthy)
Invoke-RestMethod http://localhost:8000/api/v1/health | Format-List
```

`/health` returns `status=ok`, the configured MT engine, CUDA hint, and
the current LRU cache size.

**2. First translation (cold path)**

```powershell
$body = @{ text = "Always wear the Helmet before operating the Remote control valve." } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/v1/translate `
  -Method Post -ContentType application/json -Body $body
```

Expected JSON: `translation = "Siempre use el casco antes de operar el
válvula neumática ."`, `glossary_applied = True`, `from_cache = False`.
First request takes 5–30 s because the Marian weights lazy-download into
`hf_cache`; subsequent translations are instant.

**3. LRU cache hit (warm path)**

```powershell
$body = @{ text = "Replace the inner lens." } | ConvertTo-Json
1..2 | ForEach-Object {
  $sw = [Diagnostics.Stopwatch]::StartNew()
  $r  = Invoke-RestMethod -Uri http://localhost:8000/api/v1/translate `
        -Method Post -ContentType application/json -Body $body
  $sw.Stop()
  "Request #{0}: from_cache={1}, elapsed={2:F2}s" -f $_, $r.from_cache, $sw.Elapsed.TotalSeconds
}
```

Request #1 → `from_cache: False`; request #2 → `from_cache: True` and
≈10× faster.

**4. CT2 → Marian fallback evidence**

```powershell
docker compose logs api --tail 40 | Select-String "CTranslate2|falling back"
```

You should see the warning emitted by `app/api/deps.py` when the
container starts translating without converted CT2 weights mounted.

**5. UI surfaces**

Open in a browser:

| URL | What it serves |
|---|---|
| <http://localhost:8000/> | Vanilla web UI (`frontend/index.html`) |
| <http://localhost:8000/gradio/> | Embedded Gradio demo |
| <http://localhost:8000/docs> | Swagger / OpenAPI explorer |

**6. Inspect & shut down**

```powershell
docker compose exec api bash             # poke around inside the container
docker compose restart api               # restart without losing hf_cache
docker compose down                      # stop + remove (hf_cache survives)
docker compose down -v                   # full wipe, including hf_cache
```

> **Qwen note.** The default container has Qwen 2.5 post-editing
> **enabled** — the very first `/translate` will then also lazy-download
> ~5 GB of Qwen weights (several minutes on CPU). To skip this on a
> smoke test, restart with `MT_MVP_POSTEDIT_USE_QWEN=false`. To keep
> the full pipeline but avoid the cold wait, run the prewarm profile
> once: `docker compose --profile prewarm run --rm prewarm`.

---

## Tests (`pytest`)

| Suite | File | Coverage |
|-------|------|----------|
| **Glossary (unit)** | `tests/test_glossary_protect_restore.py` | `protect_source` (case + word-boundary), `enforce_placeholders` (tolerant of MT truncation), `reassert_targets_after_edit` (accent drift, multi-drift, canonical+drifted mix, fallback-to-reference when Qwen drops a term, multi-entry round trip), verbatim-brand round-trip |
| **PDF filters (unit)** | `tests/test_pdf_extraction_filters.py` | `is_caps_parts_dump` (positive + warning whitelist + length floor), `is_titlecase_parts_dump` (positive + heading whitelist + period whitelist + instructional-sentence whitelist), unified `is_parts_dump` |
| **API (integration)** | `tests/test_api_integration.py` | `TestClient` with `FakeMTEngine`: full translate flow, second-request cache hit, `include_debug` bypass, 422 on empty text, health route, cache-clear route |

```powershell
pytest -v
```

```
============================= 37 passed in 0.22s =============================
```

---

## Configuration (`MT_MVP_*` env vars, optional `.env`)

| Variable | Role |
|----------|------|
| `MT_MVP_MT_ENGINE` | `ctranslate2` (default) or `marian_hf` |
| `MT_MVP_MT_MODEL_NAME` | HF id for tokenizer / Marian weights |
| `MT_MVP_CT2_MODEL_DIR` | Directory with `model.bin` |
| `MT_MVP_CT2_COMPUTE_TYPE` | `int8`, `float16`, `default`, etc. |
| `MT_MVP_DEVICE` | `cuda`, `cpu`, or unset (auto) |
| `MT_MVP_POSTEDIT_USE_QWEN` | `true` / `false` |
| `MT_MVP_POSTEDIT_QWEN_MODEL` | e.g. `Qwen/Qwen2.5-3B-Instruct` |
| `MT_MVP_POSTEDIT_MAX_NEW_TOKENS` | Qwen decode cap |
| `MT_MVP_POSTEDIT_QWEN_MAX_INPUT_TOKENS` | Prompt truncation |
| `MT_MVP_GLOSSARY_PATH` | Glossary JSON path |
| `MT_MVP_POSTEDIT_PROMPT_PATH` | Post-edit markdown path |
| `MT_MVP_ENABLE_GRADIO` | `1` mounts the Gradio UI at `/gradio` (Docker sets this) |

---

## Run

Three completely independent launch modes — pick one. They don't share
ports, env vars, or processes, so you can also run them in parallel.

### Mode A — FastAPI service (port 8000)

The production UI + REST API.

```powershell
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Vanilla UI: <http://127.0.0.1:8000/>
- Swagger: <http://127.0.0.1:8000/docs>

To **also** mount the Gradio UI inside this same FastAPI process:

```powershell
$env:MT_MVP_ENABLE_GRADIO = "1"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Gradio (embedded): <http://127.0.0.1:8000/gradio>

### Mode B — Standalone Gradio with public link (port 7861)

For a quick client demo: launches Gradio directly (no FastAPI), opens a
temporary `https://*.gradio.live` tunnel that anyone can open.

```powershell
.\venv\Scripts\Activate.ps1
python -m app.gradio_app --share --port 7861
```

Output:

```
* Running on local URL:  http://0.0.0.0:7861
* Running on public URL: https://abcd1234.gradio.live
```

The `gradio.live` link is valid for 7 days. Pick any free port via
`--port`; the default `7860` often collides with other dev servers.
This mode does **not** read `MT_MVP_ENABLE_GRADIO` — the env var only
matters for Mode A.

### Mode C — Docker

```bash
# CPU API + embedded Gradio on http://localhost:8000
docker compose up --build

# Same, on GPU (host needs NVIDIA Container Toolkit + CUDA 12.x driver).
# Builds the runtime-gpu target with cu124 PyTorch wheels.
docker compose --profile gpu up --build

# Standalone Gradio with --share on port 7860 (separate container)
docker compose --profile demo up --build

# One-shot: prime the hf_cache volume with Marian + Qwen weights so the
# first /translate request doesn't stall on a 5.5 GB download. Run this
# once after the initial build.
docker compose --profile prewarm run --rm prewarm
```

To confirm the container is alive and translating end-to-end, follow
the [§ Verify the container](#verify-the-container) smoke-test checklist
(health probe, sample translate, cache-hit, fallback warning).

---

## PanBlast corpus: download → extract → translate

In addition to the HTTP service, the project ships a batch pipeline that
builds a domain corpus from PanBlast technical PDF manuals and runs it
through the same engine the API uses.

### 1. Download PDFs — `scripts/download_panblast_manuals.py`

- 18 PDFs (~57 MB) across 5 categories, scraped from
  <https://www.panblast.com/manuals.acv>.
- Idempotent: existing files are skipped.
- Writes `data/manifest.json` mapping product code → PDF.

```powershell
python scripts\download_panblast_manuals.py
```

### 2. Extract PDF → CSV — `scripts/pdfs_to_csv.py`

Pipeline (per PDF):

- **PyMuPDF** text extraction (best reading order on technical layouts).
- Strip running headers / footers (lines appearing on ≥ 50 % of pages).
- Drop noise: page numbers, bare part numbers (`ZVP-PC-0027-01`), pure
  numerics, lines with < 3 letters.
- **Drop CAPS parts-list dumps**: long, all-uppercase glued noun phrases
  with no instruction verb and no internal punctuation.
- **Drop Title-Case parts-list dumps**: long, Title-Case dominant glued
  segments with ≥ 2 catalog keywords (PARTS, LIST, STOCK, CODE,
  DESCRIPTION, ITEM, ASSEMBLY, EXPLODED, …) and no instructional verb.
- Re-join PDF-wrapped lines into paragraphs → sentence-split on `.!?`.
- Cap segment length at **800 chars** (OPUS-MT 512-token ceiling).
- **fastText `lid.176.bin`** language filter on the **lowercased** segment;
  reject if `lid != "en"` OR confidence < `--lang-min-confidence`
  (default **0.55**).
- Stable SHA-1 row id over `category|pdf|page|segment_idx|source_en`.

Output: **1,334 segments / ~184 K source chars** in `data/csv_final/`.

```powershell
python scripts\pdfs_to_csv.py
python scripts\pdfs_to_csv.py --lang-min-confidence 0.40
python scripts\pdfs_to_csv.py --no-lang-filter            # keep every language
```

### 3. Batch-translate — `scripts/translate_csv.py`

Identical pipeline to `POST /api/v1/translate`:
`glossary protect → CT2 Marian → glossary restore → Qwen 2.5-3B post-edit →
glossary re-assert → spacing fixes`.

- **GPU** auto-selected (Qwen bf16 + CTranslate2 Marian ≈ 7.7 GB VRAM on
  RTX 3060).
- **Resumable**: existing non-empty `target_es` rows are reused; atomic
  flushes every 25 rows survive Ctrl-C / power loss.
- **Cross-row memo cache**: duplicate EN strings share their ES
  translation. **23.3 %** of the corpus is duplicate boilerplate, so the
  cache saves ~10 minutes per full run.

```powershell
python scripts\translate_csv.py                  # full pipeline
python scripts\translate_csv.py --no-postedit    # MT-only fast pass
python scripts\translate_csv.py --limit 50       # smoke test
python scripts\translate_csv.py --start-fresh    # ignore previously-translated rows
```

Output CSV: `id, category, pdf, page, segment_idx, source_en, target_es,
char_count, glossary_applied, postedit_applied, error`.

### 4. Where things land

```
data/
├── manifest.json                           # product code → PDF
├── <category>/*.pdf                        # 18 downloaded PDFs
└── csv_final/                              # extracted EN segments
    ├── all_segments.csv                    # 1,334 segments
    └── <category>/<pdf_stem>.csv

translated_data_final/                      # translations
├── all_segments.csv
└── <category>/<pdf_stem>.csv

results_final/                              # evaluation snapshot
├── glossary/                               # per-term & per-PDF stats
├── quality/                                # LaBSE, BERTScore, COMET-QE
├── plots/                                  # 9 PNGs
└── report.md
```

---

## Evaluation: glossary + LaBSE + BERTScore + COMET-QE

Four independent quality signals, all reference-free, all local
(`requirements-eval.txt`).

**One-shot driver** (writes `results_final/`):

```powershell
python scripts\run_full_eval.py
```

Under the hood it runs four scripts:

### a. `scripts/evaluate_glossary.py`

For every translated row: which EN sources appear, do the canonical ES
targets appear in the translation, and do any EN phrases leak through
untranslated. Writes per-row, per-term, per-PDF CSVs + stdout summary.

### b. `scripts/evaluate_quality.py` — LaBSE + BERTScore

- **LaBSE** — Google's multilingual encoder; cosine between source and
  target embeddings.
- **BERTScore F1** on `xlm-roberta-large` (layer 17) — token-level
  semantic overlap with source-as-pseudo-reference.

### c. `scripts/run_comet_on_worst.py` — COMET-QE

`Unbabel/wmt20-comet-qe-da` (2.28 GB) is the canonical reference-free MT
quality estimator. The scale is **DA z-score**, roughly −1.5..+1.0,
higher is better. Anything below −1.0 is essentially broken MT.

```powershell
python scripts\download_comet_qe.py        # one-time stage into models/
python scripts\run_comet_on_worst.py --top 1334
```

### d. `scripts/make_eval_plots.py`

9 PNGs into `results_final/plots/`: LaBSE / BERTScore distributions and
scatter, per-PDF mean LaBSE, per-term & per-PDF hit-rate heatmaps,
row-status pie, COMET buckets per PDF.

---

## Detailed analytics

### Corpus extraction & LID rejection

PanBlast's 18 PDF manuals (~57 MB) contain ~3,800 unique text segments
after PyMuPDF extraction. Most are **not English**:

| Stage | Segments | Notes |
|---|---:|---|
| PyMuPDF extracted raw | ~3,800 | running headers, page numbers, part numbers already dropped |
| After CAPS + Title-Case parts-list filters | 3,784 | ~80 glued ALL-CAPS + Title-Case dumps removed |
| After fastText LID @ 0.55 | **1,334** | only confidently-English rows pass |
| → kept too short for LID | 27 | < 8 alphabetic chars; trust the noise filter |

LID rejected **2,450 non-English segments** across 10 detected languages
(pt ~ fr ~ de ~ hu ~ nl ~ es ~ it ~ sv each ≈ 11 %; ru 9 %; low-confidence
en 2 %). Without LID, all of these would have been translated EN→ES —
wasting ~3 h of GPU and dragging every corpus-level mean downward.

### Quality metric distributions

**LaBSE cosine** (multilingual embedding similarity, scale [−1, 1]):

| Stat | Value |
|---|---:|
| mean ± 95 % CI | **0.8947 ± 0.0032** |
| std | 0.0605 |
| p05 / median / p95 | 0.7834 / 0.9095 / 0.9530 |
| min / max | 0.3597 / 1.0000 |

Bucket distribution: **59.3 % great (≥ 0.9), 35.0 % good (0.8–0.9),
3.9 % ok (0.7–0.8), 1.8 % poor (< 0.7)**.

**BERTScore F1** (xlm-roberta-large, layer 17): mean **0.9110 ± 0.0010**.
**98.8 %** of segments land in the `high (0.85–0.95)` bucket, **1.0 %** in
`very-high (≥ 0.95)`, **0.2 %** in `mid (0.7–0.85)`, **0 %** in `low (< 0.7)`.

**COMET-QE** (`Unbabel/wmt20-comet-qe-da`, DA z-score; scored on RTX
3060 in ~16 s):

| Stat | Value |
|---|---:|
| mean | **−0.0469** |
| median | −0.0626 |
| std | 0.4163 |
| min / max | −1.24 / +0.97 |

| Bucket | Count | Share |
|---|---:|---:|
| broken (< −1.0) | **16** | 1.2 % |
| very weak (−1.0 .. −0.5) | 195 | 14.6 % |
| weak (−0.5 .. −0.2) | 264 | 19.8 % |
| ok (−0.2 .. 0.1) | 350 | 26.2 % |
| good (≥ 0.1) | **509** | **38.2 %** |

### Glossary audit

**Headline:** 1,067 term occurrences in source, **1,029 applied in target
= 96.4 % term-level hit rate**. 514 of 538 rows-with-terms (95.5 %) have
**all** glossary terms applied; only **24 rows** are partially applied.

The top-20 most-used glossary terms:

| EN source | Canonical ES | Occ. | Applied | Hit rate |
|---|---|---:|---:|---:|
| Helmet | Casco | 314 | 302 | 96.2 % |
| Cape | Capa | 86 | 85 | 98.8 % |
| Panblast | Panblast | 78 | 78 | **100 %** |
| Breathing tube | Tubo Respirador | 64 | 61 | 95.3 % |
| Air Cooling Controller | Cooler de Casco | 52 | 51 | 98.1 % |
| NPT | NPT | 49 | 48 | 98.0 % |
| Spartan | Spartan | 38 | 29 | **76.3 %** ◀ |
| Remote control valve | Válvula neumática | 37 | 37 | **100 %** |
| Inner lens | Lámina interior | 36 | 32 | 88.9 % |
| AirFlo | AirFlo | 35 | 35 | **100 %** |
| Sealing Band | Banda de goma | 27 | 27 | **100 %** |
| Pipe Nipple | Niple de tubo dosificador | 26 | 26 | **100 %** |
| Blast pot | Tolva | 23 | 23 | **100 %** |
| Cosmo | Cosmo | 19 | 19 | **100 %** |
| Respirator Airline Filter | Filtro operario | 18 | 18 | **100 %** |
| BSP | BSP | 17 | 16 | 94.1 % |
| Galaxy | Galaxy | 16 | 16 | **100 %** |
| AbraFlo | AbraFlo | 15 | 15 | **100 %** |
| PBF | PBF | 14 | 14 | **100 %** |
| Outer lens | Lámina exterior | 13 | 12 | 92.3 % |

Of the 23 verbatim brand/standard supplements, **18 land at 100 % hit
rate** (Panblast, AirFlo, Galaxy, AbraFlo, Cosmo, JIC, AcoustiFlex, Pet
Cock, ISO, OSHA, Titan, …); two land at 94–98 % (NPT, BSP — lost on a
single all-caps safety paragraph each); and **Spartan** is the outlier at
76.3 %. It appears 38 times, 9 of which inside the same ALL-CAPS safety
boilerplate (`DO NOT USE THE SPARTAN SUPPLIED AIR RESPIRATOR HELMET …`).
The word-bounded protection + verbatim entry keep Spartan intact in 26 /
29 mixed-case mentions, but in all-caps blocks Marian rearranges
surrounding words so aggressively that the placeholder ends up in a
non-grammatical position; Qwen rewrites the sentence; the
fallback-to-reference reverts to the already-broken MT output. Tracked
under "All-caps Marian boilerplate" in the wishlist below.

### Per-PDF quality ranking

Median LaBSE is very close across all 18 PDFs (0.90 – 0.92); the spread
in mean is driven entirely by the bottom-10 % tail of long all-caps
warnings.

**Top 5 by mean LaBSE:**

| PDF | Rows | mean | median |
|---|---:|---:|---:|
| `ZVP-PC-0038-01.pdf` | 16 | **0.9200** | 0.9197 |
| `ZVP-PC-0041-01.pdf` | 13 | 0.9148 | 0.9203 |
| `ZVP-PC-0043-01.pdf` | 27 | 0.9120 | 0.9147 |
| `ZVP-PC-0039-01.pdf` | 22 | 0.9110 | 0.9164 |
| `ZVP-PC-0042-01.pdf` | 27 | 0.9044 | 0.9063 |

**Bottom 5 by mean LaBSE:**

| PDF | Rows | mean | median |
|---|---:|---:|---:|
| `ZVP-PC-0100-00.pdf` | 64 | **0.8813** | 0.9058 |
| `ZVP-PC-0086-00.pdf` | 126 | 0.8837 | 0.9040 |
| `ZVP-PC-0071-01.pdf` | 65 | 0.8855 | 0.9036 |
| `ZVP-PC-0111-00.pdf` | 68 | 0.8888 | 0.9201 |
| `ZVP-PC-0072-01.pdf` | 74 | 0.8904 | 0.9046 |

### Worst & best translation gallery

**Best 6 long-source examples** (COMET-QE ≥ +0.7, source ≥ 60 chars).
Note the same compressor-safety paragraph appears in 5 PDFs and is
translated identically each time via the cross-row memo cache:

| COMET-QE | LaBSE | EN | ES |
|---:|---:|---|---|
| +0.762 | 0.926 | An overheated compressor, or one that is in poor mechanical condition, may produce carbon monoxide (CO) and objectionable odours. | Un compresor sobrecalentado o uno que esté en malas condiciones mecánicas puede producir monóxido de carbono (CO) y olores desagradables. |
| +0.757 | 0.927 | An overheated compressor, or one that is in poor mechanical condition, may produce carbon monoxide (CO) and objectionable odors. | Un compresor sobrecalentado o uno que esté en malas condiciones mecánicas puede producir monóxido de carbono (CO) y olores desagradables. |
| +0.741 | 0.907 | An overheated compressor, or one that is in poor mechanical condition, may produce carbon monoxide and objectionable odors. | Un compresor sobrecalentado o uno que esté en malas condiciones mecánicas puede producir monóxido de carbono y olores desagradables. |
| +0.709 | 0.892 | overheated compressor, or one that is in poor mechanical condition may produce carbon monoxide (CO) and objectionable odors. | Compresor sobrecalentado o uno en malas condiciones mecánicas puede producir monóxido de carbono (CO) y olores desagradables. |
| +0.704 | 0.921 | The precautions described above also apply to portable compressors. | Las precauciones descritas anteriormente también se aplican a los compresores portátiles. |
| LaBSE 0.982 | — | These instructions cover the installation, operation and maintenance of the PanBlast Fina Abrasive Control Valve. | Estas instrucciones cubren la instalación, el funcionamiento y el mantenimiento de la Válvula dosificadora en formato PanBlast Fina. |

The last row shows the full glossary stack in action: `PanBlast`
(verbatim) + `Abrasive Control Valve → Válvula dosificadora` both land
cleanly, register is formal, the surrounding text is fluent Spanish.

**Worst 6 long-source examples** (bottom of COMET-QE after dropping
single-token rows):

| COMET-QE | LaBSE | EN | ES |
|---:|---:|---|---|
| −1.237 | 0.912 | NOTE: NEVER LIFT AND/OR CARRY THE SUPPLIED AIR RESPIRATOR HELMET ASSEMBLY BY THE BREATHING TUBE, AS DAMAGE TO THE SUPPLIED AIR RESPIRATOR HELMET OR BREATHING TUBE MAY OCCUR. | NOTA: NUNCA LÍNEA Y/O LLEVAR AL RESPIRATOR DE AÉREO SUMINISTRADO Casco ASAMBLEA POR EL Tubo Respirador, EN CALIDAD DE INFIERNO AL RESPIRATOR DE AÉREO SUMINISTRADO Casco O Tubo Respirador MAYO OCCUR. |
| −1.144 | 0.896 | NOTE: THE INNER LENS PROTECTIVE LAYER UST BE PEELED OFF THE LENS BEFORE FITTING INTO THE INNER WINDOW GASKET. | NOTA: EL Lámina interior PLAYER PROTECTIVO SE PELIGRA DE LAS LENS ANTES DE FITAR EN LA GASE DE LA VENTA INTERNA. |
| −1.111 | 0.943 | Carefully remove the Inner Collar and Outer Cape press studs from the Supplied Air Respirator Helmet shell, and discard the Inner Collar and Outer Cape. | Retire cuidadosamente el Collar Interior y Exterior Capa prensar los tacos del respirador de aire suministrado Casco, y descarte el Collar Interior y Exterior Capa. |
| −1.103 | 0.867 | Do not tuck the Outer Cape into the Supplied Air Respirator Helmet shell interior. | No arrope el interior exterior Capa en el respirador de aire suministrado Casco. |
| −1.075 | 0.836 | DO NOT USE THE SPARTAN SUPPLIED AIR RESPIRATOR HELMET OPERATOR VISION IS IMPAIRED IN ANY WAY DUE TO MISTING OR FOGGING. | NO USAR LA VISIÓN DEL Spartan SUMINISTRADO DE RESPIRATOR DE AÉREO Casco El operador está IMPAIRED EN CUALQUIER MANERA DESDE EL MISTING O EL FOGGING. |
| −1.060 | 0.901 | NOTE: NEVER LIFT AND/OR CARRY THE RESPIRATOR HELMET ASSEMBLY BY THE BREATHING TUBE, AS DAMAGE TO THE RESPIRATOR HELMET OR BREATHING TUBE MAY OCCUR. | NOTA: NUNCA LÍNEA Y/O LLEVA AL RESPIRATOR Casco ASAMBLEA POR LA Tubo Respirador, COMO DAÑO AL RESPIRATOR Casco O Tubo Respirador MAYO OCCUR. |

### Failure-class taxonomy

Walking the bottom-50 rows by COMET-QE, four causes account for every
broken (< −1.0) row:

1. **Marian on long ALL-CAPS sentences (~70 % of broken rows).** The
   model was trained on mixed-case news and treats `LIFT`, `MAY`, `OR`
   as standalone tokens to be transliterated rather than translated.
   `LIFT → LÍNEA` (line), `MAY → MAYO` (May, the month), `OCCUR →
   OCCUR` (no Spanish gloss found, copied through). **Fix path:**
   sentence-case preprocessing before MT, then UPPER back if the source
   was uppercase. ~30 lines of code.
2. **PDF-extractor OCR artefacts (~15 %).** Examples: `UST` for `MUST`,
   `IMPAIRED` left untranslated, `c/w Unres.` ⇒ random abbreviation.
   **Fix path:** a 20-rule lexical normaliser (`UST → MUST`, `c/w →
   with`, `Assy → Assembly`, `Unres. → Unrestricted`) ahead of MT.
3. **Brand-name + Spanish-noun chains where Qwen rearranges word order
   (~10 %).** "Inner Collar and Outer Cape" ends up as "el Collar
   Interior y Exterior Capa" — the glossary halves are protected and
   restored individually but the noun-noun-adjective ordering Qwen
   leaves intact is awkward. **Fix path:** split the compound `Inner
   Collar-Outer Cape → Capa Titan` entry into two entries with explicit
   Spanish ordering.
4. **Glossary-tagged Spanish that becomes a "leak" in audit (~5 %).**
   `Pet Cock → Pet Cock` and `Pop Up → Pop Up` are verbatim entries:
   target *is* the EN phrase, so the English-leak audit flags them
   even though they are correct. Reported as `english_leak_rows > 0`
   despite a hit rate of 88–100 %. **Fix path:** skip the english-leak
   audit when `source == target` in `evaluate_glossary.py`.

### Pipeline performance (RTX 3060 host)

| Step | Wall-clock |
|---|---:|
| PDF → CSV extraction (incl. fastText LID) | ~6 s |
| Translation (full pipeline, 1,334 rows, 23 % cache hits) | **34 m 03 s** |
| LaBSE + BERTScore eval | 25 s |
| COMET-QE eval (full corpus, ~1,334 rows) | 16 s |
| Plots | 6 s |

The full loop from `.pdf` files to a populated `results_final/` takes
**~37 minutes** end-to-end.

---

## Reproducing the full pipeline from scratch

```powershell
# (one-time) install eval deps + stage LID and COMET-QE models
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-eval.txt
Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin' -OutFile 'models\lid.176.bin'
python scripts\download_comet_qe.py

# 1. Download PanBlast PDFs (idempotent)
python scripts\download_panblast_manuals.py

# 2. Extract: CAPS + Title-Case filters + fastText LID
python scripts\pdfs_to_csv.py

# 3. Translate (~34 min on RTX 3060)
python scripts\translate_csv.py --start-fresh

# 4. Evaluate (glossary + LaBSE + BERTScore + COMET-QE + plots)
python scripts\run_full_eval.py
```

---

## Code layout

```
mt_mvp/
├── app/
│   ├── main.py                       # FastAPI factory + Gradio mount at /gradio
│   ├── gradio_app.py                 # Gradio demo UI (also runnable standalone)
│   ├── core/config.py                # Settings (env-var driven)
│   ├── api/
│   │   ├── routes.py                 # HTTP: cache wrap, health, cache clear
│   │   ├── deps.py                   # Cached singletons: MT, Glossary, Qwen, PostEditor
│   │   └── schemas.py                # Pydantic request/response
│   └── services/
│       ├── translation.py            # Core translate orchestration
│       ├── translate_cache.py        # LRU/TTL + cache key
│       ├── glossary.py               # protect / enforce / reassert (word-bounded)
│       ├── ct2_engine.py             # CTranslate2 backend
│       ├── mt_engine.py              # HuggingFace Marian backend (fallback)
│       ├── qwen_postedit.py          # Qwen 2.5 chat-template inference
│       └── postedit.py               # PostEditor (Qwen + glossary reassert + cleanup)
├── frontend/                          # Vanilla web UI
├── glossary/   prompts/   models/
├── data/
│   ├── manifest.json
│   ├── <category>/*.pdf
│   └── csv_final/                     # extracted EN segments
├── translated_data_final/             # translations
├── results_final/                     # evaluation snapshot + report.md
├── scripts/
│   ├── download_panblast_manuals.py
│   ├── pdfs_to_csv.py
│   ├── translate_csv.py
│   ├── refresh_per_pdf.py
│   ├── filter_english.py
│   ├── evaluate_glossary.py
│   ├── evaluate_quality.py
│   ├── download_comet_qe.py
│   ├── run_comet_on_worst.py
│   ├── make_eval_plots.py
│   ├── run_full_eval.py               # one-shot eval driver
│   ├── compare_versions.py            # generic diff (only useful if you iterate)
│   ├── compile_glossary_from_xlsx.py
│   ├── convert_marian_to_ct2.py
│   └── prewarm_hf_cache.py            # prime the HF cache (Marian + Qwen)
├── tests/                             # 37 tests across 3 files
├── Dockerfile                         # multi-target build (runtime + runtime-gpu), non-root, healthcheck
├── docker-compose.yml                 # api / api-gpu / gradio-demo / prewarm profiles
├── .dockerignore   .env.example   pytest.ini
├── requirements.txt   requirements-gpu.txt   requirements-eval.txt
└── README_ENG.md / README_RUS.md / PROJECT_FILES.md
```

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/translate` | Body: `text`, `apply_glossary`, `apply_postedit`, `include_debug` |
| `GET` | `/api/v1/health` | Status, GPU, cache entries, pipeline summary |
| `POST` | `/api/v1/translate/cache/clear` | Drop all in-memory cache entries |

After editing `glossary/*.json` or `prompts/*.md`, call
`POST /api/v1/translate/cache/clear` or restart Uvicorn so cached
translations refresh.

---

## TODO / wishlist

Tracked failure classes for a future iteration:

1. **Marian on all-caps safety boilerplate** — the bottom of the COMET-QE
   distribution. Two viable fixes:
   - **Sentence-case preprocessing**: detect "> 50 % upper-case letters",
     lowercase + first-letter cap before MT, re-uppercase the output if
     the source was uppercase. Small change in
     `app/services/translation.py`.
   - **Backbone upgrade for these segments**: route them through
     `facebook/nllb-200-distilled-600M`, which is far more robust on
     shouted text.
2. **OCR artefacts** — `UST` instead of `MUST`, `Assy c/w Unres.`, glued
   `SEALING BAND O-RING PRESS STUDS` mid-sentence. A short lexical
   normaliser ahead of MT (`UST → MUST`, `c/w → with`, `Assy → Assembly`,
   `Unres. → Unrestricted`) would lift several broken rows into "ok".
3. **Glossary entry splits** — the single 100 %-miss term is `Inner
   Collar-Outer Cape → Capa Titan`. Should be split into two entries
   with disambiguated targets.
4. **Per-segment register lock** — Qwen occasionally drifts from formal
   "usted" to informal "tú" inside a long sentence; tighten the
   `prompts/postedit_en_es.md` instructions.
5. **English-leak audit** — exclude `source == target` entries (Pet Cock,
   Pop Up) so they stop showing up as false-positive leaks.
6. **CI** — wire the 37 pytest suite into GitHub Actions so PRs prove
   they don't regress glossary or extraction filters.
7. **Public LaBSE / COMET-QE drift dashboard** — `make_eval_plots.py`
   already produces the inputs; a small Streamlit page on top would let
   non-engineers track score deltas across runs.
