# Blast equipment EN→ES translator (MVP)

A **local English → Spanish** stack for shot-blasting, abrasives, and operator-PPE manuals: datasheets, SOPs, catalogue lines. No parallel corpus and no hand-crafted Spanish references—just your PDFs, a domain glossary, and models on disk.

**What you get:** **CTranslate2 Marian** by default, optional **NLLB-200**, a **JSON glossary** (the optional `notes` field in each entry is for humans only; runtime uses `source` / `target`), **Qwen 2.5** post-editing, response caching, a vanilla web UI plus **Gradio**, **Docker**, and **GitHub Actions** pytest CI.

**Measured on PanBlast** (18 PDFs → **1,334** segments after filters): LaBSE **0.901**, BERTScore **0.911**, report-line **COMET-Kiwi (wmt22, ~0…1)** **0.770**, glossary term hit rate **≈96.3%**, full GPU run **≈34 min**. Deep dive: [`results_final/report.md`](results_final/report.md). File map: [`PROJECT_FILES.md`](PROJECT_FILES.md). Russian README: [`README_RUS.md`](README_RUS.md).

---

## Final pipeline at a glance

The full PanBlast corpus (18 PDFs, ~57 MB) is extracted to 1,334 confidently
English sentences, translated through the production pipeline, and scored
on four independent reference-free metrics:

| Metric | **Final v4** (n = 1,334) |
|---|---:|
| LaBSE cosine (mean ± 95 % CI) | **0.901 ± 0.003** |
| BERTScore F1 (mean) | **0.911** |
| COMET-Kiwi wmt22 (mean, ~0…1) | **0.770** |
| COMET-Kiwi "poor" rows (< 0.5) | **30 (2.2 %)** |
| COMET-Kiwi "great" rows (≥ 0.85) | **352 (26.4 %)** |
| Glossary term-level hit rate | **96.3 %** (1,037 / 1,077) |
| Rows with all glossary terms applied | **94.7 %** (518 / 547) |
| Translation wall-clock on RTX 3060 | **34 m 00 s** (23 % cross-row cache hits) |
| ALL-CAPS rows rewritten before MT | **204 (15.3 %)** |

*Table **COMET-Kiwi** matches [`results_final/report.md`](results_final/report.md) and `quality_summary.json`. The **engine sweep** table below still uses **COMET-QE wmt20 (z-score)** for the three-way Marian vs NLLB comparison.*

The full numerical breakdown — per-term hit rates, per-PDF means, COMET
(Kiwi) buckets, best/worst translation gallery, and remaining failure classes —
lives in [`results_final/report.md`](results_final/report.md).

### How we got here — iteration history

The pipeline went through **four evaluation iterations** before landing on
this final shape. The first pass translated the entire multilingual PDF
corpus (4,201 rows; LaBSE 0.887, glossary 85 %, **4 h 41 m** on GPU). Most
failures clustered into three causes: glued ALL-CAPS parts-list dumps,
two-thirds of rows that were not English, and Qwen post-edits that
silently dropped canonical Spanish terms. The second pass added a CAPS
parts-list filter, inline fastText `lid.176.bin` language identification,
and accent-tolerant glossary re-assertion — lifting glossary to 97.0 % and
shrinking runtime to 44 m. The third pass tightened all three: a
Title-Case parts-list detector caught what CAPS missed, the LID predict
became lowercased and confidence-thresholded, `protect_source` became
word-bounded so 23 verbatim brand entries could be added without false
matches, and a cross-row memo cache reused the 23 % of duplicate EN
boilerplate (COMET-QE mean −0.066 → −0.047, broken rows 26 → 16, runtime
44 m → **34 m**). The fourth pass — landed in this commit — added an
**ALL-CAPS sentence-case preprocessor** that rewrites the 204 shouted
safety warnings into mixed case before MT and UPPER-cases the output back,
and ran a **three-way engine sweep** (Marian-CT2 vs NLLB-200-distilled
600M vs 1.3B) to confirm Marian still wins on this corpus.

| Metric | v1 | v2 | v3 | **v4 (final)** | total Δ |
|---|---:|---:|---:|---:|---:|
| LaBSE mean | 0.887 | 0.892 | 0.895 | **0.901** | **+0.014** |
| BERTScore F1 mean | — | 0.910 | 0.911 | **0.911** | — |
| COMET-QE **wmt20** mean | — | −0.066 | −0.047 | **−0.005** | **+0.061** |
| Broken rows **wmt20** (< −1.0) | — | 26 | 16 | **3** | **−23 (−88 %)** |
| Glossary hit rate | 85 % | 97.0 % | 96.4 % | **96.3 %** | **+11.3 pp** |
| Wall-clock | 4 h 41 m | 44 m | 34 m | **34 m** | **8.3×** |

---

## Engine sweep — why Marian-CT2 beats NLLB on this corpus

A three-way evaluation was run on the same 1,334-segment corpus with
identical glossary + ALL-CAPS preprocessing + Qwen post-edit. Only the MT
backend changes:

| Engine | LaBSE | BERTScore | COMET w20† | Broken† | Speed | Score* |
|---|---:|---:|---:|---:|---:|---:|
| **Marian opus-mt-en-es (CT2)** | **0.9007** | **0.9111** | −0.005 | **3 (0.2 %)** | **0.65 seg/s** | **0.8234** |
| NLLB-200-distilled-600M (HF) | 0.8962 | 0.9106 | +0.008 | 12 (0.9 %) | 0.55 seg/s | 0.8217 |
| NLLB-200-distilled-1.3B (HF) | 0.8920 | 0.9089 | **+0.020** | 8 (0.6 %) | 0.50 seg/s | 0.8197 |

† **wmt20** z-score; headline table + report corpus line use **Kiwi wmt22**.

\* *Score = 0.5·LaBSE + 0.3·BERTScore + 0.2·σ(COMET-QE).*

**Interpretation.** NLLB is more *fluent* on average (higher **wmt20** COMET-QE
z-score — the output reads more like idiomatic news Spanish), but it drifts
further from source semantics (lower LaBSE / BERTScore) and has 2–4× more
catastrophic failures on that **wmt20** “broken” scale. For technical safety documentation, where literal
fidelity matters more than fluency, Marian wins. NLLB-1.3B does *not*
dominate NLLB-600M — the larger model paraphrases more aggressively,
which hurts semantic similarity to the (often terse) source. Glossary
plumbing is engine-agnostic so hit rates are roughly equal.

**Decision: Marian-CT2 stays the default.** NLLB-200 is still selectable
via `MT_MVP_MT_ENGINE=nllb` (see § Configuration).

**Re-run the sweep.** The table above uses **Marian** from your current
`results_final/` snapshot (after `run_full_eval.py` on `translated_data_final`).
The commands below **add one alternative engine** and diff it against Marian
via `sweep_summary.py`.

Minimum (Marian in `results_final` vs NLLB-600M only):

```powershell
python scripts\translate_csv.py --start-fresh --engine nllb --mt-model facebook/nllb-200-distilled-600M --output-dir translated_data_nllb600m
python scripts\run_full_eval.py --input translated_data_nllb600m\all_segments.csv --out-dir results_sweep\nllb600m
python scripts\sweep_summary.py --label marian --dir results_final --label nllb600m --dir results_sweep\nllb600m --out results_sweep\sweep_summary.md
```

Full **three-way** table (add NLLB-1.3B; pass **three** `--label` / `--dir` pairs):

```powershell
python scripts\translate_csv.py --start-fresh --engine nllb --mt-model facebook/nllb-200-distilled-1.3B --output-dir translated_data_nllb13b
python scripts\run_full_eval.py --input translated_data_nllb13b\all_segments.csv --out-dir results_sweep\nllb13b
python scripts\sweep_summary.py --label marian --dir results_final --label nllb600m --dir results_sweep\nllb600m --label nllb13b --dir results_sweep\nllb13b --out results_sweep\sweep_summary.md
```

(Use the same path in **`run_full_eval.py --out-dir …`** and in
**`sweep_summary.py --dir …`**. You usually do not need to mkdir first — the
scripts create parent directories when writing.)

---

## Features

- **FastAPI** translate / health / cache; optional **Gradio** on `/gradio` (`MT_MVP_ENABLE_GRADIO=1`) or `python -m app.gradio_app --share` for a `*.gradio.live` link.
- **Pipeline:** glossary protect → MT → restore → optional **Qwen** post-edit with glossary re-assert; **ALL-CAPS** preprocessor for long safety lines (sentence case in, uppercase out).
- **Backends:** Marian **CTranslate2** default; **Marian HF** and **NLLB-200** via settings.
- **Cache** (LRU), **Docker** (`docker compose`, `gpu` profile with NVIDIA), **CI** (pytest on push/PR for Python 3.10 & 3.11).
- **Scripts:** PDF → CSV → batch translate → eval; one-shot **`run_full_eval.py`**; engine sweeps via **`sweep_summary.py`**.
- **Hugging Face:** put `HUGGINGFACE_HUB_TOKEN` in a root **`.env`** (or `huggingface-cli login`) for gated models such as **COMET-Kiwi**; eval scripts load it via **`app/hf_env.py`**. See **`.env.example`**—never commit secrets.

---

## How the pipeline works

For **`POST /api/v1/translate`** on cache miss:

1. **Cache lookup** (bypassed when `include_debug=true`)
2. **ALL-CAPS preprocess** — detect "≥ 60 % uppercase letters" and
   sentence-case the source so the MT engine sees mixed case
3. **Glossary protect** — longest-match, word-bounded EN phrases → `__GLS{i}__`
   placeholders
4. **Machine translate** — Marian via CTranslate2 (or PyTorch Marian, or
   NLLB-200; see `MT_MVP_MT_ENGINE`)
5. **Glossary restore** — placeholders → canonical Spanish (tolerant of
   MT-induced truncation `__GLS{i}__` → `__GLS{i}_`)
6. **Post-edit** — Qwen 2.5 Instruct refines fluency, then **reassert**
   glossary targets with an accent-/case-insensitive matcher, fall back
   to the pre-Qwen MT output if Qwen deletes a canonical term, then fix
   English leaks and spacing
7. **ALL-CAPS postprocess** — if the source was uppercase, UPPER-case the
   final output

`GET /api/v1/health` — CUDA hints, MT engine, cache size, pipeline summary.
`POST /api/v1/translate/cache/clear` — empty the in-memory cache.

---

## Requirements

- **Python 3.11+**
- **PyTorch ≥ 2.6** (CPU or CUDA; `transformers` 4.46+ refuses to load
  `.bin` checkpoints on older torch due to `CVE-2025-32434`)
- Optional: NVIDIA GPU for MT + Qwen
- Optional: Docker 24+ for the containerised path

---

## Installation (host)

```powershell
cd mt_mvp
python -m venv venv
.\venv\Scripts\Activate.ps1
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
override every default; the most useful ones are
`MT_MVP_POSTEDIT_USE_QWEN=false` on memory-constrained boxes and
`MT_MVP_MT_ENGINE=nllb` to switch the MT backbone.

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

Expected JSON: `translation = "Siempre use el casco antes de operar la
válvula neumática."`, `glossary_applied = True`, `from_cache = False`.
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

**4. ALL-CAPS preprocess in action**

```powershell
$body = @{ text = "NOTE: NEVER LIFT AND/OR CARRY THE HELMET ASSEMBLY BY THE BREATHING TUBE, AS DAMAGE MAY OCCUR." ; include_debug = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/v1/translate -Method Post -ContentType application/json -Body $body | Select-Object translation, debug
```

Inspect `debug.allcaps_source_detected = True`,
`debug.source_after_allcaps_preprocess` and the upper-cased output. Without
the v4 fix Marian translated `LIFT → LÍNEA` and `MAY → MAYO`.

**5. CT2 → Marian fallback evidence**

```powershell
docker compose logs api --tail 40 | Select-String "CTranslate2|falling back"
```

You should see the warning emitted by `app/api/deps.py` when the
container starts translating without converted CT2 weights mounted.

**6. UI surfaces**

Open in a browser:

| URL | What it serves |
|---|---|
| <http://localhost:8000/> | Vanilla web UI (`frontend/index.html`) |
| <http://localhost:8000/gradio/> | Embedded Gradio demo |
| <http://localhost:8000/docs> | Swagger / OpenAPI explorer |

**7. Inspect & shut down**

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

## Tests and CI

**53** pytest cases (glossary, ALL-CAPS, PDF filters, API with a **fake MT engine** — no weight downloads in CI). Locally: `pytest -q`. On GitHub: `.github/workflows/tests.yml` (Python 3.10 and 3.11). Per-file detail lives in [`PROJECT_FILES.md`](PROJECT_FILES.md).

---

## Configuration (`MT_MVP_*` and optional `.env`)

App variables use the **`MT_MVP_`** prefix (see the table below). A root **`.env`** is **optional** — defaults live in `app/core/config.py`. Copy **`.env.example` → `.env`** when you need overrides.

For **Hugging Face** (gated COMET-Kiwi and similar), add `HUGGINGFACE_HUB_TOKEN` to the **same** `.env`; eval scripts pick it up via **`app/hf_env.py`**. Never commit secrets.

| Variable | Role |
|----------|------|
| `MT_MVP_MT_ENGINE` | `ctranslate2` (default), `marian_hf`, or `nllb` |
| `MT_MVP_MT_MODEL_NAME` | HF id for tokenizer / model weights (`Helsinki-NLP/opus-mt-en-es`, `facebook/nllb-200-distilled-600M`, …) |
| `MT_MVP_CT2_MODEL_DIR` | Directory with `model.bin` |
| `MT_MVP_CT2_COMPUTE_TYPE` | `int8`, `float16`, `default`, etc. |
| `MT_MVP_NLLB_SRC_LANG` | Flores-200 source code, default `eng_Latn` |
| `MT_MVP_NLLB_TGT_LANG` | Flores-200 target code, default `spa_Latn` |
| `MT_MVP_NLLB_NUM_BEAMS` | NLLB beam search width, default 4 |
| `MT_MVP_NLLB_DTYPE` | `auto` (bf16 on CUDA, fp32 on CPU), `fp16`, `bf16`, `fp32` |
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

Or add `MT_MVP_ENABLE_GRADIO=1` to the repo-root **`.env`** (same mechanism as other `MT_MVP_*` vars) and restart uvicorn.

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
(health probe, sample translate, cache-hit, ALL-CAPS evidence, fallback
warning).

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

Identical pipeline to `POST /api/v1/translate` (including the ALL-CAPS
preprocess + UPPER postprocess):
`allcaps sentence-case → glossary protect → CT2 Marian → glossary
restore → Qwen 2.5-3B post-edit → glossary re-assert → allcaps UPPER →
spacing fixes`.

- **GPU** auto-selected (Qwen bf16 + CTranslate2 Marian ≈ 7.7 GB VRAM on
  RTX 3060).
- **Resumable**: existing non-empty `target_es` rows are reused; atomic
  flushes every 25 rows survive Ctrl-C / power loss.
- **Cross-row memo cache**: duplicate EN strings share their ES
  translation. **23.3 %** of the corpus is duplicate boilerplate, so the
  cache saves ~10 minutes per full run.
- **Engine override**: `--engine nllb --mt-model facebook/nllb-200-distilled-600M`
  (or any HF id) lets the same script power the model sweep used in
  `results_final/report.md`.

```powershell
python scripts\translate_csv.py                  # full pipeline, Marian-CT2 default
python scripts\translate_csv.py --no-postedit    # MT-only fast pass
python scripts\translate_csv.py --limit 50       # smoke test
python scripts\translate_csv.py --start-fresh    # ignore previously-translated rows
python scripts\translate_csv.py --engine nllb --mt-model facebook/nllb-200-distilled-600M --output-dir translated_data_nllb600m --start-fresh
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

resulting_files/                            # client deliverable (2 columns)
├── all_segments.csv
└── <category>/<pdf_stem>.csv               # source_en, target_es only

results_final/                              # evaluation snapshot
├── glossary/                               # per-term & per-PDF stats
├── quality/                                # LaBSE, BERTScore, COMET (Kiwi in quality_scores; optional wmt20 in comet_qe_worst)
├── plots/                                  # 9 PNGs
└── report.md
```

### 5. Client deliverable — `resulting_files/`

A reviewer-friendly view of the winning translation: per-PDF CSVs with
just two columns, `source_en` and `target_es`, plus a consolidated
`all_segments.csv`. Regenerate any time:

```powershell
python scripts\build_resulting_files.py
python scripts\build_resulting_files.py --input translated_data_nllb600m\all_segments.csv --output resulting_files_nllb600m
```

---

## Evaluation

Install **`requirements-eval.txt`**. Full snapshot into `results_final/`:

```powershell
python scripts\run_full_eval.py
```

**What runs:**

1. **`evaluate_glossary.py`** — glossary hits and English leaks.
2. **`evaluate_quality.py`** — **LaBSE**, **BERTScore**, and **COMET**:
   - default **auto**: if an HF token is present → **COMET-Kiwi** `wmt22` (~**0…1**); else **COMET-QE** `wmt20` (**z-score**). If auto-Kiwi fails once → fallback to **wmt20**;
   - override: `--comet-model Unbabel/wmt22-cometkiwi-da` or `.../wmt20-comet-qe-da`;
   - weights: `python scripts\download_comet_qe.py` (wmt20 default) or **`--model Unbabel/wmt22-cometkiwi-da`** for Kiwi (gated repo + license on HF);
   - CSV column **`comet_qe`**: holds **Kiwi** scores when Kiwi is used, **z-scores** when wmt20 is used—don’t mix without checking `quality_summary.json → comet_model`.
3. **`run_comet_on_worst.py`** — **wmt20** on the worst-by-LaBSE slice (separate CSV; z-score buckets). [`results_final/report.md`](results_final/report.md) uses **Kiwi** from step 2 for the corpus headline.
4. **`make_eval_plots.py`** — PNGs.

Optional **`sweep_summary.py`** — ranks multiple engine runs (the *Engine sweep* section above).

---

## Detailed analytics

Everything charts-and-tables lives in [`results_final/report.md`](results_final/report.md); it refreshes with `run_full_eval.py`. This README keeps only headline numbers so it never drifts from the report.

---

## Reproducing the full pipeline from scratch

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-eval.txt
Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin' -OutFile 'models\lid.176.bin'
python scripts\download_comet_qe.py
# Optional gated COMET-Kiwi: token in .env → then
# python scripts\download_comet_qe.py --model Unbabel/wmt22-cometkiwi-da

python scripts\download_panblast_manuals.py
python scripts\pdfs_to_csv.py
python scripts\translate_csv.py --start-fresh
python scripts\run_full_eval.py
python scripts\build_resulting_files.py
```

---

## Code layout

```
mt_mvp/
├── .github/workflows/tests.yml            # CI: pytest on push + PR, Py 3.10/3.11 matrix
├── app/
│   ├── main.py                            # FastAPI factory + Gradio mount at /gradio
│   ├── gradio_app.py                      # Gradio demo UI (also runnable standalone)
│   ├── core/config.py                     # Settings (env-var driven)
│   ├── hf_env.py                          # load root .env (HF token, etc.)
│   ├── api/
│   │   ├── routes.py                      # HTTP: cache wrap, health, cache clear
│   │   ├── deps.py                        # Cached singletons: MT, Glossary, Qwen, PostEditor
│   │   └── schemas.py                     # Pydantic request/response
│   └── services/
│       ├── translation.py                 # Core translate orchestration (allcaps + glossary + MT + postedit)
│       ├── text_case.py                   # ALL-CAPS detector + sentence-case preprocessor (v4)
│       ├── translate_cache.py             # LRU/TTL + cache key
│       ├── glossary.py                    # protect / enforce / reassert (word-bounded)
│       ├── ct2_engine.py                  # CTranslate2 Marian backend (default)
│       ├── mt_engine.py                   # HuggingFace Marian backend (fallback)
│       ├── nllb_engine.py                 # HuggingFace NLLB-200 backend (alternative, v4)
│       ├── qwen_postedit.py               # Qwen 2.5 chat-template inference
│       └── postedit.py                    # PostEditor (Qwen + glossary reassert + cleanup)
├── frontend/                              # Vanilla web UI
├── glossary/   prompts/   models/
├── data/
│   ├── manifest.json
│   ├── <category>/*.pdf
│   └── csv_final/                         # extracted EN segments
├── translated_data_final/                 # winning Marian-CT2 + Qwen + ALL-CAPS translations
├── resulting_files/                       # client deliverable (2-column CSVs)
├── results_final/                         # evaluation snapshot + report.md
├── scripts/
│   ├── download_panblast_manuals.py
│   ├── pdfs_to_csv.py
│   ├── translate_csv.py                   # batch translate (supports --engine nllb)
│   ├── build_resulting_files.py           # per-PDF 2-column CSVs (v4)
│   ├── filter_english.py
│   ├── evaluate_glossary.py
│   ├── evaluate_quality.py
│   ├── download_comet_qe.py
│   ├── run_comet_on_worst.py
│   ├── make_eval_plots.py
│   ├── run_full_eval.py                   # one-shot eval driver
│   ├── sweep_summary.py                   # N-way engine ranking (v4)
│   ├── compare_versions.py                # generic 2-way diff
│   ├── compile_glossary_from_xlsx.py
│   ├── convert_marian_to_ct2.py
│   ├── prefetch_nllb.py                   # download NLLB weights into HF cache (v4)
│   └── prewarm_hf_cache.py                # prime the HF cache (Marian + Qwen)
├── tests/                                  # 53 tests across 4 files (incl. text_case)
├── Dockerfile                              # multi-target build (runtime + runtime-gpu), non-root, healthcheck
├── docker-compose.yml                      # api / api-gpu / gradio-demo / prewarm profiles
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

When `include_debug=true` the response includes
`debug.allcaps_source_detected`, `debug.source_after_allcaps_preprocess`,
`debug.stages_executed_this_request`, plus the pre/post-MT strings —
useful when comparing engines or chasing a glossary miss.

---

## TODO / wishlist

The first six items from the v3 wishlist are **closed** in v4 (ALL-CAPS
preprocessor landed, CI is wired, glossary lowercased, NLLB tested and
rejected, English-leak audit cleaned, glossary entry splits applied
selectively). Remaining tracked items:

1. **OCR artefacts** — `UST` instead of `MUST`, `Assy c/w Unres.`, glued
   `SEALING BAND O-RING PRESS STUDS` mid-sentence. A short lexical
   normaliser ahead of MT (`UST → MUST`, `c/w → with`, `Assy → Assembly`,
   `Unres. → Unrestricted`) would lift the remaining 3 broken rows into
   "ok" and probably close the gap on `inner lens` / `outer lens`.
2. **Un-deduplicated CAD parts dumps** — the worst-6 gallery is now
   dominated by CAD exports where every callout becomes a flat
   sentence ("Body Lever Screw Screw …"). These have no grammatical
   structure for any NMT model to anchor onto; the right fix is a
   PDF-side table detector or a content-aware splitter.
3. **Per-segment register lock** — Qwen occasionally drifts from formal
   "usted" to informal "tú" inside a long sentence; tighten the
   `prompts/postedit_en_es.md` instructions.
4. **Public LaBSE / COMET-QE drift dashboard** — `make_eval_plots.py`
   already produces the inputs; a small Streamlit page on top would let
   non-engineers track score deltas across runs.
5. **NLLB hybrid routing** — for the ~3 % of rows where COMET-QE on
   Marian < −0.5, retry with NLLB-1.3B and keep the higher-scoring
   output. The v4 sweep numbers suggest this would lift the corpus mean
   without dragging down the head of the distribution.
