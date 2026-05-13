# Переводчик EN→ES для дробеструйного оборудования (MVP)

MT-стек для перевода **английский → испанский** в нише **дробеструйной
обработки, абразивной обработки и индивидуальных средств защиты
оператора** (даташиты, SOP-документы, каталожные позиции). Использует
**CTranslate2 Marian** (с опцией **PyTorch Marian**), **JSON-глоссарий**,
**Qwen 2.5** для пост-редактирования, **LRU/TTL-кэш** ответов, обычный
**веб-UI**, **Gradio**-демо с публичной ссылкой и
production-готовый **Docker**-образ.

Покомпонентная карта файлов — в **[`PROJECT_FILES.md`](PROJECT_FILES.md)**.
Английская README — **[`README_ENG.md`](README_ENG.md)**.

---

## Финальный пайплайн — кратко

Корпус PanBlast (18 PDF, ~57 МБ) извлекается в 1 334 уверенно-английских
сегмента, переводится через production-пайплайн и оценивается четырьмя
независимыми reference-free метриками:

| Метрика | **Финал** (n = 1 334) |
|---|---:|
| LaBSE cosine (среднее ± 95 % ДИ) | **0.895 ± 0.003** |
| BERTScore F1 (среднее) | **0.911** |
| COMET-QE (среднее, DA z-score) | **−0.047** |
| COMET-QE «broken» строк (< −1.0) | **16 (1.2 %)** |
| Hit rate глоссария по терминам | **96.4 %** (1 029 / 1 067) |
| Строк со всеми применёнными терминами | **95.5 %** (514 / 538) |
| Wall-clock полного перевода (RTX 3060) | **34 м 03 с** (23 % cross-row кэша) |

Полный численный разбор — per-term hit rates, per-PDF средние,
COMET-QE-корзины, галерея лучших/худших переводов, оставшиеся классы
ошибок — лежит в [`results_final/report.md`](results_final/report.md) и
продублирован ниже в этом файле.

### Как сюда пришли

Пайплайн прошёл **три эволюционные итерации**, прежде чем устаканился в
финал. Первая версия переводила весь мультиязычный PDF-корпус целиком
(4 201 строка; LaBSE 0.887, hit rate глоссария 84.9 %, 4 ч 41 м на
GPU). Большинство провалов сводились к трём причинам: (1) слипшиеся
ALL-CAPS списки деталей, которые Marian не мог распарсить, (2)
две трети строк оказались не на английском, но всё равно проходили
через MT, и (3) Qwen в post-edit тихо удалял канонические испанские
термины или сбрасывал акценты. Вторая итерация добавила (1) CAPS-фильтр
parts-list на этапе извлечения PDF, (2) inline fastText `lid.176.bin`
для определения языка и (3) акцент- и регистро-толерантный re-assert
глоссария с fallback на чистый MT-выход, когда Qwen удаляет термин —
вместе это подняло глоссарий до 97.0 % и сократило runtime до 44 м.
Финальная итерация ужесточила все три: Title-Case-детектор parts-list
ловит то, что пропустил CAPS-фильтр, LID-предикт теперь идёт по
lowercase-входу с порогом 0.55 (отбрасывает 2 450 не-английских
сегментов на 10 языках), `protect_source` стал word-bounded, чтобы
безопасно добавить 23 новые verbatim-записи бренда/стандартов (Spartan,
Titan, NPT, OSHA, …) без матча внутри длинных слов, а cross-row
memo-кэш переиспользует 23 % английских предложений, повторяющихся
между PDF. **Чистый эффект:** среднее COMET-QE сдвинулось −0.066 →
−0.047 (+0.019), число broken-строк упало на 38 % (26 → 16), термин
`Remote control valve → Válvula neumática` прошёл из 100 %-промаха в
100 %-попадание (37/37), а полный перевод теперь занимает 34 минуты
вместо почти 5 часов.

---

## Возможности

- **FastAPI + Uvicorn** REST API на `/api/v1/*`
- **Глоссарий-aware MT**: защищаем EN-термины → переводим → восстанавливаем
  ES-таргеты; Qwen post-edit повторно применяет канонические термины и
  откатывается на чистый MT, если Qwen перифразирует термин
- **Backend-ы**: **CTranslate2** (по умолчанию) или **HuggingFace Marian**
- **Post-edit**: опционально **Qwen 2.5 Instruct** (3B по умолчанию, 7B
  доступна)
- **Кэш**: in-memory LRU (1 024 записи), TTL 24 ч
- **Два UI**:
  - production-вёрстка на `/`
  - **Gradio**-демо на `/gradio` или standalone-запуск `python -m
    app.gradio_app --share`, который публикует
    `https://*.gradio.live` URL для клиентов
- **Docker**: `docker compose up` поднимает API + Gradio вместе;
  `--profile gpu` использует NVIDIA Container Toolkit
- **Инструментарий**: пайплайн PDF → CSV → перевод → оценка с
  one-shot драйвером (`scripts/run_full_eval.py`)

---

## Как работает пайплайн

Для **`POST /api/v1/translate`** при cache miss:

1. **Lookup в кэше** (обходится, если `include_debug=true`)
2. **Защита глоссарием** — самый длинный word-bounded EN-матч →
   placeholder `__GLS{i}__`
3. **Машинный перевод** — Marian EN→ES через CTranslate2 (или чистый PyTorch)
4. **Восстановление глоссария** — placeholder → канонический испанский
   (regex толерантен к MT-обрезанию `__GLS{i}__` до `__GLS{i}_`)
5. **Post-edit** — Qwen 2.5 Instruct полирует беглость, затем
   **reassert** глоссария по accent- / case-insensitive-матчу, fallback
   на pre-Qwen MT-выход, если Qwen удалил канонический термин, потом
   исправление English-leak-ов и пробелов

`GET /api/v1/health` — состояние CUDA, MT-engine, размер кэша, summary
пайплайна.
`POST /api/v1/translate/cache/clear` — очистить in-memory-кэш.

---

## Скорость и качество

| Слой | Выбор |
|------|-------|
| MT | **CTranslate2** для эффективного инференса `model.bin` |
| GPU | CUDA auto-select; Qwen жёстко пинуется на `device_map={"": 0}` |
| Post-edit | Qwen 2.5 **3B**, 256 max-new-tokens, 2 048 input truncation |
| Качество | protect / restore / reassert глоссария + Qwen + чистка пробелов |
| Повторные вызовы | LRU + TTL кэш возвращает идентичные ответы без MT/Qwen |
| Batch-дедуп | Cross-row memo-кэш в `scripts/translate_csv.py` (23 % cache hit) |

Полностью отключить Qwen: `MT_MVP_POSTEDIT_USE_QWEN=false`.

---

## Требования

- **Python 3.11+**
- **PyTorch** (CPU или CUDA; CUDA 12.4 wheels — в `requirements-gpu.txt`)
- Опционально: NVIDIA GPU для MT + Qwen
- Опционально: Docker 24+ для контейнерного варианта

---

## Установка (на хост)

```powershell
cd mt_mvp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-gpu.txt   # CUDA-wheels (опционально)
python -m pip install -r requirements-eval.txt  # нужны только для воспроизведения оценки
```

**Конвертировать Marian в CTranslate2** (создаёт `models/opus-mt-en-es-ct2/model.bin`):

```powershell
python scripts\convert_marian_to_ct2.py --model Helsinki-NLP/opus-mt-en-es --output-dir models\opus-mt-en-es-ct2 --force
```

**Пересобрать глоссарий из Excel**:

```powershell
python scripts\compile_glossary_from_xlsx.py
```

---

## Установка (Docker)

```bash
# профиль по умолчанию: API + Gradio демо на http://localhost:8000 (CPU-образ,
# target `runtime` в Dockerfile, slim Python base ~1.5 GB)
docker compose up --build

# профиль GPU: собирает target `runtime-gpu` (nvidia/cuda:12.4.1 + cu124
# PyTorch wheels, ~6 GB образ). На хосте должны быть NVIDIA Container
# Toolkit и драйвер CUDA-12.x.
docker compose --profile gpu up --build

# профиль публичного демо: на старте выводит https://*.gradio.live URL
docker compose --profile demo up --build

# Опциональный one-shot: прогреть HF-кэш Marian + Qwen (~5.5 GB) в общий
# volume `hf_cache`, чтобы первый /translate шёл мгновенно, а не висел
# на скачивании. Запускается один раз после первой сборки.
docker compose --profile prewarm run --rm prewarm
```

В Dockerfile четыре стейджа — `base` / `base-gpu` ставят venv (CPU torch
или cu124 torch), `runtime` / `runtime-gpu` копируют его + код приложения,
переключаются на UID 10001 и открывают `:8000`. Модели **не** запекаются
в образ — монтируются в рантайме через `./models:/app/models`. Если
`model.bin` отсутствует, API автоматически фолбэчится на HuggingFace
Marian (см. `app/api/deps.py`).

`MT_MVP_*` env-vars (полный список — в `app/core/config.py` и
`.env.example`) перекрывают любые дефолты; самый полезный —
`MT_MVP_POSTEDIT_USE_QWEN=false` на машинах с дефицитом памяти.

### Проверка работоспособности контейнера

После `docker compose up -d --build` пройдитесь по этому чек-листу,
чтобы убедиться, что стек жив и переводит. Все команды для PowerShell
на Windows (на Linux/macOS замените `Invoke-RestMethod` на `curl`).

**1. Статус и healthcheck**

```powershell
docker compose ps                       # ожидаем: api  Up X seconds (healthy)
Invoke-RestMethod http://localhost:8000/api/v1/health | Format-List
```

`/health` возвращает `status=ok`, выбранный MT-движок, наличие CUDA и
размер LRU-кэша.

**2. Первый перевод (cold path)**

```powershell
$body = @{ text = "Always wear the Helmet before operating the Remote control valve." } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/v1/translate `
  -Method Post -ContentType application/json -Body $body
```

Ожидаемый JSON: `translation = "Siempre use el casco antes de operar el
válvula neumática ."`, `glossary_applied = True`, `from_cache = False`.
Первый запрос идёт 5–30 с, потому что Marian лениво скачивает веса в
`hf_cache`; последующие — мгновенно.

**3. Попадание в LRU-кэш (warm path)**

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

Запрос №1 → `from_cache: False`; запрос №2 → `from_cache: True` и
примерно в 10 раз быстрее.

**4. Подтверждение фолбэка CT2 → Marian**

```powershell
docker compose logs api --tail 40 | Select-String "CTranslate2|falling back"
```

Должен появиться warning из `app/api/deps.py` — он печатается, когда
контейнер начинает переводить без примонтированных CT2-весов.

**5. UI-эндпоинты**

Откройте в браузере:

| URL | Что отдаёт |
|---|---|
| <http://localhost:8000/> | Vanilla web UI (`frontend/index.html`) |
| <http://localhost:8000/gradio/> | Встроенный Gradio-демо |
| <http://localhost:8000/docs> | Swagger / OpenAPI |

**6. Заглянуть внутрь и остановить**

```powershell
docker compose exec api bash             # зайти в контейнер
docker compose restart api               # рестарт без потери hf_cache
docker compose down                      # остановить + удалить (hf_cache остаётся)
docker compose down -v                   # полная чистка вместе с hf_cache
```

> **Про Qwen.** В дефолтном контейнере Qwen 2.5 post-editing **включён**
> — поэтому самый первый `/translate` ещё и подкачает ~5 GB весов
> Qwen (несколько минут на CPU). Чтобы быстрее проверить пайплайн,
> можно поднять с `MT_MVP_POSTEDIT_USE_QWEN=false`. Или, чтобы
> сохранить полный пайплайн и не ждать на холодном старте — один раз
> прогрейте кэш: `docker compose --profile prewarm run --rm prewarm`.

---

## Тесты (`pytest`)

| Suite | Файл | Покрытие |
|-------|------|----------|
| **Глоссарий (unit)** | `tests/test_glossary_protect_restore.py` | `protect_source` (регистр + word-boundary), `enforce_placeholders` (толерантен к MT-обрезанию), `reassert_targets_after_edit` (accent drift, multi-drift, смесь canonical+drifted, fallback на reference, multi-entry round trip), verbatim-brand round trip |
| **PDF-фильтры (unit)** | `tests/test_pdf_extraction_filters.py` | `is_caps_parts_dump` (positive + whitelist предупреждений + минимальная длина), `is_titlecase_parts_dump` (positive + whitelist заголовков + period + инструкционных предложений), объединённый `is_parts_dump` |
| **API (integration)** | `tests/test_api_integration.py` | `TestClient` с `FakeMTEngine`: полный flow перевода, hit кэша на повторном запросе, обход кэша по `include_debug`, 422 на пустом тексте, health-роут, cache-clear-роут |

```powershell
pytest -v
```

```
============================= 37 passed in 0.22s =============================
```

---

## Конфигурация (`MT_MVP_*` или `.env`)

| Переменная | Назначение |
|------------|------------|
| `MT_MVP_MT_ENGINE` | `ctranslate2` (по умолчанию) или `marian_hf` |
| `MT_MVP_MT_MODEL_NAME` | HF id для токенизатора / Marian-весов |
| `MT_MVP_CT2_MODEL_DIR` | Директория с `model.bin` |
| `MT_MVP_CT2_COMPUTE_TYPE` | `int8`, `float16`, `default` и т. д. |
| `MT_MVP_DEVICE` | `cuda`, `cpu` или пусто (auto) |
| `MT_MVP_POSTEDIT_USE_QWEN` | `true` / `false` |
| `MT_MVP_POSTEDIT_QWEN_MODEL` | например `Qwen/Qwen2.5-3B-Instruct` |
| `MT_MVP_POSTEDIT_MAX_NEW_TOKENS` | Qwen decode cap |
| `MT_MVP_POSTEDIT_QWEN_MAX_INPUT_TOKENS` | Промпт-truncation |
| `MT_MVP_GLOSSARY_PATH` | Путь к JSON-глоссарию |
| `MT_MVP_POSTEDIT_PROMPT_PATH` | Путь к post-edit markdown |
| `MT_MVP_ENABLE_GRADIO` | `1` монтирует Gradio UI на `/gradio` внутри FastAPI (Docker ставит сам) |

---

## Запуск

Три полностью независимых режима — выбирай нужный. Они не делят ни
порты, ни env-vars, ни процессы, поэтому можно поднимать в параллель.

### Режим A — FastAPI-сервис (порт 8000)

Production UI + REST API.

```powershell
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- UI: <http://127.0.0.1:8000/>
- Swagger: <http://127.0.0.1:8000/docs>

Чтобы **дополнительно** монтировать Gradio внутрь того же FastAPI-процесса:

```powershell
$env:MT_MVP_ENABLE_GRADIO = "1"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Gradio (встроенный): <http://127.0.0.1:8000/gradio>

### Режим B — Standalone Gradio с публичной ссылкой (порт 7861)

Быстрая демонстрация клиенту: запускается Gradio напрямую (без FastAPI),
поднимает временный туннель `https://*.gradio.live`, который можно
открыть откуда угодно.

```powershell
.\venv\Scripts\Activate.ps1
python -m app.gradio_app --share --port 7861
```

Вывод:

```
* Running on local URL:  http://0.0.0.0:7861
* Running on public URL: https://abcd1234.gradio.live
```

Ссылка `gradio.live` действительна 7 дней. Любой свободный порт можно
выбрать через `--port`; дефолтный `7860` часто занят другими
dev-серверами. Этот режим **не** читает `MT_MVP_ENABLE_GRADIO` — env-var
работает только в режиме A.

### Режим C — Docker

```bash
# CPU-API + встроенный Gradio на http://localhost:8000
docker compose up --build

# То же, на GPU (нужен NVIDIA Container Toolkit + драйвер CUDA-12.x).
# Собирает target `runtime-gpu` с cu124 PyTorch wheels.
docker compose --profile gpu up --build

# Standalone Gradio с --share на порту 7860 (отдельный контейнер)
docker compose --profile demo up --build

# One-shot: прогреть volume `hf_cache` весами Marian + Qwen, чтобы первый
# /translate не висел на скачивании 5.5 GB. Запускается один раз после
# первой сборки.
docker compose --profile prewarm run --rm prewarm
```

Убедиться, что контейнер реально жив и переводит, можно по чек-листу
[§ Проверка работоспособности контейнера](#проверка-работоспособности-контейнера)
(healthcheck, sample-перевод, cache-hit, fallback-warning).

---

## Корпус PanBlast: скачать → извлечь → перевести

Помимо HTTP-сервиса, проект содержит batch-пайплайн, который собирает
доменный корпус из технических PDF-руководств PanBlast и прогоняет его
через тот же engine, что использует API.

### 1. Скачать PDF — `scripts/download_panblast_manuals.py`

- 18 PDF (~57 МБ) в 5 категориях, скачиваются с
  <https://www.panblast.com/manuals.acv>.
- Идемпотентный скрипт: уже скачанные файлы пропускаются.
- Пишет `data/manifest.json` с маппингом product code → PDF.

```powershell
python scripts\download_panblast_manuals.py
```

### 2. PDF → CSV — `scripts/pdfs_to_csv.py`

Пайплайн (по каждому PDF):

- **PyMuPDF** извлечение текста (лучший reading order на технических
  layout-ах).
- Срезание running-заголовков / footer-ов (строки, встречающиеся на
  ≥ 50 % страниц).
- Срез шума: номера страниц, голые номера деталей (`ZVP-PC-0027-01`),
  чистые числа, строки < 3 букв.
- **Срез CAPS-parts-list-дампов**: длинные all-uppercase глуинные
  именные фразы без инструкционных глаголов и без внутренней пунктуации.
- **Срез Title-Case-parts-list-дампов**: длинные Title-Case-dominant
  сегменты с ≥ 2 catalog-keyword-ов (PARTS, LIST, STOCK, CODE,
  DESCRIPTION, ITEM, ASSEMBLY, EXPLODED, …) и без инструкционного
  глагола.
- Склейка PDF-обрезанных строк в абзацы → split по `.!?`.
- Ограничение длины сегмента 800 симв. (OPUS-MT 512-token ceiling).
- **fastText `lid.176.bin`** язык-фильтр на **lowercased**-сегменте;
  отбрасывается, если `lid != "en"` ИЛИ confidence <
  `--lang-min-confidence` (по умолчанию **0.55**).
- Стабильный SHA-1 row-id по `category|pdf|page|segment_idx|source_en`.

Выход: **1 334 сегмента / ~184 K source-chars** в `data/csv_final/`.

```powershell
python scripts\pdfs_to_csv.py
python scripts\pdfs_to_csv.py --lang-min-confidence 0.40
python scripts\pdfs_to_csv.py --no-lang-filter            # сохранить все языки
```

### 3. Batch-перевод — `scripts/translate_csv.py`

Идентичный пайплайн с `POST /api/v1/translate`:
`защита глоссарием → CT2 Marian → восстановление глоссария → Qwen 2.5-3B
post-edit → re-assert глоссария → правка пробелов`.

- **GPU** выбирается автоматически (Qwen bf16 + CTranslate2 Marian ≈
  7.7 ГБ VRAM на RTX 3060).
- **Возобновляемый**: непустые `target_es` переиспользуются; атомарный
  flush каждые 25 строк переживёт Ctrl-C / отключение питания.
- **Cross-row memo-кэш**: одинаковые EN-строки делят ES-перевод. **23.3 %**
  корпуса — повторяющийся boilerplate, поэтому кэш экономит ~10 минут на
  каждом полном запуске.

```powershell
python scripts\translate_csv.py                  # полный пайплайн
python scripts\translate_csv.py --no-postedit    # MT-only быстрый проход
python scripts\translate_csv.py --limit 50       # smoke-тест
python scripts\translate_csv.py --start-fresh    # игнорировать уже переведённые строки
```

Выходной CSV: `id, category, pdf, page, segment_idx, source_en, target_es,
char_count, glossary_applied, postedit_applied, error`.

### 4. Куда всё пишется

```
data/
├── manifest.json                           # product code → PDF
├── <category>/*.pdf                        # 18 скачанных PDF
└── csv_final/                              # извлечённые EN-сегменты
    ├── all_segments.csv                    # 1 334 сегмента
    └── <category>/<pdf_stem>.csv

translated_data_final/                      # переводы
├── all_segments.csv
└── <category>/<pdf_stem>.csv

results_final/                              # оценка
├── glossary/                               # per-term & per-PDF stats
├── quality/                                # LaBSE, BERTScore, COMET-QE
├── plots/                                  # 9 PNG
└── report.md
```

---

## Оценка качества: глоссарий + LaBSE + BERTScore + COMET-QE

Четыре независимых сигнала качества, все reference-free, все локально
(`requirements-eval.txt`).

**Один драйвер** (пишет в `results_final/`):

```powershell
python scripts\run_full_eval.py
```

Под капотом выполняет четыре скрипта:

### a. `scripts/evaluate_glossary.py`

Для каждой строки: какие EN-источники встречаются, появляется ли
канонический ES-таргет в переводе, протекает ли какая-нибудь EN-фраза
непереведённой. Пишет per-row, per-term, per-PDF CSV + stdout-summary.

### b. `scripts/evaluate_quality.py` — LaBSE + BERTScore

- **LaBSE** — мультиязычный энкодер Google; косинус между эмбеддингами
  источника и таргета.
- **BERTScore F1** на `xlm-roberta-large` (слой 17) — token-level
  semantic-overlap с source-as-pseudo-reference.

### c. `scripts/run_comet_on_worst.py` — COMET-QE

`Unbabel/wmt20-comet-qe-da` (2.28 ГБ) — каноничный reference-free
MT-QE-эстиматор. Шкала — **DA z-score**, примерно −1.5..+1.0, выше —
лучше. Всё ниже −1.0 — фактически сломанный MT.

```powershell
python scripts\download_comet_qe.py        # один раз, кладёт в models/
python scripts\run_comet_on_worst.py --top 1334
```

### d. `scripts/make_eval_plots.py`

9 PNG в `results_final/plots/`: LaBSE / BERTScore-распределения и
scatter, per-PDF среднее LaBSE, per-term / per-PDF тепловые карты
hit-rate, pie row-status, COMET-buckets per PDF.

---

## Подробная аналитика

### Извлечение корпуса и отклонение по LID

18 PDF-руководств PanBlast (~57 МБ) содержат ~3 800 уникальных текстовых
сегментов после PyMuPDF-извлечения. Большинство — **не на английском**:

| Этап | Сегменты | Заметки |
|---|---:|---|
| PyMuPDF — сырое извлечение | ~3 800 | колонтитулы, номера страниц, part-номера уже отброшены |
| После CAPS- + Title-Case-фильтров | 3 784 | ~80 слипшихся ALL-CAPS + Title-Case дампов отброшено |
| После fastText LID @ 0.55 | **1 334** | проходят только уверенно-английские строки |
| → слишком короткие для LID | 27 | < 8 буквенных символов; полагаемся на низовой noise-фильтр |

LID отбросил **2 450 не-английских сегментов** на 10 определённых
языках (pt ~ fr ~ de ~ hu ~ nl ~ es ~ it ~ sv — каждый ≈ 11 %; ru 9 %;
low-confidence en 2 %). Без LID все они прошли бы перевод EN→ES — впустую
~3 ч GPU и просадка всех корпусных средних метрик.

### Распределения метрик качества

**LaBSE cosine** (мультиязычная эмбеддинговая схожесть, шкала [−1, 1]):

| Статистика | Значение |
|---|---:|
| среднее ± 95 % ДИ | **0.8947 ± 0.0032** |
| ст. отклонение | 0.0605 |
| p05 / медиана / p95 | 0.7834 / 0.9095 / 0.9530 |
| min / max | 0.3597 / 1.0000 |

Распределение по корзинам: **59.3 % great (≥ 0.9), 35.0 % good (0.8–0.9),
3.9 % ok (0.7–0.8), 1.8 % poor (< 0.7)**.

**BERTScore F1** (xlm-roberta-large, слой 17): среднее **0.9110 ± 0.0010**.
**98.8 %** сегментов — в корзине `high (0.85–0.95)`, **1.0 %** — в
`very-high (≥ 0.95)`, **0.2 %** — в `mid (0.7–0.85)`, **0 %** — в
`low (< 0.7)`.

**COMET-QE** (`Unbabel/wmt20-comet-qe-da`, DA z-score; оценено на RTX
3060 за ~16 с):

| Статистика | Значение |
|---|---:|
| среднее | **−0.0469** |
| медиана | −0.0626 |
| ст. отклонение | 0.4163 |
| min / max | −1.24 / +0.97 |

| Корзина | Кол-во | Доля |
|---|---:|---:|
| broken (< −1.0) | **16** | 1.2 % |
| very weak (−1.0 .. −0.5) | 195 | 14.6 % |
| weak (−0.5 .. −0.2) | 264 | 19.8 % |
| ok (−0.2 .. 0.1) | 350 | 26.2 % |
| good (≥ 0.1) | **509** | **38.2 %** |

### Аудит глоссария

**Шапка:** 1 067 вхождений терминов в источнике, **1 029 применены в
переводе = hit rate 96.4 %**. 514 из 538 строк с терминами (95.5 %)
имеют **все** глоссарийные термины применёнными; только **24 строки**
покрыты частично.

Топ-20 наиболее используемых терминов:

| EN-источник | Канонический ES | Вхожд. | Прим. | Hit rate |
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

Из 23 verbatim-добавлений (бренд/стандарт), **18 дают 100 %-hit rate**
(Panblast, AirFlo, Galaxy, AbraFlo, Cosmo, JIC, AcoustiFlex, Pet Cock,
ISO, OSHA, Titan, …); две — 94–98 % (NPT, BSP — каждая потеряна на
одном all-caps safety-абзаце); и **Spartan** — выброс на 76.3 %. Он
встречается 38 раз, 9 из них — внутри того же ALL-CAPS-safety-шаблона
(`DO NOT USE THE SPARTAN SUPPLIED AIR RESPIRATOR HELMET …`). Word-bounded
protection + verbatim-запись сохраняют Spartan нетронутым в 26 из 29
mixed-case упоминаний, но в all-caps-блоках Marian так агрессивно
переупорядочивает соседние слова, что placeholder оказывается в
неграмматичной позиции; Qwen переписывает предложение; fallback-to-
reference откатывает на и без того сломанный MT-выход. Это отмечено в
wishlist ниже под пунктом «All-caps Marian boilerplate».

### Рейтинг качества по PDF

Медиана LaBSE очень близка по всем 18 PDF (0.90 – 0.92); разброс в
среднем гонит нижний 10 %-хвост длинных all-caps-предупреждений.

**Топ-5 по среднему LaBSE:**

| PDF | Строк | среднее | медиана |
|---|---:|---:|---:|
| `ZVP-PC-0038-01.pdf` | 16 | **0.9200** | 0.9197 |
| `ZVP-PC-0041-01.pdf` | 13 | 0.9148 | 0.9203 |
| `ZVP-PC-0043-01.pdf` | 27 | 0.9120 | 0.9147 |
| `ZVP-PC-0039-01.pdf` | 22 | 0.9110 | 0.9164 |
| `ZVP-PC-0042-01.pdf` | 27 | 0.9044 | 0.9063 |

**Низ-5 по среднему LaBSE:**

| PDF | Строк | среднее | медиана |
|---|---:|---:|---:|
| `ZVP-PC-0100-00.pdf` | 64 | **0.8813** | 0.9058 |
| `ZVP-PC-0086-00.pdf` | 126 | 0.8837 | 0.9040 |
| `ZVP-PC-0071-01.pdf` | 65 | 0.8855 | 0.9036 |
| `ZVP-PC-0111-00.pdf` | 68 | 0.8888 | 0.9201 |
| `ZVP-PC-0072-01.pdf` | 74 | 0.8904 | 0.9046 |

### Галерея лучших и худших переводов

**Лучшие 6 длинных примеров** (COMET-QE ≥ +0.7, источник ≥ 60 симв.).
Один и тот же абзац про компрессор-безопасность встречается в 5 PDF и
каждый раз переводится идентично — через cross-row memo-кэш:

| COMET-QE | LaBSE | EN | ES |
|---:|---:|---|---|
| +0.762 | 0.926 | An overheated compressor, or one that is in poor mechanical condition, may produce carbon monoxide (CO) and objectionable odours. | Un compresor sobrecalentado o uno que esté en malas condiciones mecánicas puede producir monóxido de carbono (CO) y olores desagradables. |
| +0.757 | 0.927 | An overheated compressor, or one that is in poor mechanical condition, may produce carbon monoxide (CO) and objectionable odors. | Un compresor sobrecalentado o uno que esté en malas condiciones mecánicas puede producir monóxido de carbono (CO) y olores desagradables. |
| +0.741 | 0.907 | An overheated compressor, or one that is in poor mechanical condition, may produce carbon monoxide and objectionable odors. | Un compresor sobrecalentado o uno que esté en malas condiciones mecánicas puede producir monóxido de carbono y olores desagradables. |
| +0.709 | 0.892 | overheated compressor, or one that is in poor mechanical condition may produce carbon monoxide (CO) and objectionable odors. | Compresor sobrecalentado o uno en malas condiciones mecánicas puede producir monóxido de carbono (CO) y olores desagradables. |
| +0.704 | 0.921 | The precautions described above also apply to portable compressors. | Las precauciones descritas anteriormente también se aplican a los compresores portátiles. |
| LaBSE 0.982 | — | These instructions cover the installation, operation and maintenance of the PanBlast Fina Abrasive Control Valve. | Estas instrucciones cubren la instalación, el funcionamiento y el mantenimiento de la Válvula dosificadora en formato PanBlast Fina. |

Последняя строка — наглядная работа всего глоссарийного стека:
`PanBlast` (verbatim) + `Abrasive Control Valve → Válvula dosificadora`
оба корректно ложатся, регистр формальный, окружение — естественный
испанский.

**Худшие 6 длинных примеров** (низ COMET-QE после отбрасывания
однотокенных строк):

| COMET-QE | LaBSE | EN | ES |
|---:|---:|---|---|
| −1.237 | 0.912 | NOTE: NEVER LIFT AND/OR CARRY THE SUPPLIED AIR RESPIRATOR HELMET ASSEMBLY BY THE BREATHING TUBE, AS DAMAGE TO THE SUPPLIED AIR RESPIRATOR HELMET OR BREATHING TUBE MAY OCCUR. | NOTA: NUNCA LÍNEA Y/O LLEVAR AL RESPIRATOR DE AÉREO SUMINISTRADO Casco ASAMBLEA POR EL Tubo Respirador, EN CALIDAD DE INFIERNO AL RESPIRATOR DE AÉREO SUMINISTRADO Casco O Tubo Respirador MAYO OCCUR. |
| −1.144 | 0.896 | NOTE: THE INNER LENS PROTECTIVE LAYER UST BE PEELED OFF THE LENS BEFORE FITTING INTO THE INNER WINDOW GASKET. | NOTA: EL Lámina interior PLAYER PROTECTIVO SE PELIGRA DE LAS LENS ANTES DE FITAR EN LA GASE DE LA VENTA INTERNA. |
| −1.111 | 0.943 | Carefully remove the Inner Collar and Outer Cape press studs from the Supplied Air Respirator Helmet shell, and discard the Inner Collar and Outer Cape. | Retire cuidadosamente el Collar Interior y Exterior Capa prensar los tacos del respirador de aire suministrado Casco, y descarte el Collar Interior y Exterior Capa. |
| −1.103 | 0.867 | Do not tuck the Outer Cape into the Supplied Air Respirator Helmet shell interior. | No arrope el interior exterior Capa en el respirador de aire suministrado Casco. |
| −1.075 | 0.836 | DO NOT USE THE SPARTAN SUPPLIED AIR RESPIRATOR HELMET OPERATOR VISION IS IMPAIRED IN ANY WAY DUE TO MISTING OR FOGGING. | NO USAR LA VISIÓN DEL Spartan SUMINISTRADO DE RESPIRATOR DE AÉREO Casco El operador está IMPAIRED EN CUALQUIER MANERA DESDE EL MISTING O EL FOGGING. |
| −1.060 | 0.901 | NOTE: NEVER LIFT AND/OR CARRY THE RESPIRATOR HELMET ASSEMBLY BY THE BREATHING TUBE, AS DAMAGE TO THE RESPIRATOR HELMET OR BREATHING TUBE MAY OCCUR. | NOTA: NUNCA LÍNEA Y/O LLEVA AL RESPIRATOR Casco ASAMBLEA POR LA Tubo Respirador, COMO DAÑO AL RESPIRATOR Casco O Tubo Respirador MAYO OCCUR. |

### Таксономия классов ошибок

Если пройтись по нижним 50 строкам по COMET-QE, четыре причины
объясняют каждую broken (< −1.0)-строку:

1. **Marian на длинных ALL-CAPS-предложениях (~70 % broken-строк).**
   Модель училась на mixed-case-новостях и трактует `LIFT`, `MAY`, `OR`
   как отдельные токены для транслитерации, а не перевода. `LIFT →
   LÍNEA` (линия), `MAY → MAYO` (май-месяц), `OCCUR → OCCUR` (нет
   испанского глосса, копируется). **Путь фикса:** sentence-case
   препроцессинг перед MT, затем UPPER обратно, если источник был в
   верхнем регистре. ~30 строк кода.
2. **OCR-артефакты PDF-экстрактора (~15 %).** Примеры: `UST` вместо
   `MUST`, `IMPAIRED` остался без перевода, `c/w Unres.` → случайная
   аббревиатура. **Путь фикса:** 20-правильный лексический нормализатор
   (`UST → MUST`, `c/w → with`, `Assy → Assembly`, `Unres. →
   Unrestricted`) перед MT.
3. **Цепочки «бренд + испанское существительное», где Qwen переставляет
   слова (~10 %).** «Inner Collar and Outer Cape» становится «el Collar
   Interior y Exterior Capa» — обе половины защищены и восстановлены по
   отдельности, но порядок noun-noun-adjective, который оставляет Qwen,
   выглядит неуклюже. **Путь фикса:** разнести составную `Inner
   Collar-Outer Cape → Capa Titan` на две отдельные записи с явным
   испанским порядком.
4. **Глоссарийный испанский, который аудит метит как «leak» (~5 %).**
   `Pet Cock → Pet Cock` и `Pop Up → Pop Up` — verbatim: target *равен*
   EN-фразе, поэтому english-leak-аудит флагует их, хотя они корректны.
   В таблице видны как `english_leak_rows > 0` при hit rate 88–100 %.
   **Путь фикса:** пропускать english-leak-аудит, когда `source ==
   target` в `evaluate_glossary.py`.

### Производительность пайплайна (хост RTX 3060)

| Этап | Wall-clock |
|---|---:|
| PDF → CSV-экстракция (включая fastText LID) | ~6 с |
| Перевод (полный пайплайн, 1 334 строки, 23 % cache hit) | **34 м 03 с** |
| LaBSE + BERTScore оценка | 25 с |
| COMET-QE оценка (весь корпус, ~1 334 строки) | 16 с |
| Графики | 6 с |

Весь цикл от `.pdf` до полностью заполненного `results_final/` занимает
**~37 минут** end-to-end.

---

## Воспроизведение полного пайплайна с нуля

```powershell
# (один раз) eval-зависимости + загрузка LID и COMET-QE
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-eval.txt
Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin' -OutFile 'models\lid.176.bin'
python scripts\download_comet_qe.py

# 1. Скачать PDF PanBlast (идемпотентно)
python scripts\download_panblast_manuals.py

# 2. Извлечь: CAPS- + Title-Case-фильтры + fastText LID
python scripts\pdfs_to_csv.py

# 3. Перевод (~34 мин на RTX 3060)
python scripts\translate_csv.py --start-fresh

# 4. Оценка (глоссарий + LaBSE + BERTScore + COMET-QE + графики)
python scripts\run_full_eval.py
```

---

## Структура кода

```
mt_mvp/
├── app/
│   ├── main.py                       # FastAPI factory + Gradio mount на /gradio
│   ├── gradio_app.py                 # Gradio демо-UI (также запускается отдельно)
│   ├── core/config.py                # Settings (env-var driven)
│   ├── api/
│   │   ├── routes.py                 # HTTP: cache wrap, health, cache clear
│   │   ├── deps.py                   # Кэшированные singleton-ы: MT, Glossary, Qwen, PostEditor
│   │   └── schemas.py                # Pydantic request/response
│   └── services/
│       ├── translation.py            # Оркестрация перевода
│       ├── translate_cache.py        # LRU/TTL + cache key
│       ├── glossary.py               # protect / enforce / reassert (word-bounded)
│       ├── ct2_engine.py             # CTranslate2 backend
│       ├── mt_engine.py              # HuggingFace Marian (fallback)
│       ├── qwen_postedit.py          # Qwen 2.5 chat-template inference
│       └── postedit.py               # PostEditor (Qwen + reassert + чистка)
├── frontend/                          # Vanilla web UI
├── glossary/   prompts/   models/
├── data/
│   ├── manifest.json
│   ├── <category>/*.pdf
│   └── csv_final/                     # извлечённые EN-сегменты
├── translated_data_final/             # переводы
├── results_final/                     # snapshot оценки + report.md
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
│   ├── run_full_eval.py               # one-shot eval-драйвер
│   ├── compare_versions.py            # generic diff (нужен только если итерируешь)
│   ├── compile_glossary_from_xlsx.py
│   ├── convert_marian_to_ct2.py
│   └── prewarm_hf_cache.py            # прогрев HF-кэша (Marian + Qwen)
├── tests/                             # 37 тестов в 3 файлах
├── Dockerfile                         # multi-target сборка (runtime + runtime-gpu), non-root, healthcheck
├── docker-compose.yml                 # профили api / api-gpu / gradio-demo / prewarm
├── .dockerignore   .env.example   pytest.ini
├── requirements.txt   requirements-gpu.txt   requirements-eval.txt
└── README_ENG.md / README_RUS.md / PROJECT_FILES.md
```

---

## API

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/v1/translate` | Тело: `text`, `apply_glossary`, `apply_postedit`, `include_debug` |
| `GET` | `/api/v1/health` | Статус, GPU, размер кэша, summary пайплайна |
| `POST` | `/api/v1/translate/cache/clear` | Опустошить in-memory-кэш |

После правки `glossary/*.json` или `prompts/*.md` нужно вызвать
`POST /api/v1/translate/cache/clear` или перезапустить Uvicorn, чтобы
кэш отдавал обновлённые переводы.

---

## TODO / wishlist

Классы ошибок, отслеживаемые для будущей итерации:

1. **Marian на all-caps safety-boilerplate** — низ COMET-QE-распределения.
   Два жизнеспособных пути:
   - **Sentence-case препроцессинг**: детектировать «> 50 % заглавных»,
     привести к нижнему регистру с заглавной первой буквой перед MT,
     UPPER обратно на выходе, если источник был в верхнем регистре.
     Маленький патч в `app/services/translation.py`.
   - **Сменить backbone для этих сегментов** на
     `facebook/nllb-200-distilled-600M` — он гораздо устойчивее к
     «крику».
2. **OCR-артефакты** — `UST` вместо `MUST`, `Assy c/w Unres.`,
   слипшийся `SEALING BAND O-RING PRESS STUDS` посреди предложения.
   Короткий лексический нормализатор перед MT (`UST → MUST`, `c/w →
   with`, `Assy → Assembly`, `Unres. → Unrestricted`) подняло бы
   несколько broken-строк в «ok».
3. **Раздел compound-записей глоссария** — единственная 100 %-miss
   запись: `Inner Collar-Outer Cape → Capa Titan`. Разнести на две
   записи с disambiguated-таргетами.
4. **Per-segment register lock** — Qwen иногда дрейфует с формального
   «usted» на неформальное «tú» внутри длинного предложения; ужесточить
   инструкции в `prompts/postedit_en_es.md`.
5. **English-leak audit** — исключить записи, где `source == target`
   (Pet Cock, Pop Up), чтобы они не светились как ложно-положительные
   утечки.
6. **CI** — подключить 37-тестовый pytest-сьют к GitHub Actions, чтобы
   PR-ы доказывали отсутствие регрессии по глоссарию и PDF-фильтрам.
7. **Публичный дашборд дрейфа LaBSE / COMET-QE** — `make_eval_plots.py`
   уже даёт входные данные; небольшая Streamlit-страница поверх
   позволила бы не-инженерам наблюдать дельту между запусками.
