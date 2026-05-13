
# Переводчик EN→ES для дробеструйного оборудования (MVP)

Локальный **EN → ES** для мануалов и каталогов в нише дробеструйной обработки металлов, абразивов и СИЗ: даташиты, SOP, строки каталога. Перевод осуществляется локально, бесплатно, без параллельного корпуса, с использованием глоссария и при необходимости пост-эдитинг с помощью помещающейся в локальное железо LLM.

**Состав:** Marian через **CTranslate2** (дефолт), при желании **NLLB-200**, доменный **JSON-глоссарий** (в т.ч. поле `notes` в JSON только для людей, рантайм читает `source`/`target`), **Qwen 2.5** пост-редактирование, кэш 24 часа с максимумом в 1024 записи, веб-UI + **Gradio**, **Docker**, **GitHub Actions** с pytest.

**Цифры по корпусу PanBlast** (18 PDF превращаем **1 334** сегмента после фильтров): LaBSE **0.901**, BERTScore **0.911**, в отчёте — **COMET-Kiwi wmt22** (~0…1) **0.770**, глоссарий по терминам **≈96.3 %**, полный прогон на RTX 3060 **≈34 мин**. Разбор вглубь лежит в [`results_final/report.md`](results_final/report.md). Карта репозитория: [`PROJECT_FILES.md`](PROJECT_FILES.md). На английском:[`README_ENG.md`](README_ENG.md).

---

## Финальный пайплайн

Корпус PanBlast (18 PDF, ~57 МБ) извлекается в 1 334 английских
сегмента (опрелеляется fast-text детектором), переводится через production-пайплайн и оценивается четырьмя независимыми reference-free метриками:

| Метрика | **Финал v4** (n = 1 334) |
|---|---:|
| LaBSE cosine (среднее ± 95 % ДИ) | **0.901 ± 0.003** |
| BERTScore F1 (среднее) | **0.911** |
| COMET-Kiwi wmt22 (среднее, ~0…1) | **0.770** |
| COMET-Kiwi "poor" строк (< 0.5) | **30 (2.2 %)** |
| COMET-Kiwi "great" строк (≥ 0.85) | **352 (26.4 %)** |
| Hit rate глоссария по терминам | **96.3 %** (1 037 / 1 077) |
| Строк со всеми применёнными терминами | **94.7 %** (518 / 547) |
| Wall-clock полного перевода (RTX 3060) | **34 м 00 с** (23 % cross-row кэша) |
| ALL-CAPS строк, переписанных перед MT | **204 (15.3 %)** |

*В таблице **COMET-Kiwi** — как в [`results_final/report.md`](results_final/report.md) и `quality_summary.json`. Таблица **«Сравнение движков»** ниже по-прежнему использует **COMET-QE wmt20 (z-score)** для честного трёхстороннего сравнения Marian vs NLLB, так как не все переводы были прогнаны через wmt22*

Полный численный разбор - per-term hit rates, per-PDF средние,
корзины COMET (Kiwi), галерея лучших/худших переводов, оставшиеся классы
ошибок - лежит в [`results_final/report.md`](results_final/report.md).

### История итераций

Пайплайн прошёл четыре итерации, каждая версия исправляла ошибки предыдущей. Первая версия
переводила весь мультиязычный PDF-корпус целиком (4 201 строка;
LaBSE 0.887, глоссарий 85 %, **4 ч 41 м** на GPU). Большинство провалов
сводились к трём причинам: слипшиеся ALL-CAPS списки деталей, две трети
строк оказались не на английском, и Qwen-postedit тихо удалял
канонические испанские термины. Вторая итерация добавила CAPS-фильтр
parts-list, inline fastText `lid.176.bin` для определения языка и
диакритико-толерантный re-assert глоссария. Всё это подняло глоссарий до
97.0 % и сократило runtime до 44 м. Третья итерация ужесточила всё это:
Title-Case-детектор parts-list, lowercase-LID с порогом 0.55,
word-bounded `protect_source` (23 verbatim-записи бренда/стандартов),
cross-row memo-кэш для 23 % повторяющихся EN-предложений (COMET-QE
−0.066 → −0.047, broken-строки 26 → 16, runtime → **34 м**). Четвёртая
итерация это **ALL-CAPS preprocessor**, который
переписывает 204 safety-предупреждения в смешанный регистр
перед MT и UPPER-кейс на выходе, плюс **сравнение трёх движков**
(Marian-CT2 vs NLLB-200-distilled 600M vs 1.3B), которое подтвердило:
Marian-CT2 на этом корпусе впереди.

| Метрика | v1 | v2 | v3 | **v4 (финал)** | total Delta |
|---|---:|---:|---:|---:|---:|
| LaBSE среднее | 0.887 | 0.892 | 0.895 | **0.901** | **+0.014** |
| BERTScore F1 | — | 0.910 | 0.911 | **0.911** | — |
| COMET-QE **wmt20** среднее | — | −0.066 | −0.047 | **−0.005** | **+0.061** |
| Broken-строки wmt20 (< −1.0) | — | 26 | 16 | **3** | **−23 (−88 %)** |
| Hit rate глоссария | 85 % | 97.0 % | 96.4 % | **96.3 %** | **+11.3 пп** |
| Wall-clock | 4 ч 41 м | 44 м | 34 м | **34 м** | **8.3×** |

---

## Сравнение движков: почему Marian-CT2 выигрывает у NLLB

Three-way оценка на тех же 1 334 сегментах с идентичным
glossary + ALL-CAPS-препроцесс + Qwen-postedit. Меняется только
MT-бэкенд:

| Движок | LaBSE | BERTScore | COMET w20† | Broken† | Скорость | Скор* |
|---|---:|---:|---:|---:|---:|---:|
| **Marian opus-mt-en-es (CT2)** | **0.9007** | **0.9111** | −0.005 | **3 (0.2 %)** | **0.65 сег/с** | **0.8234** |
| NLLB-200-distilled-600M (HF) | 0.8962 | 0.9106 | +0.008 | 12 (0.9 %) | 0.55 сег/с | 0.8217 |
| NLLB-200-distilled-1.3B (HF) | 0.8920 | 0.9089 | **+0.020** | 8 (0.6 %) | 0.50 сег/с | 0.8197 |

† **wmt20** z-score; верхняя таблица и отчёт — **Kiwi wmt22**, так как 3 переведенных сета были прогнаны только через wmt20.

\* *Скор = 0.5·LaBSE + 0.3·BERTScore + 0.2·σ(COMET-QE).*

**Интерпретация.** NLLB в среднем *более беглый* (более высокий **wmt20** COMET-QE
z-score означает, что результат читается ближе к общему
испанскому), но дрейфует дальше от семантики оригинала (ниже
LaBSE / BERTScore) и даёт в 2–4 раза больше катастрофических провалов
(по шкале **wmt20** "broken").
Для технической документации по безопасности, где буквальная точность
важнее беглости, выигрывает Marian. NLLB-1.3B *не доминирует* над
NLLB-600M — большая модель парафразирует агрессивнее, что ухудшает
семантическое сходство с (часто лаконичным) оригиналом. Глоссарий
работает на любом движке одинаково, поэтому hit rate почти не
различается.

**Решение: Marian-CT2 остаётся дефолтом.** NLLB-200 всё ещё можно
включить через `MT_MVP_MT_ENGINE=nllb` (см. § Конфигурация).

**Повторить сравнение.** В таблице выше **Marian** — это уже текущий снимок в
`results_final/` (после `run_full_eval.py` на `translated_data_final`). В
командах ниже вы **докачиваете только альтернативный движок** и сводите его с
Marian через `sweep_summary.py`.

Минимум (Marian из `results_final` vs один NLLB‑600M):

```powershell
python scripts\translate_csv.py --start-fresh --engine nllb --mt-model facebook/nllb-200-distilled-600M --output-dir translated_data_nllb600m
python scripts\run_full_eval.py --input translated_data_nllb600m\all_segments.csv --out-dir results_sweep\nllb600m
python scripts\sweep_summary.py --label marian --dir results_final --label nllb600m --dir results_sweep\nllb600m --out results_sweep\sweep_summary.md
```

Полное **трёхстороннее** как в таблице (добавить NLLB‑1.3B и передать **три**
`--label` / `--dir`):

```powershell
python scripts\translate_csv.py --start-fresh --engine nllb --mt-model facebook/nllb-200-distilled-1.3B --output-dir translated_data_nllb13b
python scripts\run_full_eval.py --input translated_data_nllb13b\all_segments.csv --out-dir results_sweep\nllb13b
python scripts\sweep_summary.py --label marian --dir results_final --label nllb600m --dir results_sweep\nllb600m --label nllb13b --dir results_sweep\nllb13b --out results_sweep\sweep_summary.md
```

(Каталог задаётся в **`run_full_eval.py --out-dir …`** — тот же путь
передаётся в **`sweep_summary.py --dir …`**. Создавать папку вручную обычно
не нужно: скрипты создадут путь при записи.)

---

## Возможности

- **API** (FastAPI): перевод, health, кэш; опционально **Gradio** на `/gradio` (`MT_MVP_ENABLE_GRADIO=1`) или отдельно `python -m app.gradio_app --share` для ссылки `*.gradio.live`.
- **Перевод:** глоссарий (защита терминов → MT → восстановление), **ALL-CAPS** для длинных предупреждений (sentence case перед MT, верхний регистр в конце), пост-редактирование **Qwen 2.5** с повторной подтяжкой терминов из глоссария.
- **Движки:** Marian **CTranslate2** по умолчанию; **Marian HF** и **NLLB-200** — через настройки.
- **Кэш** ответов (LRU), **Docker** (`docker compose`, профиль `gpu` при NVIDIA), **CI** — pytest на push/PR (Python 3.10 и 3.11).
- **Скрипты:** PDF → CSV → пакетный перевод → оценка; one-shot — `run_full_eval.py`, сравнение движков — `sweep_summary.py`.
- **Hugging Face:** для gated-моделей (в т.ч. **COMET-Kiwi**) нужно положить токен в **`.env`** как `HUGGINGFACE_HUB_TOKEN=...` или залогиниться в `huggingface-cli login`. Корневой `.env` подхватывается скриптами через `app/hf_env.py` (секреты в git не коммитим — см. `.env.example`).

---

## Как работает пайплайн

Для **`POST /api/v1/translate`** при cache miss:

1. **Lookup в кэше** (обходится, если `include_debug=true`)
2. **ALL-CAPS препроцесс** — детектируем «≥ 60 % верхнего регистра» и
   переводим в Sentence case, чтобы MT увидел смешанный регистр
3. **Защита глоссарием** — самый длинный word-bounded EN-матч →
   placeholder `__GLS{i}__`
4. **Машинный перевод** — Marian через CTranslate2 (либо PyTorch Marian,
   либо NLLB-200; см. `MT_MVP_MT_ENGINE`)
5. **Восстановление глоссария** — placeholder → канонический испанский
   (regex толерантен к MT-обрезанию `__GLS{i}__` до `__GLS{i}_`)
6. **Post-edit** — Qwen 2.5 Instruct полирует беглость, затем
   **reassert** глоссария по accent-/case-insensitive-матчу, fallback
   на pre-Qwen MT-выход, если Qwen удалил канонический термин, затем
   правка английских утечек и пробелов
7. **ALL-CAPS постпроцесс** — если исходник был в верхнем регистре,
   результат тоже переводим в UPPER

`GET /api/v1/health` — подсказки по CUDA, MT-движок, размер кэша, краткое
описание пайплайна.
`POST /api/v1/translate/cache/clear` — очистить in-memory кэш.

---

## Требования

- **Python 3.11+**
- **PyTorch ≥ 2.6** (CPU или CUDA; `transformers` 4.46+ отказывается
  грузить `.bin`-чекпоинты на старом torch из-за `CVE-2025-32434`)
- Опционально: NVIDIA GPU для MT + Qwen
- Опционально: Docker 24+ для контейнеризованного пути

---

## Установка (host)

```powershell
cd mt_mvp
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-gpu.txt   # CUDA-wheel-ы (опционально если есть видеокарта)
python -m pip install -r requirements-eval.txt  # нужно только для оценки
```

**Конвертировать Marian в CTranslate2** (создаёт
`models/opus-mt-en-es-ct2/model.bin`):

```powershell
python scripts\convert_marian_to_ct2.py --model Helsinki-NLP/opus-mt-en-es --output-dir models\opus-mt-en-es-ct2 --force
```

**Пересобрать глоссарий из Excel**:

```powershell
python scripts\compile_glossary_from_xlsx.py
```

---

## Установка Docker

```bash
# дефолтный профиль: API + Gradio демо на http://localhost:8000
# (CPU-образ, Dockerfile target `runtime`, slim Python base ~1.5 ГБ)
docker compose up --build

# GPU-профиль: собирает target `runtime-gpu` (nvidia/cuda:12.4.1 base +
# cu124 PyTorch wheels, ~6 ГБ образ). На хосте нужен NVIDIA Container
# Toolkit и драйвер CUDA 12.x.
docker compose --profile gpu up --build

# Публичная клиентская демо: на старте печатает https://*.gradio.live
docker compose --profile demo up --build

# Опциональный one-shot: предзагрузить Marian + Qwen (~5.5 ГБ) в общий
# `hf_cache`-том, чтобы первый /translate-запрос был мгновенным, а не
# подвисал на скачивании. Запускается один раз после первой сборки.
docker compose --profile prewarm run --rm prewarm
```

В Dockerfile четыре target-а: `base` / `base-gpu` ставят venv (CPU torch
или cu124 torch соответственно), `runtime` / `runtime-gpu` копируют
этот venv плюс исходники приложения, переходят на UID 10001 и
экспонируют `:8000`. Модели **не** запекаются в образ, их нужно монтировать в
рантайме через `./models:/app/models`. Если `model.bin` отсутствует, API
автоматически откатывается на HuggingFace Marian (см. `app/api/deps.py`).

Переменные `MT_MVP_*` (полный список — в `app/core/config.py` и
`.env.example`) перекрывают любые дефолты; самые полезные —
`MT_MVP_POSTEDIT_USE_QWEN=false` для машин с малой памятью и
`MT_MVP_MT_ENGINE=nllb` для смены MT-бэкбона.

### Проверка контейнера

После `docker compose up -d --build` пройдитесь по чек-листу. Команды —
для PowerShell на Windows (на Linux/macOS замените `Invoke-RestMethod`
на `curl`).

**1. Статус и healthcheck**

```powershell
docker compose ps                       # ожидаем: api  Up X seconds (healthy)
Invoke-RestMethod http://localhost:8000/api/v1/health | Format-List
```

`/health` возвращает `status=ok`, текущий MT-движок, подсказку по CUDA и
размер LRU-кэша.

**2. Первый перевод (cold path)**

```powershell
$body = @{ text = "Always wear the Helmet before operating the remote control valve." } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/v1/translate `
  -Method Post -ContentType application/json -Body $body
```

Ожидаемый JSON: `translation = "Siempre use el casco antes de operar la
válvula neumática."`, `glossary_applied = True`, `from_cache = False`.
Первый запрос длится 5–30 с, потому что веса Marian подгружаются в
`hf_cache`; последующие переводы мгновенные.

**3. Hit LRU-кэша (warm path)**

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

Запрос #1 → `from_cache: False`; запрос #2 → `from_cache: True` и
≈10× быстрее.

**4. ALL-CAPS препроцесс в деле**

```powershell
$body = @{ text = "NOTE: NEVER LIFT AND/OR CARRY THE HELMET ASSEMBLY BY THE BREATHING TUBE, AS DAMAGE MAY OCCUR." ; include_debug = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/api/v1/translate -Method Post -ContentType application/json -Body $body | Select-Object translation, debug
```

Проверьте `debug.allcaps_source_detected = True`,
`debug.source_after_allcaps_preprocess` и UPPER-выход. До v4-фикса
Marian переводил `LIFT → LÍNEA` и `MAY → MAYO`.

**5. Свидетельства CT2 → Marian fallback**

```powershell
docker compose logs api --tail 40 | Select-String "CTranslate2|falling back"
```

Должно быть видно предупреждение из `app/api/deps.py`, когда контейнер
стартует без смонтированных CT2-весов.

**6. UI-эндпоинты**

Открыть в браузере:

| URL | Что отдаёт |
|---|---|
| <http://localhost:8000/> | Вёрстка веб-UI (`frontend/index.html`) |
| <http://localhost:8000/gradio/> | Встроенная Gradio демо |
| <http://localhost:8000/docs> | Swagger / OpenAPI |

**7. Инспекция и остановка**

```powershell
docker compose exec api bash             # зайти внутрь контейнера
docker compose restart api               # перезапуск без потери hf_cache
docker compose down                      # стоп + удаление (hf_cache живёт)
docker compose down -v                   # полный wipe, включая hf_cache
```

> **Про Qwen.** Дефолтный контейнер запускается с включённым Qwen 2.5
> post-edit, поэтому самый первый `/translate` дополнительно скачает
> ~5 ГБ Qwen-весов (несколько минут на CPU). Чтобы пропустить это в
> smoke-тесте, перезапустите с `MT_MVP_POSTEDIT_USE_QWEN=false`. Чтобы
> сохранить полный пайплайн без cold-ожидания — выполните один раз
> `docker compose --profile prewarm run --rm prewarm`.

---

## Тесты и CI

**53** теста pytest (глоссарий, ALL-CAPS, фильтры PDF, API со **заглушкой MT** — реальные веса в CI не качаются). Локально: `pytest -q`. На GitHub — workflow `.github/workflows/tests.yml` (Python 3.10 и 3.11). Детали по файлам — в [`PROJECT_FILES.md`](PROJECT_FILES.md).

---

## Конфигурация (`MT_MVP_*` и опционально `.env`)

Переменные приложения — с префиксом **`MT_MVP_`** (см. таблицу ниже). Файл **`.env`** в корне **не обязателен**: дефолты заданы в `app/core/config.py`. Скопировать **`.env.example` → `.env`**, если нужны переопределения.

Для **Hugging Face** (gated COMET-Kiwi и др.) в **тот же** `.env` можно добавить `HUGGINGFACE_HUB_TOKEN` — его подхватывают eval-скрипты через `app/hf_env.py` (не коммитьте токен).

| Переменная | Назначение |
|------------|------------|
| `MT_MVP_MT_ENGINE` | `ctranslate2` (дефолт), `marian_hf` или `nllb` |
| `MT_MVP_MT_MODEL_NAME` | HF id токенайзера / весов модели (`Helsinki-NLP/opus-mt-en-es`, `facebook/nllb-200-distilled-600M`, …) |
| `MT_MVP_CT2_MODEL_DIR` | Каталог с `model.bin` |
| `MT_MVP_CT2_COMPUTE_TYPE` | `int8`, `float16`, `default` и т. п. |
| `MT_MVP_NLLB_SRC_LANG` | Flores-200-код источника, дефолт `eng_Latn` |
| `MT_MVP_NLLB_TGT_LANG` | Flores-200-код таргета, дефолт `spa_Latn` |
| `MT_MVP_NLLB_NUM_BEAMS` | Ширина beam search в NLLB, дефолт 4 |
| `MT_MVP_NLLB_DTYPE` | `auto` (bf16 на CUDA, fp32 на CPU), `fp16`, `bf16`, `fp32` |
| `MT_MVP_DEVICE` | `cuda`, `cpu` или не задано (auto) |
| `MT_MVP_POSTEDIT_USE_QWEN` | `true` / `false` |
| `MT_MVP_POSTEDIT_QWEN_MODEL` | например `Qwen/Qwen2.5-3B-Instruct` |
| `MT_MVP_POSTEDIT_MAX_NEW_TOKENS` | Лимит декода Qwen |
| `MT_MVP_POSTEDIT_QWEN_MAX_INPUT_TOKENS` | Обрезание промпта |
| `MT_MVP_GLOSSARY_PATH` | Путь к JSON-глоссарию |
| `MT_MVP_POSTEDIT_PROMPT_PATH` | Путь к markdown-промпту post-edit |
| `MT_MVP_ENABLE_GRADIO` | `1` монтирует Gradio UI на `/gradio` (Docker ставит это сам) |

---

## Запуск

Три полностью независимых режима, можно выбрать любой. Они не делят порты,
env-переменные и процессы, поэтому можно запускать параллельно.

### Режим A — FastAPI-сервис (порт 8000)

Production UI + REST API.

```powershell
.\venv\Scripts\Activate.ps1
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- Веб-UI: <http://127.0.0.1:8000/>
- Swagger: <http://127.0.0.1:8000/docs>

Чтобы **также** примонтировать Gradio внутри того же FastAPI-процесса:

```powershell
$env:MT_MVP_ENABLE_GRADIO = "1"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Либо добавь в корневой **`.env`** строку `MT_MVP_ENABLE_GRADIO=1` (читается так же, как остальные `MT_MVP_*`) и снова запусти uvicorn.

- Gradio (встроенный): <http://127.0.0.1:8000/gradio>

### Режим B — Standalone Gradio с публичной ссылкой (порт 7861)

Для быстрой клиентской демо: Gradio стартует напрямую (без FastAPI),
открывает временный тоннель `https://*.gradio.live`, который может
открыть кто угодно.

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
задать через `--port`; дефолтный `7860` часто занят другими dev-серверами.
Этот режим **не** читает `MT_MVP_ENABLE_GRADIO` — переменная важна только
для Режима A.

### Режим C — Docker

```bash
# CPU API + встроенная Gradio на http://localhost:8000
docker compose up --build

# То же на GPU (на хосте нужен NVIDIA Container Toolkit + драйвер CUDA 12.x).
# Собирает target runtime-gpu с cu124 PyTorch-wheel-ами.
docker compose --profile gpu up --build

# Standalone Gradio с --share на порту 7860 (отдельный контейнер)
docker compose --profile demo up --build

# One-shot: прогрев hf_cache-тома весами Marian + Qwen, чтобы первый
# /translate-запрос не висел на 5.5-ГБ-скачивании. Запускается один раз
# после первой сборки.
docker compose --profile prewarm run --rm prewarm
```

Чтобы убедиться, что контейнер живой и переводит end-to-end — пройдите
чек-лист [§ Проверка контейнера](#проверка-контейнера) (health,
sample-translate, cache-hit, ALL-CAPS evidence, fallback-предупреждение).

---

## Корпус PanBlast: скачать → извлечь → перевести

В дополнение к HTTP-сервису проект ходит со batch-пайплайном, который
строит domain-корпус из технических PDF-мануалов PanBlast и прогоняет его
через тот же движок, что и API.

### 1. Скачивание PDF — `scripts/download_panblast_manuals.py`

- 18 PDF (~57 МБ) в 5 категориях, скачиваются с
  <https://www.panblast.com/manuals.acv>.
- Идемпотентно: существующие файлы пропускаются.
- Пишет `data/manifest.json` с маппингом product code → PDF.

```powershell
python scripts\download_panblast_manuals.py
```

### 2. Извлечение PDF → CSV — `scripts/pdfs_to_csv.py`

Пайплайн (на каждый PDF):

- **PyMuPDF** извлекает текст (лучший reading order на технических
  layout-ах).
- Снимаются running-заголовки / footer-ы (строки, встречающиеся на
  ≥ 50 % страниц).
- Дроп шума: номера страниц, голые part numbers (`ZVP-PC-0027-01`),
  чисто-числовые строки, строки с < 3 буквами.
- **Дроп CAPS parts-list dumps**: длинные, all-uppercase слипшиеся
  noun-phrase-ы без инструктивного глагола и без внутренней пунктуации.
- **Дроп Title-Case parts-list dumps**: длинные, доминирующе-Title-Case
  слипшиеся сегменты с ≥ 2 каталожными ключевыми словами (PARTS, LIST,
  STOCK, CODE, DESCRIPTION, ITEM, ASSEMBLY, EXPLODED, …) без
  инструктивного глагола.
- Сшивка PDF-перенесённых строк в параграфы → split предложений по `.!?`.
- Лимит длины сегмента **800 символов** (OPUS-MT cap 512 токенов).
- **fastText `lid.176.bin`** language-filter на **lowercased** сегменте;
  отбрасываем, если `lid != "en"` ИЛИ confidence < `--lang-min-confidence`
  (дефолт **0.55**).
- Стабильный SHA-1 id строки на основе
  `category|pdf|page|segment_idx|source_en`.

Выход: **1 334 сегмента / ~184 К символов исходника** в `data/csv_final/`.

```powershell
python scripts\pdfs_to_csv.py
python scripts\pdfs_to_csv.py --lang-min-confidence 0.40
python scripts\pdfs_to_csv.py --no-lang-filter            # сохранить любые языки
```

### 3. Batch-перевод — `scripts/translate_csv.py`

Идентичный `POST /api/v1/translate` пайплайн (включая ALL-CAPS
препроцесс + UPPER постпроцесс):
`allcaps sentence-case → glossary protect → CT2 Marian → glossary
restore → Qwen 2.5-3B post-edit → glossary re-assert → allcaps UPPER →
правки пробелов`.

- **GPU** выбирается автоматически (Qwen bf16 + CTranslate2 Marian
  ≈ 7.7 ГБ VRAM на RTX 3060).
- **Resumable**: непустые `target_es` повторно используются; атомарные
  flush-ы каждые 25 строк переживают Ctrl-C / выключение питания.
- **Cross-row memo-кэш**: дубликатные EN-строки делят одну ES-переводку.
  **23.3 %** корпуса — это дубликатная boilerplate, кэш экономит ~10
  минут на полный прогон.
- **Переключение движка**:
  `--engine nllb --mt-model facebook/nllb-200-distilled-600M` (или любой
  HF id) — этот же скрипт используется в model sweep для
  `results_final/report.md`.

```powershell
python scripts\translate_csv.py                  # полный пайплайн, Marian-CT2 по дефолту
python scripts\translate_csv.py --no-postedit    # MT-only быстрый прогон
python scripts\translate_csv.py --limit 50       # smoke-тест
python scripts\translate_csv.py --start-fresh    # игнорировать ранее переведённые строки
python scripts\translate_csv.py --engine nllb --mt-model facebook/nllb-200-distilled-600M --output-dir translated_data_nllb600m --start-fresh
```

Выходная CSV: `id, category, pdf, page, segment_idx, source_en, target_es,
char_count, glossary_applied, postedit_applied, error`.

### 4. Где что лежит

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

resulting_files/                            # клиентский deliverable (2 колонки)
├── all_segments.csv
└── <category>/<pdf_stem>.csv               # только source_en, target_es

results_final/                              # снапшот эвалуэйшна
├── glossary/                               # per-term и per-PDF статистика
├── quality/                                # LaBSE, BERTScore, COMET (Kiwi в quality_scores; опционально wmt20 в comet_qe_worst)
├── plots/                                  # 9 PNG-ов
└── report.md
```

### 5. Клиентский deliverable — `resulting_files/`

Удобный для review-инга вид победившего перевода: per-PDF CSV с двумя
колонками `source_en` и `target_es`, плюс сводный `all_segments.csv`.
Регенерация в любой момент:

```powershell
python scripts\build_resulting_files.py
python scripts\build_resulting_files.py --input translated_data_nllb600m\all_segments.csv --output resulting_files_nllb600m
```

---

## Оценки качества

Нужны пакеты из `requirements-eval.txt`. Полный прогон одной командой (пишет в `results_final/`):

```powershell
python scripts\run_full_eval.py
```

**По шагам:**

1. **`evaluate_glossary.py`** — насколько термины глоссария попали в испанский и не «протек» ли английский.
2. **`evaluate_quality.py`** — **LaBSE** (смысл EN↔ES), **BERTScore** (близость к формулировке источника), **COMET**:
   - по умолчанию **авто**: если есть токен HF — **COMET-Kiwi** `wmt22` (шкала порядка **0…1**); иначе **COMET-QE** `wmt20` (**z-score**). Если авто-Kiwi не завёлся — один раз пробуем **wmt20**;
   - явно: `--comet-model Unbabel/wmt22-cometkiwi-da` или `.../wmt20-comet-qe-da`;
   - веса: `python scripts\download_comet_qe.py` (wmt20 по умолчанию) или с тем же скриптом **`--model Unbabel/wmt22-cometkiwi-da`** для Kiwi (нужен доступ к gated-репо на HF);
   - в CSV колонка называется **`comet_qe`**: при Kiwi там числа **Kiwi**, при wmt20 — **z-score** (не смешивайте без подписи).
3. **`run_comet_on_worst.py`** — **wmt20** по выборке «худших» по LaBSE (отдельный CSV; z-score). В [`results_final/report.md`](results_final/report.md) **основной** корпусный COMET — **Kiwi** из шага 2.
4. **`make_eval_plots.py`** — PNG для отчёта.

Опционально **`sweep_summary.py`** — сводная таблица по нескольким прогонам движков (как в блоке «Сравнение движков» выше).

---

## Детальная аналитика

Всё в цифрах и картинках — в [`results_final/report.md`](results_final/report.md); он обновляется вместе с `run_full_eval.py`. Этот README держит только верхнеуровневые цифры, чтобы не расходиться с отчётом.

---

## Воспроизвести пайплайн с нуля

```powershell
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-eval.txt
Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin' -OutFile 'models\lid.176.bin'
python scripts\download_comet_qe.py
# Опционально COMET-Kiwi (gated): токен в .env → затем
# python scripts\download_comet_qe.py --model Unbabel/wmt22-cometkiwi-da

python scripts\download_panblast_manuals.py
python scripts\pdfs_to_csv.py
python scripts\translate_csv.py --start-fresh
python scripts\run_full_eval.py
python scripts\build_resulting_files.py
```

---

## Структура кода

```
mt_mvp/
├── .github/workflows/tests.yml            # CI: pytest на push + PR, матрица Py 3.10/3.11
├── app/
│   ├── main.py                            # FastAPI factory + Gradio-mount на /gradio
│   ├── gradio_app.py                      # Gradio демо UI (тоже запускается standalone)
│   ├── core/config.py                     # Settings (env)
│   ├── hf_env.py                          # подгрузка корневого .env (HF-токен и др.)
│   ├── api/
│   │   ├── routes.py                      # HTTP: cache-обвязка, health, cache-clear
│   │   ├── deps.py                        # Cached singletons: MT, Glossary, Qwen, PostEditor
│   │   └── schemas.py                     # Pydantic request/response
│   └── services/
│       ├── translation.py                 # Оркестрация translate-а (allcaps + glossary + MT + postedit)
│       ├── text_case.py                   # ALL-CAPS детектор + sentence-case препроцессор (v4)
│       ├── translate_cache.py             # LRU/TTL + cache-key
│       ├── glossary.py                    # protect / enforce / reassert (word-bounded)
│       ├── ct2_engine.py                  # CTranslate2 Marian (дефолт)
│       ├── mt_engine.py                   # HuggingFace Marian (fallback)
│       ├── nllb_engine.py                 # HuggingFace NLLB-200 (альтернатива, v4)
│       ├── qwen_postedit.py               # Qwen 2.5 chat-template inference
│       └── postedit.py                    # PostEditor (Qwen + glossary-reassert + cleanup)
├── frontend/                              # Веб-UI
├── glossary/   prompts/   models/
├── data/
│   ├── manifest.json
│   ├── <category>/*.pdf
│   └── csv_final/                         # извлечённые EN-сегменты
├── translated_data_final/                 # выигравшие Marian-CT2 + Qwen + ALL-CAPS переводы
├── resulting_files/                       # клиентский deliverable (2-колоночные CSV)
├── results_final/                         # снапшот эвалуэйшна + report.md
├── scripts/
│   ├── download_panblast_manuals.py
│   ├── pdfs_to_csv.py
│   ├── translate_csv.py                   # batch-перевод (поддерживает --engine nllb)
│   ├── build_resulting_files.py           # per-PDF 2-колоночные CSV (v4)
│   ├── filter_english.py
│   ├── evaluate_glossary.py
│   ├── evaluate_quality.py
│   ├── download_comet_qe.py
│   ├── run_comet_on_worst.py
│   ├── make_eval_plots.py
│   ├── run_full_eval.py                   # one-shot eval-драйвер
│   ├── sweep_summary.py                   # N-way ранжирование движков (v4)
│   ├── compare_versions.py                # generic 2-way diff
│   ├── compile_glossary_from_xlsx.py
│   ├── convert_marian_to_ct2.py
│   ├── prefetch_nllb.py                   # выкачать NLLB-веса в HF-кэш (v4)
│   └── prewarm_hf_cache.py                # прогреть HF-кэш (Marian + Qwen)
├── tests/                                 # 53 теста по 4 файлам (incl. text_case)
├── Dockerfile                             # multi-target build (runtime + runtime-gpu), non-root, healthcheck
├── docker-compose.yml                     # api / api-gpu / gradio-demo / prewarm-профили
├── .dockerignore   .env.example   pytest.ini
├── requirements.txt   requirements-gpu.txt   requirements-eval.txt
└── README_ENG.md / README_RUS.md / PROJECT_FILES.md
```

---

## API

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/v1/translate` | Тело: `text`, `apply_glossary`, `apply_postedit`, `include_debug` |
| `GET` | `/api/v1/health` | Статус, GPU, размер кэша, краткое описание пайплайна |
| `POST` | `/api/v1/translate/cache/clear` | Очистить все записи in-memory кэша |

После редактирования `glossary/*.json` или `prompts/*.md` вызовите
`POST /api/v1/translate/cache/clear` или рестартуйте Uvicorn, чтобы
кэшированные переводы обновились.

Когда `include_debug=true`, ответ включает
`debug.allcaps_source_detected`, `debug.source_after_allcaps_preprocess`,
`debug.stages_executed_this_request` плюс pre/post-MT-строки — полезно
для сравнения движков или ловли пропуска глоссария.

---

## TODO / wishlist

1) Протестировать другие пост-эдитинг модели
2) Провести умное распознавание, чтобы таблицы тоже пеерводились. Сейчас переводим только тексты, чтобы быстро читались и копировались. 
3) Расширить глоссарий.
