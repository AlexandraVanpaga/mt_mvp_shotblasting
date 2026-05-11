# Переводчик EN→ES для дробеструйного оборудования (MVP)

Стек машинного перевода **английский → испанский** для технической документации в области **дробеструйного оборудования, абразивной обработки и СИЗ оператора** (паспорта изделий, СОПы, каталожные описания). Объединяет **CTranslate2 Marian** (или опциональный **PyTorch Marian**), **JSON-глоссарий**, **пост-редактирование через Qwen2.5**, **LRU/TTL-кэш ответов** и **тематический веб-интерфейс** с **OpenAPI (Swagger)**.

Подробное описание каждого файла репозитория — в **[PROJECT_FILES.md](PROJECT_FILES.md)**.

---

## Возможности

- **FastAPI + Uvicorn** REST API
- **МТ с поддержкой глоссария**: защита английских терминов → перевод → восстановление испанских эквивалентов; пост-редактирование **переутверждает** эти термины, чтобы LLM не подменял каталожные формулировки
- **Бэкенды**: CTranslate2 (по умолчанию) или Hugging Face Marian
- **Пост-редактирование**: опциональный **Qwen2.5 Instruct** (по умолчанию 3B + ограничения генерации для снижения latency)
- **Кэш**: in-memory LRU (**1024** записи), TTL **24ч**; `include_debug: true` обходит кэш
- **Фронтенд**: тематическая страница в стиле «blast-room» на `/` с примерами и ссылкой на `/docs`
- **Утилиты**: `raw_glossary.xlsx` → `glossary/en_es_shotblasting.json`; скрипт конвертации Marian → CT2

---

## Как работает пайплайн

Для **`POST /api/v1/translate`** когда ответ **не найден в кэше**:

1. **Проверка кэша** (пропускается если `include_debug: true`)
2. **Глоссарная защита** — английские фразы по принципу наибольшего совпадения → плейсхолдеры `__GLS0__`
3. **Машинный перевод** — Marian EN→ES
4. **Восстановление глоссария** — плейсхолдеры → канонический испанский (regex допускает отсутствие пробелов вокруг токенов)
5. **Пост-редактирование** — Qwen (опционально) → **переутверждение** глоссарных терминов из pre-Qwen испанского → исправление английских утечек → пробелы вокруг многословных терминов

`GET /api/v1/health` — статус CUDA, режим МТ, размер кэша, сводка пайплайна.

`POST /api/v1/translate/cache/clear` — очистка кэша переводов

---

## Скорость и качество

| Слой | Решение |
|---|---|
| МТ | **CTranslate2** — быстрый инференс `model.bin` |
| GPU | Автовыбор CUDA; Qwen закреплён через `device_map={"": 0}` |
| Пост-редактирование | **3B** Qwen, **256** max new tokens, truncation **2048** токенов |
| Качество | Глоссарный пайплайн + **переутверждение** после Qwen + исправление пробелов |
| Повторные запросы | **LRU + TTL** кэш возвращает готовый ответ без вызова МТ/Qwen |

Отключить Qwen полностью: `MT_MVP_POSTEDIT_USE_QWEN=false`.

---

## Требования

- **Python 3.11+**
- **PyTorch** (CPU или CUDA — см. `requirements-gpu.txt` для cu124)
- Опционально: **NVIDIA GPU** для МТ + Qwen

---

## Установка

```bash
git clone https://github.com/AlexandraVanpaga/mt_mvp_shotblasting.git
cd mt_mvp_shotblasting

python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# Linux / Mac
source .venv/bin/activate

python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-gpu.txt   # при использовании CUDA
```

**Конвертация Marian в CTranslate2** (создаёт `models/opus-mt-en-es-ct2/model.bin`):

```bash
python scripts/convert_marian_to_ct2.py \
  --model Helsinki-NLP/opus-mt-en-es \
  --output-dir models/opus-mt-en-es-ct2 \
  --force
```

> Используйте актуальную версию `transformers` (см. `requirements.txt`; версия 5.x устраняет ряд проблем с конвертером Marian).

**Пересборка глоссария из Excel**:

```bash
python scripts/compile_glossary_from_xlsx.py
```

---

## Тесты (`pytest`)

Настройка: **`pytest.ini`** (`pythonpath = .`, `testpaths = tests`). Пакет **`pytest`** указан в **`requirements.txt`**.

| Набор | Файл | Что проверяется |
|-------|------|-----------------|
| **Глоссарий (юнит)** | `tests/test_glossary_protect_restore.py` | Методы `Glossary.protect_source` и `Glossary.enforce_placeholders`: подстановка плейсхолдеров, регистронезависимость, приоритет **более длинной** англ. фразы, восстановление при «склеенных» `__GLS*__` без пробелов, неизвестные плейсхолдеры не ломают строку, нормализация пробелов, сквозной сценарий с двумя терминами **без** реального МТ |
| **API (интеграция)** | `tests/test_api_integration.py` | HTTP-клиент **`TestClient`** к реальным маршрутам FastAPI с **подменённым МТ** (`FakeMTEngine` возвращает защищённую строку как есть): полный цикл перевода с флагами глоссария и пост-редакта, отключение глоссария/пост-редакта, **второй идентичный запрос** из **LRU-кэша** (`from_cache`), **`include_debug: true`** обходит кэш и отдаёт **`debug.protected_source`**, **422** при пустом `text`, **`GET /api/v1/health`**, **`POST /api/v1/translate/cache/clear`** |
| **Фикстуры** | `tests/conftest.py` | Временные `glossary.json` и промпт пост-редактора, `Settings(..., _env_file=None)`, `dependency_overrides` для настроек/МТ/глоссария/пост-редактора, `postedit_use_qwen=false`, очистка кэша переводов до/после API-тестов |

**Команды** (из корня репозитория; при необходимости активируйте виртуальное окружение):

```bash
pytest
pytest -v
pytest tests/test_glossary_protect_restore.py -v
pytest tests/test_api_integration.py -v
```

На Windows в PowerShell команды те же.

**Пример вывода** (16 тестов; платформа и время зависят от машины):

```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Users\Alexandra\Desktop\mt_mvp
configfile: pytest.ini
testpaths: tests
collected 16 items

tests\test_api_integration.py .......
tests\test_glossary_protect_restore.py .........

============================= 16 passed in 0.29s ==============================
```

---

## Конфигурация (`MT_MVP_*` или `.env`)

| Переменная | Назначение |
|---|---|
| `MT_MVP_MT_ENGINE` | `ctranslate2` (по умолчанию) или `marian_hf` |
| `MT_MVP_MT_MODEL_NAME` | HF id для токенизатора / весов Marian |
| `MT_MVP_CT2_MODEL_DIR` | Папка с `model.bin` |
| `MT_MVP_CT2_COMPUTE_TYPE` | Например `int8`, `float16`, `default` |
| `MT_MVP_DEVICE` | `cuda`, `cpu` или не задано (авто) |
| `MT_MVP_POSTEDIT_USE_QWEN` | `true` / `false` |
| `MT_MVP_POSTEDIT_QWEN_MODEL` | Например `Qwen/Qwen2.5-3B-Instruct` |
| `MT_MVP_POSTEDIT_MAX_NEW_TOKENS` | Лимит токенов генерации Qwen |
| `MT_MVP_POSTEDIT_QWEN_MAX_INPUT_TOKENS` | Обрезка промпта |
| `MT_MVP_GLOSSARY_PATH` | Путь к JSON-глоссарию |
| `MT_MVP_POSTEDIT_PROMPT_PATH` | Путь к markdown-инструкции пост-редактора |

---

## Запуск

```bash
# Windows
.\.venv\Scripts\Activate.ps1

# Linux / Mac
source .venv/bin/activate

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- **UI**: http://127.0.0.1:8000/
- **Swagger**: http://127.0.0.1:8000/docs

---

## Структура проекта

```
mt_mvp_shotblasting/
├── app/
│   ├── main.py                 # FastAPI: настройка, CORS, статика
│   ├── core/config.py          # Настройки через Pydantic BaseSettings
│   ├── api/
│   │   ├── routes.py           # HTTP: кэш, health, очистка кэша
│   │   ├── deps.py             # Синглтоны: MT-движок, глоссарий, Qwen, PostEditor
│   │   └── schemas.py          # Pydantic-схемы запросов и ответов
│   └── services/
│       ├── translation.py      # Основной пайплайн перевода (глоссарий + МТ + пост-ред.)
│       ├── translate_cache.py  # LRU/TTL-кэш + построение ключа кэша
│       ├── glossary.py
│       ├── ct2_engine.py / mt_engine.py
│       ├── qwen_postedit.py / postedit.py
│       └── …
├── frontend/                   # Тематический статический UI
├── glossary/
├── prompts/
├── models/
├── scripts/
├── tests/                      # pytest: юнит глоссария + интеграция API
├── pytest.ini                 # pytest: pythonpath, testpaths
├── requirements.txt
├── requirements-gpu.txt
└── README.md
```

---

## API

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/api/v1/translate` | Тело: `text`, `apply_glossary`, `apply_postedit`, `include_debug` |
| `GET` | `/api/v1/health` | Статус, GPU, записи кэша, пайплайн |
| `POST` | `/api/v1/translate/cache/clear` | Очистить кэш переводов |

---

## Примечания

> После изменения `glossary/*.json` или `prompts/*.md` вызовите `POST /api/v1/translate/cache/clear` или перезапустите Uvicorn, чтобы обновить кэшированные переводы. Текст промпта пост-редактора кэшируется в процессе (`lru_cache` по пути к файлу) — при редактировании markdown без смены пути требуется перезапуск.

> Добавьте файл `LICENSE` перед публичным распространением.
