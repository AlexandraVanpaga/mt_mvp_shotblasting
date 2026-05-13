# syntax=docker/dockerfile:1.6
#
# Multi-stage build for the EN→ES blast-equipment translator.
#
# Two parallel build targets are provided:
#
#   * ``runtime``      — CPU only, slim Python base (~1.5 GB final image).
#                        Default for ``docker build .`` / ``docker compose up``.
#
#   * ``runtime-gpu``  — CUDA 12.4 base with cu124 PyTorch wheels (~6 GB final
#                        image). Requires the NVIDIA Container Toolkit on the
#                        host. Selected via ``--target runtime-gpu`` or the
#                        ``gpu`` Compose profile.
#
# Build:    docker build -t mt-mvp:latest .
# Build:    docker build --target runtime-gpu -t mt-mvp:gpu .
# Run:      docker run --rm -p 8000:8000 mt-mvp:latest
# Run GPU:  docker run --rm -p 8000:8000 --gpus all mt-mvp:gpu
#
# Models are NOT baked into the image (they are large and license-bound). Mount
# them at runtime; see docker-compose.yml for canonical bind-mounts.

ARG PYTHON_VERSION=3.11

# ── Stage 1a: CPU dependency layer ───────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS base

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt
# torch >= 2.6 is required by transformers >= 4.46 to load .bin checkpoints
# (CVE-2025-32434). The Helsinki-NLP Marian repos still ship pytorch_model.bin.
RUN pip install --upgrade pip \
 && pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0+cpu \
 && pip install -r /tmp/requirements.txt

# ── Stage 1b: GPU dependency layer (CUDA 12.4) ───────────────────────────────
#
# Built on the CUDA 12.4 runtime image so the resulting torch wheel actually
# sees the host's GPU. PyTorch bundles its own cuDNN, so the plain ``runtime``
# variant (not ``cudnn-runtime``) is enough.
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base-gpu

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install --extra-index-url https://download.pytorch.org/whl/cu124 \
        torch==2.6.0 \
 && pip install -r /tmp/requirements.txt

# ── Stage 2a: CPU runtime ────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=base /opt/venv /opt/venv

RUN useradd --create-home --uid 10001 --shell /bin/bash app
WORKDIR /app
COPY --chown=app:app app/ /app/app/
COPY --chown=app:app frontend/ /app/frontend/
COPY --chown=app:app prompts/ /app/prompts/
COPY --chown=app:app glossary/ /app/glossary/
COPY --chown=app:app scripts/ /app/scripts/
COPY --chown=app:app README_ENG.md README_RUS.md /app/

# Pre-initialise the HF cache directory with correct ownership. When a named
# Docker volume is first mounted at /home/app/.cache/huggingface, Docker copies
# the underlying image directory's ownership (UID 10001 = app) into the empty
# volume — without this, the volume would be owned by root and the non-root
# `app` user could not write to it.
RUN mkdir -p /app/models /app/data /home/app/.cache/huggingface \
 && chown -R app:app /app /home/app/.cache

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl --silent --fail http://localhost:8000/api/v1/health || exit 1

ENV MT_MVP_ENABLE_GRADIO=1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ── Stage 2b: GPU runtime ────────────────────────────────────────────────────
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime-gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        python3.11 \
        python3.11-venv \
 && rm -rf /var/lib/apt/lists/*

COPY --from=base-gpu /opt/venv /opt/venv

RUN useradd --create-home --uid 10001 --shell /bin/bash app
WORKDIR /app
COPY --chown=app:app app/ /app/app/
COPY --chown=app:app frontend/ /app/frontend/
COPY --chown=app:app prompts/ /app/prompts/
COPY --chown=app:app glossary/ /app/glossary/
COPY --chown=app:app scripts/ /app/scripts/
COPY --chown=app:app README_ENG.md README_RUS.md /app/

# Pre-initialise HF cache dir so a fresh named volume inherits app:app
# ownership (see CPU runtime stage for full explanation).
RUN mkdir -p /app/models /app/data /home/app/.cache/huggingface \
 && chown -R app:app /app /home/app/.cache

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl --silent --fail http://localhost:8000/api/v1/health || exit 1

ENV MT_MVP_ENABLE_GRADIO=1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
