from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as translate_router
from app.core.config import settings

logger = logging.getLogger(__name__)


def _enable_gradio() -> bool:
    """Whether to mount the Gradio demo UI at ``/gradio``.

    Disabled by default in unit tests / lean deployments so importing
    ``app.main`` stays cheap (Gradio is a heavy import). Enable with
    ``MT_MVP_ENABLE_GRADIO=1`` in ``.env`` or the process environment (Docker sets this).
    """
    return settings.enable_gradio


def create_app() -> FastAPI:
    app = FastAPI(
        title="Blast equipment EN to ES translator",
        description="Glossary-backed MT for shotblasting / PPE catalog copy.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(translate_router, prefix="/api/v1")

    frontend_dir = Path(__file__).resolve().parents[1] / "frontend"

    @app.get("/")
    def index_page() -> FileResponse:
        return FileResponse(frontend_dir / "index.html")

    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    if _enable_gradio():
        try:
            import gradio as gr  # noqa: PLC0415 - heavy optional import

            from app.gradio_app import build_ui  # noqa: PLC0415

            ui = build_ui()
            # Use a queue so concurrent demo users don't pile onto the model.
            ui.queue()
            gr.mount_gradio_app(app, ui, path="/gradio")
            logger.info("Gradio UI mounted at /gradio")
        except Exception as exc:  # noqa: BLE001 — never block API startup
            logger.warning("Gradio UI failed to mount (%s); /gradio disabled.", exc)

    logger.info(
        "Post-edit Qwen: %s (set MT_MVP_POSTEDIT_USE_QWEN=true to enable; see .env.example).",
        "enabled" if settings.postedit_use_qwen else "disabled by default",
    )

    return app


app = create_app()
