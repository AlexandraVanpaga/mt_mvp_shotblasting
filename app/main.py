from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as translate_router


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

    return app


app = create_app()
