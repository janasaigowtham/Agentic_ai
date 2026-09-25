"""FastAPI entrypoint. Run with: uvicorn trh.api.main:app --reload --port 8000

Mock mode needs no environment variables. Live mode activates only when both
GEMINI_API_KEY and ANTHROPIC_API_KEY are set before the server starts.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from trh.api import routes_gate, routes_reviews, routes_trajectories
from trh.config import load_config

DEFAULT_DB_PATH = os.environ.get("TRH_DB_PATH", "trh.sqlite3")


def create_app(db_path: str = DEFAULT_DB_PATH) -> FastAPI:
    app = FastAPI(title="Trajectory Review Harness")
    app.state.db_path = db_path

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(routes_trajectories.router)
    app.include_router(routes_reviews.router)
    app.include_router(routes_gate.router)

    @app.get("/api/config")
    def get_mode():
        config = load_config()
        return {"mode": config.mode.value}

    return app


app = create_app()
