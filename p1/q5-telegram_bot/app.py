"""FastAPI webhook service."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, ORJSONResponse
from telegram import Update

from agent import DataAgent
from conversation import ConversationStore
from settings import Settings, validate_settings
from telegram_handler import build_telegram_application


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = validate_settings()
    client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)
    agent = DataAgent(settings, client, ConversationStore())
    telegram = build_telegram_application(settings.telegram_bot_token.get_secret_value(), agent)
    await telegram.initialize()
    app.state.settings, app.state.client, app.state.agent, app.state.telegram = settings, client, agent, telegram
    yield
    await telegram.shutdown()
    await client.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="Telegram Data Analyst", default_response_class=ORJSONResponse, lifespan=lifespan)

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"service": "telegram-data-agent", "status": "ok"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.get("/status")
    async def status(request: Request) -> dict[str, Any]:
        settings: Settings = request.app.state.settings
        return {"status": "running", "model": settings.model}

    @app.post("/webhook")
    async def webhook(request: Request) -> dict[str, bool]:
        try:
            update = Update.de_json(await request.json(), request.app.state.telegram.bot)
            await request.app.state.telegram.process_update(update)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid Telegram update") from exc
        return {"ok": True}

    @app.get("/logs/{run_id}.jsonl")
    async def logs(run_id: str, request: Request) -> FileResponse:
        if not run_id.startswith("run_") or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in run_id):
            raise HTTPException(status_code=404, detail="Log not found")
        path: Path = request.app.state.settings.log_directory / f"{run_id}.jsonl"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Log not found")
        return FileResponse(path, media_type="application/x-ndjson")

    return app


app = create_app()
