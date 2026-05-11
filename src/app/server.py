"""
src/app/server.py

Точка запуска uvicorn. Модели загружаются ОДИН РАЗ при старте процесса.
Перезагрузка страницы браузером не вызывает повторную загрузку.

Запуск:
    uvicorn src.app.server:app --host 0.0.0.0 --port 8000
    uvicorn src.app.server:app --host 0.0.0.0 --port 8000 --reload  # для разработки
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.app import api
from src.app.models_state import ModelsState

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — загрузка и выгрузка моделей
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    state: ModelsState = app.state.models_state
    log.info("MTQE: loading models (~30–60s on CPU)")
    t0 = time.monotonic()
    try:
        state.load()
        elapsed = time.monotonic() - t0
        log.info("MTQE: models ready in %.1f s", elapsed)
    except Exception as exc:
        log.error("КРИТИЧЕСКАЯ ОШИБКА при загрузке моделей: %s", exc, exc_info=True)
        state.mark_error(str(exc))
        # Сервер стартует, но /api/status вернёт ready=false

    yield  # сервер работает

    log.info("MTQE: shutdown")


# ---------------------------------------------------------------------------
# Создание приложения
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    application = FastAPI(
        title="MTQE — Machine Translation Quality Estimator",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Синглтон состояния моделей — живёт всё время работы процесса
    application.state.models_state = ModelsState()

    # Роуты API
    application.include_router(api.router)

    # Статические файлы
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        application.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return application


app = create_app()
