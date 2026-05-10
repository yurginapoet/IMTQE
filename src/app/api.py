"""
src/app/api.py

FastAPI роутер. Все эндпоинты системы MTQE.

Эндпоинты:
    GET  /                      — HTML интерфейс
    GET  /api/status            — статус загрузки моделей
    POST /api/evaluate          — оценка одного предложения
    POST /api/evaluate_batch    — оценка списка (не используется в UI, для API)
    POST /api/feedback          — сохранение ручной разметки
    POST /api/reload_models     — горячая перезагрузка sentence/span моделей
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.app.feedback import DEFAULT_FEEDBACK_PATH, count_feedback, save_feedback
from src.app.models_state import ModelsState

log = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Шаблоны
# ---------------------------------------------------------------------------

_templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_templates_dir))


# ---------------------------------------------------------------------------
# Dependency: получить ModelsState из app.state
# ---------------------------------------------------------------------------

def get_models_state(request: Request) -> ModelsState:
    return request.app.state.models_state


def get_predictor(request: Request):
    state: ModelsState = request.app.state.models_state
    return state.get_predictor()  # бросает RuntimeError если не готов


# ---------------------------------------------------------------------------
# Pydantic схемы запросов / ответов
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    src: str = Field(..., description="Исходное предложение на английском")
    mt:  str = Field(..., description="Машинный перевод на русском")


class EvaluateBatchRequest(BaseModel):
    pairs: list[dict[str, str]] = Field(
        ..., description="Список пар [{src: str, mt: str}]"
    )


class FeedbackRequest(BaseModel):
    src:          str
    mt:           str
    start_char:   int = Field(..., ge=0)
    end_char:     int = Field(..., ge=0)
    error_type:   str
    severity:     str = Field(..., pattern="^(BAD-minor|BAD-major)$")
    features:     dict[str, Any] = Field(default_factory=dict)
    word_logprobs: list[float]   = Field(default_factory=list)


class StatusResponse(BaseModel):
    ready:            bool
    status:           str
    models_loaded_at: float | None = None
    error:            str | None   = None
    feedback_count:   int          = 0


# ---------------------------------------------------------------------------
# HTML интерфейс
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------------------------------------------------------
# /api/status
# ---------------------------------------------------------------------------

@router.get("/api/status", response_model=StatusResponse)
async def status(state: ModelsState = Depends(get_models_state)):
    return StatusResponse(
        ready=state.ready,
        status=state.status.value,
        models_loaded_at=state.loaded_at if state.ready else None,
        error=state.error_message if state.error_message else None,
        feedback_count=count_feedback(),
    )


# ---------------------------------------------------------------------------
# /api/evaluate
# ---------------------------------------------------------------------------

@router.post("/api/evaluate")
async def evaluate(
    req: EvaluateRequest,
    predictor=Depends(get_predictor),
) -> dict[str, Any]:
    """
    Оценка одного предложения. Основной эндпоинт UI.

    Возвращает SentenceUIResult.to_dict() плюс поле debug
    с features и word_logprobs для кэширования на фронтенде
    (нужно для последующего сохранения feedback без пересчёта признаков).
    """
    if not req.src.strip() and not req.mt.strip():
        raise HTTPException(status_code=422, detail="src и mt не могут быть оба пустыми")

    t0 = time.monotonic()
    try:
        result = predictor.predict_sentence(req.src, req.mt)
    except Exception as e:
        log.error("Ошибка при evaluate: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {e}")

    elapsed = time.monotonic() - t0
    log.info("evaluate: %.2f сек  score=%.3f  mqm=%.3f",
             elapsed, result.score, result.mqm_score)

    data = result.to_dict()
    data["elapsed_sec"] = round(elapsed, 3)
    return data


# ---------------------------------------------------------------------------
# /api/evaluate_batch
# ---------------------------------------------------------------------------

@router.post("/api/evaluate_batch")
async def evaluate_batch(
    req: EvaluateBatchRequest,
    predictor=Depends(get_predictor),
) -> list[dict[str, Any]]:
    """
    Батчевая оценка (для внешних клиентов API, UI использует /api/evaluate).
    Максимум 50 пар за раз.
    """
    pairs = [(p.get("src", ""), p.get("mt", "")) for p in req.pairs]
    if len(pairs) > 50:
        raise HTTPException(status_code=422, detail="Максимум 50 пар за один запрос")
    if not pairs:
        return []

    try:
        results = predictor.predict_batch(pairs)
    except Exception as e:
        log.error("Ошибка при evaluate_batch: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {e}")

    return [r.to_dict() for r in results]


# ---------------------------------------------------------------------------
# /api/feedback
# ---------------------------------------------------------------------------

@router.post("/api/feedback")
async def feedback(req: FeedbackRequest) -> dict[str, Any]:
    """
    Сохраняет ручную разметку ошибки в data/feedback/feedback.jsonl.
    features и word_logprobs берутся из кэша фронтенда — тяжёлые модели
    НЕ запускаются повторно.
    """
    if req.end_char < req.start_char:
        raise HTTPException(
            status_code=422,
            detail="end_char не может быть меньше start_char"
        )

    try:
        feedback_id = save_feedback(
            src=req.src,
            mt=req.mt,
            start_char=req.start_char,
            end_char=req.end_char,
            error_type=req.error_type,
            severity=req.severity,
            features=req.features,
            word_logprobs=req.word_logprobs,
        )
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Ошибка записи: {e}")

    return {"saved": True, "feedback_id": feedback_id}


# ---------------------------------------------------------------------------
# /api/reload_models
# ---------------------------------------------------------------------------

@router.post("/api/reload_models")
async def reload_models(
    state: ModelsState = Depends(get_models_state),
) -> dict[str, Any]:
    """
    Горячая перезагрузка sentence model и span model после дообучения.
    LaBSE и ruGPT-3 НЕ перезагружаются — они остаются в RAM.

    Вызывать после: python scripts/finetune_from_feedback.py
    """
    if not state.ready:
        raise HTTPException(status_code=503, detail="Модели ещё не загружены")

    try:
        predictor = state.get_predictor()
        predictor.reload_light_models()
        log.info("Sentence model и span model успешно перезагружены")
    except Exception as e:
        log.error("Ошибка перезагрузки моделей: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return {"reloaded": True}
