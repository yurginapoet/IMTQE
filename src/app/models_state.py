"""
src/app/models_state.py

Синглтон состояния моделей. Хранится в app.state.models_state.
Разделяет загрузку моделей (server.py) и их использование (api.py).
"""

from __future__ import annotations

import logging
import time
from enum import Enum
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


class LoadStatus(str, Enum):
    NOT_STARTED = "not_started"
    LOADING     = "loading"
    READY       = "ready"
    ERROR       = "error"


class ModelsState:
    """
    Хранит Predictor и статус его загрузки.
    Потокобезопасность: uvicorn использует один event loop,
    lifespan выполняется до начала обработки запросов — race condition исключён.
    """

    def __init__(self) -> None:
        self._predictor   = None
        self._status      = LoadStatus.NOT_STARTED
        self._error_msg   = ""
        self._loaded_at   = 0.0

    # ------------------------------------------------------------------
    # Загрузка
    # ------------------------------------------------------------------

    def load(
        self,
        models_dir: str | Path = "models",
        mqm_weights_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self._status = LoadStatus.LOADING
        # Импорт здесь — тяжёлые зависимости (torch, transformers, sentence_transformers)
        # загружаются только один раз при вызове load()
        from src.predict import Predictor
        self._predictor = Predictor(
            models_dir=models_dir,
            mqm_weights_path=mqm_weights_path,
            device=device,
        )
        self._status    = LoadStatus.READY
        self._loaded_at = time.time()

    def mark_error(self, message: str) -> None:
        self._status    = LoadStatus.ERROR
        self._error_msg = message

    # ------------------------------------------------------------------
    # Доступ
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._status == LoadStatus.READY

    @property
    def status(self) -> LoadStatus:
        return self._status

    @property
    def error_message(self) -> str:
        return self._error_msg

    @property
    def loaded_at(self) -> float:
        return self._loaded_at

    def get_predictor(self):
        """
        Возвращает Predictor или бросает RuntimeError если не готов.
        """
        if self._status != LoadStatus.READY:
            raise RuntimeError(
                f"Модели не готовы (status={self._status}). "
                f"Дождитесь завершения загрузки."
            )
        return self._predictor
