"""
src/app/feedback.py

Сохранение пользовательской разметки ошибок для офлайн дообучения.

Принцип: тяжёлые признаки (LaBSE, ruGPT-3) НЕ пересчитываются.
Фронтенд кэширует features из последней оценки и присылает их
вместе с разметкой. Скрипт дообучения читает features прямо из jsonl.

Файл: data/feedback/feedback.jsonl
Формат: одна JSON-строка на запись (newline-delimited JSON).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEFAULT_FEEDBACK_PATH = Path("data/feedback/feedback.jsonl")


def save_feedback(
    src: str,
    mt: str,
    start_char: int,
    end_char: int,
    error_type: str,
    severity: str,
    features: dict[str, Any],
    word_logprobs: list[float],
    feedback_path: Path = DEFAULT_FEEDBACK_PATH,
) -> str:
    """
    Сохраняет одну запись разметки в feedback.jsonl.

    Возвращает feedback_id (UUID4) для подтверждения клиенту.
    """
    feedback_id = str(uuid.uuid4())
    record = {
        "id":           feedback_id,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "src":          src,
        "mt":           mt,
        "start_char":   start_char,
        "end_char":     end_char,
        "error_type":   error_type,
        "severity":     severity,
        "features":     features,       # 22 признака — не пересчитываем при дообучении
        "word_logprobs": word_logprobs, # per-word logprobs — не пересчитываем
    }

    feedback_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log.info("Feedback сохранён: id=%s  error_type=%s  severity=%s",
                 feedback_id, error_type, severity)
    except OSError as e:
        log.error("Ошибка записи feedback: %s", e)
        raise

    return feedback_id


def count_feedback(feedback_path: Path = DEFAULT_FEEDBACK_PATH) -> int:
    """Возвращает число записей в feedback файле."""
    if not feedback_path.exists():
        return 0
    with open(feedback_path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
