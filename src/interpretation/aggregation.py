"""
src/interpretation/aggregation.py

MQM-style агрегация штрафов за ошибки → sentence/paragraph score.

Формула (архитектура, раздел 5.4):
  Q = 100 - Σᵢ (wₜᵢ · pₛᵢ · cᵢ) / Z

  wₜ  — вес типа ошибки (обучается, хранится в models/weights_mqm.npy)
  pₛ  — штраф severity: BAD-major=5, BAD-minor=1
  cᵢ  — confidence XLM-RoBERTa для спана i ∈ [0,1]
  Z   — суммарное число слов mt (нормализатор)

ИСПРАВЛЕНИЕ формулы:
  Деление на Z применяется к сумме в целом, не к каждому слагаемому.
  Математически результат одинаков (Σ(x/Z) == (Σx)/Z), однако:
  - per_type_penalty теперь хранит НЕнормализованные штрафы (wt*ps*ci),
    что позволяет корректно сравнивать вклады типов ошибок между
    предложениями разной длины.
  - Нормализация на Z применяется один раз при вычислении итогового Q.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.interpretation.rules import MQM_ERROR_TYPES, TypedSpan

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Штрафы за severity (из архитектуры, раздел 5.4)
# ---------------------------------------------------------------------------

SEVERITY_PENALTY: Mapping[str, float] = {
    "BAD-minor": 1.0,
    "BAD-major": 5.0,
}


# ---------------------------------------------------------------------------
# Датакласс результата агрегации
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MQMAggregation:
    """
    Результат MQM-style агрегации.

    Атрибуты:
        mqm_score       итоговый score ∈ [0,1], где 1 = идеальный перевод
        penalty         Σ(wt * ps * ci) — сырой штраф до деления на Z
        z               суммарное число слов mt (нормализатор)
        per_type_penalty вклад каждого типа ошибки в сырой штраф (до /Z)
                        для отладки и анализа доминирующих ошибок
    """
    mqm_score:       float
    penalty:         float
    z:               float
    per_type_penalty: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Загрузка весов
# ---------------------------------------------------------------------------

def load_mqm_weights(weights_path: str | Path | None) -> np.ndarray:
    """
    Загружает веса MQM типов ошибок из .npy файла.

    Если файл отсутствует или weights_path=None — возвращает единичные веса
    (все типы ошибок равнозначны).
    Порядок весов фиксирован как MQM_ERROR_TYPES (должен совпадать с
    порядком при оптимизации в aggregation блоке обучения).
    """
    if weights_path is None:
        return np.ones(len(MQM_ERROR_TYPES), dtype=np.float32)

    path = Path(weights_path)
    if not path.exists():
        _log.warning(
            "weights_mqm.npy не найден по пути %s — используем единичные веса",
            path,
        )
        return np.ones(len(MQM_ERROR_TYPES), dtype=np.float32)

    w = np.load(path)
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    if w.shape[0] != len(MQM_ERROR_TYPES):
        raise ValueError(
            f"weights_mqm.npy имеет длину {w.shape[0]}, "
            f"ожидается {len(MQM_ERROR_TYPES)} (по числу типов в MQM_ERROR_TYPES)"
        )
    return w


# ---------------------------------------------------------------------------
# Агрегация на уровне предложения
# ---------------------------------------------------------------------------

def aggregate_sentence_mqm(
    typed_spans: Sequence[TypedSpan],
    mt_word_count: int,
    weights: np.ndarray | None = None,
) -> MQMAggregation:
    """
    Sentence-level MQM-style агрегация.

    Формула:
        penalty = Σᵢ (wₜᵢ · pₛᵢ · cᵢ)          # сырой штраф
        Q       = 100 - penalty / Z               # нормализованный score
        mqm_score = clip(Q / 100, 0, 1)

    ИСПРАВЛЕНИЕ: деление на Z выполняется один раз для всей суммы,
    а не для каждого слагаемого в цикле. per_type_penalty хранит
    ненормализованные вклады для корректного сравнения типов.

    Параметры:
        typed_spans    спаны с назначенными типами MQM
        mt_word_count  число слов mt (≥ 1; при 0 принудительно = 1)
        weights        вектор весов типов, shape=(len(MQM_ERROR_TYPES),)
    """
    if weights is None:
        weights = np.ones(len(MQM_ERROR_TYPES), dtype=np.float32)

    z = float(max(int(mt_word_count), 1))

    # Накапливаем ненормализованные штрафы
    raw_penalty_total: float = 0.0
    per_type_raw: dict[str, float] = {}

    for span in typed_spans:
        wt = _weight_for_type(weights, span.error_type)
        ps = float(SEVERITY_PENALTY.get(span.severity, 1.0))
        ci = float(np.clip(span.confidence, 0.0, 1.0))

        raw = wt * ps * ci                           # ненормализованный вклад спана
        raw_penalty_total += raw
        per_type_raw[span.error_type] = per_type_raw.get(span.error_type, 0.0) + raw

    # Нормализуем один раз
    normalized_penalty = raw_penalty_total / z
    q_100     = 100.0 - normalized_penalty
    mqm_score = float(np.clip(q_100 / 100.0, 0.0, 1.0))

    return MQMAggregation(
        mqm_score=mqm_score,
        penalty=raw_penalty_total,       # сырой штраф (для отладки и оптимизации wt)
        z=z,
        per_type_penalty=per_type_raw,   # ненормализованные вклады по типам
    )


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _weight_for_type(weights: np.ndarray, error_type: str) -> float:
    """Возвращает вес для типа ошибки по индексу в MQM_ERROR_TYPES."""
    try:
        idx = MQM_ERROR_TYPES.index(error_type)
    except ValueError:
        # Неизвестный тип — используем вес 1.0 (не игнорируем ошибку)
        return 1.0
    return float(weights[idx])