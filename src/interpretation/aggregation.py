from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.interpretation.rules import MQM_ERROR_TYPES, TypedSpan


SEVERITY_PENALTY: Mapping[str, float] = {
    "BAD-minor": 1.0,
    "BAD-major": 5.0,
}


@dataclass(frozen=True)
class MQMAggregation:
    mqm_score: float
    penalty: float
    z: float
    per_type_penalty: dict[str, float] = field(default_factory=dict)


def load_mqm_weights(weights_path: str | Path | None) -> np.ndarray:
    """
    Загружает веса MQM типов ошибок.

    Если файл отсутствует или weights_path=None — возвращает единичные веса.
    Порядок весов фиксирован как MQM_ERROR_TYPES.
    """
    if weights_path is None:
        return np.ones(len(MQM_ERROR_TYPES), dtype=np.float32)

    path = Path(weights_path)
    if not path.exists():
        return np.ones(len(MQM_ERROR_TYPES), dtype=np.float32)

    w = np.load(path)
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    if w.shape[0] != len(MQM_ERROR_TYPES):
        raise ValueError(
            f"weights_mqm.npy имеет длину {w.shape[0]}, ожидается {len(MQM_ERROR_TYPES)}"
        )
    return w


def aggregate_sentence_mqm(
    typed_spans: Sequence[TypedSpan],
    mt_word_count: int,
    weights: np.ndarray | None = None,
) -> MQMAggregation:
    """
    Sentence-level MQM-style агрегация (без абзаца).

    Формула из архитектуры:
      Q = 100 - Σ_i (w_ti * p_si * c_i) / Z
      Z = число слов mt
    Возвращаем mqm_score в [0,1] как Q/100, clipped.
    """
    if weights is None:
        weights = np.ones(len(MQM_ERROR_TYPES), dtype=np.float32)

    z = float(max(int(mt_word_count), 1))
    penalty_total = 0.0
    per_type: dict[str, float] = {}

    for span in typed_spans:
        wt = _weight_for_type(weights, span.error_type)
        ps = float(SEVERITY_PENALTY.get(span.severity, 1.0))
        ci = float(np.clip(span.confidence, 0.0, 1.0))
        p = (wt * ps * ci) / z
        penalty_total += p
        per_type[span.error_type] = per_type.get(span.error_type, 0.0) + p

    q_100 = 100.0 - penalty_total
    mqm_score = float(np.clip(q_100 / 100.0, 0.0, 1.0))

    return MQMAggregation(
        mqm_score=mqm_score,
        penalty=penalty_total,
        z=z,
        per_type_penalty=per_type,
    )


def _weight_for_type(weights: np.ndarray, error_type: str) -> float:
    try:
        idx = MQM_ERROR_TYPES.index(error_type)
    except ValueError:
        return 1.0
    return float(weights[idx])

