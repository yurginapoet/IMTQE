"""
src/interpretation/overall.py

Сборка финального результата для одного предложения:
  sentence score (NGBoost/XGBoost) + CI + SHAP explanation
  + BAD-спаны с типами MQM (XLM-R + rules.py)
  + MQM-style агрегированный score

ИСПРАВЛЕНИЕ:
  Поле overall_score убрано из OverallSentenceResult — оно было избыточным
  и вводило в заблуждение (дублировало sentence_score без изменений).
  predict.py использует sentence_score и mqm.mqm_score раздельно —
  это правильный подход, т.к. они несут разную информацию.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.interpretation.aggregation import (
    MQMAggregation,
    aggregate_sentence_mqm,
    load_mqm_weights,
)
from src.interpretation.rules import TypedSpan, assign_mqm_types
from src.models.sentence_model import SentencePrediction
from src.models.span_model import SpanPrediction


# ---------------------------------------------------------------------------
# Датакласс результата
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OverallSentenceResult:
    """
    Финальный результат для одного предложения (sentence + word level).

    Атрибуты:
        sentence_score  оценка качества из NGBoost/XGBoost ∈ [0,1]
        uncertainty     Var[Beta] или аппроксимация для XGBoost
        ci_low          нижняя граница CI₉₅
        ci_high         верхняя граница CI₉₅
        explanation     вклады MQM-категорий из SHAP (Accuracy, Fluency, ...)
        mqm             результат MQM-style агрегации (mqm_score + per_type)
        spans           BAD-спаны с назначенными типами MQM
    """
    sentence_score: float
    uncertainty:    float
    ci_low:         float
    ci_high:        float
    explanation:    Mapping[str, float]
    mqm:            MQMAggregation
    spans:          Sequence[TypedSpan] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Оценщик
# ---------------------------------------------------------------------------

class OverallSentenceEvaluator:
    """
    Блок общей оценки для режима sentence + word-level (без абзаца).

    Собирает:
    - sentence score + CI + explanation из SentenceModel (SHAP → категории)
    - BAD-спаны из SpanModel + тип MQM через rules.py
    - MQM-style штраф → mqm_score (0..1)

    Загружает веса MQM один раз при инициализации.
    """

    def __init__(self, weights_path: str | Path | None = None) -> None:
        self._weights = load_mqm_weights(weights_path)

    def evaluate(
        self,
        sentence_pred: SentencePrediction,
        span_pred: SpanPrediction,
        mt_words: Sequence[str],
        sentence_features: Mapping[str, Any] | None = None,
    ) -> OverallSentenceResult:
        """
        Параметры:
            sentence_pred     результат SentenceModel.predict()
            span_pred         результат SpanModel.predict()
            mt_words          список слов mt (list[str])
            sentence_features словарь raw-признаков из FeatureExtractor
                              (нужен для правил: digit_match_ratio, oov_ratio, ...)
        """
        # 1. Назначаем типы MQM спанам детерминированно через rules.py
        typed = assign_mqm_types(
            mt_words=mt_words,
            spans=span_pred.spans,
            sentence_features=sentence_features,
        )

        # 2. MQM-style агрегация штрафов
        mqm = aggregate_sentence_mqm(
            typed_spans=typed,
            mt_word_count=len(mt_words),
            weights=self._weights,
        )

        return OverallSentenceResult(
            sentence_score=float(np.clip(sentence_pred.score, 0.0, 1.0)),
            uncertainty=float(max(sentence_pred.uncertainty, 0.0)),
            ci_low=float(np.clip(sentence_pred.ci_low, 0.0, 1.0)),
            ci_high=float(np.clip(sentence_pred.ci_high, 0.0, 1.0)),
            explanation=dict(sentence_pred.explanation),
            mqm=mqm,
            spans=typed,
        )