from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.interpretation.aggregation import MQMAggregation, aggregate_sentence_mqm, load_mqm_weights
from src.interpretation.rules import TypedSpan, assign_mqm_types
from src.models.sentence_model import SentencePrediction
from src.models.span_model import SpanPrediction


@dataclass(frozen=True)
class OverallSentenceResult:
    sentence_score: float
    uncertainty: float
    ci_low: float
    ci_high: float
    explanation: Mapping[str, float]
    mqm: MQMAggregation
    spans: Sequence[TypedSpan] = field(default_factory=list)
    overall_score: float = 0.0


class OverallSentenceEvaluator:
    """
    Блок общей оценки для режима sentence+word-level (без абзаца).

    Собирает:
    - sentence score + CI + explanation из SentenceModel (SHAP→категории)
    - BAD-спаны из SpanModel + тип MQM через rules.py
    - MQM-style штраф → mqm_score (0..1)
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
        typed = assign_mqm_types(mt_words=mt_words, spans=span_pred.spans, sentence_features=sentence_features)
        mqm = aggregate_sentence_mqm(typed_spans=typed, mt_word_count=len(mt_words), weights=self._weights)

        overall_score = float(np.clip(sentence_pred.score, 0.0, 1.0))

        return OverallSentenceResult(
            sentence_score=float(sentence_pred.score),
            uncertainty=float(sentence_pred.uncertainty),
            ci_low=float(sentence_pred.ci_low),
            ci_high=float(sentence_pred.ci_high),
            explanation=dict(sentence_pred.explanation),
            mqm=mqm,
            spans=typed,
            overall_score=overall_score,
        )

