from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.features.extractor import FeatureExtractor
from src.interpretation.overall import OverallSentenceEvaluator, OverallSentenceResult
from src.interpretation.rules import describe_error_type_ru
from src.models.sentence_model import SentenceModel
from src.models.sentence_model import MQM_CATEGORY_RU
from src.models.span_model import SpanModel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SentenceErrorItem:
    severity: str
    error_type: str
    confidence: float
    span_text: str
    start_idx: int
    end_idx: int


@dataclass(frozen=True)
class SentenceUIResult:
    src: str
    mt: str
    score: float
    ci_low: float
    ci_high: float
    uncertainty: float
    mqm_score: float
    highlighted_mt_html: str
    errors: Sequence[SentenceErrorItem] = field(default_factory=list)
    explanation: Mapping[str, float] = field(default_factory=dict)
    debug: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Predictor:
    """
    Единая точка входа для инференса MTQE (sentence+word-level).

    Загружает модели один раз:
    - FeatureExtractor (spaCy + LaBSE + ruGPT-3)
    - SentenceModel (XGBoost/NGBoost) + SHAP explainer
    - SpanModel (XLM-R token classification)
    - OverallSentenceEvaluator (rules + MQM aggregation)
    """

    def __init__(
        self,
        models_dir: str | Path = "models",
        sentence_model_path: str | Path | None = None,
        shap_explainer_path: str | Path | None = None,
        span_model_dir: str | Path | None = None,
        mqm_weights_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        models_dir = Path(models_dir)

        if sentence_model_path is None:
            model_candidate = models_dir / "xgboost_sentence.model"
            if model_candidate.exists():
                sentence_model_path = model_candidate
            else:
                sentence_model_path = models_dir / "ngboost_sentence.pkl"

        if shap_explainer_path is None:
            shap_explainer_path = models_dir / "shap_explainer.pkl"

        if span_model_dir is None:
            span_model_dir = models_dir / "xlm_roberta_span"

        self.extractor = FeatureExtractor()
        self.extractor.load_heavy_models()

        self.sentence_model = SentenceModel(sentence_model_path, shap_explainer_path)
        self.span_model = SpanModel(span_model_dir, device=device)
        self.overall = OverallSentenceEvaluator(weights_path=mqm_weights_path)

    def predict_sentence(self, src: str, mt: str) -> SentenceUIResult:
        src = (src or "").strip()
        mt = (mt or "").strip()

        if not src and not mt:
            return SentenceUIResult(
                src=src,
                mt=mt,
                score=0.0,
                ci_low=0.0,
                ci_high=0.0,
                uncertainty=0.0,
                mqm_score=1.0,
                highlighted_mt_html=_render_highlighted_mt([], []),
                errors=[],
                explanation={},
            )

        feats = self.extractor.extract(src, mt)
        vec = feats["vector"]
        sentence_pred = self.sentence_model.predict(vec)

        mt_words = mt.split()
        if mt_words and feats.get("word_logprobs"):
            word_logprobs = feats["word_logprobs"]
        else:
            word_logprobs = None

        span_pred = self.span_model.predict(src, mt, word_logprobs=word_logprobs)

        overall: OverallSentenceResult = self.overall.evaluate(
            sentence_pred=sentence_pred,
            span_pred=span_pred,
            mt_words=mt_words,
            sentence_features=feats.get("raw", {}),
        )

        errors: list[SentenceErrorItem] = []
        for span in overall.spans:
            span_text = " ".join(mt_words[span.start_idx : span.end_idx + 1]) if mt_words else ""
            errors.append(
                SentenceErrorItem(
                    severity=span.severity,
                    error_type=f"{span.error_type} — {describe_error_type_ru(span.error_type)}",
                    confidence=float(np.clip(span.confidence, 0.0, 1.0)),
                    span_text=span_text,
                    start_idx=span.start_idx,
                    end_idx=span.end_idx,
                )
            )

        highlighted = _render_highlighted_mt(mt_words, overall.spans)
        explanation_sorted = _sort_explanation_ru(overall.explanation)

        return SentenceUIResult(
            src=src,
            mt=mt,
            score=float(np.clip(overall.sentence_score, 0.0, 1.0)),
            ci_low=float(np.clip(overall.ci_low, 0.0, 1.0)),
            ci_high=float(np.clip(overall.ci_high, 0.0, 1.0)),
            uncertainty=float(max(overall.uncertainty, 0.0)),
            mqm_score=float(np.clip(overall.mqm.mqm_score, 0.0, 1.0)),
            highlighted_mt_html=highlighted,
            errors=errors,
            explanation=explanation_sorted,
        )

    def predict_batch(self, pairs: Sequence[tuple[str, str]]) -> list[SentenceUIResult]:
        clean_pairs: list[tuple[str, str]] = []
        for src, mt in pairs:
            s = (src or "").strip()
            m = (mt or "").strip()
            if not s and not m:
                continue
            clean_pairs.append((s, m))

        if not clean_pairs:
            return []

        feats_list = self.extractor.extract_batch(clean_pairs)
        vectors = np.stack([f["vector"] for f in feats_list])
        sentence_preds = self.sentence_model.predict_batch(vectors)

        results: list[SentenceUIResult] = []
        for (src, mt), feats, sentence_pred in zip(clean_pairs, feats_list, sentence_preds):
            mt_words = mt.split()
            word_logprobs = feats.get("word_logprobs") or None
            span_pred = self.span_model.predict(src, mt, word_logprobs=word_logprobs)
            overall = self.overall.evaluate(
                sentence_pred=sentence_pred,
                span_pred=span_pred,
                mt_words=mt_words,
                sentence_features=feats.get("raw", {}),
            )

            errors: list[SentenceErrorItem] = []
            for span in overall.spans:
                span_text = " ".join(mt_words[span.start_idx : span.end_idx + 1]) if mt_words else ""
                errors.append(
                    SentenceErrorItem(
                        severity=span.severity,
                        error_type=f"{span.error_type} — {describe_error_type_ru(span.error_type)}",
                        confidence=float(np.clip(span.confidence, 0.0, 1.0)),
                        span_text=span_text,
                        start_idx=span.start_idx,
                        end_idx=span.end_idx,
                    )
                )

            highlighted = _render_highlighted_mt(mt_words, overall.spans)
            explanation_sorted = _sort_explanation_ru(overall.explanation)

            results.append(
                SentenceUIResult(
                    src=src,
                    mt=mt,
                    score=float(np.clip(overall.sentence_score, 0.0, 1.0)),
                    ci_low=float(np.clip(overall.ci_low, 0.0, 1.0)),
                    ci_high=float(np.clip(overall.ci_high, 0.0, 1.0)),
                    uncertainty=float(max(overall.uncertainty, 0.0)),
                    mqm_score=float(np.clip(overall.mqm.mqm_score, 0.0, 1.0)),
                    highlighted_mt_html=highlighted,
                    errors=errors,
                    explanation=explanation_sorted,
                )
            )

        return results


def _sort_explanation(expl: Mapping[str, float]) -> dict[str, float]:
    items = sorted(expl.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
    return {k: float(v) for k, v in items}


def _sort_explanation_ru(expl: Mapping[str, float]) -> dict[str, float]:
    items = sorted(expl.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
    out: dict[str, float] = {}
    for k, v in items:
        label = MQM_CATEGORY_RU.get(k, k)
        out[label] = float(v)
    return out


def _render_highlighted_mt(mt_words: Sequence[str], spans: Sequence[Any]) -> str:
    if not mt_words:
        return ""

    severities = ["OK"] * len(mt_words)
    for span in spans:
        for i in range(max(span.start_idx, 0), min(span.end_idx, len(mt_words) - 1) + 1):
            if span.severity == "BAD-major":
                severities[i] = "BAD-major"
            elif span.severity == "BAD-minor" and severities[i] != "BAD-major":
                severities[i] = "BAD-minor"

    parts: list[str] = []
    for word, sev in zip(mt_words, severities):
        if sev == "BAD-major":
            parts.append(f'<span style="background:#ffb3b3;padding:2px 4px;border-radius:4px;">{_escape_html(word)}</span>')
        elif sev == "BAD-minor":
            parts.append(f'<span style="background:#ffe3a3;padding:2px 4px;border-radius:4px;">{_escape_html(word)}</span>')
        else:
            parts.append(_escape_html(word))

    return " ".join(parts)


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )
