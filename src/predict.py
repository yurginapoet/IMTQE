"""Инференс MTQE: признаки, sentence-модели, span-модель, агрегация MQM."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from src.features.extractor import FeatureExtractor
from src.features.schema import FEATURE_NAMES_CLASSIC, FEATURE_NAMES_LIGHT
from src.interpretation.explanation_loss import shap_categories_to_loss_shares
from src.interpretation.overall import OverallSentenceEvaluator, OverallSentenceResult
from src.interpretation.rules import describe_error_type_ru
from src.models.sentence_model import (
    FEATURE_TO_MQM,
    MQM_CATEGORY_RU,
    SUPPORTED_MODEL_TYPES,
    SentenceModel,
    SentencePrediction,
    resolve_sentence_artifacts,
)
from src.models.span_model import SpanModel, SpanPrediction

log = logging.getLogger(__name__)

ENSEMBLE_MODEL_NAME = "ensemble"
MODEL_DISPLAY_ORDER = (ENSEMBLE_MODEL_NAME, "xgboost", "rf", "ridge")
DEFAULT_ENSEMBLE_WEIGHTS: dict[str, float] = {
    "xgboost": 0.45,
    "rf": 0.35,
    "ridge": 0.20,
}
EXPLANATION_FEATURE_BLACKLIST = {"avg_token_length", "src_length", "mt_length"}
SPAN_PENALTY_MINOR = 0.03
SPAN_PENALTY_MAJOR = 0.06
SPAN_PENALTY_EXTRA_SPAN = 0.01
SPAN_PENALTY_CAP = 0.12


@dataclass(frozen=True)
class SentenceErrorItem:
    severity: str
    error_type: str
    error_label: str
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
    selected_model: str = ENSEMBLE_MODEL_NAME
    model_scores: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Predictor:
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
        self._models_dir = models_dir
        self._device = device
        self._sentence_models: dict[str, SentenceModel] = {}

        default_model_name = "xgboost"
        if sentence_model_path is not None:
            default_model_path = Path(sentence_model_path)
            default_explainer_path = Path(shap_explainer_path) if shap_explainer_path is not None else None
            self._model_artifacts = {
                default_model_name: (default_model_path, default_explainer_path),
            }
        else:
            self._model_artifacts = {
                name: resolve_sentence_artifacts(models_dir, name)
                for name in SUPPORTED_MODEL_TYPES
            }

        if span_model_dir is None:
            span_model_dir = models_dir / "xlm_roberta_span"

        sentence_model = self._get_sentence_model(default_model_name)
        expected_feature_count = max(
            sentence_model.expected_feature_count,
            self._max_available_feature_count(),
        )

        log.info("Инициализация FeatureExtractor...")
        self.extractor = FeatureExtractor()
        require_neural = expected_feature_count > len(FEATURE_NAMES_CLASSIC)
        self.extractor.load_heavy_models(require_neural=require_neural)
        self._validate_extractor_features(expected_feature_count)

        self._span_model_dir = span_model_dir

        log.info("Загрузка SpanModel из %s", span_model_dir)
        self.span_model = SpanModel(span_model_dir, device=device)

        log.info("Инициализация OverallSentenceEvaluator...")
        self.overall = OverallSentenceEvaluator(weights_path=mqm_weights_path)
        log.info("Predictor готов.")

    def predict_sentence(
        self,
        src: str,
        mt: str,
        model_name: str = "xgboost",
        compare_all: bool = False,
        confidence_threshold: float = 0.0,
    ) -> SentenceUIResult:
        src = (src or "").strip()
        mt = (mt or "").strip()

        if not src and not mt:
            return _empty_result(src, mt, model_name=model_name)

        confidence_threshold = float(np.clip(confidence_threshold, 0.0, 1.0))
        feats = self.extractor.extract(src, mt)

        requested_models = _ordered_available_model_names(SUPPORTED_MODEL_TYPES)
        predictions = self._predict_sentence_models(feats["vector"], requested_models)
        if not predictions:
            raise RuntimeError("Не удалось получить ни одной sentence-модели для инференса")

        mt_words = _get_mt_words(feats, mt)
        word_logprobs = feats.get("word_logprobs") or None
        span_pred = self.span_model.predict(
            src,
            mt,
            word_logprobs=word_logprobs,
            mt_words=mt_words,
        )
        span_pred = _filter_span_prediction(span_pred, confidence_threshold)

        ensemble_pred = _build_ensemble_prediction(predictions)
        span_penalty = _compute_span_penalty(span_pred)
        penalized_ensemble_pred = _apply_score_penalty(ensemble_pred, span_penalty["penalty"])

        overall: OverallSentenceResult = self.overall.evaluate(
            sentence_pred=penalized_ensemble_pred,
            span_pred=span_pred,
            mt_words=mt_words,
            sentence_features=feats.get("raw", {}),
        )
        overall = _with_loss_explanation(
            overall,
            self._display_explanation_en(penalized_ensemble_pred),
        )

        model_scores = _build_model_scores(predictions, penalized_ensemble_pred)
        primary_model_name = model_name if model_name in predictions else next(iter(predictions))
        primary_model, primary_pred = predictions[primary_model_name]
        debug_info = build_sentence_debug_payload(
            feats,
            primary_model,
            primary_pred,
            model_scores=model_scores,
            selected_model=ENSEMBLE_MODEL_NAME,
            compare_all=compare_all,
            confidence_threshold=confidence_threshold,
            all_predictions=predictions,
            ensemble_prediction=penalized_ensemble_pred,
            ensemble_weights=_normalized_model_weights(predictions.keys()),
            span_penalty=span_penalty,
        )
        return _build_ui_result(
            src,
            mt,
            mt_words,
            overall,
            debug_info,
            selected_model=ENSEMBLE_MODEL_NAME,
            model_scores=model_scores,
        )

    def predict_batch(self, pairs: Sequence[tuple[str, str]]) -> list[SentenceUIResult]:
        clean_pairs: list[tuple[str, str]] = []
        for src, mt in pairs:
            s = (src or "").strip()
            m = (mt or "").strip()
            if s or m:
                clean_pairs.append((s, m))

        if not clean_pairs:
            return []

        feats_list = self.extractor.extract_batch(clean_pairs)
        results: list[SentenceUIResult] = []
        for (src, mt), feats in zip(clean_pairs, feats_list):
            predictions = self._predict_sentence_models(feats["vector"], SUPPORTED_MODEL_TYPES)
            mt_words = _get_mt_words(feats, mt)
            word_logprobs = feats.get("word_logprobs") or None
            span_pred = self.span_model.predict(
                src,
                mt,
                word_logprobs=word_logprobs,
                mt_words=mt_words,
            )

            ensemble_pred = _build_ensemble_prediction(predictions)
            span_penalty = _compute_span_penalty(span_pred)
            penalized_ensemble_pred = _apply_score_penalty(ensemble_pred, span_penalty["penalty"])

            overall = self.overall.evaluate(
                sentence_pred=penalized_ensemble_pred,
                span_pred=span_pred,
                mt_words=mt_words,
                sentence_features=feats.get("raw", {}),
            )
            overall = _with_loss_explanation(
                overall,
                self._display_explanation_en(penalized_ensemble_pred),
            )

            model_scores = _build_model_scores(predictions, penalized_ensemble_pred)
            primary_model, primary_pred = predictions["xgboost"] if "xgboost" in predictions else next(iter(predictions.values()))
            debug_info = build_sentence_debug_payload(
                feats,
                primary_model,
                primary_pred,
                model_scores=model_scores,
                selected_model=ENSEMBLE_MODEL_NAME,
                compare_all=True,
                confidence_threshold=0.0,
                all_predictions=predictions,
                ensemble_prediction=penalized_ensemble_pred,
                ensemble_weights=_normalized_model_weights(predictions.keys()),
                span_penalty=span_penalty,
            )
            results.append(
                _build_ui_result(
                    src,
                    mt,
                    mt_words,
                    overall,
                    debug_info,
                    selected_model=ENSEMBLE_MODEL_NAME,
                    model_scores=model_scores,
                )
            )

        return results

    def reload_light_models(self) -> None:
        log.info("Перезагрузка SentenceModel и SpanModel...")
        self._sentence_models.clear()
        self.span_model = SpanModel(self._span_model_dir, device=self._device)
        log.info("SentenceModel и SpanModel успешно перезагружены.")

    def _get_sentence_model(self, model_name: str) -> SentenceModel:
        if model_name not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Неизвестная sentence-модель: {model_name}")
        if model_name not in self._sentence_models:
            model_path, explainer_path = self._model_artifacts.get(model_name, resolve_sentence_artifacts(self._models_dir, model_name))
            log.info("Загрузка SentenceModel[%s] из %s", model_name, model_path)
            self._sentence_models[model_name] = SentenceModel(
                model_path,
                explainer_path,
                model_type=model_name,
            )
        return self._sentence_models[model_name]

    def _predict_sentence_models(
        self,
        features: np.ndarray,
        model_names: Sequence[str],
    ) -> dict[str, tuple[SentenceModel, Any]]:
        out: dict[str, tuple[SentenceModel, Any]] = {}
        for model_name in model_names:
            sentence_model = self._get_sentence_model(model_name)
            out[model_name] = (sentence_model, sentence_model.predict(features))
        return out

    def _display_explanation_en(self, sentence_pred: Any) -> dict[str, float]:
        return shap_categories_to_loss_shares(
            sentence_pred.explanation,
            loss_budget=max(0.0, 1.0 - float(sentence_pred.score)),
            min_share=0.005,
        )

    def _max_available_feature_count(self) -> int:
        counts = []
        for model_name in SUPPORTED_MODEL_TYPES:
            model_path, _explainer_path = self._model_artifacts.get(model_name, (None, None))
            if model_path is not None and Path(model_path).exists():
                try:
                    counts.append(self._get_sentence_model(model_name).expected_feature_count)
                except FileNotFoundError:
                    continue
        return max(counts) if counts else len(FEATURE_NAMES_CLASSIC)

    def _validate_extractor_features(self, expected_feature_count: int) -> None:
        active_feature_count = len(self.extractor.active_feature_names)
        if active_feature_count >= expected_feature_count:
            return

        if expected_feature_count <= len(FEATURE_NAMES_LIGHT):
            required_label = "light"
        elif expected_feature_count <= len(FEATURE_NAMES_CLASSIC):
            required_label = "classic"
        else:
            required_label = "semantic-extended"

        raise RuntimeError(
            "FeatureExtractor не может собрать достаточно признаков для текущей "
            f"sentence-модели: нужно {expected_feature_count}, доступно {active_feature_count} "
            f"({required_label} model requirement). Проверь загрузку LaBSE/ruGPT-3 "
            "и актуальность обученной sentence-модели."
        )


def _with_loss_explanation(
    overall: OverallSentenceResult | Any,
    loss_shares_en: dict[str, float],
) -> OverallSentenceResult | Any:
    if is_dataclass(overall) and not isinstance(overall, type):
        return replace(overall, explanation=loss_shares_en)
    return SimpleNamespace(**{**vars(overall), "explanation": loss_shares_en})


def _get_mt_words(feats: dict[str, Any], mt_fallback: str) -> list[str]:
    if "mt_words" in feats and feats["mt_words"]:
        return list(feats["mt_words"])
    return mt_fallback.split()


def _build_ui_result(
    src: str,
    mt: str,
    mt_words: list[str],
    overall: OverallSentenceResult,
    debug_info: dict[str, Any] | None = None,
    selected_model: str = "xgboost",
    model_scores: Mapping[str, float] | None = None,
) -> SentenceUIResult:
    errors: list[SentenceErrorItem] = []
    for span in overall.spans:
        span_text = _safe_span_text(mt_words, span.start_idx, span.end_idx)
        errors.append(
            SentenceErrorItem(
                severity=span.severity,
                error_type=span.error_type,
                error_label=describe_error_type_ru(span.error_type),
                confidence=float(np.clip(span.confidence, 0.0, 1.0)),
                span_text=span_text,
                start_idx=span.start_idx,
                end_idx=span.end_idx,
            )
        )

    highlighted = _render_highlighted_mt(mt_words, overall.spans)
    explanation_out = _build_explanation_ru(overall.explanation)

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
        explanation=explanation_out,
        debug=debug_info or {},
        selected_model=selected_model,
        model_scores=dict(model_scores or {selected_model: float(np.clip(overall.sentence_score, 0.0, 1.0))}),
    )


def _empty_result(src: str, mt: str, model_name: str = ENSEMBLE_MODEL_NAME) -> SentenceUIResult:
    return SentenceUIResult(
        src=src,
        mt=mt,
        score=0.0,
        ci_low=0.0,
        ci_high=0.0,
        uncertainty=0.0,
        mqm_score=1.0,
        highlighted_mt_html="",
        errors=[],
        explanation={},
        debug={},
        selected_model=ENSEMBLE_MODEL_NAME,
        model_scores={ENSEMBLE_MODEL_NAME: 0.0, model_name: 0.0},
    )


def _filter_span_prediction(span_pred: SpanPrediction, confidence_threshold: float) -> SpanPrediction:
    if confidence_threshold <= 0.0:
        return span_pred
    filtered_spans = [
        span
        for span in span_pred.spans
        if float(getattr(span, "confidence", 0.0)) >= confidence_threshold
    ]
    return SpanPrediction(
        word_labels=list(span_pred.word_labels),
        word_probs=list(span_pred.word_probs),
        spans=filtered_spans,
    )


def _safe_span_text(mt_words: Sequence[str], start_idx: int, end_idx: int) -> str:
    if not mt_words:
        return ""
    start = max(start_idx, 0)
    end = min(end_idx, len(mt_words) - 1)
    if start > end:
        return ""
    return " ".join(mt_words[start : end + 1])


def _render_highlighted_mt(mt_words: Sequence[str], spans: Sequence[Any]) -> str:
    if not mt_words:
        return ""

    n = len(mt_words)
    severities = ["OK"] * n

    for span in spans:
        start = max(int(span.start_idx), 0)
        end = min(int(span.end_idx), n - 1)
        if start > end:
            continue
        for i in range(start, end + 1):
            if span.severity == "BAD-major":
                severities[i] = "BAD-major"
            elif span.severity == "BAD-minor" and severities[i] != "BAD-major":
                severities[i] = "BAD-minor"

    parts: list[str] = []
    for word, sev in zip(mt_words, severities):
        escaped = _escape_html(word)
        if sev == "BAD-major":
            parts.append(
                f'<span style="background:#ffb3b3;padding:2px 4px;'
                f'border-radius:4px;" title="BAD-major">{escaped}</span>'
            )
        elif sev == "BAD-minor":
            parts.append(
                f'<span style="background:#ffe3a3;padding:2px 4px;'
                f'border-radius:4px;" title="BAD-minor">{escaped}</span>'
            )
        else:
            parts.append(escaped)

    return " ".join(parts)


def _build_explanation_ru(expl: Mapping[str, float]) -> dict[str, float]:
    items = sorted(expl.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
    return {MQM_CATEGORY_RU.get(k, k): float(v) for k, v in items}


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def build_sentence_debug_payload(
    feats: Mapping[str, Any],
    sentence_model: SentenceModel,
    sentence_pred: Any,
    model_scores: Mapping[str, float],
    selected_model: str,
    compare_all: bool,
    confidence_threshold: float,
    all_predictions: Mapping[str, tuple[SentenceModel, Any]] | None = None,
    ensemble_prediction: Any | None = None,
    ensemble_weights: Mapping[str, float] | None = None,
    span_penalty: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    wlp = feats.get("word_logprobs") or None
    if hasattr(sentence_model, "explain_prediction"):
        model_debug = sentence_model.explain_prediction(feats["vector"], sentence_pred)
    else:
        shap_values = _serialize_shap_values(
            getattr(sentence_pred, "shap_values", None),
            getattr(sentence_model, "feature_names", []),
        )
        model_debug = {
            "feature_contributions": shap_values or {},
            "shap_values": shap_values,
            "model_type": getattr(sentence_model, "model_type", selected_model),
        }
    model_details = _build_model_details(
        feats,
        sentence_model,
        sentence_pred,
        all_predictions=all_predictions,
        ensemble_prediction=ensemble_prediction,
    )
    out: dict[str, Any] = {
        "features": feats.get("raw", {}),
        "word_logprobs": wlp if wlp else [],
        "feature_explanation": model_debug["feature_contributions"],
        "shap_values": model_debug["shap_values"],
        "model_type": model_debug["model_type"],
        "selected_model": selected_model,
        "model_scores": dict(model_scores),
        "compare_all": bool(compare_all),
        "confidence_threshold": float(confidence_threshold),
        "models": model_details,
        "ensemble_weights": dict(ensemble_weights or {}),
        "span_penalty": dict(span_penalty or {}),
    }
    return out


def _serialize_shap_values(
    shap_values: Any,
    feature_names: Sequence[str],
) -> dict[str, float] | list[float] | None:
    if shap_values is None:
        return None
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 1 and len(shap_values) == len(feature_names):
            return {
                name: float(value)
                for name, value in zip(feature_names, shap_values, strict=False)
            }
        return shap_values.tolist()
    return shap_values


def _ordered_available_model_names(model_names: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for name in MODEL_DISPLAY_ORDER:
        if name == ENSEMBLE_MODEL_NAME:
            continue
        if name in model_names and name not in seen:
            ordered.append(name)
            seen.add(name)
    for name in model_names:
        if name != ENSEMBLE_MODEL_NAME and name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def _normalized_model_weights(model_names: Sequence[str]) -> dict[str, float]:
    available = [name for name in _ordered_available_model_names(model_names) if name in DEFAULT_ENSEMBLE_WEIGHTS]
    if not available:
        return {}
    total = sum(DEFAULT_ENSEMBLE_WEIGHTS[name] for name in available)
    if total <= 0.0:
        uniform = 1.0 / len(available)
        return {name: uniform for name in available}
    return {name: DEFAULT_ENSEMBLE_WEIGHTS[name] / total for name in available}


def _build_ensemble_prediction(
    predictions: Mapping[str, tuple[SentenceModel, SentencePrediction | Any]],
) -> SentencePrediction:
    if not predictions:
        raise ValueError("Для ансамбля не передано ни одной модели")

    weights = _normalized_model_weights(predictions.keys())
    if not weights:
        fallback_name, (_fallback_model, fallback_pred) = next(iter(predictions.items()))
        return _clone_prediction(fallback_pred, model_name=fallback_name)

    first_name = next(iter(weights))
    first_model, first_pred = predictions[first_name]
    feature_names = getattr(first_model, "feature_names", [])
    first_shap = np.asarray(getattr(first_pred, "shap_values", []), dtype=np.float32).reshape(-1)
    expected_len = int(first_shap.shape[0] or len(feature_names))

    score = 0.0
    uncertainty = 0.0
    ci_low = 0.0
    ci_high = 0.0
    combined_shap = np.zeros(expected_len, dtype=np.float32)

    for name, weight in weights.items():
        _model, pred = predictions[name]
        score += float(pred.score) * weight
        uncertainty += float(pred.uncertainty) * weight
        ci_low += float(pred.ci_low) * weight
        ci_high += float(pred.ci_high) * weight
        shap_vals = np.asarray(getattr(pred, "shap_values", np.zeros(expected_len)), dtype=np.float32).reshape(-1)
        if shap_vals.shape[0] < expected_len:
            padded = np.zeros(expected_len, dtype=np.float32)
            padded[: shap_vals.shape[0]] = shap_vals
            shap_vals = padded
        elif shap_vals.shape[0] > expected_len:
            shap_vals = shap_vals[:expected_len]
        combined_shap += shap_vals * weight

    return SentencePrediction(
        score=float(np.clip(score, 0.0, 1.0)),
        uncertainty=float(max(uncertainty, 0.0)),
        ci_low=float(np.clip(ci_low, 0.0, 1.0)),
        ci_high=float(np.clip(ci_high, 0.0, 1.0)),
        alpha=None,
        beta_param=None,
        shap_values=combined_shap,
        explanation=_aggregate_category_contributions(feature_names, combined_shap),
    )


def _aggregate_category_contributions(feature_names: Sequence[str], shap_values: np.ndarray) -> dict[str, float]:
    expl: dict[str, float] = {}
    for name, value in zip(feature_names, np.asarray(shap_values).reshape(-1), strict=False):
        if name in EXPLANATION_FEATURE_BLACKLIST:
            continue
        category = FEATURE_TO_MQM.get(name)
        if not category:
            continue
        expl[category] = expl.get(category, 0.0) + float(value)
    return expl


def _compute_span_penalty(span_pred: SpanPrediction) -> dict[str, Any]:
    spans = list(getattr(span_pred, "spans", []) or [])
    major = sum(1 for span in spans if getattr(span, "severity", "") == "BAD-major")
    minor = sum(1 for span in spans if getattr(span, "severity", "") == "BAD-minor")
    has_bad_label = bool(spans)
    penalty = 0.0
    if has_bad_label:
        penalty += SPAN_PENALTY_MAJOR if major else SPAN_PENALTY_MINOR
        penalty += max(0, len(spans) - 1) * SPAN_PENALTY_EXTRA_SPAN
    penalty = float(min(SPAN_PENALTY_CAP, penalty))
    return {
        "has_bad_label": has_bad_label,
        "bad_spans": len(spans),
        "major_spans": major,
        "minor_spans": minor,
        "penalty": penalty,
    }


def _apply_score_penalty(pred: SentencePrediction, penalty: float) -> SentencePrediction:
    clamped_penalty = max(0.0, float(penalty))
    new_score = float(np.clip(float(pred.score) - clamped_penalty, 0.0, 1.0))
    return SentencePrediction(
        score=new_score,
        uncertainty=float(pred.uncertainty),
        ci_low=float(np.clip(pred.ci_low, 0.0, 1.0)),
        ci_high=float(np.clip(pred.ci_high, 0.0, 1.0)),
        alpha=pred.alpha,
        beta_param=pred.beta_param,
        shap_values=np.asarray(pred.shap_values, dtype=np.float32),
        explanation=dict(pred.explanation),
    )


def _clone_prediction(pred: SentencePrediction | Any, model_name: str | None = None) -> SentencePrediction:
    shap_values = np.asarray(getattr(pred, "shap_values", []), dtype=np.float32)
    feature_names = []
    if model_name:
        feature_names = []
    explanation = getattr(pred, "explanation", {})
    if explanation and not any(isinstance(v, float) and abs(v) <= 1.0 for v in explanation.values()):
        explanation = dict(explanation)
    return SentencePrediction(
        score=float(np.clip(float(getattr(pred, "score", 0.0)), 0.0, 1.0)),
        uncertainty=float(max(float(getattr(pred, "uncertainty", 0.0)), 0.0)),
        ci_low=float(np.clip(float(getattr(pred, "ci_low", 0.0)), 0.0, 1.0)),
        ci_high=float(np.clip(float(getattr(pred, "ci_high", 0.0)), 0.0, 1.0)),
        alpha=getattr(pred, "alpha", None),
        beta_param=getattr(pred, "beta_param", None),
        shap_values=shap_values,
        explanation=dict(explanation),
    )


def _build_model_scores(
    predictions: Mapping[str, tuple[SentenceModel, Any]],
    ensemble_prediction: SentencePrediction,
) -> dict[str, float]:
    scores = {
        ENSEMBLE_MODEL_NAME: float(np.clip(ensemble_prediction.score, 0.0, 1.0)),
    }
    for name, (_model, pred) in predictions.items():
        scores[name] = float(np.clip(pred.score, 0.0, 1.0))
    ordered: dict[str, float] = {}
    for name in MODEL_DISPLAY_ORDER:
        if name in scores:
            ordered[name] = scores[name]
    for name, score in scores.items():
        if name not in ordered:
            ordered[name] = score
    return ordered


def _build_model_details(
    feats: Mapping[str, Any],
    sentence_model: SentenceModel,
    sentence_pred: Any,
    all_predictions: Mapping[str, tuple[SentenceModel, Any]] | None = None,
    ensemble_prediction: Any | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    raw_features = feats.get("raw", {})
    if all_predictions:
        for name, (model, pred) in all_predictions.items():
            details[name] = _build_single_model_detail(feats, model, pred, raw_features)
    else:
        details[getattr(sentence_model, "model_type", "xgboost")] = _build_single_model_detail(
            feats,
            sentence_model,
            sentence_pred,
            raw_features,
        )
    if ensemble_prediction is not None:
        details[ENSEMBLE_MODEL_NAME] = _build_ensemble_model_detail(
            ensemble_prediction,
            raw_features,
            template_model=sentence_model,
        )
    ordered: dict[str, Any] = {}
    for name in MODEL_DISPLAY_ORDER:
        if name in details:
            ordered[name] = details[name]
    for name, value in details.items():
        if name not in ordered:
            ordered[name] = value
    return ordered


def _build_single_model_detail(
    feats: Mapping[str, Any],
    model: SentenceModel,
    pred: Any,
    raw_features: Mapping[str, Any],
) -> dict[str, Any]:
    if hasattr(model, "explain_prediction"):
        model_debug = model.explain_prediction(feats["vector"], pred)
    else:
        shap_values = _serialize_shap_values(
            getattr(pred, "shap_values", None),
            getattr(model, "feature_names", []),
        ) or {}
        model_debug = {
            "feature_values": dict(raw_features),
            "feature_contributions": shap_values,
            "shap_values": shap_values,
            "category_contributions": dict(getattr(pred, "explanation", {})),
            "model_type": getattr(model, "model_type", "xgboost"),
        }
    return {
        "score": float(np.clip(float(getattr(pred, "score", 0.0)), 0.0, 1.0)),
        "ci_low": float(np.clip(float(getattr(pred, "ci_low", 0.0)), 0.0, 1.0)),
        "ci_high": float(np.clip(float(getattr(pred, "ci_high", 0.0)), 0.0, 1.0)),
        "uncertainty": float(max(float(getattr(pred, "uncertainty", 0.0)), 0.0)),
        "feature_values": dict(model_debug.get("feature_values") or raw_features),
        "feature_contributions": dict(model_debug.get("feature_contributions") or {}),
        "shap_values": dict(model_debug.get("shap_values") or {}),
        "category_contributions": dict(model_debug.get("category_contributions") or {}),
    }


def _build_ensemble_model_detail(
    pred: Any,
    raw_features: Mapping[str, Any],
    template_model: SentenceModel,
) -> dict[str, Any]:
    shap_values = _serialize_shap_values(
        getattr(pred, "shap_values", None),
        getattr(template_model, "feature_names", []),
    ) or {}
    return {
        "score": float(np.clip(float(getattr(pred, "score", 0.0)), 0.0, 1.0)),
        "ci_low": float(np.clip(float(getattr(pred, "ci_low", 0.0)), 0.0, 1.0)),
        "ci_high": float(np.clip(float(getattr(pred, "ci_high", 0.0)), 0.0, 1.0)),
        "uncertainty": float(max(float(getattr(pred, "uncertainty", 0.0)), 0.0)),
        "feature_values": dict(raw_features),
        "feature_contributions": dict(shap_values),
        "shap_values": dict(shap_values),
        "category_contributions": dict(getattr(pred, "explanation", {})),
    }
