"""
src/predict.py

Единая точка входа для инференса MTQE (sentence + word-level).

Загружает все модели один раз при создании Predictor:
  - FeatureExtractor   (spaCy + LaBSE + ruGPT-3)
  - SentenceModel      (XGBoost/NGBoost) + SHAP explainer
  - SpanModel          (XLM-R token classification)
  - OverallSentenceEvaluator (rules.py + MQM-aggregation)

Публичный API:
  predictor.predict_sentence(src, mt) → SentenceUIResult
  predictor.predict_batch([(src, mt), ...]) → list[SentenceUIResult]

ИСПРАВЛЕНИЯ:
  1. _render_highlighted_mt — индексация end_idx теперь безопасна
     для всех крайних случаев (end_idx >= len(mt_words)).
  2. В SentenceErrorItem убрано дублирование error_type + русского описания
     в одну строку через дефис — теперь хранятся раздельно для гибкости UI.
  3. predict_batch — слова берутся через FeatureExtractor токенизацию
     (согласовано с span_model), а не через .split() который может
     расходиться с spaCy-токенизацией span-модели.
  4. ДОБАВЛЕНО: поле debug в SentenceUIResult, содержащее features,
     word_logprobs и shap_values для детального отображения на фронтенде.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.features.extractor import FeatureExtractor
from src.interpretation.overall import OverallSentenceEvaluator, OverallSentenceResult
from src.interpretation.rules import describe_error_type_ru
from src.models.sentence_model import SentenceModel, MQM_CATEGORY_RU
from src.models.span_model import SpanModel

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Датаклассы выходного формата
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SentenceErrorItem:
    """Один найденный BAD-спан с типом ошибки и координатами."""
    severity:    str
    error_type:  str    # например "Fluency/Agreement"
    error_label: str    # русское описание, например "Грамматика: ошибки согласования"
    confidence:  float
    span_text:   str
    start_idx:   int
    end_idx:     int


@dataclass(frozen=True)
class SentenceUIResult:
    """Финальный результат одного предложения для UI и API."""
    src:                  str
    mt:                   str
    score:                float          # NGBoost/XGBoost score ∈ [0,1]
    ci_low:               float
    ci_high:              float
    uncertainty:          float
    mqm_score:            float          # MQM-style score ∈ [0,1]
    highlighted_mt_html:  str
    errors:               Sequence[SentenceErrorItem] = field(default_factory=list)
    explanation:          Mapping[str, float] = field(default_factory=dict)
    debug:                Mapping[str, Any]   = field(default_factory=dict)   # ← добавлено

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """
    Единая точка входа для инференса MTQE (sentence + word-level).

    Загружает модели один раз — держать один экземпляр на процесс.

    Параметры:
        models_dir           корневая директория моделей (по умолчанию "models/")
        sentence_model_path  путь к .model (XGBoost) или .pkl (NGBoost)
        shap_explainer_path  путь к shap_explainer.pkl
        span_model_dir       путь к директории xlm_roberta_span (HF формат)
        mqm_weights_path     путь к weights_mqm.npy (None → единичные веса)
        device               "cpu" или "cuda" для SpanModel
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

        # Автовыбор модели предложения: XGBoost приоритетнее NGBoost
        if sentence_model_path is None:
            candidate = models_dir / "xgboost_sentence.model"
            sentence_model_path = candidate if candidate.exists() else models_dir / "ngboost_sentence.pkl"

        if shap_explainer_path is None:
            shap_explainer_path = models_dir / "shap_explainer.pkl"

        if span_model_dir is None:
            span_model_dir = models_dir / "xlm_roberta_span"

        log.info("Инициализация FeatureExtractor...")
        self.extractor = FeatureExtractor()
        self.extractor.load_heavy_models()

        # Сохраняем пути для перезагрузки
        self._sentence_model_path = sentence_model_path
        self._shap_explainer_path = shap_explainer_path
        self._span_model_dir = span_model_dir
        self._device = device

        log.info("Загрузка SentenceModel из %s", sentence_model_path)
        self.sentence_model = SentenceModel(sentence_model_path, shap_explainer_path)

        log.info("Загрузка SpanModel из %s", span_model_dir)
        self.span_model = SpanModel(span_model_dir, device=device)

        log.info("Инициализация OverallSentenceEvaluator...")
        self.overall = OverallSentenceEvaluator(weights_path=mqm_weights_path)

        log.info("Predictor готов.")

    # ------------------------------------------------------------------
    # predict_sentence
    # ------------------------------------------------------------------

    def predict_sentence(self, src: str, mt: str) -> SentenceUIResult:
        """
        Инференс для одного предложения (src, mt).

        Возвращает SentenceUIResult с score, CI, MQM score,
        подсвеченным HTML и списком ошибок.
        """
        src = (src or "").strip()
        mt  = (mt or "").strip()

        if not src and not mt:
            return _empty_result(src, mt)

        # 1. Извлечение признаков
        feats = self.extractor.extract(src, mt)

        # 2. Sentence-level score + SHAP
        sentence_pred = self.sentence_model.predict(feats["vector"])
        # 3. Слова mt для span-модели и рендера
        #    Используем токенизацию FeatureExtractor (spaCy) для согласованности
        mt_words = _get_mt_words(feats, mt)

        # 4. Span-level: severity каждого слова → BAD-спаны
        word_logprobs = feats.get("word_logprobs") or None
        span_pred = self.span_model.predict(src, mt, word_logprobs=word_logprobs)

        # 5. Сборка финального результата
        overall: OverallSentenceResult = self.overall.evaluate(
            sentence_pred=sentence_pred,
            span_pred=span_pred,
            mt_words=mt_words,
            sentence_features=feats.get("raw", {}),
        )

        # Собираем debug-информацию: признаки, word_logprobs, SHAP values
        debug_info = {
            "features": feats.get("raw", {}),
            "word_logprobs": word_logprobs if word_logprobs else [],
        }
        debug_info["shap_values"] = _serialize_shap_values(
            getattr(sentence_pred, "shap_values", None),
            self.sentence_model.feature_names,
        )

        return _build_ui_result(src, mt, mt_words, overall, debug_info)

    # ------------------------------------------------------------------
    # predict_batch
    # ------------------------------------------------------------------

    def predict_batch(self, pairs: Sequence[tuple[str, str]]) -> list[SentenceUIResult]:
        """
        Батчевый инференс для списка (src, mt) пар.

        Признаки вычисляются батчами (эффективно для LaBSE и ruGPT-3).
        SpanModel вызывается последовательно (будущее: батчинг).
        """
        clean_pairs: list[tuple[str, str]] = []
        for src, mt in pairs:
            s = (src or "").strip()
            m = (mt or "").strip()
            if s or m:  # пропускаем только полностью пустые пары
                clean_pairs.append((s, m))

        if not clean_pairs:
            return []

        # Батчевое извлечение признаков
        feats_list = self.extractor.extract_batch(clean_pairs)

        # Батчевый sentence-level score
        vectors = np.stack([f["vector"] for f in feats_list])
        sentence_preds = self.sentence_model.predict_batch(vectors)

        results: list[SentenceUIResult] = []
        for (src, mt), feats, sentence_pred in zip(clean_pairs, feats_list, sentence_preds):
            mt_words      = _get_mt_words(feats, mt)
            word_logprobs = feats.get("word_logprobs") or None
            span_pred     = self.span_model.predict(src, mt, word_logprobs=word_logprobs)

            overall = self.overall.evaluate(
                sentence_pred=sentence_pred,
                span_pred=span_pred,
                mt_words=mt_words,
                sentence_features=feats.get("raw", {}),
            )

            # Собираем debug для текущего предложения
            debug_info = {
                "features": feats.get("raw", {}),
                "word_logprobs": word_logprobs if word_logprobs else [],
            }
            debug_info["shap_values"] = _serialize_shap_values(
                getattr(sentence_pred, "shap_values", None),
                self.sentence_model.feature_names,
            )

            results.append(_build_ui_result(src, mt, mt_words, overall, debug_info))

        return results

    # ------------------------------------------------------------------
    # reload_light_models — для горячей перезагрузки после дообучения
    # ------------------------------------------------------------------
    def reload_light_models(self) -> None:
        """
        Горячая перезагрузка лёгких моделей (SentenceModel + SpanModel).
        LaBSE и ruGPT-3 остаются в памяти.
        Вызывается из /api/reload_models.
        """
        log.info("Перезагрузка SentenceModel и SpanModel...")
        self.sentence_model = SentenceModel(
            self._sentence_model_path,
            self._shap_explainer_path
        )
        self.span_model = SpanModel(
            self._span_model_dir,
            device=self._device
        )
        log.info("SentenceModel и SpanModel успешно перезагружены.")


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _get_mt_words(feats: dict[str, Any], mt_fallback: str) -> list[str]:
    """
    Получает список слов mt из результата FeatureExtractor.

    Если FeatureExtractor хранит токены в feats["mt_words"] — берём оттуда
    (spaCy-токенизация, согласована со span-моделью и word_logprobs).
    Иначе fallback через .split().
    """
    if "mt_words" in feats and feats["mt_words"]:
        return list(feats["mt_words"])
    return mt_fallback.split()


def _build_ui_result(
    src: str,
    mt: str,
    mt_words: list[str],
    overall: OverallSentenceResult,
    debug_info: dict[str, Any] | None = None,
) -> SentenceUIResult:
    """Собирает SentenceUIResult из OverallSentenceResult и дополнительной debug-информации."""
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

    highlighted     = _render_highlighted_mt(mt_words, overall.spans)
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
    )


def _empty_result(src: str, mt: str) -> SentenceUIResult:
    """Возвращает нейтральный результат для пустого входа."""
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
    )


def _safe_span_text(mt_words: Sequence[str], start_idx: int, end_idx: int) -> str:
    """Безопасное получение текста спана без IndexError."""
    if not mt_words:
        return ""
    start = max(start_idx, 0)
    end   = min(end_idx, len(mt_words) - 1)
    if start > end:
        return ""
    return " ".join(mt_words[start : end + 1])


def _render_highlighted_mt(mt_words: Sequence[str], spans: Sequence[Any]) -> str:
    """
    Генерирует HTML с подсветкой BAD-слов по severity.

    Цветовая схема:
      BAD-major → красный фон (#ffb3b3)
      BAD-minor → жёлтый фон (#ffe3a3)

    ИСПРАВЛЕНИЕ: безопасная работа с end_idx:
    - end_idx трактуется как включительный индекс (0-based)
    - min(end_idx, len(mt_words) - 1) предотвращает выход за границы
    - BAD-major перезаписывает BAD-minor (приоритет severity)
    """
    if not mt_words:
        return ""

    n = len(mt_words)
    severities = ["OK"] * n

    for span in spans:
        start = max(int(span.start_idx), 0)
        end   = min(int(span.end_idx), n - 1)   # включительный, безопасный
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
    """
    Переводит ключи explanation (MQM-категории) на русский
    и сортирует по убыванию абсолютного вклада.
    """
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
