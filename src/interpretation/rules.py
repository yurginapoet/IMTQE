"""
src/interpretation/rules.py

Детерминированное назначение типов ошибок MQM для BAD-спанов.

Принцип: XLM-RoBERTa определяет ТОЛЬКО severity.
Тип MQM определяется исключительно правилами здесь.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

try:
    import pymorphy2
    _morph = pymorphy2.MorphAnalyzer()
except Exception:
    _morph = None


# ---------------------------------------------------------------------------
# Типология MQM
# ---------------------------------------------------------------------------

MQM_ERROR_TYPES: tuple[str, ...] = (
    "Accuracy/Mistranslation",
    "Accuracy/Omission",
    "Accuracy/Addition",
    "Accuracy/Untranslated",
    "Accuracy/Hallucination",
    "Fluency/Morphology",
    "Fluency/Agreement",
    "Fluency/Spelling",
    "Fluency/LexicalChoice",
    "Fluency/Repetition",
    "Terminology/WrongTerm",
    "Terminology/Inconsistency",
    "Locale/NumberFormat",
    "Locale/DateFormat",
    "Locale/Quotes",
    "Locale/Currency",
    "Style/Register",
)

MQM_ERROR_TYPE_RU: dict[str, str] = {
    "Accuracy/Mistranslation":   "Искажение смысла: неверный перевод",
    "Accuracy/Omission":         "Пропуск: часть смысла отсутствует",
    "Accuracy/Addition":         "Добавление: лишняя информация в переводе",
    "Accuracy/Untranslated":     "Непереведено: фрагмент оставлен на исходном языке",
    "Accuracy/Hallucination":    "Галлюцинация: добавлен выдуманный смысл",
    "Fluency/Morphology":        "Грамматика: ошибки морфологии",
    "Fluency/Agreement":         "Грамматика: ошибки согласования",
    "Fluency/Spelling":          "Орфография: опечатки/неверное написание",
    "Fluency/LexicalChoice":     "Стиль/лексика: неудачный выбор слов",
    "Fluency/Repetition":        "Стиль/лексика: тавтология и повторы",
    "Terminology/WrongTerm":     "Терминология: неверный термин",
    "Terminology/Inconsistency": "Терминология: непоследовательность терминов",
    "Locale/NumberFormat":       "Локаль: числа/формат чисел",
    "Locale/DateFormat":         "Локаль: формат дат",
    "Locale/Quotes":             "Локаль: кавычки",
    "Locale/Currency":           "Локаль: валюты/обозначения",
    "Style/Register":            "Стиль: неуместный регистр",
}


def describe_error_type_ru(error_type: str) -> str:
    """Возвращает русское описание типа MQM-ошибки с безопасным fallback."""
    return MQM_ERROR_TYPE_RU.get(error_type, error_type)

# ---------------------------------------------------------------------------
# Регулярные выражения
# ---------------------------------------------------------------------------

_RE_DIGIT            = re.compile(r"\d")
_RE_DATE             = re.compile(
    r"\b\d{1,2}/\d{1,2}/\d{4}\b"
    r"|"
    r"\b(?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
)
_RE_STRAIGHT_QUOTES  = re.compile(r'["\']')
_RE_CURRENCY         = re.compile(r"[$€£¥₽]")
_RE_LATIN            = re.compile(r"[A-Za-z]")
_RE_CYRILLIC         = re.compile(r"[а-яёА-ЯЁ]")
_RE_WEIRD_CONSONANTS = re.compile(r"[бвгджзйклмнпрстфхцчшщ]{4,}", re.IGNORECASE)

_LOGPROB_FLUENCY_THRESHOLD = -6.0


# ---------------------------------------------------------------------------
# Датаклассы
# ---------------------------------------------------------------------------

@dataclass
class SpanResult:
    start_idx:          int
    end_idx:            int
    severity:           str
    confidence:         float
    word_logprobs_span: list[float] = field(default_factory=list)


@dataclass
class TypedSpan:
    start_idx:          int
    end_idx:            int
    severity:           str
    confidence:         float
    word_logprobs_span: list[float]
    error_type:         str = "Accuracy/Mistranslation"

    @classmethod
    def from_span(cls, span: SpanResult, error_type: str) -> "TypedSpan":
        return cls(
            start_idx=span.start_idx,
            end_idx=span.end_idx,
            severity=span.severity,
            confidence=span.confidence,
            word_logprobs_span=list(span.word_logprobs_span),
            error_type=error_type,
        )


# ---------------------------------------------------------------------------
# Публичная функция
# ---------------------------------------------------------------------------

def assign_mqm_types(
    mt_words: Sequence[str],
    spans: Sequence[SpanResult],
    sentence_features: Mapping[str, Any] | None = None,
) -> list[TypedSpan]:
    feats = sentence_features or {}

    # Новые и важные признаки
    untranslated_ratio    = float(feats.get("untranslated_ratio", 0.0))
    morphology_error_rate = float(feats.get("morphology_error_rate", 0.0))
    repetition_ratio      = float(feats.get("repetition_ratio", 0.0))
    named_entity_missing  = float(feats.get("named_entity_missing_ratio", 0.0))
    latin_ratio           = float(feats.get("latin_ratio", 0.0))

    digit_match_ratio     = float(feats.get("digit_match_ratio", 1.0))
    quotes_mismatch       = int(feats.get("quotes_mismatch", 0))
    date_format_error     = int(feats.get("date_format_error", 0))
    agreement_errors      = float(feats.get("agreement_errors", 0.0))
    oov_ratio             = float(feats.get("oov_ratio", 0.0))

    results: list[TypedSpan] = []
    for span in spans:
        span_text = _span_text(mt_words, span.start_idx, span.end_idx)
        error_type = _infer_type(
            span_text=span_text,
            word_logprobs_span=span.word_logprobs_span,
            untranslated_ratio=untranslated_ratio,
            morphology_error_rate=morphology_error_rate,
            repetition_ratio=repetition_ratio,
            named_entity_missing=named_entity_missing,
            latin_ratio=latin_ratio,
            digit_match_ratio=digit_match_ratio,
            quotes_mismatch=quotes_mismatch,
            date_format_error=date_format_error,
            agreement_errors=agreement_errors,
            oov_ratio=oov_ratio,
        )
        results.append(TypedSpan.from_span(span, error_type))

    return results


def _span_text(mt_words: Sequence[str], start_idx: int, end_idx: int) -> str:
    if not mt_words:
        return ""
    start = max(start_idx, 0)
    end = min(end_idx, len(mt_words) - 1)
    if start > end:
        return ""
    return " ".join(mt_words[start:end + 1])


def _infer_type(
    span_text: str,
    word_logprobs_span: Sequence[float],
    untranslated_ratio: float,
    morphology_error_rate: float,
    repetition_ratio: float,
    named_entity_missing: float,
    latin_ratio: float,
    digit_match_ratio: float,
    quotes_mismatch: int,
    date_format_error: int,
    agreement_errors: float,
    oov_ratio: float,
) -> str:
    text = span_text.strip()
    if not text:
        return "Accuracy/Mistranslation"

    # === Высокий приоритет ===
    if untranslated_ratio > 0.22 or latin_ratio > 0.30:
        return "Accuracy/Untranslated"

    if _RE_CURRENCY.search(text):
        return "Locale/Currency"

    if quotes_mismatch and _RE_STRAIGHT_QUOTES.search(text):
        return "Locale/Quotes"

    if date_format_error and _RE_DATE.search(text):
        return "Locale/DateFormat"

    if digit_match_ratio < 0.75 and _RE_DIGIT.search(text):
        return "Locale/NumberFormat"

    # === Fluency ===
    if morphology_error_rate > 0.20 or _has_morphology_error(text):
        return "Fluency/Morphology"

    if oov_ratio > 0.25 and _has_spelling_issue(text):
        return "Fluency/Spelling"

    if repetition_ratio > 0.24:
        return "Fluency/Repetition"

    if agreement_errors > 0 and _looks_like_agreement_error(word_logprobs_span):
        return "Fluency/Agreement"

    # === Accuracy ===
    if named_entity_missing > 0.40 and len(text.split()) <= 3:
        return "Accuracy/Omission"

    if _is_low_fluency(word_logprobs_span):
        return "Fluency/LexicalChoice"

    # Дефолт
    return "Accuracy/Mistranslation"


# ---------------------------------------------------------------------------
# Вспомогательные функции (оставляем как были)
# ---------------------------------------------------------------------------

def _looks_untranslated(span_text: str) -> bool:
    has_latin = bool(_RE_LATIN.search(span_text))
    has_cyr   = bool(_RE_CYRILLIC.search(span_text))
    return has_latin and not has_cyr


def _has_morphology_error(span_text: str) -> bool:
    if _morph is None:
        return False
    words = [w for w in span_text.split() if _RE_CYRILLIC.search(w)]
    if not words:
        return False
    for w in words:
        parses = _morph.parse(w)
        if not parses or parses[0].tag.POS is None:
            return True
    return False


def _has_spelling_issue(span_text: str) -> bool:
    words = span_text.split()
    for w in words:
        if len(w) > 20 and "-" not in w:
            return True
        if 2 <= len(w) <= 8 and _RE_WEIRD_CONSONANTS.search(w):
            return True
    return False


def _looks_like_agreement_error(word_logprobs_span: Sequence[float]) -> bool:
    if not word_logprobs_span:
        return False
    mean_lp = float(np.mean(word_logprobs_span))
    return -6.0 < mean_lp < -3.5


def _is_low_fluency(word_logprobs_span: Sequence[float]) -> bool:
    if not word_logprobs_span:
        return False
    mean_lp = float(np.mean(word_logprobs_span))
    return mean_lp < _LOGPROB_FLUENCY_THRESHOLD
