from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.models.span_model import SpanResult


MQM_ERROR_TYPES: tuple[str, ...] = (
    # Accuracy
    "Accuracy/Mistranslation",
    "Accuracy/Omission",
    "Accuracy/Addition",
    "Accuracy/Untranslated",
    "Accuracy/Hallucination",
    # Fluency
    "Fluency/Morphology",
    "Fluency/Agreement",
    "Fluency/Spelling",
    "Fluency/LexicalChoice",
    # Terminology
    "Terminology/WrongTerm",
    "Terminology/Inconsistency",
    # Locale
    "Locale/NumberFormat",
    "Locale/DateFormat",
    "Locale/Quotes",
    "Locale/Currency",
    # Style
    "Style/Register",
)

MQM_ERROR_TYPE_RU: dict[str, str] = {
    "Accuracy/Mistranslation": "Искажение смысла: неверный перевод",
    "Accuracy/Omission": "Пропуск: часть смысла отсутствует",
    "Accuracy/Addition": "Добавление: лишняя информация в переводе",
    "Accuracy/Untranslated": "Непереведено: фрагмент оставлен на исходном языке",
    "Accuracy/Hallucination": "Галлюцинация: добавлен выдуманный смысл",
    "Fluency/Morphology": "Грамматика: ошибки морфологии",
    "Fluency/Agreement": "Грамматика: ошибки согласования",
    "Fluency/Spelling": "Орфография: опечатки/неверное написание",
    "Fluency/LexicalChoice": "Стиль/лексика: неудачный выбор слов",
    "Terminology/WrongTerm": "Терминология: неверный термин",
    "Terminology/Inconsistency": "Терминология: непоследовательность терминов",
    "Locale/NumberFormat": "Локаль: числа/формат чисел",
    "Locale/DateFormat": "Локаль: формат дат",
    "Locale/Quotes": "Локаль: кавычки",
    "Locale/Currency": "Локаль: валюты/обозначения",
    "Style/Register": "Стиль: неуместный регистр (официальность/разговорность)",
}

_RE_DIGIT = re.compile(r"\d")
_RE_DATE = re.compile(
    r"\b\d{1,2}/\d{1,2}/\d{4}\b"
    r"|"
    r"\b(?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
)
_RE_STRAIGHT_QUOTES = re.compile(r"[\"']")
_RE_CURRENCY = re.compile(r"[$€£¥₽]")
_RE_LATIN = re.compile(r"[A-Za-z]")


@dataclass
class TypedSpan(SpanResult):
    """SpanResult + детерминированный тип MQM."""

    error_type: str = "Accuracy/Mistranslation"


def assign_mqm_types(
    mt_words: Sequence[str],
    spans: Sequence[SpanResult],
    sentence_features: Mapping[str, Any] | None = None,
) -> list[TypedSpan]:
    """
    Назначает error_type каждому BAD-спану детерминированно.

    Сейчас доступен только sentence/word-level режим:
    - используем span текст (по mt_words)
    - опционально используем sentence-level признаки из FeatureExtractor.extract()["raw"]

    Параметры:
        mt_words          список слов mt (длина должна совпадать с индексацией spans)
        spans             BAD-спаны (start_idx/end_idx относительно mt_words)
        sentence_features словарь признаков уровня предложения (опционально)
    """
    feats = sentence_features or {}
    digit_match_ratio = float(feats.get("digit_match_ratio", 1.0))
    quotes_mismatch = int(feats.get("quotes_mismatch", 0))
    date_format_error = int(feats.get("date_format_error", 0))

    results: list[TypedSpan] = []
    for span in spans:
        span_text = _span_text(mt_words, span.start_idx, span.end_idx)
        error_type = _infer_type(span_text, span.word_logprobs_span, digit_match_ratio, quotes_mismatch, date_format_error)
        results.append(
            TypedSpan(
                start_idx=span.start_idx,
                end_idx=span.end_idx,
                severity=span.severity,
                confidence=span.confidence,
                word_logprobs_span=list(span.word_logprobs_span),
                error_type=error_type,
            )
        )
    return results


def describe_error_type_ru(error_type: str) -> str:
    return MQM_ERROR_TYPE_RU.get(error_type, error_type)


def _span_text(mt_words: Sequence[str], start_idx: int, end_idx: int) -> str:
    if start_idx < 0 or end_idx < start_idx or start_idx >= len(mt_words):
        return ""
    end_idx = min(end_idx, len(mt_words) - 1)
    return " ".join(mt_words[start_idx : end_idx + 1])


def _infer_type(
    span_text: str,
    word_logprobs_span: Sequence[float],
    digit_match_ratio: float,
    quotes_mismatch: int,
    date_format_error: int,
) -> str:
    text = span_text.strip()
    if not text:
        return "Accuracy/Mistranslation"

    if _RE_CURRENCY.search(text):
        return "Locale/Currency"

    if quotes_mismatch and _RE_STRAIGHT_QUOTES.search(text):
        return "Locale/Quotes"

    if date_format_error and _RE_DATE.search(text):
        return "Locale/DateFormat"

    if digit_match_ratio < 1.0 and _RE_DIGIT.search(text):
        return "Locale/NumberFormat"

    if _looks_untranslated(text):
        return "Accuracy/Untranslated"

    if _is_low_fluency(word_logprobs_span):
        return "Fluency/LexicalChoice"

    return "Accuracy/Mistranslation"


def _looks_untranslated(span_text: str) -> bool:
    text = span_text.strip()
    if not text:
        return False
    latin = bool(_RE_LATIN.search(text))
    cyr = any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in text)
    return latin and not cyr


def _is_low_fluency(word_logprobs_span: Sequence[float]) -> bool:
    if not word_logprobs_span:
        return False
    arr = np.asarray(word_logprobs_span, dtype=np.float32)
    mean_lp = float(arr.mean())
    return mean_lp < -8.0
