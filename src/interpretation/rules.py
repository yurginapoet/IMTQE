"""
src/interpretation/rules.py

Детерминированное назначение типов ошибок MQM для BAD-спанов.

Принцип (НФТ-2):
  XLM-RoBERTa определяет ТОЛЬКО severity (OK/BAD-minor/BAD-major).
  Тип MQM ВСЕГДА определяется через правила здесь — никакой нейросети
  для классификации типа.

Иерархия правил (в порядке убывания приоритета):
  1. Locale/Currency      — символы валют ($€£¥₽)
  2. Locale/Quotes        — прямые кавычки в русском тексте
  3. Locale/DateFormat    — дата не адаптирована
  4. Locale/NumberFormat  — числа из src не найдены в mt
  5. Accuracy/Untranslated — латиница без кириллицы → не переведено
  6. Fluency/LexicalChoice — очень низкий logprob → неестественная лексика
  7. Fluency/Agreement    — если sentence_features сигнализирует об ошибках согласования
  8. Accuracy/Mistranslation — дефолт (низкое косинусное сходство и т.п.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Типология MQM (порядок фиксирован — weights_mqm.npy индексируется по нему)
# ---------------------------------------------------------------------------

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
    "Accuracy/Mistranslation":   "Искажение смысла: неверный перевод",
    "Accuracy/Omission":         "Пропуск: часть смысла отсутствует",
    "Accuracy/Addition":         "Добавление: лишняя информация в переводе",
    "Accuracy/Untranslated":     "Непереведено: фрагмент оставлен на исходном языке",
    "Accuracy/Hallucination":    "Галлюцинация: добавлен выдуманный смысл",
    "Fluency/Morphology":        "Грамматика: ошибки морфологии",
    "Fluency/Agreement":         "Грамматика: ошибки согласования",
    "Fluency/Spelling":          "Орфография: опечатки/неверное написание",
    "Fluency/LexicalChoice":     "Стиль/лексика: неудачный выбор слов",
    "Terminology/WrongTerm":     "Терминология: неверный термин",
    "Terminology/Inconsistency": "Терминология: непоследовательность терминов",
    "Locale/NumberFormat":       "Локаль: числа/формат чисел",
    "Locale/DateFormat":         "Локаль: формат дат",
    "Locale/Quotes":             "Локаль: кавычки",
    "Locale/Currency":           "Локаль: валюты/обозначения",
    "Style/Register":            "Стиль: неуместный регистр (официальность/разговорность)",
}

# ---------------------------------------------------------------------------
# Скомпилированные регулярные выражения
# ---------------------------------------------------------------------------

_RE_DIGIT          = re.compile(r"\d")
_RE_DATE           = re.compile(
    r"\b\d{1,2}/\d{1,2}/\d{4}\b"
    r"|"
    r"\b(?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
)
_RE_STRAIGHT_QUOTES = re.compile(r'["\']')
_RE_CURRENCY       = re.compile(r"[$€£¥₽]")
_RE_LATIN          = re.compile(r"[A-Za-z]")
_RE_CYRILLIC       = re.compile(r"[а-яёА-ЯЁ]")

# ---------------------------------------------------------------------------
# Порог logprob для Fluency/LexicalChoice
#
# ИСПРАВЛЕНИЕ: порог -8.0 слишком агрессивен для ruGPT-3 Small.
# Нормальный диапазон mean logprob для русского текста: -3.5 … -6.5.
# При mean < -6.0 слово считается лексически аномальным.
# Порог выбирается на val-сете WMT21 (рекомендуется верифицировать
# при обучении через percentile распределения word_logprobs BAD-слов).
# ---------------------------------------------------------------------------
_LOGPROB_FLUENCY_THRESHOLD = -6.0


# ---------------------------------------------------------------------------
# Датаклассы
# ---------------------------------------------------------------------------

@dataclass
class SpanResult:
    """
    Результат span-level модели (XLM-RoBERTa) для одного BAD-спана.
    Определяет severity, но НЕ тип ошибки.
    """
    start_idx:         int
    end_idx:           int
    severity:          str          # "BAD-minor" | "BAD-major"
    confidence:        float        # p(BAD) из XLM-RoBERTa
    word_logprobs_span: list[float] = field(default_factory=list)


@dataclass
class TypedSpan:
    """
    SpanResult + детерминированный тип MQM, назначенный rules.py.

    ИСПРАВЛЕНИЕ: не наследуем SpanResult через @dataclass, чтобы избежать
    проблемы Python с полями-дефолтами в цепочке наследования датаклассов.
    Все поля явно перечислены.
    """
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
# Публичные функции
# ---------------------------------------------------------------------------

def assign_mqm_types(
    mt_words: Sequence[str],
    spans: Sequence[SpanResult],
    sentence_features: Mapping[str, Any] | None = None,
) -> list[TypedSpan]:
    """
    Назначает error_type каждому BAD-спану детерминированно.

    Параметры:
        mt_words          список слов mt (len должен совпадать с макс. end_idx)
        spans             BAD-спаны (start_idx/end_idx включительные, 0-indexed)
        sentence_features признаки уровня предложения из FeatureExtractor (опционально)
    """
    feats = sentence_features or {}

    digit_match_ratio = float(feats.get("digit_match_ratio", 1.0))
    quotes_mismatch   = int(feats.get("quotes_mismatch", 0))
    date_format_error = int(feats.get("date_format_error", 0))
    agreement_errors  = float(feats.get("agreement_errors", 0.0))
    oov_ratio         = float(feats.get("oov_ratio", 0.0))

    results: list[TypedSpan] = []
    for span in spans:
        span_text  = _span_text(mt_words, span.start_idx, span.end_idx)
        error_type = _infer_type(
            span_text=span_text,
            word_logprobs_span=span.word_logprobs_span,
            digit_match_ratio=digit_match_ratio,
            quotes_mismatch=quotes_mismatch,
            date_format_error=date_format_error,
            agreement_errors=agreement_errors,
            oov_ratio=oov_ratio,
        )
        results.append(TypedSpan.from_span(span, error_type))

    return results


def describe_error_type_ru(error_type: str) -> str:
    """Возвращает русское описание типа ошибки MQM."""
    return MQM_ERROR_TYPE_RU.get(error_type, error_type)


# ---------------------------------------------------------------------------
# Внутренние функции
# ---------------------------------------------------------------------------

def _span_text(mt_words: Sequence[str], start_idx: int, end_idx: int) -> str:
    """Безопасная сборка текста спана."""
    if not mt_words:
        return ""
    start = max(start_idx, 0)
    end   = min(end_idx, len(mt_words) - 1)
    if start > end:
        return ""
    return " ".join(mt_words[start : end + 1])


def _infer_type(
    span_text: str,
    word_logprobs_span: Sequence[float],
    digit_match_ratio: float,
    quotes_mismatch: int,
    date_format_error: int,
    agreement_errors: float,
    oov_ratio: float,
) -> str:
    """
    Детерминированная иерархия правил для определения типа MQM.

    Правила проверяются в порядке убывания специфичности:
    специфичные (Locale) → общие (Accuracy/Mistranslation).
    """
    text = span_text.strip()
    if not text:
        return "Accuracy/Mistranslation"

    # 1. Валюта — наиболее специфичный сигнал
    if _RE_CURRENCY.search(text):
        return "Locale/Currency"

    # 2. Кавычки — только если sentence-level флаг установлен
    if quotes_mismatch and _RE_STRAIGHT_QUOTES.search(text):
        return "Locale/Quotes"

    # 3. Дата не адаптирована
    if date_format_error and _RE_DATE.search(text):
        return "Locale/DateFormat"

    # 4. Число из src не найдено в mt
    if digit_match_ratio < 1.0 and _RE_DIGIT.search(text):
        return "Locale/NumberFormat"

    # 5. Не переведено: только латиница, нет кириллицы
    if _looks_untranslated(text):
        return "Accuracy/Untranslated"

    # 6. Орфография: высокий OOV в спане
    if oov_ratio > 0.3 and _has_spelling_issue(text):
        return "Fluency/Spelling"

    # 7. Согласование: sentence-level признак сигнализирует об ошибках,
    #    и спан содержит глагол/прилагательное (косвенная проверка)
    if agreement_errors > 0 and _looks_like_agreement_error(text, word_logprobs_span):
        return "Fluency/Agreement"

    # 8. Fluency/LexicalChoice: очень низкий logprob → неестественная лексика
    #    ИСПРАВЛЕНИЕ: порог изменён с -8.0 на -6.0
    if _is_low_fluency(word_logprobs_span):
        return "Fluency/LexicalChoice"

    # 9. Дефолт — искажение смысла
    return "Accuracy/Mistranslation"


def _looks_untranslated(span_text: str) -> bool:
    """Возвращает True если спан содержит только латиницу без кириллицы."""
    text = span_text.strip()
    if not text:
        return False
    has_latin = bool(_RE_LATIN.search(text))
    has_cyr   = bool(_RE_CYRILLIC.search(text))
    # Только если есть латиница И нет кириллицы вообще
    return has_latin and not has_cyr


def _has_spelling_issue(span_text: str) -> bool:
    """
    Эвристика: спан выглядит как орфографическая ошибка.
    Длинное слово без дефиса и без пробелов (слитное написание)
    или слово с нетипичным сочетанием букв.
    """
    words = span_text.split()
    for w in words:
        # Очень длинное слово без дефиса — подозрительно
        if len(w) > 20 and "-" not in w:
            return True
    return False


def _looks_like_agreement_error(
    span_text: str,
    word_logprobs_span: Sequence[float],
) -> bool:
    """
    Эвристика: спан вероятно содержит ошибку согласования.
    Используем logprob как прокси: если среднее умеренно плохое
    (не экстремальное, т.к. экстремальное → LexicalChoice),
    и sentence_features уже зафиксировал agreement_errors > 0.
    """
    if not word_logprobs_span:
        return False
    arr = np.asarray(word_logprobs_span, dtype=np.float32)
    mean_lp = float(arr.mean())
    # Умеренно плохой logprob: хуже нормального, но лучше порога LexicalChoice
    return -6.0 < mean_lp < -4.0


def _is_low_fluency(word_logprobs_span: Sequence[float]) -> bool:
    """
    Возвращает True если среднее logprob спана ниже порога.

    ИСПРАВЛЕНИЕ: порог изменён с -8.0 на _LOGPROB_FLUENCY_THRESHOLD (-6.0).
    При необходимости откалибровать на val WMT21 по percentile BAD-слов.
    """
    if not word_logprobs_span:
        return False
    arr = np.asarray(word_logprobs_span, dtype=np.float32)
    mean_lp = float(arr.mean())
    return mean_lp < _LOGPROB_FLUENCY_THRESHOLD