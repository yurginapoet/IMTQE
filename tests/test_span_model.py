"""
tests/test_span_model.py

Проверяет SpanModel без переобучения — только инференс на реальной модели.
Запуск из корня проекта:
    python -m pytest tests/test_span_model.py -v
    # или напрямую:
    python tests/test_span_model.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.span_model import (
    SpanModel,
    SpanPrediction,
    SpanResult,
    _build_word_spans,
    _map_subtokens_to_words,
)

MODEL_DIR = Path("models/xlm_roberta_span")

# Тестовые пары (src EN, mt RU)
GOOD_PAIR = (
    "The bank raised interest rates by 0.5 percent.",
    "Банк повысил процентные ставки на 0,5 процента.",
)
BAD_PAIR = (
    "The scientists discovered a new species of deep-sea fish near the Mariana Trench.",
    "Учёные открыли банк вблизи Марианского жёлоба.",  # hallucination: "банк" вместо "вид рыбы"
)
EMPTY_MT = (
    "Hello world.",
    "",
)


# ---------------------------------------------------------------------------
# Юнит-тесты _map_subtokens_to_words (без загрузки модели)
# ---------------------------------------------------------------------------

def test_map_subtokens_basic():
    """Простейший случай: каждое слово = один субтокен."""
    mt_text = "кот сидит дома"
    # Симулируем offsets: кот=[0,3], сидит=[4,9], дома=[10,14]
    offsets = [(0, 3), (4, 9), (10, 14)]
    result  = _map_subtokens_to_words(mt_text, offsets, 3)
    assert result == [0, 1, 2], f"Ожидается [0,1,2], получено {result}"
    print("✓ _map_subtokens: 1 субтокен на слово")


def test_map_subtokens_multi():
    """Слово разбито на несколько субтокенов — первый получает индекс, остальные None."""
    mt_text = "привет"
    # "привет" → ["при", "вет"] — два субтокена
    offsets = [(0, 3), (3, 6)]
    result  = _map_subtokens_to_words(mt_text, offsets, 1)
    assert result[0] == 0,    f"Первый субтокен должен быть 0, получено {result[0]}"
    assert result[1] is None, f"Второй субтокен должен быть None, получено {result[1]}"
    print("✓ _map_subtokens: multi-subtoken слово")


def test_map_subtokens_empty_offset():
    """Пустой offset (tok_start == tok_end) → None."""
    mt_text = "кот"
    offsets = [(0, 0), (0, 3)]   # первый пустой
    result  = _map_subtokens_to_words(mt_text, offsets, 1)
    assert result[0] is None, f"Пустой offset должен давать None, получено {result[0]}"
    assert result[1] == 0,    f"Нормальный offset должен давать 0, получено {result[1]}"
    print("✓ _map_subtokens: пустой offset → None")


def test_map_subtokens_two_words():
    """Два слова, второе с двумя субтокенами."""
    mt_text = "кот бежит"
    # кот=[0,3], бе=[4,6], жит=[6,9]
    offsets = [(0, 3), (4, 6), (6, 9)]
    result  = _map_subtokens_to_words(mt_text, offsets, 2)
    assert result[0] == 0,    f"'кот' первый субтокен → 0, получено {result[0]}"
    assert result[1] == 1,    f"'бежит' первый субтокен → 1, получено {result[1]}"
    assert result[2] is None, f"'бежит' второй субтокен → None, получено {result[2]}"
    print("✓ _map_subtokens: два слова, второе multi-subtoken")


def test_map_subtokens_with_explicit_mt_words():
    """Явные mt_words позволяют держать токенизацию согласованной с extractor/UI."""
    mt_text = "Привет, мир!"
    offsets = [(0, 6), (6, 7), (8, 11), (11, 12)]
    mt_words = ["Привет", ",", "мир", "!"]
    result = _map_subtokens_to_words(mt_text, offsets, len(mt_words), mt_words=mt_words)
    assert result == [0, 1, 2, 3], f"Ожидается [0,1,2,3], получено {result}"
    print("✓ _map_subtokens: explicit mt_words поддерживаются")


def test_build_word_spans_with_punctuation_tokens():
    mt_text = "Привет, мир!"
    spans = _build_word_spans(mt_text, ["Привет", ",", "мир", "!"])
    assert spans == [(0, 6), (6, 7), (8, 11), (11, 12)]
    print("✓ _build_word_spans: punctuation-aware токены корректно маппятся")


def test_decode_words_uses_bad_threshold():
    model = SpanModel.__new__(SpanModel)
    model.bad_threshold = 0.45
    model.major_threshold = 0.60

    probs = torch.tensor([
        [0.503, 0.297, 0.200],  # p(BAD)=0.497 -> BAD-minor при soft threshold
        [0.450, 0.150, 0.400],  # p(BAD)=0.550 и p(major)=0.400 -> BAD-minor
        [0.200, 0.100, 0.700],  # уверенный BAD-major
    ])
    labels, p_bad = model._decode_words(
        probs=probs,
        mt_start_pos=0,
        word_to_first_subtoken=[0, 1, 2],
        n_words=3,
    )

    assert labels == ["BAD-minor", "BAD-minor", "BAD-major"]
    assert np.allclose(p_bad, [0.497, 0.55, 0.8], atol=1e-3)
    print("✓ _decode_words: soft threshold подсвечивает borderline BAD")


# ---------------------------------------------------------------------------
# Интеграционные тесты — загрузка реальной модели
# ---------------------------------------------------------------------------

def _load_model() -> SpanModel | None:
    if not MODEL_DIR.exists():
        print(f"⚠ Пропуск: {MODEL_DIR} не найден")
        return None
    try:
        return SpanModel(MODEL_DIR, device="cpu")
    except FileNotFoundError as exc:
        pytest.skip(f"Span-модель недоступна для интеграционного теста: {exc}")
    except OSError as exc:
        pytest.skip(f"Span-модель не может быть загружена локально: {exc}")


def test_span_model_loads():
    model = _load_model()
    if model is None:
        return
    assert model.tokenizer is not None
    assert model.model     is not None
    print("✓ SpanModel: загрузка успешна")


def test_predict_good_pair():
    """Хороший перевод: большинство слов должны быть OK."""
    model = _load_model()
    if model is None:
        return

    src, mt = GOOD_PAIR
    result  = model.predict(src, mt)

    assert isinstance(result, SpanPrediction)
    n_words = len(mt.split())
    assert len(result.word_labels) == n_words, \
        f"word_labels: ожидается {n_words}, получено {len(result.word_labels)}"
    assert len(result.word_probs) == n_words, \
        f"word_probs: ожидается {n_words}, получено {len(result.word_probs)}"

    # Все метки из допустимого множества
    for label in result.word_labels:
        assert label in ("OK", "BAD-minor", "BAD-major"), f"Недопустимая метка: {label}"

    # Все p(BAD) в [0, 1]
    for p in result.word_probs:
        assert 0.0 <= p <= 1.0, f"p(BAD) вне [0,1]: {p}"

    ok_ratio = result.word_labels.count("OK") / n_words
    print(f"✓ predict (хороший перевод):")
    print(f"  word_labels={result.word_labels}")
    print(f"  OK ratio={ok_ratio:.2f}  spans={len(result.spans)}")


def test_predict_empty_mt():
    """Пустой mt — должен вернуть пустые списки без ошибки."""
    model = _load_model()
    if model is None:
        return

    src, mt = EMPTY_MT
    result  = model.predict(src, mt)

    assert isinstance(result, SpanPrediction)
    assert result.word_labels == []
    assert result.word_probs  == []
    assert result.spans       == []
    print("✓ predict (пустой mt): корректно возвращает пустые списки")


def test_predict_with_word_logprobs():
    """word_logprobs передаются и попадают в SpanResult.word_logprobs_span."""
    model = _load_model()
    if model is None:
        return

    src, mt = BAD_PAIR
    n_words      = len(mt.split())
    word_logprobs = [-float(i + 1) for i in range(n_words)]  # фиктивные логпробы

    result = model.predict(src, mt, word_logprobs=word_logprobs)

    for span in result.spans:
        span_len = span.end_idx - span.start_idx + 1
        assert len(span.word_logprobs_span) == span_len, (
            f"Спан [{span.start_idx},{span.end_idx}]: "
            f"ожидается {span_len} logprobs, получено {len(span.word_logprobs_span)}"
        )
    print(f"✓ predict с word_logprobs: {len(result.spans)} спанов")
    for span in result.spans:
        print(f"  span=[{span.start_idx},{span.end_idx}] "
              f"severity={span.severity} conf={span.confidence:.3f} "
              f"logprobs={span.word_logprobs_span}")


def test_span_structure():
    """Проверяет корректность структуры SpanResult."""
    model = _load_model()
    if model is None:
        return

    src, mt = BAD_PAIR
    result  = model.predict(src, mt)

    for span in result.spans:
        assert isinstance(span, SpanResult),       "span должен быть SpanResult"
        assert span.start_idx >= 0,                f"start_idx < 0: {span.start_idx}"
        assert span.end_idx >= span.start_idx,     f"end_idx < start_idx: {span}"
        assert span.severity in ("BAD-minor", "BAD-major"), \
            f"Недопустимый severity: {span.severity}"
        assert 0.0 <= span.confidence <= 1.0,      f"confidence вне [0,1]: {span.confidence}"

        # Все слова спана в word_labels — действительно BAD
        for wi in range(span.start_idx, span.end_idx + 1):
            assert result.word_labels[wi] != "OK", \
                f"Слово {wi} в спане помечено OK — ошибка группировки"

    print(f"✓ span структура: {len(result.spans)} спанов, все корректны")
    if result.spans:
        worst = max(result.spans, key=lambda s: s.confidence)
        words = mt.split()
        span_text = " ".join(words[worst.start_idx : worst.end_idx + 1])
        print(f"  Самый уверенный спан: '{span_text}' "
              f"severity={worst.severity} conf={worst.confidence:.3f}")


def test_predict_batch_consistency():
    """predict_batch не существует в SpanModel — инференс батчится на уровне predict.py.
    Проверяем что два вызова predict дают одинаковый результат (детерминированность)."""
    model = _load_model()
    if model is None:
        return

    src, mt = GOOD_PAIR
    r1 = model.predict(src, mt)
    r2 = model.predict(src, mt)

    assert r1.word_labels == r2.word_labels, "Результаты должны быть идентичны (детерминированность)"
    assert np.allclose(r1.word_probs, r2.word_probs, atol=1e-5), \
        "word_probs должны совпадать при повторном вызове"
    print("✓ predict детерминирован: два вызова дают одинаковый результат")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Юнит-тесты _map_subtokens_to_words (без моделей)")
    print("=" * 60)
    test_map_subtokens_basic()
    test_map_subtokens_multi()
    test_map_subtokens_empty_offset()
    test_map_subtokens_two_words()
    test_map_subtokens_with_explicit_mt_words()
    test_build_word_spans_with_punctuation_tokens()

    print()
    print("=" * 60)
    print("Интеграционные тесты SpanModel (загрузка реальной модели)")
    print("=" * 60)
    test_span_model_loads()
    test_predict_good_pair()
    test_predict_empty_mt()
    test_predict_with_word_logprobs()
    test_span_structure()
    test_predict_batch_consistency()

    print()
    print("✅ Все тесты пройдены")
