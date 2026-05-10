"""
tests/test_sentence_model.py

Запуск:
    python -m tests.test_sentence_model
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.sentence_model import (
    FEATURE_NAMES,
    FEATURE_TO_MQM,
    SentenceModel,
    SentencePrediction,
    _aggregate_shap,
    _beta_stats,
    _xgboost_uncertainty,
)

XGBOOST_PATH = Path("models/xgboost_sentence.model")   # изменено с .pkl на .model
EXPLAINER_PATH = Path("models/shap_explainer.pkl")

DUMMY_FEATURES = np.array([
    1.05, 1.0, 1.0, 12.0, 13.0,
    1.0, 0.95, 0.0, 0.0,
    0.02, 0.65, 5.5, 0.9, 0.0, 4.0, 0.3,
    0.88, 0.35,
    45.0, -3.8, 0.5, -6.2,
], dtype=np.float32)

assert len(DUMMY_FEATURES) == 22


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_feature_names_length():
    assert len(FEATURE_NAMES) == 22
    print("OK  FEATURE_NAMES: 22 признака")


def test_feature_to_mqm_coverage():
    for name in FEATURE_NAMES:
        assert name in FEATURE_TO_MQM, f"Признак '{name}' отсутствует в FEATURE_TO_MQM"
    print("OK  FEATURE_TO_MQM: все 22 признака покрыты")


def test_beta_stats_basic():
    score, uncertainty, ci_low, ci_high = _beta_stats(2.0, 2.0)
    assert abs(score - 0.5) < 1e-6
    assert uncertainty > 0
    assert 0.0 <= ci_low < score < ci_high <= 1.0
    print(f"OK  _beta_stats(2,2): score={score:.3f} unc={uncertainty:.4f} CI=[{ci_low:.3f},{ci_high:.3f}]")


def test_xgboost_uncertainty_bounds():
    for s in [0.1, 0.5, 0.9]:
        unc, ci_low, ci_high = _xgboost_uncertainty(s)
        assert unc >= 0
        assert 0.0 <= ci_low <= ci_high <= 1.0
    print("OK  _xgboost_uncertainty: bounds корректны для 0.1, 0.5, 0.9")


def test_aggregate_shap_categories():
    agg = _aggregate_shap(np.ones(22, dtype=np.float32))
    assert set(agg.keys()) == {"Accuracy", "Fluency", "Terminology", "Locale", "Style"}
    assert abs(sum(agg.values()) - 20.0) < 1e-5
    print(f"OK  _aggregate_shap: {list(agg.keys())} сумма={sum(agg.values()):.1f}")


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_xgboost_predict():
    model = SentenceModel(XGBOOST_PATH, EXPLAINER_PATH)
    pred = model.predict(DUMMY_FEATURES)

    assert isinstance(pred, SentencePrediction)
    assert 0.0 <= pred.score <= 1.0, f"score вне [0,1]: {pred.score}"
    assert pred.uncertainty >= 0
    assert 0.0 <= pred.ci_low <= pred.ci_high <= 1.0
    assert pred.alpha is None
    assert pred.beta_param is None
    assert len(pred.shap_values) == 22
    assert len(pred.explanation) > 0

    print(f"OK  XGBoost predict:")
    print(f"    score={pred.score:.4f}  unc={pred.uncertainty:.6f}  CI=[{pred.ci_low:.3f},{pred.ci_high:.3f}]")
    print(f"    explanation={pred.explanation}")


def test_xgboost_predict_batch():
    model = SentenceModel(XGBOOST_PATH, EXPLAINER_PATH)

    bad = DUMMY_FEATURES.copy()
    bad[16] = 0.3    # cosine_similarity низкое
    bad[0]  = 0.4    # length_ratio — omission
    bad[18] = 150.0  # perplexity высокое

    preds = model.predict_batch(np.stack([DUMMY_FEATURES, bad]))

    assert len(preds) == 2
    assert preds[0].score > preds[1].score, (
        f"Хороший перевод должен иметь score выше: {preds[0].score:.3f} vs {preds[1].score:.3f}"
    )
    print(f"OK  XGBoost predict_batch: хороший={preds[0].score:.4f}  плохой={preds[1].score:.4f}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Unit tests")
    print("=" * 60)
    test_feature_names_length()
    test_feature_to_mqm_coverage()
    test_beta_stats_basic()
    test_xgboost_uncertainty_bounds()
    test_aggregate_shap_categories()

    print()
    print("=" * 60)
    print("Integration tests")
    print("=" * 60)
    test_xgboost_predict()
    test_xgboost_predict_batch()

    print()
    print("✅ Все тесты пройдены")