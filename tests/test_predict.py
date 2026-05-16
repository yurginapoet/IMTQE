import sys
import importlib
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_predict_module():
    sys.modules.pop("src.predict", None)

    extractor_mod = ModuleType("src.features.extractor")
    extractor_mod.FeatureExtractor = object

    overall_mod = ModuleType("src.interpretation.overall")
    overall_mod.OverallSentenceEvaluator = object
    overall_mod.OverallSentenceResult = object

    rules_mod = ModuleType("src.interpretation.rules")
    rules_mod.describe_error_type_ru = lambda error_type: error_type

    sentence_mod = ModuleType("src.models.sentence_model")
    sentence_mod.SentenceModel = object
    class DummySentencePrediction:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    sentence_mod.SentencePrediction = DummySentencePrediction
    sentence_mod.SUPPORTED_MODEL_TYPES = ("xgboost", "ridge", "rf")
    sentence_mod.FEATURE_TO_MQM = {"f0": "Accuracy", "f1": "Locale"}
    sentence_mod.MQM_CATEGORY_RU = {
        "Accuracy": "Accuracy",
        "Locale": "Locale",
    }
    sentence_mod.resolve_sentence_artifacts = lambda models_dir, model_name="xgboost": (
        Path(models_dir) / f"{model_name}.model",
        Path(models_dir) / f"{model_name}.explainer.pkl",
    )

    span_mod = ModuleType("src.models.span_model")
    span_mod.SpanModel = object
    span_mod.SpanPrediction = object

    sys.modules["src.features.extractor"] = extractor_mod
    sys.modules["src.interpretation.overall"] = overall_mod
    sys.modules["src.interpretation.rules"] = rules_mod
    sys.modules["src.models.sentence_model"] = sentence_mod
    sys.modules["src.models.span_model"] = span_mod

    return importlib.import_module("src.predict")


def test_predictor_requires_full_feature_set_for_sentence_model(tmp_path, monkeypatch):
    predict_mod = load_predict_module()

    class DummySentenceModel:
        def __init__(self, *_args, **_kwargs):
            self.expected_feature_count = 33
            self.feature_names = [f"f{i}" for i in range(33)]

    class DummyFeatureExtractor:
        require_neural_calls = []

        def __init__(self, *_args, **_kwargs):
            self.active_feature_names = [f"f{i}" for i in range(22)]

        def load_heavy_models(self, require_neural: bool = False):
            self.require_neural_calls.append(require_neural)

    monkeypatch.setattr(predict_mod, "SentenceModel", DummySentenceModel)
    monkeypatch.setattr(predict_mod, "FeatureExtractor", DummyFeatureExtractor)
    monkeypatch.setattr(predict_mod, "SpanModel", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(predict_mod, "OverallSentenceEvaluator", lambda *_args, **_kwargs: object())

    with pytest.raises(RuntimeError, match="нужно 33, доступно 22"):
        predict_mod.Predictor(models_dir=tmp_path)

    assert DummyFeatureExtractor.require_neural_calls == [False]


def test_predict_sentence_passes_extractor_tokens_to_span_model(tmp_path, monkeypatch):
    predict_mod = load_predict_module()
    mt_words = ["Привет", ",", "мир", "!"]

    class DummyFeatureExtractor:
        def __init__(self, *_args, **_kwargs):
            self.active_feature_names = [f"f{i}" for i in range(33)]

        def load_heavy_models(self, require_neural: bool = False):
            self.require_neural = require_neural

        def extract(self, src: str, mt: str) -> dict:
            return {
                "vector": np.zeros(33, dtype=np.float32),
                "formal_ratio": 0.0,
                "word_logprobs": [-1.0, -2.0, -3.0, -4.0],
                "mt_words": mt_words,
                "raw": {"formal_ratio": 0.0},
            }

    class DummySentenceModel:
        def __init__(self, *_args, **_kwargs):
            self.expected_feature_count = 33
            self.feature_names = ["f0", "f1"]

        def predict(self, _features):
            return SimpleNamespace(
                score=0.8,
                uncertainty=0.02,
                ci_low=0.7,
                ci_high=0.9,
                alpha=None,
                beta_param=None,
                shap_values=np.array([0.25, -0.15], dtype=np.float32),
                explanation={"Accuracy": 0.4},
            )

    class DummySpanModel:
        def __init__(self, *_args, **_kwargs):
            self.last_call = None

        def predict(self, src: str, mt: str, word_logprobs=None, mt_words=None):
            self.last_call = {
                "src": src,
                "mt": mt,
                "word_logprobs": word_logprobs,
                "mt_words": list(mt_words) if mt_words is not None else None,
            }
            return SimpleNamespace(word_labels=["OK"] * 4, word_probs=[0.0] * 4, spans=[])

    class DummyOverallSentenceEvaluator:
        def __init__(self, *_args, **_kwargs):
            pass

        def evaluate(self, sentence_pred, span_pred, mt_words, sentence_features=None):
            return SimpleNamespace(
                sentence_score=sentence_pred.score,
                uncertainty=sentence_pred.uncertainty,
                ci_low=sentence_pred.ci_low,
                ci_high=sentence_pred.ci_high,
                explanation=sentence_pred.explanation,
                mqm=SimpleNamespace(mqm_score=0.76),
                spans=[
                    SimpleNamespace(
                        start_idx=0,
                        end_idx=1,
                        severity="BAD-minor",
                        error_type="Locale/Quotes",
                        confidence=0.61,
                    )
                ],
            )

    monkeypatch.setattr(predict_mod, "FeatureExtractor", DummyFeatureExtractor)
    monkeypatch.setattr(predict_mod, "SentenceModel", DummySentenceModel)
    monkeypatch.setattr(predict_mod, "SpanModel", DummySpanModel)
    monkeypatch.setattr(predict_mod, "OverallSentenceEvaluator", DummyOverallSentenceEvaluator)

    predictor = predict_mod.Predictor(models_dir=tmp_path)
    result = predictor.predict_sentence("Hello, world!", "Привет, мир!")

    assert predictor.span_model.last_call["mt_words"] == mt_words
    assert predictor.span_model.last_call["word_logprobs"] == [-1.0, -2.0, -3.0, -4.0]
    assert result.errors[0].span_text == "Привет ,"
    assert result.debug["shap_values"]["f0"] == pytest.approx(0.25)
    assert result.debug["shap_values"]["f1"] == pytest.approx(-0.15)
    assert result.model_scores["ensemble"] == pytest.approx(0.8)
    assert result.debug["models"]["ensemble"]["score"] == pytest.approx(0.8)
    assert result.debug["span_penalty"]["has_bad_label"] is False
