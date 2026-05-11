from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "train_sentence_model.py"
SPEC = spec_from_file_location("train_sentence_model_script", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
train_sentence_model = module_from_spec(SPEC)
SPEC.loader.exec_module(train_sentence_model)

_PearsonCallback = train_sentence_model._PearsonCallback
_build_train_sample_weights = train_sentence_model._build_train_sample_weights


def test_build_train_sample_weights_downweights_synthetic_rows() -> None:
    train_df = pd.DataFrame(
        {
            "score_norm": [0.80, 0.20, 0.30, 0.90],
            "is_synthetic": [False, True, True, False],
        }
    )

    weights = _build_train_sample_weights(
        train_df,
        synthetic_weight=0.1,
        low_score_tau=0.5,
        low_score_weight=3.0,
    )

    assert np.allclose(weights, np.array([1.0, 0.3, 0.3, 1.0], dtype=np.float32))


def test_pearson_callback_stops_only_after_patience() -> None:
    X_val = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y_val = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    callback = _PearsonCallback(X_val, y_val, patience=2)

    class DummyModel:
        def __init__(self, preds: np.ndarray) -> None:
            self._preds = preds

        def predict(self, _dval):
            return self._preds

    improving_preds = np.array([0.2, 0.6, 1.0], dtype=np.float32)

    assert callback.after_iteration(DummyModel(improving_preds), 0, {}) is False
    assert callback.no_improve == 0

    assert callback.after_iteration(DummyModel(improving_preds), 1, {}) is False
    assert callback.no_improve == 1

    assert callback.after_iteration(DummyModel(improving_preds), 2, {}) is True
    assert callback.no_improve == 2
