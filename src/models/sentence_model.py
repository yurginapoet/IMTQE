"""
src/models/sentence_model.py

Инференс-обёртка над обученной sentence-level моделью (XGBoost или NGBoost).
Загружает модель из .pkl (NGBoost) или из .model (XGBoost).
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
from scipy.stats import beta as scipy_beta

# Импорт xgboost будет только при необходимости, но для type hints
try:
    import xgboost as xgb
except ImportError:
    xgb = None

log = logging.getLogger(__name__)

# ------------------------------------------------------------
# Маппинг признак → категория MQM (из архитектуры, раздел 5.5)
# ------------------------------------------------------------

FEATURE_TO_MQM: Dict[str, str] = {
    # Structural
    "length_ratio":         "Accuracy",
    "abs_length_diff":      "Accuracy",
    "token_count_diff":     "Accuracy",
    "src_length":           None,
    "mt_length":            None,
    # Formatting
    "digit_match_ratio":    "Locale",
    "punct_ratio":          "Locale",
    "quotes_mismatch":      "Locale",
    "date_format_error":    "Locale",
    # Linguistic
    "oov_ratio":            "Fluency",
    "type_token_ratio":     "Fluency",
    "avg_token_length":     "Fluency",
    "entity_overlap_ratio": "Terminology",
    "agreement_errors":     "Fluency",
    "syntax_depth":         "Fluency",
    "formal_ratio":         "Style",
    # Semantic
    "cosine_similarity":    "Accuracy",
    "embedding_distance":   "Accuracy",
    # Fluency (ruGPT-3)
    "perplexity":           "Fluency",
    "mean_log_prob":        "Fluency",
    "token_ppl_variance":   "Fluency",
    "min_token_log_prob":   "Fluency",
}

FEATURE_NAMES = [
    "length_ratio", "abs_length_diff", "token_count_diff", "src_length", "mt_length",
    "digit_match_ratio", "punct_ratio", "quotes_mismatch", "date_format_error",
    "oov_ratio", "type_token_ratio", "avg_token_length", "entity_overlap_ratio",
    "agreement_errors", "syntax_depth", "formal_ratio",
    "cosine_similarity", "embedding_distance",
    "perplexity", "mean_log_prob", "token_ppl_variance", "min_token_log_prob",
]
assert len(FEATURE_NAMES) == 22, f"Ожидается 22 признака, получено {len(FEATURE_NAMES)}"


@dataclass
class SentencePrediction:
    score:       float
    uncertainty: float
    ci_low:      float
    ci_high:     float
    alpha:       Optional[float]
    beta_param:  Optional[float]
    shap_values: np.ndarray
    explanation: Dict[str, float] = field(default_factory=dict)


class SentenceModel:
    """
    Загружает модель: .pkl (NGBoost) или .model (XGBoost).
    Также загружает SHAP explainer (pickle).
    """

    def __init__(self, model_path: Union[str, Path], explainer_path: Union[str, Path]):
        model_path = Path(model_path)
        explainer_path = Path(explainer_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        if not explainer_path.exists():
            raise FileNotFoundError(f"SHAP explainer не найден: {explainer_path}")

        # Загрузка SHAP (всегда pickle)
        with open(explainer_path, "rb") as f:
            self._explainer = pickle.load(f)

        # Загрузка модели в зависимости от расширения
        if model_path.suffix == ".model":
            # XGBoost в бинарном формате
            if xgb is None:
                raise ImportError("xgboost не установлен, но требуется для .model файла")
            self._model = xgb.XGBRegressor()
            self._model.load_model(str(model_path))
            self._model_type = "xgboost"
            log.info("Загружена XGBoost модель из %s", model_path)
        else:
            # Предполагаем pickle (NGBoost или старый XGBoost)
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
            cls_name = type(self._model).__name__
            if "NGBRegressor" in cls_name:
                self._model_type = "ngboost"
            else:
                self._model_type = "xgboost"
            log.info("Загружена модель типа %s из %s", self._model_type, model_path)

    # ------------------------------------------------------------
    # Публичные методы
    # ------------------------------------------------------------

    def predict(self, features: np.ndarray) -> SentencePrediction:
        if features.ndim == 1:
            X = features.reshape(1, -1)
        else:
            X = features

        if self._model_type == "ngboost":
            return self._predict_ngboost(X)[0]
        else:
            return self._predict_xgboost(X)[0]

    def predict_batch(self, features: np.ndarray) -> list[SentencePrediction]:
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if self._model_type == "ngboost":
            return self._predict_ngboost(features)
        else:
            return self._predict_xgboost(features)

    # ------------------------------------------------------------
    # NGBoost (не используется, но оставлено для совместимости)
    # ------------------------------------------------------------

    def _predict_ngboost(self, X: np.ndarray) -> list[SentencePrediction]:
        dist = self._model.pred_dist(X)
        alphas = dist.params["alpha"]
        betas = dist.params["beta"]
        shap_vals = self._shap_values(X)

        results = []
        for i in range(len(alphas)):
            a, b = float(alphas[i]), float(betas[i])
            score, unc, lo, hi = _beta_stats(a, b)
            sv = shap_vals[i] if shap_vals.ndim == 2 else shap_vals
            results.append(SentencePrediction(
                score=score, uncertainty=unc,
                ci_low=lo, ci_high=hi,
                alpha=a, beta_param=b,
                shap_values=sv, explanation=_aggregate_shap(sv)
            ))
        return results

    # ------------------------------------------------------------
    # XGBoost – основная логика
    # ------------------------------------------------------------

    def _predict_xgboost(self, X: np.ndarray) -> list[SentencePrediction]:
        scores = np.clip(self._model.predict(X), 0.0, 1.0)
        shap_vals = self._shap_values(X)

        results = []
        for i, sc in enumerate(scores):
            unc, lo, hi = _xgboost_uncertainty(float(sc))
            sv = shap_vals[i] if shap_vals.ndim == 2 else shap_vals
            results.append(SentencePrediction(
                score=float(sc), uncertainty=unc,
                ci_low=lo, ci_high=hi,
                alpha=None, beta_param=None,
                shap_values=sv, explanation=_aggregate_shap(sv)
            ))
        return results

    # ------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------

    def _shap_values(self, X: np.ndarray) -> np.ndarray:
        try:
            vals = self._explainer.shap_values(X)
            arr = np.array(vals)
            if arr.ndim == 2 and X.shape[0] == 1:
                return arr[0]   # (22,)
            if arr.ndim == 1 and X.shape[0] == 1:
                return arr       # (22,)
            return arr
        except Exception as e:
            log.warning("SHAP не удался: %s", e)
            n = X.shape[0]
            return np.zeros((n, 22) if n > 1 else (22,), dtype=np.float32)


# ------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------

def _beta_stats(alpha: float, beta_p: float):
    s = alpha + beta_p
    score = alpha / s
    uncertainty = (alpha * beta_p) / (s * s * (s + 1))
    ci_low = float(scipy_beta.ppf(0.025, alpha, beta_p))
    ci_high = float(scipy_beta.ppf(0.975, alpha, beta_p))
    return score, uncertainty, ci_low, ci_high

def _xgboost_uncertainty(score: float):
    concentration = 10.0
    a = max(score * concentration, 1e-4)
    b = max((1.0 - score) * concentration, 1e-4)
    _, unc, lo, hi = _beta_stats(a, b)
    return unc, lo, hi

def _aggregate_shap(shap_vals: np.ndarray) -> Dict[str, float]:
    cat = {}
    for i, name in enumerate(FEATURE_NAMES):
        if i >= len(shap_vals):
            break
        cat_name = FEATURE_TO_MQM.get(name)
        if cat_name:
            cat[cat_name] = cat.get(cat_name, 0.0) + float(shap_vals[i])
    return cat