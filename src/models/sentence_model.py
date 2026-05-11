"""
Инференс sentence-level модели (XGBoost) и SHAP explainer.

Поддерживает sentence-вектор с semantic PCA, классику без PCA, старые размерности.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import xgboost as xgb
from scipy.stats import beta as scipy_beta

from src.features.schema import (
    FEATURE_NAMES as FULL_FEATURE_NAMES,
    FEATURE_NAMES_CLASSIC as LEGACY_FEATURE_NAMES,
    FEATURE_NAMES_LIGHT,
    SEMANTIC_FEATURE_NAMES,
    SENTENCE_FEATURE_NAMES,
    SENTENCE_FEATURE_NAMES_CLASSIC,
)

log = logging.getLogger(__name__)

FEATURE_NAMES = FULL_FEATURE_NAMES

FEATURE_TO_MQM: Dict[str, str | None] = {
    "length_ratio": "Accuracy",
    "abs_length_diff": "Accuracy",
    "token_count_diff": "Accuracy",
    "src_length": None,
    "mt_length": None,
    "digit_match_ratio": "Locale",
    "punct_ratio": "Locale",
    "quotes_mismatch": "Locale",
    "date_format_error": "Locale",
    "oov_ratio": "Fluency",
    "type_token_ratio": "Fluency",
    "avg_token_length": "Fluency",
    "entity_overlap_ratio": "Terminology",
    "agreement_errors": "Fluency",
    "syntax_depth": "Fluency",
    "formal_ratio": "Style",
    "cosine_similarity": "Accuracy",
    "embedding_distance": "Accuracy",
    "perplexity": "Fluency",
    "mean_log_prob": "Fluency",
    "token_ppl_variance": "Fluency",
    "min_token_log_prob": "Fluency",
}
FEATURE_TO_MQM.update({name: "Semantic" for name in SEMANTIC_FEATURE_NAMES})
FEATURE_TO_MQM.update(
    {
        "cosine_x_length_ok": "Accuracy",
        "log_perplexity": "Fluency",
        "cosine_per_logppl": "Accuracy",
        "entity_x_cosine": "Terminology",
        "oov_x_bad_cosine": "Fluency",
        "logprob_spike": "Fluency",
        "variance_x_bad_cosine": "Fluency",
        "normed_length_diff": "Accuracy",
        "digit_x_entity": "Locale",
        "formal_x_cosine": "Style",
        "dist_x_logppl": "Accuracy",
        "xgb_score": "EnsembleBase",
    }
)

MQM_CATEGORY_RU: Dict[str, str] = {
    "Accuracy": "Точность (смысл)",
    "Fluency": "Грамотность/плавность",
    "Terminology": "Терминология",
    "Locale": "Локаль/форматирование",
    "Style": "Стиль/регистр",
    "Semantic": "Семантическая согласованность",
    "EnsembleBase": "База градиентного бустинга",
    "Other": "Прочее",
}


@dataclass
class SentencePrediction:
    score: float
    uncertainty: float
    ci_low: float
    ci_high: float
    alpha: Optional[float]
    beta_param: Optional[float]
    shap_values: np.ndarray
    explanation: Dict[str, float] = field(default_factory=dict)


class SentenceModel:
    def __init__(self, model_path: Union[str, Path], explainer_path: Union[str, Path]):
        model_path = Path(model_path)
        explainer_path = Path(explainer_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        if not explainer_path.exists():
            raise FileNotFoundError(f"SHAP explainer не найден: {explainer_path}")

        self._explainer_feature_names: list[str] | None = None
        try:
            with open(explainer_path, "rb") as f:
                loaded_expl = pickle.load(f)
        except ModuleNotFoundError as exc:
            log.warning(
                "Не удалось загрузить SHAP explainer (%s). Продолжаем без SHAP.",
                exc,
            )
            loaded_expl = None

        if isinstance(loaded_expl, dict) and "explainer" in loaded_expl:
            self._explainer = loaded_expl["explainer"]
            fn = loaded_expl.get("feature_names")
            self._explainer_feature_names = list(fn) if fn else None
        else:
            self._explainer = loaded_expl

        if model_path.suffix == ".model":
            self._model = xgb.XGBRegressor()
            self._model.load_model(str(model_path))
            log.info("Загружена XGBoost модель из %s", model_path)
        else:
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
            if not isinstance(self._model, xgb.XGBRegressor):
                raise TypeError(
                    "Поддерживается только XGBoost (файл .model или pickle XGBRegressor)."
                )
            log.info("Загружена XGBoost модель (pickle) из %s", model_path)

        self._expected_feature_count = self._infer_feature_count()
        if (
            self._explainer_feature_names is not None
            and len(self._explainer_feature_names) == self._expected_feature_count
        ):
            self._feature_names = list(self._explainer_feature_names)
        else:
            self._feature_names = _infer_feature_names(self._expected_feature_count)
        log.info("SentenceModel ожидает %d признаков", self._expected_feature_count)

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def expected_feature_count(self) -> int:
        return self._expected_feature_count

    def predict(self, features: np.ndarray) -> SentencePrediction:
        X = self._prepare_features(features)
        return self._predict_xgboost(X)[0]

    def predict_batch(self, features: np.ndarray) -> list[SentencePrediction]:
        X = self._prepare_features(features)
        return self._predict_xgboost(X)

    def _infer_feature_count(self) -> int:
        try:
            return int(self._model.get_booster().num_features())
        except Exception:
            pass

        for attr_name in ("n_features_in_", "n_features_", "n_features"):
            value = getattr(self._model, attr_name, None)
            if value is not None:
                return int(value)

        if self._explainer_feature_names is not None:
            return len(self._explainer_feature_names)

        feature_names = getattr(self._explainer, "feature_names", None)
        if feature_names:
            return len(feature_names)

        background_data = getattr(self._explainer, "data", None)
        if background_data is not None and getattr(background_data, "shape", None):
            return int(background_data.shape[1])

        return len(FULL_FEATURE_NAMES)

    def _prepare_features(self, features: np.ndarray) -> np.ndarray:
        X = features.reshape(1, -1) if features.ndim == 1 else features
        if X.shape[1] == self._expected_feature_count:
            return X
        if X.shape[1] > self._expected_feature_count:
            log.warning(
                "Получено %d признаков, модель ожидает %d — берём первые %d.",
                X.shape[1],
                self._expected_feature_count,
                self._expected_feature_count,
            )
            return X[:, : self._expected_feature_count]
        raise ValueError(
            f"Недостаточно признаков: нужно {self._expected_feature_count}, получено {X.shape[1]}"
        )

    def _predict_xgboost(self, X: np.ndarray) -> list[SentencePrediction]:
        scores = np.clip(self._model.predict(X), 0.0, 1.0)
        shap_vals = self._shap_values(X)

        results = []
        for idx, score in enumerate(scores):
            uncertainty, ci_low, ci_high = _xgboost_uncertainty(float(score))
            sv = shap_vals[idx] if shap_vals.ndim == 2 else shap_vals
            results.append(
                SentencePrediction(
                    score=float(score),
                    uncertainty=uncertainty,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    alpha=None,
                    beta_param=None,
                    shap_values=sv,
                    explanation=_aggregate_shap(sv, self._feature_names),
                )
            )
        return results

    def _shap_values(self, X: np.ndarray) -> np.ndarray:
        if self._explainer is None:
            n_rows = X.shape[0]
            shape = (n_rows, self._expected_feature_count) if n_rows > 1 else (self._expected_feature_count,)
            return np.zeros(shape, dtype=np.float32)
        try:
            values = self._explainer.shap_values(X)
            arr = np.array(values)
            if arr.ndim == 2 and X.shape[0] == 1:
                return arr[0]
            if arr.ndim == 1 and X.shape[0] == 1:
                return arr
            return arr
        except Exception as exc:
            log.warning("SHAP не удался: %s", exc)
            n_rows = X.shape[0]
            shape = (n_rows, self._expected_feature_count) if n_rows > 1 else (self._expected_feature_count,)
            return np.zeros(shape, dtype=np.float32)


def _infer_feature_names(count: int) -> list[str]:
    if count == len(SENTENCE_FEATURE_NAMES):
        return list(SENTENCE_FEATURE_NAMES)
    if count == len(SENTENCE_FEATURE_NAMES_CLASSIC):
        return list(SENTENCE_FEATURE_NAMES_CLASSIC)
    if count == len(FULL_FEATURE_NAMES):
        return list(FULL_FEATURE_NAMES)
    if count == len(LEGACY_FEATURE_NAMES):
        return list(LEGACY_FEATURE_NAMES)
    if count == len(FEATURE_NAMES_LIGHT):
        return list(FEATURE_NAMES_LIGHT)
    if count < len(FULL_FEATURE_NAMES):
        return list(FULL_FEATURE_NAMES[:count])
    extra = [f"feature_{idx:03d}" for idx in range(len(FULL_FEATURE_NAMES), count)]
    return list(FULL_FEATURE_NAMES) + extra


def _beta_stats(alpha: float, beta_p: float):
    total = alpha + beta_p
    score = alpha / total
    uncertainty = (alpha * beta_p) / (total * total * (total + 1))
    ci_low = float(scipy_beta.ppf(0.025, alpha, beta_p))
    ci_high = float(scipy_beta.ppf(0.975, alpha, beta_p))
    return score, uncertainty, ci_low, ci_high


def _xgboost_uncertainty(score: float):
    concentration = 10.0
    alpha = max(score * concentration, 1e-4)
    beta_p = max((1.0 - score) * concentration, 1e-4)
    _, uncertainty, ci_low, ci_high = _beta_stats(alpha, beta_p)
    return uncertainty, ci_low, ci_high


def _aggregate_shap(
    shap_vals: np.ndarray,
    feature_names: list[str] | None = None,
) -> Dict[str, float]:
    names = feature_names or FULL_FEATURE_NAMES
    aggregated: Dict[str, float] = {}
    for idx, name in enumerate(names):
        if idx >= len(shap_vals):
            break
        category = FEATURE_TO_MQM.get(name)
        if category:
            aggregated[category] = aggregated.get(category, 0.0) + float(shap_vals[idx])
    return aggregated
