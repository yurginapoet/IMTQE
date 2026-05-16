"""
Инференс sentence-level моделей и feature-level explainability.

Поддерживаются:
- XGBoost
- Ridge
- RandomForestRegressor
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
from scipy.stats import beta as scipy_beta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
try:
    import xgboost as xgb
except ModuleNotFoundError:  # pragma: no cover - optional in lightweight test env
    xgb = None

from src.features.schema import (
    FEATURE_NAMES_CLASSIC,
    FEATURE_NAMES_LIGHT,
    SENTENCE_FEATURE_NAMES,
)

log = logging.getLogger(__name__)

FEATURE_NAMES = FEATURE_NAMES_CLASSIC
LEGACY_FEATURE_NAMES = FEATURE_NAMES_LIGHT
SUPPORTED_MODEL_TYPES = ("xgboost", "ridge", "rf")

MODEL_ARTIFACTS: dict[str, dict[str, str]] = {
    "xgboost": {
        "model": "sentence_xgboost.model",
        "explainer": "sentence_xgboost_explainer.pkl",
        "legacy_model": "xgboost_sentence.model",
        "legacy_explainer": "shap_explainer.pkl",
    },
    "ridge": {
        "model": "sentence_ridge.pkl",
        "explainer": "sentence_ridge_explainer.pkl",
    },
    "rf": {
        "model": "sentence_rf.pkl",
        "explainer": "sentence_rf_explainer.pkl",
    },
}

FEATURE_TO_MQM: Dict[str, str | None] = {
    "length_ratio": "Accuracy",
    "abs_length_diff": "Accuracy",
    "token_count_diff": "Accuracy",
    "src_length": None,
    "mt_length": None,
    "compression_ratio": "Accuracy",
    "sentence_count_diff": "Accuracy",
    "digit_match_ratio": "Locale",
    "punct_ratio": "Locale",
    "quotes_mismatch": "Locale",
    "date_format_error": "Locale",
    "number_count_diff": "Locale",
    "capitalization_mismatch": "Style",
    "currency_symbol_mismatch": "Locale",
    "oov_ratio": "Fluency",
    "type_token_ratio": "Fluency",
    "avg_token_length": "Fluency",
    "entity_overlap_ratio": "Terminology",
    "agreement_errors": "Fluency",
    "syntax_depth": "Fluency",
    "formal_ratio": "Style",
    "morphology_error_rate": "Fluency",
    "repetition_ratio": "Style",
    "named_entity_missing_ratio": "Terminology",
    "latin_ratio": "Accuracy",
    "avg_word_rank": "Fluency",
    "untranslated_ratio": "Accuracy",
    "cosine_similarity": "Accuracy",
    "embedding_distance": "Accuracy",
    "perplexity": "Fluency",
    "mean_log_prob": "Fluency",
    "token_ppl_variance": "Fluency",
    "min_token_log_prob": "Fluency",
}
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
    }
)

MQM_CATEGORY_RU: Dict[str, str] = {
    "Accuracy": "Точность (смысл)",
    "Fluency": "Грамотность/плавность",
    "Terminology": "Терминология",
    "Locale": "Локаль/форматирование",
    "Style": "Стиль/регистр",
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
    def __init__(
        self,
        model_path: Union[str, Path],
        explainer_path: Union[str, Path, None] = None,
        model_type: str | None = None,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")

        self.model_type = model_type or infer_model_type(model_path)
        self._explainer_feature_names: list[str] | None = None
        self._feature_names: list[str] = list(FEATURE_NAMES)
        self._explainer = None
        self._explainer_meta: dict[str, Any] = {}

        resolved_explainer = Path(explainer_path) if explainer_path is not None else None
        if resolved_explainer is not None and resolved_explainer.exists():
            self._load_explainer(resolved_explainer)
        elif resolved_explainer is not None:
            log.warning("Explainer не найден: %s. Продолжаем без него.", resolved_explainer)

        self._model = self._load_model(model_path)
        self._expected_feature_count = self._infer_feature_count()
        if (
            self._explainer_feature_names is not None
            and len(self._explainer_feature_names) == self._expected_feature_count
        ):
            self._feature_names = list(self._explainer_feature_names)
        else:
            self._feature_names = _infer_feature_names(self._expected_feature_count)
        log.info(
            "SentenceModel[%s] ожидает %d признаков",
            self.model_type,
            self._expected_feature_count,
        )

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def expected_feature_count(self) -> int:
        return self._expected_feature_count

    def predict(self, features: np.ndarray) -> SentencePrediction:
        X = self._prepare_features(features)
        return self._predict_batch_internal(X)[0]

    def predict_batch(self, features: np.ndarray) -> list[SentencePrediction]:
        X = self._prepare_features(features)
        return self._predict_batch_internal(X)

    def explain_prediction(
        self,
        features: np.ndarray,
        prediction: SentencePrediction | None = None,
    ) -> dict[str, Any]:
        X = self._prepare_features(features)
        if X.shape[0] != 1:
            raise ValueError("explain_prediction ожидает один вектор признаков")

        pred = prediction or self._predict_batch_internal(X)[0]
        feature_values = {
            name: float(value)
            for name, value in zip(self._feature_names, X[0], strict=False)
        }
        contributions = {
            name: float(value)
            for name, value in zip(self._feature_names, np.asarray(pred.shap_values), strict=False)
        }
        return {
            "model_type": self.model_type,
            "feature_values": feature_values,
            "feature_contributions": contributions,
            "shap_values": contributions,
            "category_contributions": dict(pred.explanation),
        }

    @classmethod
    def from_models_dir(cls, models_dir: str | Path, model_type: str = "xgboost") -> SentenceModel:
        model_path, explainer_path = resolve_sentence_artifacts(models_dir, model_type)
        return cls(model_path, explainer_path, model_type=model_type)

    def _load_explainer(self, explainer_path: Path) -> None:
        try:
            with open(explainer_path, "rb") as f:
                loaded_expl = pickle.load(f)
        except ModuleNotFoundError as exc:
            log.warning("Не удалось загрузить explainer (%s). Продолжаем без SHAP.", exc)
            loaded_expl = None

        if isinstance(loaded_expl, dict):
            self._explainer = loaded_expl.get("explainer")
            self._explainer_meta = dict(loaded_expl)
            fn = loaded_expl.get("feature_names")
            self._explainer_feature_names = list(fn) if fn else None
        else:
            self._explainer = loaded_expl
            self._explainer_meta = {}

    def _load_model(self, model_path: Path) -> Any:
        if self.model_type == "xgboost":
            if xgb is None:
                raise ModuleNotFoundError("Для XGBoost sentence-модели требуется пакет xgboost")
            model = xgb.XGBRegressor()
            model.load_model(str(model_path))
            log.info("Загружена XGBoost модель из %s", model_path)
            return model

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        if self.model_type == "ridge" and not isinstance(model, Ridge):
            raise TypeError("Ожидалась Ridge-модель.")
        if self.model_type == "rf" and not isinstance(model, RandomForestRegressor):
            raise TypeError("Ожидалась RandomForestRegressor-модель.")
        log.info("Загружена %s модель из %s", self.model_type, model_path)
        return model

    def _infer_feature_count(self) -> int:
        if self.model_type == "xgboost":
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

        coef = getattr(self._model, "coef_", None)
        if coef is not None:
            return int(np.asarray(coef).shape[-1])

        return len(FEATURE_NAMES)

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

    def _predict_batch_internal(self, X: np.ndarray) -> list[SentencePrediction]:
        raw_scores = self._predict_scores(X)
        scores = np.clip(raw_scores, 0.0, 1.0)
        shap_vals = self._feature_contributions(X)

        results = []
        for idx, score in enumerate(scores):
            uncertainty, ci_low, ci_high = _score_uncertainty(self.model_type, float(score), X, idx, self._model)
            sv = shap_vals[idx] if shap_vals.ndim == 2 else shap_vals
            results.append(
                SentencePrediction(
                    score=float(score),
                    uncertainty=uncertainty,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    alpha=None,
                    beta_param=None,
                    shap_values=np.asarray(sv, dtype=np.float32),
                    explanation=_aggregate_shap(sv, self._feature_names),
                )
            )
        return results

    def _predict_scores(self, X: np.ndarray) -> np.ndarray:
        preds = self._model.predict(X)
        return np.asarray(preds, dtype=np.float32).reshape(-1)

    def _feature_contributions(self, X: np.ndarray) -> np.ndarray:
        if self.model_type in {"xgboost", "rf"}:
            return self._tree_shap_values(X)
        if self.model_type == "ridge":
            return self._ridge_contributions(X)
        raise ValueError(f"Неизвестный тип модели: {self.model_type}")

    def _tree_shap_values(self, X: np.ndarray) -> np.ndarray:
        if self._explainer is None:
            return np.zeros((X.shape[0], self._expected_feature_count), dtype=np.float32)
        try:
            values = self._explainer.shap_values(X)
            arr = np.array(values, dtype=np.float32)
            if arr.ndim == 1:
                return arr.reshape(1, -1)
            return arr
        except Exception as exc:
            log.warning("SHAP не удался для %s: %s", self.model_type, exc)
            return np.zeros((X.shape[0], self._expected_feature_count), dtype=np.float32)

    def _ridge_contributions(self, X: np.ndarray) -> np.ndarray:
        coef = np.asarray(getattr(self._model, "coef_", np.zeros(self._expected_feature_count)), dtype=np.float32)
        if coef.ndim > 1:
            coef = coef[0]
        return X.astype(np.float32) * coef.reshape(1, -1)


def resolve_sentence_artifacts(models_dir: str | Path, model_type: str = "xgboost") -> tuple[Path, Path | None]:
    models_dir = Path(models_dir)
    if model_type not in MODEL_ARTIFACTS:
        raise ValueError(f"Неизвестный тип модели: {model_type}")

    spec = MODEL_ARTIFACTS[model_type]
    candidates = [models_dir / spec["model"]]
    explainer_candidates = [models_dir / spec.get("explainer", "")]

    legacy_model = spec.get("legacy_model")
    legacy_explainer = spec.get("legacy_explainer")
    if legacy_model:
        candidates.append(models_dir / legacy_model)
    if legacy_explainer:
        explainer_candidates.append(models_dir / legacy_explainer)

    model_path = next((p for p in candidates if p and p.exists()), candidates[0])
    explainer_path = next((p for p in explainer_candidates if p and p.exists()), None)
    return model_path, explainer_path


def infer_model_type(model_path: str | Path) -> str:
    name = Path(model_path).name.lower()
    if "ridge" in name:
        return "ridge"
    if "rf" in name or "forest" in name:
        return "rf"
    if name.endswith(".model") or "xgboost" in name:
        return "xgboost"
    raise ValueError(f"Не удалось определить тип модели по имени файла: {model_path}")


def _infer_feature_names(count: int) -> list[str]:
    if count == len(SENTENCE_FEATURE_NAMES):
        return list(SENTENCE_FEATURE_NAMES)
    if count == len(LEGACY_FEATURE_NAMES):
        return list(LEGACY_FEATURE_NAMES)
    if count == len(FEATURE_NAMES):
        return list(FEATURE_NAMES)
    if count == len(FEATURE_NAMES_LIGHT):
        return list(FEATURE_NAMES_LIGHT)
    if count < len(FEATURE_NAMES):
        return list(FEATURE_NAMES[:count])
    extra = [f"feature_{idx:03d}" for idx in range(len(FEATURE_NAMES), count)]
    return list(FEATURE_NAMES) + extra


def _beta_stats(alpha: float, beta_p: float) -> tuple[float, float, float, float]:
    total = alpha + beta_p
    score = alpha / total
    uncertainty = (alpha * beta_p) / (total * total * (total + 1))
    ci_low = float(scipy_beta.ppf(0.025, alpha, beta_p))
    ci_high = float(scipy_beta.ppf(0.975, alpha, beta_p))
    return score, uncertainty, ci_low, ci_high


def _xgboost_uncertainty(score: float) -> tuple[float, float, float]:
    concentration = 10.0
    alpha = max(score * concentration, 1e-4)
    beta_p = max((1.0 - score) * concentration, 1e-4)
    _, uncertainty, ci_low, ci_high = _beta_stats(alpha, beta_p)
    return uncertainty, ci_low, ci_high


def _score_uncertainty(
    model_type: str,
    score: float,
    X: np.ndarray,
    idx: int,
    model: Any,
) -> tuple[float, float, float]:
    if model_type == "rf" and hasattr(model, "estimators_"):
        tree_preds = np.array([est.predict(X[idx : idx + 1])[0] for est in model.estimators_], dtype=np.float32)
        tree_preds = np.clip(tree_preds, 0.0, 1.0)
        std = float(np.std(tree_preds))
        ci_low = float(np.clip(score - 1.96 * std, 0.0, 1.0))
        ci_high = float(np.clip(score + 1.96 * std, 0.0, 1.0))
        return std * std, ci_low, ci_high
    return _xgboost_uncertainty(score)


def _aggregate_shap(
    shap_vals: np.ndarray,
    feature_names: list[str] | None = None,
) -> Dict[str, float]:
    names = feature_names or FEATURE_NAMES
    aggregated: Dict[str, float] = {}
    for idx, name in enumerate(names):
        if idx >= len(shap_vals):
            break
        category = FEATURE_TO_MQM.get(name)
        if category:
            aggregated[category] = aggregated.get(category, 0.0) + float(shap_vals[idx])
    return aggregated
