"""Обучение sentence-level моделей на sentence_da_features.parquet."""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.stats import zscore as scipy_zscore
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
try:
    import shap
except ModuleNotFoundError:  # pragma: no cover - optional dependency in lightweight test env
    shap = None

try:
    import xgboost as xgb
    from xgboost import DMatrix
    from xgboost import XGBRegressor
    from xgboost.callback import TrainingCallback as _XGBTrainingCallback
except ModuleNotFoundError:  # pragma: no cover - optional dependency in lightweight test env
    xgb = None

    def DMatrix(data: Any, label: Any = None, weight: Any = None) -> Any:  # type: ignore[misc]
        return data

    class XGBRegressor:  # type: ignore[override]
        def load_model(self, *_args: Any, **_kwargs: Any) -> None:
            raise ModuleNotFoundError("xgboost не установлен")

    class _XGBTrainingCallback:  # type: ignore[override]
        pass

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.bootstrap import init_script_runtime
from src.determinism import seed_everything
from src.settings import get_settings
from src.features.interactions import add_interaction_columns_to_dataframe
from src.features.schema import (
    FEATURE_NAMES_CLASSIC,
    INTERACTION_FEATURE_NAMES,
    SENTENCE_FEATURE_NAMES,
)

log = logging.getLogger(__name__)

RANDOM_SEED = 42
DEFAULT_SYNTHETIC_WEIGHT = 0.1
DEFAULT_SYNTHETIC_FRAC = 0.3

MODEL_FILES = {
    "xgboost": "sentence_xgboost.model",
    "ridge": "sentence_ridge.pkl",
    "rf": "sentence_rf.pkl",
}
EXPLAINER_FILES = {
    "xgboost": "sentence_xgboost_explainer.pkl",
    "ridge": "sentence_ridge_explainer.pkl",
    "rf": "sentence_rf_explainer.pkl",
}
LEGACY_XGBOOST_MODEL_FILE = "xgboost_sentence.model"
LEGACY_XGBOOST_EXPLAINER_FILE = "shap_explainer.pkl"


def add_interaction_features(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Гарантирует interaction-признаки и порядок SENTENCE_FEATURE_NAMES."""
    df = df.copy()
    have_all_inter = all(name in df.columns for name in INTERACTION_FEATURE_NAMES)
    if not have_all_inter:
        df = add_interaction_columns_to_dataframe(df)
    tail = [n for n in INTERACTION_FEATURE_NAMES if n not in feature_cols]
    extended_cols = list(feature_cols) + tail
    log.info("Interaction признаков: %d (итого колонок модели: %d)", len(INTERACTION_FEATURE_NAMES), len(extended_cols))
    return df, extended_cols


def build_explainer_payload(
    model_name: str,
    model: Any,
    df: pd.DataFrame,
    feature_cols: list[str],
    models_dir: Path,
) -> None:
    log.info("Строим explainer для %s", model_name)

    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) != len(feature_cols):
        missing = set(feature_cols) - set(available_cols)
        log.warning("В df отсутствуют колонки для explainer: %s", missing)

    payload: dict[str, Any] = {
        "model_type": model_name,
        "feature_names": available_cols,
    }

    if model_name in {"xgboost", "rf"}:
        if shap is None:
            raise ModuleNotFoundError("Для SHAP explainability требуется пакет shap")
        explainer = shap.TreeExplainer(model)
        X_sample = df[df["split"] == "train"][available_cols].values[:100]
        log.info("X_sample shape для SHAP: %s", X_sample.shape)
        shap_values = explainer.shap_values(X_sample)
        log.info("SHAP values shape: %s", np.array(shap_values).shape)
        payload["explainer"] = explainer
    elif model_name == "ridge":
        payload["explainer"] = None
        payload["method"] = "coef_times_feature"
        payload["intercept"] = float(getattr(model, "intercept_", 0.0))
    else:
        raise ValueError(f"Неизвестная модель для explainer: {model_name}")

    models_dir.mkdir(parents=True, exist_ok=True)
    explainer_path = models_dir / EXPLAINER_FILES[model_name]
    with open(explainer_path, "wb") as f:
        pickle.dump(payload, f)
    log.info("Explainer сохранён: %s", explainer_path)

    if model_name == "xgboost":
        legacy_path = models_dir / LEGACY_XGBOOST_EXPLAINER_FILE
        with open(legacy_path, "wb") as f:
            pickle.dump(payload, f)
        log.info("Legacy explainer обновлён: %s", legacy_path)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Берёт признаки в порядке SENTENCE_FEATURE_NAMES либо базовые classic."""
    if all(f in df.columns for f in SENTENCE_FEATURE_NAMES):
        log.info("Parquet содержит полный sentence-вектор: %d признаков", len(SENTENCE_FEATURE_NAMES))
        return list(SENTENCE_FEATURE_NAMES)
    available = [f for f in FEATURE_NAMES_CLASSIC if f in df.columns]
    missing = [f for f in FEATURE_NAMES_CLASSIC if f not in df.columns]
    if missing:
        log.warning(
            "Отсутствуют признаки в датасете: %s\n"
            "Убедись что extract_features.py был запущен без --only и без флагов.",
            missing,
        )
    log.info(
        "Признаков (база): %d / %d; interaction будут добавлены в RAM",
        len(available),
        len(FEATURE_NAMES_CLASSIC),
    )
    return available


def load_data(processed_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    path = processed_dir / "sentence_da_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл: {path}\n"
            "Запусти: python scripts/extract_features.py"
        )

    df = pd.read_parquet(path)
    log.info("Загружено: %d строк, %d колонок", len(df), len(df.columns))

    y = df["score_norm"]
    log.info(
        "score_norm: mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
        y.mean(), y.std(), y.min(), y.max(),
    )
    log.info("Доля score_norm < 0.5: %.1f%%", (y < 0.5).mean() * 100)

    feature_cols = get_feature_cols(df)
    return df, feature_cols


def _split_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()
    log.info("train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def _build_train_sample_weights(
    train_df: pd.DataFrame,
    synthetic_weight: float = 1.0,
    low_score_tau: float | None = None,
    low_score_weight: float = 1.0,
) -> np.ndarray:
    """Веса для train-set."""
    weights = np.ones(len(train_df), dtype=np.float32)

    if "is_synthetic" in train_df.columns:
        synthetic_mask = train_df["is_synthetic"].fillna(False).to_numpy(dtype=bool)
        synthetic_count = int(synthetic_mask.sum())
        if synthetic_count:
            weights[synthetic_mask] *= float(synthetic_weight)
            effective_share = weights[synthetic_mask].sum() / max(weights.sum(), 1e-8)
            log.info(
                "Synthetic train rows: %d / %d  weight=%.3f  effective_share=%.1f%%",
                synthetic_count, len(train_df), synthetic_weight, effective_share * 100,
            )

    if low_score_tau is not None and low_score_weight > 1.0:
        low_score_mask = train_df["score_norm"].to_numpy() < float(low_score_tau)
        affected = int(low_score_mask.sum())
        weights[low_score_mask] *= float(low_score_weight)
        log.info(
            "Low-score upweighting: tau=%.2f  weight=%.2f  affected=%d / %d (%.1f%%)",
            low_score_tau, low_score_weight, affected, len(train_df), 100 * affected / len(train_df),
        )

    return weights


def _downsample_synthetic_rows(
    train_df: pd.DataFrame,
    synthetic_frac: float,
    seed: int,
) -> pd.DataFrame:
    """Ограничивает долю synthetic-строк до обучения."""
    if "is_synthetic" not in train_df.columns:
        return train_df

    synthetic_frac = float(np.clip(synthetic_frac, 0.0, 1.0))
    synthetic_df = train_df[train_df["is_synthetic"] == True]
    real_df = train_df[train_df["is_synthetic"] != True]
    if synthetic_df.empty or synthetic_frac >= 0.999:
        return train_df

    sampled_synth = synthetic_df.sample(frac=synthetic_frac, random_state=seed)
    reduced = pd.concat([real_df, sampled_synth], ignore_index=True)
    log.info(
        "После downsampling synthetic: train=%d (real=%d synthetic=%d, frac=%.2f)",
        len(reduced),
        len(real_df),
        len(sampled_synth),
        synthetic_frac,
    )
    return reduced


def _metric_dict(y_true: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    pearson, _ = pearsonr(y_true, preds)
    spearman, _ = spearmanr(y_true, preds)
    mae = mean_absolute_error(y_true, preds)
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    return {
        "pearson": float(pearson),
        "spearman": float(spearman),
        "mae": float(mae),
        "rmse": rmse,
    }


def _log_metrics(y_true: np.ndarray, preds: np.ndarray, label: str) -> None:
    metrics = _metric_dict(y_true, preds)
    log.info(
        "%s — Pearson r=%.4f  Spearman rho=%.4f  MAE=%.4f  RMSE=%.4f",
        label,
        metrics["pearson"],
        metrics["spearman"],
        metrics["mae"],
        metrics["rmse"],
    )


class _PearsonCallback(_XGBTrainingCallback):
    """Early stopping по val Pearson r."""

    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, patience: int = 150) -> None:
        super().__init__()
        self.dval = DMatrix(X_val)
        self.y_val = y_val
        self.patience = patience
        self.best_r = -np.inf
        self.best_iter = 0
        self.no_improve = 0

    def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
        preds = model.predict(self.dval)
        r, _ = pearsonr(self.y_val, preds)
        if r > self.best_r + 1e-5:
            self.best_r = r
            self.best_iter = epoch
            self.no_improve = 0
        else:
            self.no_improve += 1
        if epoch % 100 == 0:
            log.info(
                "  [%d] val Pearson=%.4f  best=%.4f @ iter %d",
                epoch, r, self.best_r, self.best_iter,
            )
        if self.no_improve >= self.patience:
            log.info(
                "Early stopping @ iter %d: Pearson не улучшался %d итераций. Best=%.4f",
                epoch, self.patience, self.best_r,
            )
            return True
        return False


def train_xgboost(
    df: pd.DataFrame,
    feature_cols: list[str],
    models_dir: Path,
    synthetic_weight: float = 0.12,
    synthetic_frac: float = DEFAULT_SYNTHETIC_FRAC,
    seed: int = RANDOM_SEED,
) -> tuple[Any, list[str]]:
    train_df, val_df, test_df = _split_frames(df)
    if xgb is None:
        raise ModuleNotFoundError("Для обучения XGBoost требуется пакет xgboost")
    train_df = _downsample_synthetic_rows(train_df, synthetic_frac=synthetic_frac, seed=seed)

    X_train = train_df[feature_cols].values
    y_train = train_df["score_norm"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["score_norm"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["score_norm"].values

    train_weights = _build_train_sample_weights(
        train_df,
        synthetic_weight=synthetic_weight,
        low_score_tau=0.15,
        low_score_weight=1.0,
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 1.0,
        "reg_alpha": 0.05,
        "gamma": 0.05,
        "tree_method": "hist",
        "seed": seed,
    }

    pearson_cb = _PearsonCallback(X_val, y_val, patience=120)

    log.info("Запуск обучения XGBoost...")
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=4000,
        evals=[(dtrain, "train"), (dval, "val")],
        callbacks=[pearson_cb],
        verbose_eval=100,
    )

    best_iter = pearson_cb.best_iter
    if best_iter + 1 < booster.num_boosted_rounds():
        booster = booster[:best_iter + 1]

    log.info("Best iter=%d  val Pearson=%.4f", best_iter, pearson_cb.best_r)

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILES["xgboost"]
    legacy_model_path = models_dir / LEGACY_XGBOOST_MODEL_FILE
    booster.save_model(str(model_path))
    booster.save_model(str(legacy_model_path))

    dtest = xgb.DMatrix(X_test)
    _log_metrics(y_test, booster.predict(dtest), "DA test")

    model = XGBRegressor()
    model.load_model(str(model_path))
    return model, feature_cols


def train_ridge(
    df: pd.DataFrame,
    feature_cols: list[str],
    models_dir: Path,
    synthetic_weight: float = DEFAULT_SYNTHETIC_WEIGHT,
    synthetic_frac: float = DEFAULT_SYNTHETIC_FRAC,
    seed: int = RANDOM_SEED,
) -> tuple[Any, list[str]]:
    train_df, val_df, test_df = _split_frames(df)
    train_df = _downsample_synthetic_rows(train_df, synthetic_frac=synthetic_frac, seed=seed)
    X_train = train_df[feature_cols].values
    y_train = train_df["score_norm"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["score_norm"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["score_norm"].values

    train_weights = _build_train_sample_weights(
        train_df,
        synthetic_weight=synthetic_weight,
        low_score_tau=0.15,
        low_score_weight=1.0,
    )

    model = Ridge(alpha=2.0, random_state=RANDOM_SEED)
    model.fit(X_train, y_train, sample_weight=train_weights)

    _log_metrics(y_val, np.clip(model.predict(X_val), 0.0, 1.0), "DA val")
    _log_metrics(y_test, np.clip(model.predict(X_test), 0.0, 1.0), "DA test")

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILES["ridge"]
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info("Ridge сохранён: %s", model_path)
    return model, feature_cols


def train_rf(
    df: pd.DataFrame,
    feature_cols: list[str],
    models_dir: Path,
    synthetic_weight: float = DEFAULT_SYNTHETIC_WEIGHT,
    synthetic_frac: float = DEFAULT_SYNTHETIC_FRAC,
    seed: int = RANDOM_SEED,
) -> tuple[Any, list[str]]:
    train_df, val_df, test_df = _split_frames(df)
    train_df = _downsample_synthetic_rows(train_df, synthetic_frac=synthetic_frac, seed=seed)
    X_train = train_df[feature_cols].values
    y_train = train_df["score_norm"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["score_norm"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["score_norm"].values

    train_weights = _build_train_sample_weights(
        train_df,
        synthetic_weight=synthetic_weight,
        low_score_tau=0.15,
        low_score_weight=1.0,
    )

    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=seed,
        verbose=10,
    )
    log.info("Запуск обучения RandomForest...")
    model.fit(X_train, y_train, sample_weight=train_weights)

    _log_metrics(y_val, np.clip(model.predict(X_val), 0.0, 1.0), "DA val")
    _log_metrics(y_test, np.clip(model.predict(X_test), 0.0, 1.0), "DA test")

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / MODEL_FILES["rf"]
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info("RandomForest сохранён: %s", model_path)
    return model, feature_cols


def train_model(
    model_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    models_dir: Path,
    synthetic_weight: float,
    synthetic_frac: float,
    seed: int,
) -> tuple[Any, list[str]]:
    if model_name == "xgboost":
        return train_xgboost(
            df,
            feature_cols,
            models_dir,
            synthetic_weight=synthetic_weight,
            synthetic_frac=synthetic_frac,
            seed=seed,
        )
    if model_name == "ridge":
        return train_ridge(
            df,
            feature_cols,
            models_dir,
            synthetic_weight=synthetic_weight,
            synthetic_frac=synthetic_frac,
            seed=seed,
        )
    if model_name == "rf":
        return train_rf(
            df,
            feature_cols,
            models_dir,
            synthetic_weight=synthetic_weight,
            synthetic_frac=synthetic_frac,
            seed=seed,
        )
    raise ValueError(f"Неизвестная модель: {model_name}")


def external_test(
    model: Any,
    processed_dir: Path,
    feature_cols: list[str],
) -> None:
    mqm_path = processed_dir / "hf_mqm_features.parquet"
    if not mqm_path.exists():
        log.warning(
            "hf_mqm_features.parquet не найден — пропускаем внешний тест.\n"
            "Запусти: python scripts/extract_features.py --only mqm"
        )
        return

    mqm = pd.read_parquet(mqm_path)
    log.info("MQM датасет: %d строк, %d колонок", len(mqm), len(mqm.columns))

    missing = [f for f in feature_cols if f not in mqm.columns]
    if missing and all(name in INTERACTION_FEATURE_NAMES for name in missing):
        mqm = add_interaction_columns_to_dataframe(mqm)
        missing = [f for f in feature_cols if f not in mqm.columns]
    if missing:
        log.error("В hf_mqm_features.parquet отсутствуют признаки: %s", missing)
        return

    X_mqm = mqm[feature_cols].values
    preds = np.clip(model.predict(X_mqm), 0.0, 1.0)
    mqm_score_raw = mqm["score"].values

    log.info(
        "MQM score raw stats: mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
        mqm_score_raw.mean(), mqm_score_raw.std(), mqm_score_raw.min(), mqm_score_raw.max(),
    )

    if "system" in mqm.columns:
        mqm_score = (
            mqm.groupby("system")["score"]
            .transform(lambda x: scipy_zscore(x, ddof=1))
            .fillna(0)
            .values
        )
        log.info("MQM score нормализован zscore по системе.")
    else:
        mqm_score = scipy_zscore(mqm_score_raw, ddof=1)
        log.info("MQM score нормализован глобальным zscore (нет колонки 'system').")

    log.info(
        "MQM score norm stats: mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
        mqm_score.mean(), mqm_score.std(), mqm_score.min(), mqm_score.max(),
    )
    log.info(
        "Preds stats: mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
        preds.mean(), preds.std(), preds.min(), preds.max(),
    )

    rho, pvalue = spearmanr(mqm_score, preds)
    log.info("MQM внешний тест — Spearman ρ=%.4f  p=%.4f", rho, pvalue)

    threshold_lo = np.percentile(mqm_score, 15)
    threshold_hi = np.percentile(mqm_score, 85)
    mask = (mqm_score <= threshold_lo) | (mqm_score >= threshold_hi)
    if mask.sum() > 100:
        rho_extreme, _ = spearmanr(mqm_score[mask], preds[mask])
        log.info(
            "MQM extremes (bottom/top 20%%, n=%d) — Spearman ρ=%.4f",
            mask.sum(), rho_extreme,
        )


def _load_model_for_eval(models_dir: Path, model_name: str) -> Any:
    model_path = models_dir / MODEL_FILES[model_name]
    if model_name == "xgboost" and not model_path.exists():
        legacy_path = models_dir / LEGACY_XGBOOST_MODEL_FILE
        if legacy_path.exists():
            model_path = legacy_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Не найден файл модели: {model_path}\n"
            "Сначала обучи: poetry run imtqe train-sentence"
        )

    if model_name == "xgboost":
        if xgb is None:
            raise ModuleNotFoundError("Для загрузки XGBoost требуется пакет xgboost")
        model = XGBRegressor()
        model.load_model(str(model_path))
        return model

    with open(model_path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    init_script_runtime()
    s = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=s.data_dir)
    parser.add_argument("--models-dir", type=Path, default=s.models_dir)
    parser.add_argument("--seed", type=int, default=s.random_seed)
    parser.add_argument(
        "--model",
        choices=["xgboost", "ridge", "rf"],
        default="xgboost",
        help="Какую sentence-модель обучать/оценивать.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Только внешний тест на MQM, без переобучения",
    )
    parser.add_argument(
        "--synthetic-weight",
        type=float,
        default=DEFAULT_SYNTHETIC_WEIGHT,
        help=(
            "Вес synthetic train-строк относительно реальных. "
            "0.1 по умолчанию: synthetic negatives помогают, но не доминируют."
        ),
    )
    parser.add_argument(
        "--synthetic-frac",
        type=float,
        default=DEFAULT_SYNTHETIC_FRAC,
        help=(
            "Какую долю synthetic train-строк оставить до обучения. "
            "0.25 по умолчанию: уменьшаем перекос в сторону искусственных примеров."
        ),
    )
    args = parser.parse_args()
    seed_everything(args.seed)

    processed_dir = args.data_dir / "processed"
    df, feature_cols = load_data(processed_dir)
    df, feature_cols = add_interaction_features(df, feature_cols)
    log.info("Итого признаков после interaction: %d", len(feature_cols))

    if args.eval_only:
        model = _load_model_for_eval(args.models_dir, args.model)
        external_test(model, processed_dir, feature_cols)
        return

    model, feature_cols = train_model(
        args.model,
        df,
        feature_cols,
        args.models_dir,
        synthetic_weight=args.synthetic_weight,
        synthetic_frac=args.synthetic_frac,
        seed=args.seed,
    )
    build_explainer_payload(args.model, model, df, feature_cols, args.models_dir)
    external_test(model, processed_dir, feature_cols)
    log.info("train_sentence_model: finished (%s)", args.model)


if __name__ == "__main__":
    main()
