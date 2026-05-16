"""
scripts/tune_ensemble.py

Двухуровневая оптимизация ансамбля XGBoost + RF + Ridge.

Уровень 1 — параллельный поиск гиперпараметров каждой модели
            через Optuna (Bayesian optimization, TPE sampler).
            Все три study запускаются в отдельных процессах.

Уровень 2 — оптимизация весов ансамбля через scipy.optimize
            поверх предсказаний лучших моделей на val-сплите.

Метрика оптимизации уровня 1: Pearson r на val-сплите (5-fold CV
внутри train-сплита, чтобы не переподогнать под конкретный val).

Метрика уровня 2 и финальная отчётность: Pearson r и Spearman rho
на val-сплите (DA) и на MQM external test при наличии.

Лучшие гиперпараметры сохраняются в models/tune_results.json.
Лучшие модели записываются поверх текущих файлов в models/.
Лучшие веса ансамбля сохраняются в models/ensemble_weights.json
и автоматически подхватываются predict.py при наличии этого файла.

Запуск:
  python scripts/tune_ensemble.py
  python scripts/tune_ensemble.py --trials 60 --cv-folds 3
  python scripts/tune_ensemble.py --only xgboost --trials 40
  python scripts/tune_ensemble.py --no-parallel   # последовательно
  python scripts/tune_ensemble.py --skip-level1    # только веса ансамбля
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ModuleNotFoundError:
    print(
        "Требуется пакет optuna: pip install optuna",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    import xgboost as xgb
    from xgboost.callback import TrainingCallback as _XGBCallback
except ModuleNotFoundError:
    xgb = None
    _XGBCallback = object

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from src.bootstrap import init_script_runtime
from src.features.interactions import add_interaction_columns_to_dataframe
from src.features.schema import INTERACTION_FEATURE_NAMES, SENTENCE_FEATURE_NAMES
from src.settings import get_settings

log = logging.getLogger(__name__)

RANDOM_SEED = 42

MODEL_FILES = {
    "xgboost": "sentence_xgboost.model",
    "ridge": "sentence_ridge.pkl",
    "rf": "sentence_rf.pkl",
}
LEGACY_XGBOOST_FILE = "xgboost_sentence.model"
TUNE_RESULTS_FILE = "tune_results.json"
ENSEMBLE_WEIGHTS_FILE = "ensemble_weights.json"


# ---------------------------------------------------------------------------
# Загрузка и подготовка данных
# ---------------------------------------------------------------------------

def load_data(processed_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    path = processed_dir / "sentence_da_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл: {path}\n"
            "Запусти сначала: python scripts/extract_features.py"
        )
    df = pd.read_parquet(path)
    log.info("Загружено: %d строк, %d колонок", len(df), len(df.columns))

    feature_cols = (
        list(SENTENCE_FEATURE_NAMES)
        if all(f in df.columns for f in SENTENCE_FEATURE_NAMES)
        else [f for f in SENTENCE_FEATURE_NAMES if f in df.columns]
    )

    have_all_inter = all(n in df.columns for n in INTERACTION_FEATURE_NAMES)
    if not have_all_inter:
        df = add_interaction_columns_to_dataframe(df)
        for name in INTERACTION_FEATURE_NAMES:
            if name not in feature_cols and name in df.columns:
                feature_cols.append(name)

    log.info("Признаков: %d", len(feature_cols))
    return df, feature_cols


def _split_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    X_tr = train_df[feature_cols].values.astype(np.float32)
    y_tr = train_df["score_norm"].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["score_norm"].values.astype(np.float32)
    X_te = test_df[feature_cols].values.astype(np.float32)
    y_te = test_df["score_norm"].values.astype(np.float32)
    log.info(
        "train=%d val=%d test=%d",
        len(X_tr), len(X_val), len(X_te),
    )
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def _sample_weights(train_df: pd.DataFrame, synthetic_weight: float) -> np.ndarray:
    weights = np.ones(len(train_df), dtype=np.float32)
    if "is_synthetic" in train_df.columns:
        mask = train_df["is_synthetic"].fillna(False).to_numpy(dtype=bool)
        weights[mask] *= float(synthetic_weight)
    return weights


def _downsample_synthetic(
    train_df: pd.DataFrame,
    synthetic_frac: float,
    seed: int,
) -> pd.DataFrame:
    if "is_synthetic" not in train_df.columns:
        return train_df
    synth = train_df[train_df["is_synthetic"] == True]
    real = train_df[train_df["is_synthetic"] != True]
    if synth.empty or synthetic_frac >= 0.999:
        return train_df
    return pd.concat(
        [real, synth.sample(frac=float(synthetic_frac), random_state=seed)],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Вспомогательный callback для XGBoost: early stopping по Pearson
# ---------------------------------------------------------------------------

class _PearsonStop(_XGBCallback):
    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, patience: int) -> None:
        super().__init__()
        if xgb is not None:
            self._dval = xgb.DMatrix(X_val)
        self._y_val = y_val
        self._patience = patience
        self._best = -np.inf
        self._no_improve = 0
        self.best_iter = 0

    def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
        if xgb is None:
            return False
        preds = model.predict(self._dval)
        r, _ = pearsonr(self._y_val, preds)
        if r > self._best + 1e-5:
            self._best = r
            self.best_iter = epoch
            self._no_improve = 0
        else:
            self._no_improve += 1
        return self._no_improve >= self._patience


# ---------------------------------------------------------------------------
# CV-оценка: усреднённый Pearson r на k фолдах внутри train
# Используется как objective для Optuna, чтобы не переподогнать под val.
# ---------------------------------------------------------------------------

def _cv_pearson(
    model_name: str,
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_weights: np.ndarray,
    n_folds: int,
    seed: int,
) -> float:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = []
    for fold_train_idx, fold_val_idx in kf.split(X_train):
        Xf_tr, yf_tr = X_train[fold_train_idx], y_train[fold_train_idx]
        Xf_val, yf_val = X_train[fold_val_idx], y_train[fold_val_idx]
        wf_tr = train_weights[fold_train_idx]

        if model_name == "xgboost":
            if xgb is None:
                raise ModuleNotFoundError("xgboost не установлен")
            dtrain = xgb.DMatrix(Xf_tr, label=yf_tr, weight=wf_tr)
            cb = _PearsonStop(Xf_val, yf_val, patience=params.pop("_patience", 50))
            booster = xgb.train(
                {k: v for k, v in params.items() if not k.startswith("_")},
                dtrain,
                num_boost_round=params.get("_num_round", 1500),
                callbacks=[cb],
                verbose_eval=False,
            )
            dval = xgb.DMatrix(Xf_val)
            preds = booster.predict(dval)

        elif model_name == "ridge":
            m = Ridge(alpha=params["alpha"], random_state=seed)
            m.fit(Xf_tr, yf_tr, sample_weight=wf_tr)
            preds = np.clip(m.predict(Xf_val), 0.0, 1.0)

        elif model_name == "rf":
            m = RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                n_jobs=-1,
                random_state=seed,
            )
            m.fit(Xf_tr, yf_tr, sample_weight=wf_tr)
            preds = np.clip(m.predict(Xf_val), 0.0, 1.0)

        else:
            raise ValueError(f"Неизвестная модель: {model_name}")

        r, _ = pearsonr(yf_val, preds)
        scores.append(r)

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Objective-функции для Optuna
# ---------------------------------------------------------------------------

def _objective_xgboost(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_weights: np.ndarray,
    n_folds: int,
    seed: int,
) -> float:
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "seed": seed,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "_num_round": trial.suggest_int("num_round", 500, 2500),
        "_patience": 60,
    }
    return _cv_pearson("xgboost", params, X_train, y_train, train_weights, n_folds, seed)


def _objective_ridge(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_weights: np.ndarray,
    n_folds: int,
    seed: int,
) -> float:
    params = {
        "alpha": trial.suggest_float("alpha", 1e-2, 100.0, log=True),
    }
    return _cv_pearson("ridge", params, X_train, y_train, train_weights, n_folds, seed)


def _objective_rf(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_weights: np.ndarray,
    n_folds: int,
    seed: int,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 6, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }
    return _cv_pearson("rf", params, X_train, y_train, train_weights, n_folds, seed)


# ---------------------------------------------------------------------------
# Запуск одного study (выполняется в отдельном процессе)
# ---------------------------------------------------------------------------

def _run_study(
    model_name: str,
    n_trials: int,
    n_folds: int,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_weights: np.ndarray,
    storage_url: str | None,
) -> dict[str, Any]:
    sampler = optuna.samplers.TPESampler(seed=seed)
    study_name = f"tune_{model_name}"

    if storage_url:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
            storage=storage_url,
            load_if_exists=True,
        )
    else:
        study = optuna.create_study(direction="maximize", sampler=sampler)

    objectives = {
        "xgboost": _objective_xgboost,
        "ridge": _objective_ridge,
        "rf": _objective_rf,
    }
    obj_fn = objectives[model_name]

    study.optimize(
        lambda trial: obj_fn(trial, X_train, y_train, train_weights, n_folds, seed),
        n_trials=n_trials,
        show_progress_bar=False,
        n_jobs=1,
    )

    log.info(
        "[%s] Лучший trial #%d: CV Pearson=%.4f, params=%s",
        model_name,
        study.best_trial.number,
        study.best_value,
        study.best_params,
    )
    return {
        "model": model_name,
        "best_cv_pearson": study.best_value,
        "best_params": study.best_params,
        "n_trials": n_trials,
    }


def _run_study_subprocess(kwargs: dict) -> dict[str, Any]:
    """Точка входа для multiprocessing.Process."""
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    return _run_study(**kwargs)


# ---------------------------------------------------------------------------
# Финальное обучение с лучшими параметрами
# ---------------------------------------------------------------------------

def _train_best_xgboost(
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    train_weights: np.ndarray,
    seed: int,
) -> Any:
    if xgb is None:
        raise ModuleNotFoundError("xgboost не установлен")

    xgb_params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "seed": seed,
        "learning_rate": params["learning_rate"],
        "max_depth": params["max_depth"],
        "min_child_weight": params["min_child_weight"],
        "subsample": params["subsample"],
        "colsample_bytree": params["colsample_bytree"],
        "reg_lambda": params["reg_lambda"],
        "reg_alpha": params["reg_alpha"],
        "gamma": params["gamma"],
    }
    num_round = params.get("num_round", 1500)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
    cb = _PearsonStop(X_val, y_val, patience=80)
    booster = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=num_round,
        callbacks=[cb],
        verbose_eval=False,
    )
    if cb.best_iter + 1 < booster.num_boosted_rounds():
        booster = booster[: cb.best_iter + 1]
    log.info(
        "XGBoost финальное обучение: best_iter=%d  val Pearson=%.4f",
        cb.best_iter,
        cb._best,
    )
    return booster


def _train_best_ridge(
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_weights: np.ndarray,
    seed: int,
) -> Ridge:
    m = Ridge(alpha=params["alpha"], random_state=seed)
    m.fit(X_train, y_train, sample_weight=train_weights)
    return m


def _train_best_rf(
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_weights: np.ndarray,
    seed: int,
) -> RandomForestRegressor:
    m = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        n_jobs=-1,
        random_state=seed,
        verbose=0,
    )
    m.fit(X_train, y_train, sample_weight=train_weights)
    return m


# ---------------------------------------------------------------------------
# Предсказание обученной моделью
# ---------------------------------------------------------------------------

def _predict(model_name: str, model: Any, X: np.ndarray) -> np.ndarray:
    if model_name == "xgboost":
        return np.clip(model.predict(xgb.DMatrix(X)), 0.0, 1.0)
    return np.clip(model.predict(X), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Оптимизация весов ансамбля
# ---------------------------------------------------------------------------

def _optimize_weights(
    val_preds: dict[str, np.ndarray],
    y_val: np.ndarray,
    model_names: list[str],
) -> dict[str, float]:
    """
    Ищет веса w_i >= 0, sum(w_i) = 1, максимизирующие Pearson r
    ансамблевого предсказания на val-сплите.

    Используется scipy.optimize.minimize с методом SLSQP.
    Дополнительно проверяется равномерное распределение и текущие
    захардкоженные веса — берётся лучший вариант.
    """
    preds_matrix = np.stack(
        [val_preds[name] for name in model_names], axis=1
    )  # (n_val, n_models)

    def neg_pearson(w: np.ndarray) -> float:
        ensemble = preds_matrix @ w
        r, _ = pearsonr(y_val, ensemble)
        return -float(r)

    n = len(model_names)
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(0.0, 1.0)] * n

    # Несколько стартовых точек: равномерная, текущая и случайные
    starts = [
        np.ones(n) / n,
    ]
    current_map = {"xgboost": 0.45, "rf": 0.35, "ridge": 0.20}
    current_start = np.array(
        [current_map.get(name, 1.0 / n) for name in model_names],
        dtype=float,
    )
    current_start /= current_start.sum()
    starts.append(current_start)

    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(8):
        w = rng.dirichlet(np.ones(n))
        starts.append(w)

    best_r = -np.inf
    best_w = starts[0].copy()

    for w0 in starts:
        result = minimize(
            neg_pearson,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )
        if result.success and -result.fun > best_r:
            best_r = -result.fun
            best_w = result.x.copy()

    best_w = np.clip(best_w, 0.0, 1.0)
    best_w /= best_w.sum()

    weights = {name: float(w) for name, w in zip(model_names, best_w)}
    log.info(
        "Оптимальные веса ансамбля (val Pearson=%.4f): %s",
        best_r,
        {k: round(v, 4) for k, v in weights.items()},
    )
    return weights


# ---------------------------------------------------------------------------
# Метрики и отчёт
# ---------------------------------------------------------------------------

def _log_metrics(
    y_true: np.ndarray,
    preds: np.ndarray,
    label: str,
) -> dict[str, float]:
    r, _ = pearsonr(y_true, preds)
    rho, _ = spearmanr(y_true, preds)
    mae = float(mean_absolute_error(y_true, preds))
    out = {"pearson": float(r), "spearman": float(rho), "mae": mae}
    log.info(
        "%s — Pearson=%.4f  Spearman=%.4f  MAE=%.4f",
        label, r, rho, mae,
    )
    return out


def _external_mqm_test(
    model_name: str,
    model: Any,
    processed_dir: Path,
    feature_cols: list[str],
) -> float | None:
    from scipy.stats import zscore as scipy_zscore

    mqm_path = processed_dir / "hf_mqm_features.parquet"
    if not mqm_path.exists():
        log.warning("hf_mqm_features.parquet не найден, пропускаем MQM тест.")
        return None

    mqm = pd.read_parquet(mqm_path)
    missing = [f for f in feature_cols if f not in mqm.columns]
    if missing:
        mqm = add_interaction_columns_to_dataframe(mqm)
        missing = [f for f in feature_cols if f not in mqm.columns]
    if missing:
        log.warning("MQM: отсутствуют признаки %s, пропускаем.", missing)
        return None

    X_mqm = mqm[feature_cols].values.astype(np.float32)
    preds = _predict(model_name, model, X_mqm)
    mqm_score_raw = mqm["score"].values

    if "system" in mqm.columns:
        mqm_score = (
            mqm.groupby("system")["score"]
            .transform(lambda x: scipy_zscore(x, ddof=1))
            .fillna(0)
            .values
        )
    else:
        mqm_score = scipy_zscore(mqm_score_raw, ddof=1)

    rho, _ = spearmanr(mqm_score, preds)
    log.info("[%s] MQM external — Spearman rho=%.4f", model_name, rho)
    return float(rho)


# ---------------------------------------------------------------------------
# Сохранение моделей
# ---------------------------------------------------------------------------

def _save_models(
    models: dict[str, Any],
    models_dir: Path,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        path = models_dir / MODEL_FILES[name]
        if name == "xgboost":
            model.save_model(str(path))
            model.save_model(str(models_dir / LEGACY_XGBOOST_FILE))
            log.info("XGBoost сохранён: %s", path)
        else:
            with open(path, "wb") as f:
                pickle.dump(model, f)
            log.info("%s сохранён: %s", name, path)


# ---------------------------------------------------------------------------
# Основной поток
# ---------------------------------------------------------------------------

def run_level1(
    model_names: list[str],
    n_trials: int,
    n_folds: int,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_df: pd.DataFrame,
    synthetic_weight: float,
    synthetic_frac: float,
    parallel: bool,
    storage_url: str | None,
) -> dict[str, dict[str, Any]]:
    """
    Запускает Optuna study для каждой модели.
    При parallel=True — в отдельных процессах через multiprocessing.
    """
    # Веса для CV (downsampled train)
    train_df_ds = _downsample_synthetic(train_df, synthetic_frac, seed)
    X_tr_ds = train_df_ds[[c for c in train_df_ds.columns
                            if c in SENTENCE_FEATURE_NAMES or c in INTERACTION_FEATURE_NAMES]
                           ].values.astype(np.float32)
    y_tr_ds = train_df_ds["score_norm"].values.astype(np.float32)
    w_tr_ds = _sample_weights(train_df_ds, synthetic_weight)

    study_kwargs = [
        {
            "model_name": name,
            "n_trials": n_trials,
            "n_folds": n_folds,
            "seed": seed,
            "X_train": X_tr_ds,
            "y_train": y_tr_ds,
            "train_weights": w_tr_ds,
            "storage_url": storage_url,
        }
        for name in model_names
    ]

    results: dict[str, dict[str, Any]] = {}

    if parallel and len(model_names) > 1:
        log.info(
            "Параллельный поиск для %s (%d процессов)",
            model_names,
            len(model_names),
        )
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=len(model_names)) as pool:
            raw = pool.map(_run_study_subprocess, study_kwargs)
        for r in raw:
            results[r["model"]] = r
    else:
        for kwargs in study_kwargs:
            r = _run_study(**kwargs)
            results[r["model"]] = r

    return results


def run_level2(
    model_names: list[str],
    best_params: dict[str, dict[str, Any]],
    df: pd.DataFrame,
    feature_cols: list[str],
    processed_dir: Path,
    models_dir: Path,
    seed: int,
    synthetic_weight: float,
    synthetic_frac: float,
) -> dict[str, float]:
    """
    Обучает финальные модели с лучшими параметрами,
    оптимизирует веса ансамбля, сохраняет всё.
    """
    train_df_full = df[df["split"] == "train"].copy()
    train_df_ds = _downsample_synthetic(train_df_full, synthetic_frac, seed)
    X_train = train_df_ds[feature_cols].values.astype(np.float32)
    y_train = train_df_ds["score_norm"].values.astype(np.float32)
    train_weights = _sample_weights(train_df_ds, synthetic_weight)

    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["score_norm"].values.astype(np.float32)
    X_te = test_df[feature_cols].values.astype(np.float32)
    y_te = test_df["score_norm"].values.astype(np.float32)

    trained_models: dict[str, Any] = {}
    val_preds: dict[str, np.ndarray] = {}
    te_preds: dict[str, np.ndarray] = {}
    mqm_rho: dict[str, float | None] = {}

    for name in model_names:
        params = best_params[name]
        log.info("Финальное обучение [%s] с params=%s", name, params)

        if name == "xgboost":
            model = _train_best_xgboost(
                params, X_train, y_train, X_val, y_val, train_weights, seed
            )
        elif name == "ridge":
            model = _train_best_ridge(params, X_train, y_train, train_weights, seed)
        elif name == "rf":
            model = _train_best_rf(params, X_train, y_train, train_weights, seed)
        else:
            raise ValueError(f"Неизвестная модель: {name}")

        trained_models[name] = model
        val_preds[name] = _predict(name, model, X_val)
        te_preds[name] = _predict(name, model, X_te)

        _log_metrics(y_val, val_preds[name], f"[{name}] val")
        _log_metrics(y_te, te_preds[name], f"[{name}] test")
        mqm_rho[name] = _external_mqm_test(name, model, processed_dir, feature_cols)

    _save_models(trained_models, models_dir)

    weights = _optimize_weights(val_preds, y_val, model_names)

    # Проверяем ансамбль на val и test
    ens_val = sum(val_preds[n] * weights[n] for n in model_names)
    ens_te = sum(te_preds[n] * weights[n] for n in model_names)
    _log_metrics(y_val, ens_val, "[ensemble] val")
    _log_metrics(y_te, ens_te, "[ensemble] test")

    return weights


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    init_script_runtime()
    s = get_settings()
    parser = argparse.ArgumentParser(
        description="Двухуровневая оптимизация ансамбля sentence-моделей."
    )
    parser.add_argument("--data-dir", type=Path, default=s.data_dir)
    parser.add_argument("--models-dir", type=Path, default=s.models_dir)
    parser.add_argument("--seed", type=int, default=s.random_seed)
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Число Optuna trials на каждую модель.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Число фолдов для CV внутри train-сплита.",
    )
    parser.add_argument(
        "--only",
        choices=["xgboost", "ridge", "rf"],
        default=None,
        help="Оптимизировать только одну модель (веса ансамбля не ищутся).",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Запускать study последовательно (по умолчанию — параллельно).",
    )
    parser.add_argument(
        "--skip-level1",
        action="store_true",
        help=(
            "Пропустить поиск гиперпараметров. Загрузить лучшие параметры "
            "из models/tune_results.json и сразу перейти к уровню 2."
        ),
    )
    parser.add_argument(
        "--synthetic-weight",
        type=float,
        default=0.12,
        help="Вес synthetic-строк в train.",
    )
    parser.add_argument(
        "--synthetic-frac",
        type=float,
        default=0.3,
        help="Доля synthetic-строк после downsampling.",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help=(
            "URL хранилища Optuna для сохранения trials между запусками. "
            "Например: sqlite:///data/optuna.db"
        ),
    )
    args = parser.parse_args()

    processed_dir = args.data_dir / "processed"
    models_dir = args.models_dir
    results_path = models_dir / TUNE_RESULTS_FILE
    weights_path = models_dir / ENSEMBLE_WEIGHTS_FILE

    df, feature_cols = load_data(processed_dir)

    model_names = (
        [args.only] if args.only
        else ["xgboost", "ridge", "rf"]
    )

    # --------------- Уровень 1: поиск гиперпараметров ---------------

    if args.skip_level1:
        if not results_path.exists():
            log.error(
                "--skip-level1 указан, но %s не найден. "
                "Запусти сначала без этого флага.",
                results_path,
            )
            sys.exit(1)
        with open(results_path, encoding="utf-8") as f:
            saved = json.load(f)
        level1_results: dict[str, dict[str, Any]] = {
            r["model"]: r for r in saved["level1"]
        }
        log.info("Загружены сохранённые результаты уровня 1: %s", results_path)
    else:
        train_df = df[df["split"] == "train"].copy()
        # Строим feature_cols-совместимый X_train для передачи в process
        train_df_ds = _downsample_synthetic(train_df, args.synthetic_frac, args.seed)
        X_tr_ds = train_df_ds[feature_cols].values.astype(np.float32)
        y_tr_ds = train_df_ds["score_norm"].values.astype(np.float32)

        t0 = time.monotonic()
        level1_results = run_level1(
            model_names=model_names,
            n_trials=args.trials,
            n_folds=args.cv_folds,
            seed=args.seed,
            X_train=X_tr_ds,
            y_train=y_tr_ds,
            train_df=train_df,
            synthetic_weight=args.synthetic_weight,
            synthetic_frac=args.synthetic_frac,
            parallel=not args.no_parallel,
            storage_url=args.storage,
        )
        log.info("Уровень 1 завершён за %.1f сек.", time.monotonic() - t0)

    # Сохраняем результаты уровня 1
    models_dir.mkdir(parents=True, exist_ok=True)
    save_payload: dict[str, Any] = {
        "level1": list(level1_results.values()),
    }

    if args.only:
        # Только одна модель — уровень 2 не запускаем
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(save_payload, f, ensure_ascii=False, indent=2)
        log.info("Результаты уровня 1 сохранены: %s", results_path)
        log.info(
            "Лучшие параметры [%s]: %s",
            args.only,
            level1_results[args.only]["best_params"],
        )
        return

    # --------------- Уровень 2: финальное обучение и веса ансамбля -------

    best_params = {
        name: level1_results[name]["best_params"]
        for name in model_names
        if name in level1_results
    }

    t0 = time.monotonic()
    weights = run_level2(
        model_names=model_names,
        best_params=best_params,
        df=df,
        feature_cols=feature_cols,
        processed_dir=processed_dir,
        models_dir=models_dir,
        seed=args.seed,
        synthetic_weight=args.synthetic_weight,
        synthetic_frac=args.synthetic_frac,
    )
    log.info("Уровень 2 завершён за %.1f сек.", time.monotonic() - t0)

    # Сохраняем итоговые веса
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)
    log.info("Веса ансамбля сохранены: %s", weights_path)

    save_payload["level2_weights"] = weights
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(save_payload, f, ensure_ascii=False, indent=2)
    log.info("Полные результаты сохранены: %s", results_path)

    log.info("tune_ensemble: finished")
    log.info(
        "Чтобы сервер использовал новые веса, убедись что predict.py "
        "читает %s (или добавь загрузку вручную).",
        weights_path,
    )


if __name__ == "__main__":
    main()
