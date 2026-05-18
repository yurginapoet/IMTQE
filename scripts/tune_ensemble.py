"""
scripts/tune_rf_then_ensemble.py

Тюнинг только RF (в главном процессе, n_jobs=-1 работает нормально),
затем level2 с уже известными параметрами XGBoost и Ridge.

Запуск в Colab:
  !python scripts/tune_rf_then_ensemble.py --gpu --trials 20 --cv-folds 3
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

# RF: пробуем cuML (GPU), fallback на sklearn
try:
    from cuml.ensemble import RandomForestRegressor as CumlRF
    _CUML_AVAILABLE = True
except ImportError:
    _CUML_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor as SklearnRF

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ModuleNotFoundError:
    print("pip install optuna", file=sys.stderr)
    sys.exit(1)

try:
    import xgboost as xgb
    from xgboost.callback import TrainingCallback as _XGBCallback
except ModuleNotFoundError:
    xgb = None
    _XGBCallback = object

from src.bootstrap import init_script_runtime
from src.features.interactions import add_interaction_columns_to_dataframe
from src.features.schema import INTERACTION_FEATURE_NAMES, SENTENCE_FEATURE_NAMES
from src.settings import get_settings

log = logging.getLogger(__name__)

RANDOM_SEED = 42
MODEL_FILES = {
    "xgboost": "sentence_xgboost.model",
    "ridge":   "sentence_ridge.pkl",
    "rf":      "sentence_rf.pkl",
}
TUNE_RESULTS_FILE    = "tune_results.json"
ENSEMBLE_WEIGHTS_FILE = "ensemble_weights.json"

# ---------------------------------------------------------------------------
# Уже найденные параметры
# ---------------------------------------------------------------------------

BEST_PARAMS_XGBOOST: dict[str, Any] = {
    "learning_rate":    0.03494340983920317,
    "max_depth":        6,
    "min_child_weight": 3,
    "subsample":        0.8919802732741521,
    "colsample_bytree": 0.9526101587075015,
    "reg_lambda":       9.63635810298425,
    "reg_alpha":        0.00891493807417657,
    "gamma":            0.003291999108334797,
    "num_round":        2367,
}

BEST_PARAMS_RIDGE: dict[str, Any] = {
    "alpha": 0.5877857271827573,
}

# ---------------------------------------------------------------------------
# Данные
# ---------------------------------------------------------------------------

def load_data(processed_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    path = processed_dir / "sentence_da_features.parquet"
    if not path.exists():
        log.error("Файл не найден: %s", path)
        sys.exit(1)
    df = pd.read_parquet(path)
    if "interaction" not in " ".join(df.columns):
        df = add_interaction_columns_to_dataframe(df)
    feature_cols = [c for c in df.columns if c in SENTENCE_FEATURE_NAMES or c in INTERACTION_FEATURE_NAMES]
    log.info("Загружено %d строк, %d признаков", len(df), len(feature_cols))
    return df, feature_cols


def _sample_weights(df: pd.DataFrame, synthetic_weight: float) -> np.ndarray:
    w = np.ones(len(df), dtype=np.float32)
    if "is_synthetic" in df.columns:
        mask = df["is_synthetic"].fillna(False).to_numpy(dtype=bool)
        w[mask] *= float(synthetic_weight)
    return w


def _downsample_synthetic(df: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    if "is_synthetic" not in df.columns:
        return df
    synth = df[df["is_synthetic"] == True]
    real  = df[df["is_synthetic"] != True]
    if synth.empty or frac >= 0.999:
        return df
    return pd.concat(
        [real, synth.sample(frac=float(frac), random_state=seed)],
        ignore_index=True,
    )

# ---------------------------------------------------------------------------
# CV для RF objective
# ---------------------------------------------------------------------------

def _cv_pearson_rf(
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    n_folds: int,
    seed: int,
    use_gpu: bool = False,
) -> float:
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = []
    for tr_idx, val_idx in kf.split(X):
        m = _make_rf(params, seed, use_gpu)
        if use_gpu and _CUML_AVAILABLE:
            m.fit(X[tr_idx], y[tr_idx])  # cuML не поддерживает sample_weight
        else:
            m.fit(X[tr_idx], y[tr_idx], sample_weight=w[tr_idx])
        preds = np.clip(m.predict(X[val_idx]), 0.0, 1.0)
        if hasattr(preds, "to_numpy"):
            preds = preds.to_numpy()
        r, _ = pearsonr(y[val_idx], preds)
        scores.append(r)
    return float(np.mean(scores))

# ---------------------------------------------------------------------------
# Тюнинг RF
# ---------------------------------------------------------------------------

def tune_rf(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    n_trials: int,
    n_folds: int,
    seed: int,
    use_gpu: bool = False,
) -> dict[str, Any]:
    if use_gpu and _CUML_AVAILABLE:
        log.info("RF тюнинг на GPU (cuML)")
    elif use_gpu:
        log.warning("--gpu указан, но cuML не найден — используется sklearn CPU")
    else:
        log.info("RF тюнинг на CPU (sklearn, n_jobs=-1)")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 600),
            "max_depth":        trial.suggest_int("max_depth", 10, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_features":     trial.suggest_float("max_features", 0.3, 1.0),
        }
        return _cv_pearson_rf(params, X, y, w, n_folds, seed, use_gpu)

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)

    log.info(
        "[rf] Лучший trial #%d: CV Pearson=%.4f, params=%s",
        study.best_trial.number,
        study.best_value,
        study.best_params,
    )
    return study.best_params

# ---------------------------------------------------------------------------
# XGBoost callback
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
# Level 2: финальное обучение + оптимизация весов ансамбля
# ---------------------------------------------------------------------------

def _make_rf(params: dict[str, Any], seed: int, use_gpu: bool) -> Any:
    """Создаёт RF на GPU (cuML) или CPU (sklearn)."""
    if use_gpu and _CUML_AVAILABLE:
        return CumlRF(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            random_state=seed,
            n_streams=4,  # параллелизм внутри cuML
        )
    return SklearnRF(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        n_jobs=-1,
        random_state=seed,
    )


    if name == "xgboost":
        return model.predict(xgb.DMatrix(X))
    return np.clip(model.predict(X), 0.0, 1.0)


def _log_metrics(y_true: np.ndarray, preds: np.ndarray, label: str) -> None:
    r, _   = pearsonr(y_true, preds)
    rho, _ = spearmanr(y_true, preds)
    mae    = float(mean_absolute_error(y_true, preds))
    log.info("%s — Pearson=%.4f  Spearman=%.4f  MAE=%.4f", label, r, rho, mae)


def _predict(name: str, model: Any, X: np.ndarray) -> np.ndarray:
    if name == "xgboost":
        return model.predict(xgb.DMatrix(X))
    preds = model.predict(X)
    if hasattr(preds, "to_numpy"):
        preds = preds.to_numpy()
    return np.clip(preds, 0.0, 1.0)



def run_level2(
    best_params: dict[str, dict[str, Any]],
    df: pd.DataFrame,
    feature_cols: list[str],
    processed_dir: Path,
    models_dir: Path,
    seed: int,
    synthetic_weight: float,
    synthetic_frac: float,
    use_gpu: bool,
) -> dict[str, float]:
    model_names = ["xgboost", "ridge", "rf"]

    train_df = _downsample_synthetic(df[df["split"] == "train"].copy(), synthetic_frac, seed)
    X_train  = train_df[feature_cols].values.astype(np.float32)
    y_train  = train_df["score_norm"].values.astype(np.float32)
    w_train  = _sample_weights(train_df, synthetic_weight)

    val_df  = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    X_val   = val_df[feature_cols].values.astype(np.float32)
    y_val   = val_df["score_norm"].values.astype(np.float32)
    X_te    = test_df[feature_cols].values.astype(np.float32)
    y_te    = test_df["score_norm"].values.astype(np.float32)

    trained: dict[str, Any] = {}
    val_preds: dict[str, np.ndarray] = {}
    te_preds:  dict[str, np.ndarray] = {}

    for name in model_names:
        params = best_params[name]
        log.info("Финальное обучение [%s] params=%s", name, params)

        if name == "xgboost":
            if xgb is None:
                raise ModuleNotFoundError("xgboost не установлен")
            xgb_params = {
                "objective":        "reg:squarederror",
                "tree_method":      "hist",
                "device":           "cuda" if use_gpu else "cpu",
                "nthread":          2 if use_gpu else -1,
                "seed":             seed,
                "learning_rate":    params["learning_rate"],
                "max_depth":        params["max_depth"],
                "min_child_weight": params["min_child_weight"],
                "subsample":        params["subsample"],
                "colsample_bytree": params["colsample_bytree"],
                "reg_lambda":       params["reg_lambda"],
                "reg_alpha":        params["reg_alpha"],
                "gamma":            params["gamma"],
            }
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
            cb = _PearsonStop(X_val, y_val, patience=80)
            model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=params.get("num_round", 1500),
                callbacks=[cb],
                verbose_eval=False,
            )
            log.info("XGBoost best_iter=%d val_pearson=%.4f", cb.best_iter, cb._best)
            save_path = models_dir / MODEL_FILES["xgboost"]
            model.save_model(str(save_path))

        elif name == "ridge":
            model = Ridge(alpha=params["alpha"], random_state=seed)
            model.fit(X_train, y_train, sample_weight=w_train)
            save_path = models_dir / MODEL_FILES["ridge"]
            with open(save_path, "wb") as f:
                pickle.dump(model, f)

        elif name == "rf":
            model = _make_rf(params, seed, use_gpu)
            if use_gpu and _CUML_AVAILABLE:
                model.fit(X_train, y_train)  # cuML не поддерживает sample_weight
            else:
                model.fit(X_train, y_train, sample_weight=w_train)
            save_path = models_dir / MODEL_FILES["rf"]
            with open(save_path, "wb") as f:
                pickle.dump(model, f)

        trained[name] = model
        val_preds[name] = _predict(name, model, X_val)
        te_preds[name]  = _predict(name, model, X_te)
        _log_metrics(y_val, val_preds[name], f"[{name}] val")
        _log_metrics(y_te,  te_preds[name],  f"[{name}] test")
        log.info("Модель сохранена: %s", save_path)

    # Оптимизация весов ансамбля
    n = len(model_names)
    V = np.stack([val_preds[nm] for nm in model_names], axis=1)

    def neg_pearson(w: np.ndarray) -> float:
        ensemble = V @ w
        r, _ = pearsonr(y_val, ensemble)
        return -r

    best_r, best_w = -np.inf, np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1.0}

    rng = np.random.default_rng(RANDOM_SEED)
    starts = [np.ones(n) / n] + [rng.dirichlet(np.ones(n)) for _ in range(12)]
    for w0 in starts:
        res = minimize(neg_pearson, w0, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-9, "maxiter": 500})
        if res.success and -res.fun > best_r:
            best_r = -res.fun
            best_w = res.x.copy()

    best_w = np.clip(best_w, 0.0, 1.0)
    best_w /= best_w.sum()
    weights = {nm: float(w) for nm, w in zip(model_names, best_w)}
    log.info("Веса ансамбля (val Pearson=%.4f): %s", best_r,
             {k: round(v, 4) for k, v in weights.items()})

    # Метрики ансамбля
    T = np.stack([te_preds[nm] for nm in model_names], axis=1)
    ensemble_val  = V @ best_w
    ensemble_test = T @ best_w
    _log_metrics(y_val, ensemble_val,  "Ансамбль val")
    _log_metrics(y_te,  ensemble_test, "Ансамбль test")

    return weights

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Тюнинг RF + level2 ансамбль")
    parser.add_argument("--trials",   type=int, default=20, help="Trials для RF (default: 20)")
    parser.add_argument("--cv-folds", type=int, default=3,  help="Фолды CV (default: 3)")
    parser.add_argument("--seed",     type=int, default=RANDOM_SEED)
    parser.add_argument("--gpu",      action="store_true",
                        help="GPU для XGBoost в level2 (device=cuda)")
    parser.add_argument("--data-dir",  type=Path, default=None)
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--synthetic-weight", type=float, default=0.12)
    parser.add_argument("--synthetic-frac",   type=float, default=0.3)
    args = parser.parse_args()

    settings = get_settings()
    init_script_runtime()

    processed_dir = (args.data_dir or Path(settings.data_dir)) / "processed"
    models_dir    = args.models_dir or Path(settings.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    df, feature_cols = load_data(processed_dir)

    # --- RF тюнинг ---
    train_df = _downsample_synthetic(
        df[df["split"] == "train"].copy(), args.synthetic_frac, args.seed
    )
    X_tr = train_df[feature_cols].values.astype(np.float32)
    y_tr = train_df["score_norm"].values.astype(np.float32)
    w_tr = _sample_weights(train_df, args.synthetic_weight)

    log.info("Тюнинг RF: %d trials, %d folds ...", args.trials, args.cv_folds)
    t0 = time.monotonic()
    rf_params = tune_rf(X_tr, y_tr, w_tr, args.trials, args.cv_folds, args.seed, args.gpu)
    log.info("RF тюнинг завершён за %.1f сек.", time.monotonic() - t0)

    best_params = {
        "xgboost": BEST_PARAMS_XGBOOST,
        "ridge":   BEST_PARAMS_RIDGE,
        "rf":      rf_params,
    }

    # Сохраняем tune_results.json
    results_path = models_dir / TUNE_RESULTS_FILE
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {"level1": [{"model": k, "best_params": v} for k, v in best_params.items()]},
            f, ensure_ascii=False, indent=2,
        )
    log.info("tune_results.json сохранён: %s", results_path)

    # --- Level 2 ---
    log.info("Запуск level2 ...")
    t0 = time.monotonic()
    weights = run_level2(
        best_params=best_params,
        df=df,
        feature_cols=feature_cols,
        processed_dir=processed_dir,
        models_dir=models_dir,
        seed=args.seed,
        synthetic_weight=args.synthetic_weight,
        synthetic_frac=args.synthetic_frac,
        use_gpu=args.gpu,
    )
    log.info("Level2 завершён за %.1f сек.", time.monotonic() - t0)

    weights_path = models_dir / ENSEMBLE_WEIGHTS_FILE
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(weights, f, ensure_ascii=False, indent=2)
    log.info("Веса ансамбля сохранены: %s", weights_path)


if __name__ == "__main__":
    main()
