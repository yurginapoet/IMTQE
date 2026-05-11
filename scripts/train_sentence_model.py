"""
scripts/train_sentence_model.py

Шаг 5 из пайплайна MTQE.
Обучает sentence-level модель на признаках из sentence_da_features.parquet.
Целевая переменная: score_norm (DA score нормализованный в [0,1]).

Поддерживаемые модели (флаг --model):
  xgboost  — XGBRegressor, eval_metric=Pearson, точечные предсказания
  ngboost  — NGBRegressor + Beta distribution, предсказывает (α, β), CI₉₅

Выходные файлы:
  models/xgboost_sentence.pkl   или   models/ngboost_sentence.pkl
  models/shap_explainer.pkl

Метрики:
  Pearson r  — на DA test (5%)
  Spearman ρ — на HF MQM dedup (внешний тест, только ранговая корреляция)

Запуск:
  python scripts/train_sentence_model.py                        # xgboost по умолчанию
  python scripts/train_sentence_model.py --model ngboost        # ngboost + Beta
  python scripts/train_sentence_model.py --eval-only            # только внешний тест
  python scripts/train_sentence_model.py --eval-only --model ngboost
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from scipy.stats import pearsonr, spearmanr
from scipy.stats import zscore as scipy_zscore
import xgboost as xgb
from xgboost import DMatrix
from xgboost import XGBRegressor
from xgboost.callback import TrainingCallback as _XGBTrainingCallback

from ngboost import NGBRegressor
from ngboost.distns import Beta

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features.schema import FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RANDOM_SEED = 42
# NGBoost: Beta требует строго (0, 1), не включая границы
BETA_EPS = 1e-4
DEFAULT_SYNTHETIC_WEIGHT = 0.10


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """
    Берём все признаки из FEATURE_NAMES которые есть в датасете.
    Импорт здесь чтобы не падать если src.features недоступен при --eval-only.
    """
    available = [f for f in FEATURE_NAMES if f in df.columns]
    missing   = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        log.warning(
            "Отсутствуют признаки в датасете: %s\n"
            "Убедись что extract_features.py был запущен без --only и без флагов.",
            missing,
        )
    log.info("Признаков для обучения: %d / %d", len(available), len(FEATURE_NAMES))
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

    # Диагностика целевой переменной
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


def _split_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, ...]:
    """Возвращает X_train, y_train, X_val, y_val, X_test, y_test."""
    train_df, val_df, test_df = _split_frames(df)

    return (
        train_df[feature_cols].values,
        train_df["score_norm"].values,
        val_df[feature_cols].values,
        val_df["score_norm"].values,
        test_df[feature_cols].values,
        test_df["score_norm"].values,
    )


def _build_train_sample_weights(
    train_df: pd.DataFrame,
    synthetic_weight: float = 1.0,
    low_score_tau: float | None = None,
    low_score_weight: float = 1.0,
) -> np.ndarray:
    """
    Веса для train-set.
    """
    weights = np.ones(len(train_df), dtype=np.float32)

    # Synthetic negatives — снижаем их влияние
    if "is_synthetic" in train_df.columns:
        synthetic_mask = train_df["is_synthetic"].fillna(False).to_numpy(dtype=bool)
        synthetic_count = int(synthetic_mask.sum())
        if synthetic_count:
            weights[synthetic_mask] *= float(synthetic_weight)
            effective_share = weights[synthetic_mask].sum() / max(weights.sum(), 1e-8)
            log.info(
                "Synthetic train rows: %d / %d  weight=%.3f  effective_share=%.1f%%",
                synthetic_count, len(train_df), synthetic_weight, effective_share * 100
            )

    # Low-score upweighting — делаем более targeted
    if low_score_tau is not None and low_score_weight > 1.0:
        low_score_mask = train_df["score_norm"].to_numpy() < float(low_score_tau)
        affected = int(low_score_mask.sum())
        weights[low_score_mask] *= float(low_score_weight)
        
        log.info(
            "Low-score upweighting: tau=%.2f  weight=%.2f  affected=%d / %d (%.1f%%)",
            low_score_tau, low_score_weight, affected, len(train_df), 100 * affected / len(train_df)
        )

    return weights


def _log_metrics(y_true: np.ndarray, preds: np.ndarray, label: str) -> None:
    r,   _ = pearsonr(y_true, preds)
    rho, _ = spearmanr(y_true, preds)
    log.info("%s — Pearson r=%.4f  Spearman rho=%.4f", label, r, rho)


class _PearsonCallback(_XGBTrainingCallback):
    """
    Early stopping по val Pearson r.
    Определён на уровне модуля чтобы pickle мог сериализовать модель.
    Наследуется от TrainingCallback — это требование XGBoost.
    Внутри after_iteration model — Booster, predict требует DMatrix.
    """

    def __init__(self, X_val: np.ndarray, y_val: np.ndarray, patience: int = 50) -> None:
        super().__init__()
        self.dval      = DMatrix(X_val)
        self.y_val     = y_val
        self.patience  = patience
        self.best_r    = -np.inf
        self.best_iter = 0
        self.no_improve = 0

    def after_iteration(self, model: Any, epoch: int, evals_log: Any) -> bool:
        preds = model.predict(self.dval)
        r, _  = pearsonr(self.y_val, preds)
        if r > self.best_r + 1e-5:
            self.best_r     = r
            self.best_iter  = epoch
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
) -> Any:
    train_df, val_df, test_df = _split_frames(df)
    
    X_train = train_df[feature_cols].values
    y_train = train_df["score_norm"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["score_norm"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["score_norm"].values

    # Очень мягкий upweighting
    train_weights = _build_train_sample_weights(
        train_df,
        synthetic_weight=synthetic_weight,
        low_score_tau=0.25,      # только самые плохие ~30%
        low_score_weight=1.8,
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=train_weights)
    dval   = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "reg:squarederror",
        "learning_rate": 0.045,
        "max_depth": 7,
        "min_child_weight": 3,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_lambda": 0.08,
        "reg_alpha": 0.02,
        "max_bin": 1024,
        "tree_method": "hist",
        "seed": RANDOM_SEED,
        "verbosity": 0,
    }

    pearson_cb = _PearsonCallback(X_val, y_val, patience=120)

    log.info("Запуск обучения XGBoost...")
    booster = xgb.train(
        params, dtrain, num_boost_round=1400,
        evals=[(dtrain, "train"), (dval, "val")],
        callbacks=[pearson_cb],
        verbose_eval=100,
    )

    best_iter = pearson_cb.best_iter
    if best_iter + 1 < booster.num_boosted_rounds():
        booster = booster[:best_iter + 1]

    log.info(f"Best iter={best_iter}  val Pearson={pearson_cb.best_r:.4f}")

    # === Strong Hybrid bias toward cosine ===
    dval_pred = booster.predict(dval)
    cos_idx = feature_cols.index("cosine_similarity")
    cos_val = X_val[:, cos_idx].astype(np.float32)

    from scipy.optimize import minimize_scalar
    from scipy.stats import pearsonr

    def objective(w):
        hybrid = w * dval_pred + (1 - w) * cos_val
        return -pearsonr(y_val, hybrid)[0]

    res = minimize_scalar(objective, bounds=(0.45, 0.70), method='bounded')
    best_w_tree = float(res.x)

    log.info(f"Hybrid → w_tree={best_w_tree:.3f} | w_cos={1-best_w_tree:.3f}")

    hybrid_meta = {
        "w_tree": best_w_tree,
        "w_cos": 1.0 - best_w_tree,
        "hybrid_enabled": True
    }
    import json
    (models_dir / "hybrid_meta.json").write_text(json.dumps(hybrid_meta, indent=2))

    model_path = models_dir / "xgboost_sentence.model"
    booster.save_model(str(model_path))

    dtest = xgb.DMatrix(X_test)
    _log_metrics(y_test, booster.predict(dtest), "DA test")

    model = XGBRegressor()
    model.load_model(str(model_path))
    return model

# ---------------------------------------------------------------------------
# NGBoost
# ---------------------------------------------------------------------------

def train_ngboost(
    df: pd.DataFrame,
    feature_cols: list[str],
    models_dir: Path,
    synthetic_weight: float,
) -> Any:
    train_df, val_df, test_df = _split_frames(df)
    X_train = train_df[feature_cols].values
    y_train = train_df["score_norm"].values
    X_val = val_df[feature_cols].values
    y_val = val_df["score_norm"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["score_norm"].values

    # Beta требует строго (0, 1): клиппинг с запасом
    y_train = y_train.clip(BETA_EPS, 1 - BETA_EPS)
    y_val   = y_val.clip(BETA_EPS, 1 - BETA_EPS)
    y_test  = y_test.clip(BETA_EPS, 1 - BETA_EPS)

    # Asymmetric sample weights: плохие переводы важнее (из архитектуры)
    TAU, W_HIGH, W_LOW = 0.5, 3.0, 1.0
    sample_weight = _build_train_sample_weights(
        train_df,
        synthetic_weight=synthetic_weight,
        low_score_tau=TAU,
        low_score_weight=W_HIGH,
    )
    log.info(
        "Asymmetric weights: tau=%.2f  w_high=%.1f  w_low=%.1f  "
        "(плохих примеров: %d / %d, итоговый mean_weight=%.3f)",
        TAU, W_HIGH, W_LOW,
        (y_train < TAU).sum(), len(y_train),
        float(sample_weight.mean()),
    )

    model = NGBRegressor(
        Dist=Beta,
        n_estimators=800,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
        verbose=100,
        verbose_eval=100,
    )

    log.info("Обучение NGBoost (Dist=Beta, asymmetric weights)...")
    model.fit(
        X_train, y_train,
        X_val=X_val, Y_val=y_val,
        early_stopping_rounds=50,
        sample_weight=sample_weight,
    )

    # Предсказание: ожидание Beta-распределения E[q] = α/(α+β)
    dist_test = model.pred_dist(X_test)
    preds_test = dist_test.mean()
    _log_metrics(y_test, preds_test, "DA test [NGBoost]")

    # Дополнительно: показываем неопределённость на первых 5 примерах
    alpha = dist_test.params["alpha"][:5]
    beta  = dist_test.params["beta"][:5]
    var   = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
    log.info("Uncertainty (Var первых 5 примеров): %s", np.round(var, 4))

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "ngboost_sentence.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info("Модель сохранена: %s", model_path)

    return model


# ---------------------------------------------------------------------------
# SHAP (общий для обеих моделей — TreeExplainer работает с обеими)
# ---------------------------------------------------------------------------

def build_shap_explainer(
    model: Any,
    df: pd.DataFrame,
    feature_cols: list[str],
    models_dir: Path,
    model_type: str,
) -> None:
    log.info("Строим SHAP explainer (model_type=%s)...", model_type)

    if model_type == "ngboost":
        # NGBoost: TreeExplainer строится на базовых деревьях (stage 0 = E[α])
        # shap поддерживает NGBoost начиная с версии 0.40
        base_learner = model.learners_[0]  # деревья для первого параметра (loc)
        explainer = shap.TreeExplainer(base_learner)
    else:
        explainer = shap.TreeExplainer(model)

    X_sample = df[df["split"] == "train"][feature_cols].values[:100]
    shap_values = explainer.shap_values(X_sample)
    log.info("SHAP values shape: %s", np.array(shap_values).shape)

    models_dir.mkdir(parents=True, exist_ok=True)
    explainer_path = models_dir / "shap_explainer.pkl"
    with open(explainer_path, "wb") as f:
        pickle.dump(explainer, f)
    log.info("SHAP explainer сохранён: %s", explainer_path)


# ---------------------------------------------------------------------------
# Внешний тест на MQM
# ---------------------------------------------------------------------------

def external_test(
    model: Any,
    processed_dir: Path,
    feature_cols: list[str],
    model_type: str,
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
    if missing:
        log.error(
            "В hf_mqm_features.parquet отсутствуют признаки: %s",
            missing,
        )
        return

    X_mqm = mqm[feature_cols].values

    if model_type == "ngboost":
        preds = model.pred_dist(X_mqm).mean()
    else:
        preds = model.predict(X_mqm)

    mqm_score_raw = mqm["score"].values

    log.info(
        "MQM score raw stats: mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
        mqm_score_raw.mean(), mqm_score_raw.std(),
        mqm_score_raw.min(), mqm_score_raw.max(),
    )

    # --- Нормализация MQM score ---
    # Вариант 1: zscore по системе (убирает межсистемные сдвиги).
    # Это стандарт WMT — сравниваем ранги внутри системы, не абсолютные числа.
    # Нужна колонка "system" в датасете.
    if "system" in mqm.columns:
        mqm_score = (
            mqm.groupby("system")["score"]
            .transform(lambda x: scipy_zscore(x, ddof=1))
            .fillna(0)
            .values
        )
        log.info("MQM score нормализован zscore по системе.")
    else:
        # Вариант 2: глобальный zscore если нет колонки system
        mqm_score = scipy_zscore(mqm_score_raw, ddof=1)
        log.info("MQM score нормализован глобальным zscore (нет колонки 'system').")

    log.info(
        "MQM score norm stats: mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
        mqm_score.mean(), mqm_score.std(),
        mqm_score.min(), mqm_score.max(),
    )
    log.info(
        "Preds stats: mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
        preds.mean(), preds.std(), preds.min(), preds.max(),
    )

    # Spearman по нормализованному score
    # MQM: больше = лучше (0=идеально, отрицательное=плохо)
    # DA preds: больше = лучше → направление совпадает, инверсия не нужна
    rho, pvalue = spearmanr(mqm_score, preds)
    log.info(
        "MQM внешний тест [%s] — Spearman ρ=%.4f  p=%.4f",
        model_type, rho, pvalue,
    )

    # Дополнительно: тест по квантилям (топ-20% vs bottom-20%)
    # Помогает понять различает ли модель явно плохие и хорошие переводы
    threshold_lo = np.percentile(mqm_score, 20)
    threshold_hi = np.percentile(mqm_score, 80)
    mask = (mqm_score <= threshold_lo) | (mqm_score >= threshold_hi)
    if mask.sum() > 100:
        rho_extreme, _ = spearmanr(mqm_score[mask], preds[mask])
        log.info(
            "MQM extremes (bottom/top 20%%, n=%d) — Spearman ρ=%.4f",
            mask.sum(), rho_extreme,
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _model_pkl_name(model_type: str) -> str:
    return f"{model_type}_sentence.pkl"


def _load_model_for_eval(model_type: str, models_dir: Path) -> Any:
    if model_type == "xgboost":
        model_path = models_dir / "xgboost_sentence.model"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Не найден файл модели: {model_path}\n"
                "Сначала обучи: python scripts/train_sentence_model.py --model xgboost"
            )
        model = XGBRegressor()
        model.load_model(str(model_path))
        return model

    model_path = models_dir / _model_pkl_name(model_type)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Не найден файл модели: {model_path}\n"
            f"Сначала обучи: python scripts/train_sentence_model.py --model {model_type}"
        )
    with open(model_path, "rb") as f:
        return pickle.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--model",
        choices=["xgboost", "ngboost"],
        default="xgboost",
        help="Какую модель обучать: xgboost (default) или ngboost+Beta",
    )
    parser.add_argument(
        "--eval-only", action="store_true",
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
    args = parser.parse_args()

    processed_dir = args.data_dir / "processed"
    log.info("=== train_sentence_model.py  [model=%s] ===", args.model)

    if args.eval_only:
        model = _load_model_for_eval(args.model, args.models_dir)
        df, feature_cols = load_data(processed_dir)
        external_test(model, processed_dir, feature_cols, args.model)
        return

    df, feature_cols = load_data(processed_dir)

    if args.model == "xgboost":
        model = train_xgboost(
            df,
            feature_cols,
            args.models_dir,
            synthetic_weight=args.synthetic_weight,
        )
    else:
        model = train_ngboost(
            df,
            feature_cols,
            args.models_dir,
            synthetic_weight=args.synthetic_weight,
        )

    build_shap_explainer(model, df, feature_cols, args.models_dir, args.model)
    external_test(model, processed_dir, feature_cols, args.model)

    log.info("=== Готово [%s]. Следующий шаг: scripts/train_span_model.py ===", args.model)


if __name__ == "__main__":
    main()
