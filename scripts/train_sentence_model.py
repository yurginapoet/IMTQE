"""
scripts/train_sentence_model.py

Шаг 5 из пайплайна MTQE.
Обучает XGBoost регрессор на признаках из sentence_da_features.parquet.
Целевая переменная: score_norm (DA score нормализованный в [0,1]).

Выходные файлы:
  models/xgboost_sentence.pkl
  models/shap_explainer.pkl

Метрики:
  Pearson r  — на DA test (5%)
  Spearman ρ — на HF MQM dedup (внешний тест, только ранговая корреляция)

Запуск:
  python scripts/train_sentence_model.py
  python scripts/train_sentence_model.py --data-dir data --models-dir models
"""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from scipy.stats import pearsonr, spearmanr
from xgboost import XGBRegressor

from src.features.extractor import FEATURE_NAMES_LIGHT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RANDOM_SEED = 42

# признаки которые есть в parquet — лёгкие (16) или все 22 если считались с --heavy
# скрипт автоматически берёт те что есть
HEAVY_FEATURES = [
    "cosine_similarity", "embedding_distance",
    "perplexity", "mean_log_prob", "token_ppl_variance", "min_token_log_prob",
]


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Берём все признаки которые есть в датасете."""
    from src.features.extractor import FEATURE_NAMES
    available = [f for f in FEATURE_NAMES if f in df.columns]
    log.info("Признаков для обучения: %d", len(available))
    return available


def load_data(processed_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    path = processed_dir / "sentence_da_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}. Запусти extract_features.py")

    df = pd.read_parquet(path)
    log.info("Загружено: %d строк, %d колонок", len(df), len(df.columns))

    feature_cols = get_feature_cols(df)
    return df, feature_cols


def train(
    df: pd.DataFrame,
    feature_cols: list[str],
    models_dir: Path,
) -> XGBRegressor:
    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "val"]
    test_df  = df[df["split"] == "test"]

    log.info("train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    X_train = train_df[feature_cols].values
    y_train = train_df["score_norm"].values
    X_val   = val_df[feature_cols].values
    y_val   = val_df["score_norm"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["score_norm"].values

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )

    log.info("Обучение XGBoost...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    # метрики на test
    preds_test = model.predict(X_test)
    r, _   = pearsonr(y_test, preds_test)
    rho, _ = spearmanr(y_test, preds_test)
    log.info("DA test — Pearson r=%.4f  Spearman rho=%.4f", r, rho)

    # сохранение модели
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "xgboost_sentence.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    log.info("Модель сохранена: %s", model_path)

    return model


def build_shap_explainer(
    model: XGBRegressor,
    df: pd.DataFrame,
    feature_cols: list[str],
    models_dir: Path,
) -> None:
    log.info("Строим SHAP explainer...")
    explainer = shap.TreeExplainer(model)

    # проверяем на train сете
    X_train = df[df["split"] == "train"][feature_cols].values
    shap_values = explainer.shap_values(X_train[:100])  # 100 примеров для проверки
    log.info("SHAP values shape: %s", shap_values.shape)

    explainer_path = models_dir / "shap_explainer.pkl"
    with open(explainer_path, "wb") as f:
        pickle.dump(explainer, f)
    log.info("SHAP explainer сохранён: %s", explainer_path)


def external_test(
    model: XGBRegressor,
    processed_dir: Path,
    feature_cols: list[str],
) -> None:
    mqm_path = processed_dir / "hf_mqm_dedup.parquet"
    if not mqm_path.exists():
        log.warning("hf_mqm_dedup.parquet не найден - пропускаем внешний тест")
        return

    mqm = pd.read_parquet(mqm_path)

    # берём только признаки которые есть в MQM датасете
    missing = [f for f in feature_cols if f not in mqm.columns]
    if missing:
        log.warning(
            "MQM датасет не содержит признаков: %s. "
            "Запусти extract_features.py на MQM данных.", missing
        )
        return

    X_mqm = mqm[feature_cols].values
    preds = model.predict(X_mqm)
    rho, _ = spearmanr(mqm["score"].values, preds)
    log.info("MQM внешний тест — Spearman rho=%.4f", rho)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    processed_dir = args.data_dir / "processed"

    log.info("=== train_sentence_model.py ===")

    df, feature_cols = load_data(processed_dir)
    model = train(df, feature_cols, args.models_dir)
    build_shap_explainer(model, df, feature_cols, args.models_dir)
    external_test(model, processed_dir, feature_cols)

    log.info("=== Готово. Следующий шаг: scripts/train_span_model.py ===")


if __name__ == "__main__":
    main()