# scripts/train_neural_head.py

"""
Обучение нейронной головы (FeatureAttentionHead) на признаках из
sentence_da_features.parquet.

Вход:  data/processed/sentence_da_features.parquet
       (столбцы schema.SENTENCE_FEATURE_NAMES — база + interaction, см. extract_features)
Выход: models/neural_head.pt
       models/neural_head_config.json

На инференсе Predictor подхватывает neural_head (если файлы есть) и строит поле
explanation как доли «потери» до 100%%: (1 − score_головы) распределяется по
attention-весам и суммируется по MQM-категориям (см. FeatureAttentionHead.explain_mqm_loss_shares).

Запуск:
  python scripts/train_neural_head.py
  python scripts/train_neural_head.py --epochs 150 --lr 1e-3
  python scripts/train_neural_head.py --eval-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features.schema import SENTENCE_FEATURE_NAMES
from src.models.neural_head import FeatureAttentionHead
from src.models.sentence_model import MQM_CATEGORY_RU

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RANDOM_SEED = 42


def load_data(processed_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    path = processed_dir / "sentence_da_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Не найден файл: {path}\n"
            "Сначала запусти: python scripts/extract_features.py"
        )

    df = pd.read_parquet(path)
    log.info("Загружено: %d строк, %d колонок", len(df), len(df.columns))

    # Берём только те признаки из SENTENCE_FEATURE_NAMES которые есть в датасете
    feature_cols = [f for f in SENTENCE_FEATURE_NAMES if f in df.columns]
    missing = [f for f in SENTENCE_FEATURE_NAMES if f not in df.columns]
    if missing:
        log.warning("Отсутствуют признаки: %s", missing)
    log.info("Признаков для обучения: %d", len(feature_cols))

    return df, feature_cols


def make_tensors(
    df: pd.DataFrame,
    feature_cols: list[str],
    split: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    frame = df[df["split"] == split]
    X = torch.tensor(frame[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(frame["score_norm"].values, dtype=torch.float32)
    return X, y


def log_metrics(y_true: np.ndarray, preds: np.ndarray, label: str) -> None:
    r, _ = pearsonr(y_true, preds)
    rho, _ = spearmanr(y_true, preds)
    log.info("%s — Pearson r=%.4f  Spearman ρ=%.4f", label, r, rho)

def make_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    frame = df[df["split"] == split]
    X = frame[feature_cols].values.astype(np.float32)
    y = frame["score_norm"].values.astype(np.float32)
    return X, y

def train(
    df: pd.DataFrame,
    feature_cols: list[str],
    models_dir: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
) -> FeatureAttentionHead:

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # --- Загружаем XGBoost и считаем его предсказания ---
    import xgboost as xgb
    xgb_model_path = models_dir / "xgboost_sentence.model"
    if not xgb_model_path.exists():
        raise FileNotFoundError(
            f"Не найден {xgb_model_path}. "
            "Сначала обучи XGBoost: python scripts/train_sentence_model.py"
        )
    booster = xgb.Booster()
    booster.load_model(str(xgb_model_path))
    log.info("XGBoost загружен для стекинга")

    def get_xgb_preds(X_np: np.ndarray) -> np.ndarray:
        dm = xgb.DMatrix(X_np)
        return booster.predict(dm).astype(np.float32)

    X_tr_np, y_tr_np = make_arrays(df, feature_cols, "train")
    X_val_np, y_val_np = make_arrays(df, feature_cols, "val")
    X_te_np, y_te_np = make_arrays(df, feature_cols, "test")

    xgb_tr  = get_xgb_preds(X_tr_np)   # (N_train,)
    xgb_val = get_xgb_preds(X_val_np)
    xgb_te  = get_xgb_preds(X_te_np)

    log.info(
        "XGBoost val Pearson до стекинга: %.4f",
        pearsonr(y_val_np, xgb_val)[0],
    )

    # --- Собираем расширенный вход: признаки + xgb_score ---
    # Вход нейронной головы = [len(feature_cols) признаков, xgb_score]
    def concat_with_xgb(X_np: np.ndarray, xgb_preds: np.ndarray) -> torch.Tensor:
        xgb_col = xgb_preds.reshape(-1, 1)
        return torch.tensor(
            np.concatenate([X_np, xgb_col], axis=1),
            dtype=torch.float32,
        )

    X_tr  = concat_with_xgb(X_tr_np, xgb_tr)
    X_val = concat_with_xgb(X_val_np, xgb_val)
    X_te  = concat_with_xgb(X_te_np, xgb_te)

    y_tr  = torch.tensor(y_tr_np,  dtype=torch.float32)
    y_val = torch.tensor(y_val_np, dtype=torch.float32)
    y_te  = torch.tensor(y_te_np,  dtype=torch.float32)

    input_dim = X_tr.shape[1]
    log.info("Вход нейронной головы: %d (sentence признаки + xgb_score)", input_dim)

    loader = DataLoader(
        TensorDataset(X_tr, y_tr),
        batch_size=batch_size,
        shuffle=True,
    )

    model = FeatureAttentionHead(input_dim=input_dim)
    log.info("Параметров: %d", sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01,
    )
    criterion = nn.HuberLoss(delta=0.1)

    best_pearson = -1.0
    best_state: dict | None = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            preds, _ = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_preds, _ = model(X_val)
        r, _ = pearsonr(y_val.numpy(), val_preds.numpy())

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                "Epoch %3d/%d  loss=%.4f  val_Pearson=%.4f  best=%.4f",
                epoch, epochs, epoch_loss / len(loader), r, best_pearson,
            )

        if r > best_pearson + 1e-4:
            best_pearson = r
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            log.info("Early stopping @ epoch %d", epoch)
            break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_preds, _ = model(X_te)
    log_metrics(y_te.numpy(), test_preds.numpy(), "TEST (стекинг)")

    # Сохраняем с расширенным списком признаков
    stacking_feature_names = feature_cols + ["xgb_score"]
    model.save(models_dir, stacking_feature_names)

    return model


def eval_only(
    df: pd.DataFrame,
    feature_cols: list[str],
    models_dir: Path,
) -> None:
    model, loaded_features = FeatureAttentionHead.load(models_dir)
    log.info("Загружена модель с %d признаками", len(loaded_features))

    # Используем признаки из сохранённого конфига
    feature_cols = [f for f in loaded_features if f in df.columns]

    X_te, y_te = make_tensors(df, feature_cols, "test")
    with torch.no_grad():
        preds, _ = model(X_te)
    log_metrics(y_te.numpy(), preds.numpy(), "TEST (eval_only)")

    # Пример объяснения на первых 3 примерах
    log.info("Пример attention по признакам (топ-5) и доли потерь по MQM:")
    for i in range(min(3, len(X_te))):
        x_one = X_te[i].numpy()
        explanation = model.explain(x_one, feature_cols)
        top5 = list(explanation.items())[:5]
        loss_cat = model.explain_mqm_loss_shares(x_one, feature_cols, min_category_share=0.005)
        loss_ru = {MQM_CATEGORY_RU.get(k, k): float(v) for k, v in loss_cat.items()}
        log.info("  Пример %d attention: %s", i, top5)
        log.info("  Пример %d доли потерь (RU): %s", i, loss_ru)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs",     type=int,  default=120)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int,  default=256)
    parser.add_argument("--patience",   type=int,  default=20)
    parser.add_argument("--eval-only",  action="store_true")
    args = parser.parse_args()

    processed_dir = args.data_dir / "processed"
    log.info("=== train_neural_head.py ===")

    df, feature_cols = load_data(processed_dir)

    if args.eval_only:
        eval_only(df, feature_cols, args.models_dir)
        return

    train(
        df=df,
        feature_cols=feature_cols,
        models_dir=args.models_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
    )
    log.info("=== Готово. Модель в models/neural_head.pt ===")


if __name__ == "__main__":
    main()