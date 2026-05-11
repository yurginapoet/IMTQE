"""
scripts/prepare_data.py

Шаг 1 из пайплайна MTQE:
  - Загрузка HF DA  (RicardoRei/wmt-da-human-evaluation, EN-RU)
  - Загрузка HF MQM (RicardoRei/wmt-mqm-human-evaluation, EN-RU)
  - Нормализация DA score в [0, 1] по train-сету (min-max)
  - Стратифицированный split 85/10/5 по квантилям score
  - Сохранение parquet-файлов в data/processed/

Выходные файлы:
  data/processed/sentence_da.parquet     — HF DA с нормализованным score
  data/processed/hf_mqm_raw.parquet      — HF MQM (score, до дедупликации)

Запуск:
  python scripts/prepare_data.py [--data-dir data] [--seed 42]
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.bootstrap import init_script_runtime
from src.determinism import seed_everything
from src.settings import get_settings

log = logging.getLogger(__name__)

# ── константы────

RANDOM_SEED = 42
EPS = 1e-4          # клиппинг для Beta: score ∈ [EPS, 1-EPS]
N_STRAT_BINS = 5    # число бинов для стратификации split
SPLIT_TRAIN  = 0.85
SPLIT_VAL    = 0.10
# SPLIT_TEST   = 0.05  # остаток

# HF DA: поле с raw оценкой называется "score" (Direct Assessment [0, 100])
DA_SCORE_COL = "score"
DA_SRC_COL   = "src"
DA_MT_COL    = "mt"
DA_LP_COL    = "lp"           # language pair
DA_TARGET_LP = "en-ru"

# HF MQM: оценка качества (штрафная, чем ниже — тем хуже)
MQM_SCORE_COL = "score"
MQM_SRC_COL   = "src"
MQM_MT_COL    = "mt"
MQM_LP_COL    = "lp"
MQM_TARGET_LP = "en-ru"


# ── утилиты──────

def make_hash(src: str, mt: str) -> str:
    """sha256-хэш пары (src, mt) — для дедупликации в dedup_mqm.py."""
    key = src.strip() + "|||" + mt.strip()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def stratified_split(
    df: pd.DataFrame,
    score_col: str,
    n_bins: int = N_STRAT_BINS,
    train_frac: float = SPLIT_TRAIN,
    val_frac: float   = SPLIT_VAL,
    seed: int         = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Стратифицированный split по квантилям score_col.
    Добавляет колонку 'split': 'train' | 'val' | 'test'.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    # квантильные бины (одинаковое число примеров в каждом)
    df["_bin"] = pd.qcut(df[score_col], q=n_bins, labels=False, duplicates="drop")

    splits = np.full(len(df), "test", dtype=object)

    for bin_id, group in df.groupby("_bin"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_frac)
        n_val   = int(n * val_frac)
        splits[idx[:n_train]]              = "train"
        splits[idx[n_train:n_train+n_val]] = "val"
        # остаток → "test"

    df["split"] = splits
    df.drop(columns=["_bin"], inplace=True)
    return df


# ── шаг 1: HF DA

def load_hf_da(processed_dir: Path, seed: int, force: bool = False) -> pd.DataFrame:
    """
    Загружает RicardoRei/wmt-da-human-evaluation, фильтрует EN-RU,
    нормализует score min-max по train-сету, добавляет split.
    """
    out_path = processed_dir / "sentence_da.parquet"
    if out_path.exists() and not force:
        log.info("HF DA уже существует: %s — пропускаем загрузку", out_path)
        return pd.read_parquet(out_path)

    log.info("Загрузка RicardoRei/wmt-da-human-evaluation …")
    ds = load_dataset("RicardoRei/wmt-da-human-evaluation", split="train", trust_remote_code=True)
    df = ds.to_pandas()
    log.info("Загружено %d строк (все LP)", len(df))

    # фильтр языковой пары────────────
    # Колонка lp может называться 'lp' или 'language_pair' — проверим оба
    lp_col = DA_LP_COL if DA_LP_COL in df.columns else "language_pair"
    if lp_col not in df.columns:
        raise KeyError(
            f"Не найдена колонка языковой пары. Доступные колонки: {list(df.columns)}"
        )

    df = df[df[lp_col] == DA_TARGET_LP].reset_index(drop=True)
    log.info("После фильтрации EN-RU: %d строк", len(df))

    if len(df) == 0:
        raise ValueError(
            f"Нет данных для LP={DA_TARGET_LP}. "
            f"Доступные LP: {df[lp_col].unique().tolist()}"
        )

    # проверка нужных колонок─────────
    for col in [DA_SCORE_COL, DA_SRC_COL, DA_MT_COL]:
        if col not in df.columns:
            raise KeyError(f"Отсутствует колонка '{col}'. Колонки: {list(df.columns)}")

    # стратифицированный split (до нормализации, чтобы бины по raw score) ─
    df = stratified_split(df, score_col=DA_SCORE_COL, seed=seed)
    log.info(
        "Split: train=%d  val=%d  test=%d",
        (df["split"] == "train").sum(),
        (df["split"] == "val").sum(),
        (df["split"] == "test").sum(),
    )

    # min-max нормализация ТОЛЬКО по train-сету ─────────────────────────
    train_mask = df["split"] == "train"
    score_min  = df.loc[train_mask, DA_SCORE_COL].min()
    score_max  = df.loc[train_mask, DA_SCORE_COL].max()
    log.info("DA score train: min=%.2f  max=%.2f", score_min, score_max)

    df["score_norm"] = (df[DA_SCORE_COL] - score_min) / (score_max - score_min)
    # клиппинг для Beta распределения (требует строго (0,1))
    df["score_norm"] = df["score_norm"].clip(EPS, 1 - EPS)

    # хэш для дедупликации с MQM─────
    df["pair_hash"] = df.apply(
        lambda r: make_hash(str(r[DA_SRC_COL]), str(r[DA_MT_COL])), axis=1
    )

    # статистика──────────────────────
    log.info(
        "score_norm (train): mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
        df.loc[train_mask, "score_norm"].mean(),
        df.loc[train_mask, "score_norm"].std(),
        df.loc[train_mask, "score_norm"].min(),
        df.loc[train_mask, "score_norm"].max(),
    )

    # сохранение──────────────────────
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("HF DA сохранён: %s  (%d строк, %d колонок)", out_path, len(df), len(df.columns))
    return df


# ── шаг 2: HF MQM───────────────────────

def load_hf_mqm(processed_dir: Path, force: bool = False) -> pd.DataFrame:
    """
    Загружает RicardoRei/wmt-mqm-human-evaluation, фильтрует EN-RU,
    отбирает строки с валидным score, сохраняет как hf_mqm_raw.parquet.
    Дедупликация (пересечение с DA train) выполняется в dedup_mqm.py.
    """
    out_path = processed_dir / "hf_mqm_raw.parquet"
    if out_path.exists() and not force:
        log.info("HF MQM уже существует: %s — пропускаем загрузку", out_path)
        return pd.read_parquet(out_path)

    log.info("Загрузка RicardoRei/wmt-mqm-human-evaluation …")
    ds = load_dataset("RicardoRei/wmt-mqm-human-evaluation", split="train", trust_remote_code=True)
    df = ds.to_pandas()
    log.info("Загружено %d строк (все LP)", len(df))

    # фильтр языковой пары────────────
    lp_col = MQM_LP_COL if MQM_LP_COL in df.columns else "language_pair"
    if lp_col not in df.columns:
        raise KeyError(
            f"Не найдена колонка языковой пары. Доступные колонки: {list(df.columns)}"
        )

    df = df[df[lp_col] == MQM_TARGET_LP].reset_index(drop=True)
    log.info("После фильтрации EN-RU: %d строк", len(df))

    if len(df) == 0:
        raise ValueError(
            f"Нет данных для LP={MQM_TARGET_LP}. "
            f"Доступные LP: {df[lp_col].unique().tolist()}"
        )

    # проверка нужных колонок─────────
    for col in [MQM_SCORE_COL, MQM_SRC_COL, MQM_MT_COL]:
        if col not in df.columns:
            raise KeyError(
                f"Отсутствует колонка '{col}'. Доступные колонки: {list(df.columns)}"
            )

    # фильтрация строк с NaN в score
    before = len(df)
    df = df.dropna(subset=[MQM_SCORE_COL]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        log.warning("Удалено %d строк с NaN в score", dropped)

    # хэш для дедупликации (dedup_mqm.py будет его использовать)
    df["pair_hash"] = df.apply(
        lambda r: make_hash(str(r[MQM_SRC_COL]), str(r[MQM_MT_COL])), axis=1
    )

    log.info(
        "MQM score: mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
        df[MQM_SCORE_COL].mean(),
        df[MQM_SCORE_COL].std(),
        df[MQM_SCORE_COL].min(),
        df[MQM_SCORE_COL].max(),
    )

    # сохранение──────────────────────
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("HF MQM сохранён: %s  (%d строк, %d колонок)", out_path, len(df), len(df.columns))
    return df


# ── main─────────

def parse_args() -> argparse.Namespace:
    s = get_settings()
    p = argparse.ArgumentParser(description="Загрузка и нормализация HF DA и HF MQM")
    p.add_argument(
        "--data-dir", type=Path, default=s.data_dir,
        help="Корневая директория данных",
    )
    p.add_argument(
        "--seed", type=int, default=s.random_seed,
        help="Random seed",
    )
    p.add_argument(
        "--skip-da",  action="store_true", help="Пропустить загрузку HF DA"
    )
    p.add_argument(
        "--skip-mqm", action="store_true", help="Пропустить загрузку HF MQM"
    )
    p.add_argument(
        "--force", action="store_true",
        help="Пересоздать parquet-файлы даже если они уже существуют",
    )
    return p.parse_args()


def main() -> None:
    init_script_runtime()
    args = parse_args()
    seed_everything(args.seed)
    processed_dir = args.data_dir / "processed"

    log.info("prepare_data: data_dir=%s seed=%d", args.data_dir, args.seed)

    if not args.skip_da:
        da_df = load_hf_da(processed_dir, seed=args.seed, force=args.force)
        log.info(
            "HF DA: %d rows (train=%d val=%d test=%d)",
            len(da_df),
            (da_df["split"] == "train").sum(),
            (da_df["split"] == "val").sum(),
            (da_df["split"] == "test").sum(),
        )
    else:
        log.info("HF DA skipped (--skip-da)")

    if not args.skip_mqm:
        mqm_df = load_hf_mqm(processed_dir, force=args.force)
        log.info("HF MQM: %d rows", len(mqm_df))
    else:
        log.info("HF MQM skipped (--skip-mqm)")

    log.info("prepare_data: finished")


if __name__ == "__main__":
    main()
