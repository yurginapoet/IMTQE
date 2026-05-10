"""
scripts/extract_features.py

Шаг feature extraction для sentence- и word-level датасетов.

Новый формат:
  22 handcrafted/classic признака
  + 64 semantic PCA признака
  = 86 признаков
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features.extractor import FEATURE_NAMES, FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def resolve_sentence_input(processed_dir: Path) -> Path:
    augmented = processed_dir / "sentence_da_augmented.parquet"
    if augmented.exists():
        return augmented
    return processed_dir / "sentence_da.parquet"


def extract_for_df(
    df: pd.DataFrame,
    src_col: str,
    mt_col: str,
    extractor: FeatureExtractor,
    batch_size: int,
    progress_desc: str,
) -> pd.DataFrame:
    """
    Извлекает все активные признаки для всех строк df батчами.
    Возвращает df с добавленными числовыми колонками и word_logprobs.
    """
    feature_names = extractor.active_feature_names
    n = len(df)
    all_vectors = np.zeros((n, len(feature_names)), dtype=np.float32)
    all_word_logprobs = [None] * n

    pairs = list(zip(df[src_col].astype(str), df[mt_col].astype(str), strict=False))

    for start in tqdm(range(0, n, batch_size), desc=progress_desc, unit="batch"):
        batch = pairs[start : start + batch_size]
        results = extractor.extract_batch(batch)
        for idx, result in enumerate(results):
            all_vectors[start + idx] = result["vector"]
            all_word_logprobs[start + idx] = result["word_logprobs"]

    df = df.copy()
    for feat_idx, name in enumerate(feature_names):
        df[name] = all_vectors[:, feat_idx]
    df["word_logprobs"] = all_word_logprobs

    log.info(
        "Признаки добавлены: %d числовых + word_logprobs. Итого колонок: %d",
        len(feature_names),
        len(df.columns),
    )
    return df


def process_dataset(
    name: str,
    in_path: Path,
    out_path: Path,
    src_col: str,
    mt_col: str,
    extractor: FeatureExtractor,
    batch_size: int,
    force: bool,
) -> None:
    if out_path.exists() and not force:
        log.info("%s уже существует - пропускаем. Используй --force для пересчёта.", out_path.name)
        return

    if not in_path.exists():
        log.warning("Входной файл не найден: %s - пропускаем %s", in_path, name)
        return

    log.info("--- %s ---", name)
    log.info("Загрузка: %s", in_path)
    df = pd.read_parquet(in_path)
    log.info("Строк: %d", len(df))

    df = extract_for_df(
        df,
        src_col=src_col,
        mt_col=mt_col,
        extractor=extractor,
        batch_size=batch_size,
        progress_desc=f"{name} feature batches",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("Сохранено: %s (%d строк, %d колонок)", out_path, len(df), len(df.columns))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--only", choices=["da", "wl", "mqm"], help="Обработать только один датасет")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Пересчитать признаки даже если выходной файл уже существует",
    )
    args = parser.parse_args()

    processed_dir = args.data_dir / "processed"
    sentence_input = resolve_sentence_input(processed_dir)

    log.info("=== extract_features.py ===")
    log.info("Sentence-level источник: %s", sentence_input.name)

    extractor = FeatureExtractor()
    extractor.load_heavy_models(require_neural=True)
    if extractor.active_feature_names != FEATURE_NAMES:
        raise RuntimeError(
            "FeatureExtractor не активировал полный 86-мерный набор признаков. "
            "Проверь наличие models/semantic_pca.pkl и доступность MiniLM."
        )
    log.info("Все модели загружены. Считаем %d признаков.", len(FEATURE_NAMES))

    datasets = {
        "da": (
            sentence_input,
            processed_dir / "sentence_da_features.parquet",
            "src",
            "mt",
        ),
        "wl": (
            processed_dir / "wordlevel_train.parquet",
            processed_dir / "wordlevel_features.parquet",
            "src",
            "mt",
        ),
        "mqm": (
            processed_dir / "hf_mqm_dedup.parquet",
            processed_dir / "hf_mqm_features.parquet",
            "src",
            "mt",
        ),
    }

    for key, (in_path, out_path, src_col, mt_col) in datasets.items():
        if args.only and args.only != key:
            continue
        process_dataset(
            name=key,
            in_path=in_path,
            out_path=out_path,
            src_col=src_col,
            mt_col=mt_col,
            extractor=extractor,
            batch_size=args.batch_size,
            force=args.force,
        )

    log.info("=== Готово. Следующий шаг: scripts/train_sentence_model.py ===")


if __name__ == "__main__":
    main()
