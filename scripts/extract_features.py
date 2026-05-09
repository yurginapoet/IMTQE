"""
scripts/extract_features.py

Шаг 4 из пайплайна MTQE.
Прогоняет все три датасета через FeatureExtractor и сохраняет все 22 признака.

LaBSE и ruGPT-3 загружаются всегда — все 22 признака считаются за один проход.
Запускать на Colab T4 (GPU ускоряет LaBSE и ruGPT-3).

Выходные файлы:
  data/processed/sentence_da_features.parquet   — DA датасет + 22 признака
  data/processed/wordlevel_features.parquet     — WMT21 word-level + 22 признака
  data/processed/hf_mqm_features.parquet        — MQM dedup + 22 признака

Запуск:
  python scripts/extract_features.py
  python scripts/extract_features.py --only da   # только DA
  python scripts/extract_features.py --only wl   # только wordlevel
  python scripts/extract_features.py --only mqm  # только MQM
  python scripts/extract_features.py --force     # пересчитать даже если файл есть
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.extractor import FEATURE_NAMES, FeatureExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BATCH_SIZE = 64


def extract_for_df(
    df: pd.DataFrame,
    src_col: str,
    mt_col: str,
    extractor: FeatureExtractor,
) -> pd.DataFrame:
    """
    Извлекает все 22 признака для всех строк df батчами.
    Возвращает df с добавленными колонками признаков и word_logprobs.
    """
    n = len(df)
    all_vectors       = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    all_word_logprobs = [None] * n

    pairs = list(zip(df[src_col], df[mt_col]))

    for start in tqdm(range(0, n, BATCH_SIZE), desc="Батчи"):
        batch   = pairs[start : start + BATCH_SIZE]
        results = extractor.extract_batch(batch)
        for i, res in enumerate(results):
            all_vectors[start + i]       = res["vector"]
            all_word_logprobs[start + i] = res["word_logprobs"]

    df = df.copy()
    for j, name in enumerate(FEATURE_NAMES):
        df[name] = all_vectors[:, j]
    df["word_logprobs"] = all_word_logprobs

    log.info(
        "Признаки добавлены: %d признаков + word_logprobs. Итого колонок: %d",
        len(FEATURE_NAMES), len(df.columns),
    )
    return df


def process_dataset(
    name: str,
    in_path: Path,
    out_path: Path,
    src_col: str,
    mt_col: str,
    extractor: FeatureExtractor,
    force: bool,
) -> None:
    if out_path.exists() and not force:
        log.info("%s уже существует — пропускаем. Используй --force для пересчёта.", out_path.name)
        return

    if not in_path.exists():
        log.warning("Входной файл не найден: %s — пропускаем %s", in_path, name)
        return

    log.info("--- %s ---", name)
    log.info("Загрузка: %s", in_path)
    df = pd.read_parquet(in_path)
    log.info("Строк: %d", len(df))

    df = extract_for_df(df, src_col=src_col, mt_col=mt_col, extractor=extractor)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("Сохранено: %s  (%d строк, %d колонок)", out_path, len(df), len(df.columns))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--only", choices=["da", "wl", "mqm"],
        help="Обработать только один датасет"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Пересчитать признаки даже если выходной файл уже существует"
    )
    args = parser.parse_args()

    processed_dir = args.data_dir / "processed"

    log.info("=== extract_features.py ===")

    log.info("Загрузка spaCy моделей...")
    extractor = FeatureExtractor()

    log.info("Загрузка LaBSE и ruGPT-3 (тяжёлые модели)...")
    extractor.load_heavy_models()
    log.info("Все модели загружены. Считаем все 22 признака.")

    datasets = {
        "da": (
            processed_dir / "sentence_da.parquet",
            processed_dir / "sentence_da_features.parquet",
            "src", "mt",
        ),
        "wl": (
            processed_dir / "wordlevel_train.parquet",
            processed_dir / "wordlevel_features.parquet",
            "src", "mt",
        ),
        "mqm": (
            processed_dir / "hf_mqm_dedup.parquet",
            processed_dir / "hf_mqm_features.parquet",
            "src", "mt",
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
            force=args.force,
        )

    log.info("=== Готово. Следующий шаг: scripts/train_sentence_model.py ===")


if __name__ == "__main__":
    main()