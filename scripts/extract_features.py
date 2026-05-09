"""
scripts/extract_features.py

Шаг 4 из пайплайна MTQE.
Прогоняет sentence_da и wordlevel_train через FeatureExtractor,
сохраняет признаки в parquet.

Лёгкие признаки (16) считаются всегда.
Тяжёлые (6): LaBSE + ruGPT-3 — только с флагом --heavy.
На Colab запускать с --heavy, локально без него.

Выходные файлы:
  data/processed/sentence_da_features.parquet
  data/processed/wordlevel_features.parquet

Запуск:
  python scripts/extract_features.py              # только лёгкие
  python scripts/extract_features.py --heavy      # все 22 признака
  python scripts/extract_features.py --only da    # только DA датасет
  python scripts/extract_features.py --only wl    # только wordlevel
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.features.extractor import FEATURE_NAMES, FEATURE_NAMES_LIGHT, FeatureExtractor

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
    Извлекает признаки для всех строк df батчами.
    Возвращает df с добавленными колонками признаков.
    """
    feature_names = FEATURE_NAMES if extractor.heavy_loaded else FEATURE_NAMES_LIGHT
    n = len(df)
    all_vectors = np.zeros((n, len(feature_names)), dtype=np.float32)
    all_word_logprobs = [None] * n  # только если heavy

    pairs = list(zip(df[src_col], df[mt_col]))

    for start in tqdm(range(0, n, BATCH_SIZE), desc="Батчи"):
        batch = pairs[start : start + BATCH_SIZE]
        results = extractor.extract_batch(batch)
        for i, res in enumerate(results):
            all_vectors[start + i] = res["vector"]
            if extractor.heavy_loaded:
                all_word_logprobs[start + i] = res["word_logprobs"]

    # добавляем признаки как отдельные колонки
    df = df.copy()
    for j, name in enumerate(feature_names):
        df[name] = all_vectors[:, j]

    if extractor.heavy_loaded:
        df["word_logprobs"] = all_word_logprobs

    return df


def process_da(processed_dir: Path, extractor: FeatureExtractor, heavy: bool) -> None:
    out_path = processed_dir / "sentence_da_features.parquet"

    if out_path.exists():
        log.info("sentence_da_features.parquet уже существует - пропускаем")
        return

    log.info("Загрузка sentence_da.parquet...")
    df = pd.read_parquet(processed_dir / "sentence_da.parquet")
    log.info("Строк: %d", len(df))

    df = extract_for_df(df, src_col="src", mt_col="mt", extractor=extractor)

    df.to_parquet(out_path, index=False)
    log.info("Сохранено: %s  (%d строк, %d колонок)", out_path, len(df), len(df.columns))


def process_wordlevel(processed_dir: Path, extractor: FeatureExtractor, heavy: bool) -> None:
    out_path = processed_dir / "wordlevel_features.parquet"

    if out_path.exists():
        log.info("wordlevel_features.parquet уже существует - пропускаем")
        return

    log.info("Загрузка wordlevel_train.parquet...")
    df = pd.read_parquet(processed_dir / "wordlevel_train.parquet")
    log.info("Строк: %d", len(df))

    df = extract_for_df(df, src_col="src", mt_col="mt", extractor=extractor)

    df.to_parquet(out_path, index=False)
    log.info("Сохранено: %s  (%d строк, %d колонок)", out_path, len(df), len(df.columns))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--heavy", action="store_true", help="Загрузить LaBSE и ruGPT-3")
    parser.add_argument("--only", choices=["da", "wl"], help="Обработать только один датасет")
    args = parser.parse_args()

    processed_dir = args.data_dir / "processed"

    log.info("=== extract_features.py ===")
    log.info("heavy=%s", args.heavy)

    log.info("Загрузка spaCy моделей...")
    extractor = FeatureExtractor()

    if args.heavy:
        extractor.load_heavy_models()

    log.info("Признаков будет: %d", len(FEATURE_NAMES if args.heavy else FEATURE_NAMES_LIGHT))

    if args.only != "wl":
        process_da(processed_dir, extractor, args.heavy)

    if args.only != "da":
        process_wordlevel(processed_dir, extractor, args.heavy)

    log.info("=== Готово. Следующий шаг: notebooks/05_train_sentence_model.ipynb ===")


if __name__ == "__main__":
    main()