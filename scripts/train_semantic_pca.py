"""
scripts/train_semantic_pca.py

Обучает PCA для semantic difference vectors MiniLM.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features import neural

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RANDOM_SEED = 42


def resolve_input_path(processed_dir: Path) -> Path:
    augmented = processed_dir / "sentence_da_augmented.parquet"
    if augmented.exists():
        return augmented
    return processed_dir / "sentence_da.parquet"


def load_training_pairs(path: Path, max_rows: int | None = None) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Не найден входной файл: {path}")

    df = pd.read_parquet(path)
    if "split" in df.columns:
        df = df[df["split"] == "train"].copy()
    if max_rows is not None:
        df = df.iloc[:max_rows].copy()

    pairs = list(zip(df["src"].astype(str), df["mt"].astype(str), strict=False))
    if not pairs:
        raise ValueError("Не найдено обучающих пар для PCA")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--n-components", type=int, default=neural.SEMANTIC_VECTOR_SIZE)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    models_dir = args.models_dir
    out_path = models_dir / "semantic_pca.pkl"
    meta_path = models_dir / "semantic_pca_meta.json"

    if out_path.exists() and not args.force:
        log.info("%s уже существует - пропускаем. Используй --force для пересчёта.", out_path)
        return

    processed_dir = args.data_dir / "processed"
    input_path = resolve_input_path(processed_dir)
    log.info("=== train_semantic_pca.py ===")
    log.info("Источник пар: %s", input_path)

    pairs = load_training_pairs(input_path, max_rows=args.max_rows)
    log.info("Пар для PCA: %d", len(pairs))

    encoder = neural.load_encoder()
    diff_vectors = neural.build_difference_vectors(
        pairs,
        encoder,
        batch_size=args.batch_size,
    )
    log.info("Матрица semantic difference: %s", diff_vectors.shape)

    n_components = min(args.n_components, diff_vectors.shape[0], diff_vectors.shape[1])
    if n_components != args.n_components:
        log.warning(
            "Снижаем число компонент PCA с %d до %d из-за размера данных",
            args.n_components,
            n_components,
        )

    pca = PCA(
        n_components=n_components,
        random_state=args.seed,
        svd_solver="randomized",
    )
    pca.fit(diff_vectors)

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, out_path)
    meta = {
        "input_path": str(input_path),
        "num_pairs": int(len(pairs)),
        "n_components": int(n_components),
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("PCA сохранён: %s", out_path)
    log.info(
        "Explained variance ratio sum: %.4f",
        float(np.sum(pca.explained_variance_ratio_)),
    )
    log.info("Метаданные сохранены: %s", meta_path)


if __name__ == "__main__":
    main()
