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
import joblib
import logging
import sys
import tempfile
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
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 100,
) -> pd.DataFrame:
    """
    Извлекает все активные признаки для всех строк df батчами.
    Возвращает df с добавленными числовыми колонками и word_logprobs.
    """
    feature_names = extractor.active_feature_names
    n = len(df)
    resume_state = None
    if checkpoint_path is not None and checkpoint_path.exists():
        log.info("Найден checkpoint: %s", checkpoint_path)
        resume_state = joblib.load(checkpoint_path)

    if resume_state is not None:
        if (
            int(resume_state["total_rows"]) == n
            and list(resume_state["feature_names"]) == list(feature_names)
            and int(resume_state["batch_size"]) == batch_size
        ):
            all_vectors = resume_state["all_vectors"]
            all_word_logprobs = resume_state["all_word_logprobs"]
            resume_start = int(resume_state["next_start"])
            log.info("Возобновляем с позиции %d / %d", resume_start, n)
        else:
            log.warning("Checkpoint несовместим с текущим запуском, начинаем заново")
            all_vectors = np.zeros((n, len(feature_names)), dtype=np.float32)
            all_word_logprobs = [None] * n
            resume_start = 0
    else:
        all_vectors = np.zeros((n, len(feature_names)), dtype=np.float32)
        all_word_logprobs = [None] * n
        resume_start = 0

    pairs = list(zip(df[src_col].astype(str), df[mt_col].astype(str), strict=False))

    for batch_idx, start in enumerate(
        tqdm(
            range(resume_start, n, batch_size),
            desc=progress_desc,
            unit="batch",
            initial=resume_start // batch_size,
            total=(n + batch_size - 1) // batch_size,
        ),
        start=1,
    ):
        batch = pairs[start : start + batch_size]
        results = extractor.extract_batch(batch)
        for idx, result in enumerate(results):
            all_vectors[start + idx] = result["vector"]
            all_word_logprobs[start + idx] = result["word_logprobs"]

        if checkpoint_path is not None and checkpoint_every > 0 and (batch_idx % checkpoint_every == 0):
            save_checkpoint_atomic(
                checkpoint_path,
                {
                    "feature_names": list(feature_names),
                    "batch_size": batch_size,
                    "total_rows": n,
                    "next_start": start + len(batch),
                    "all_vectors": all_vectors,
                    "all_word_logprobs": all_word_logprobs,
                },
            )
            log.info("Checkpoint сохранён: %s (next_start=%d)", checkpoint_path, start + len(batch))

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


def save_checkpoint_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    joblib.dump(payload, tmp_path)
    tmp_path.replace(path)


def process_dataset(
    name: str,
    in_path: Path,
    out_path: Path,
    src_col: str,
    mt_col: str,
    extractor: FeatureExtractor,
    batch_size: int,
    force: bool,
    checkpoint_dir: Path,
    checkpoint_every: int,
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
    checkpoint_path = checkpoint_dir / f"{name}_features.resume.joblib"
    if force and checkpoint_path.exists():
        checkpoint_path.unlink()

    df = extract_for_df(
        df,
        src_col=src_col,
        mt_col=mt_col,
        extractor=extractor,
        batch_size=batch_size,
        progress_desc=f"{name} feature batches",
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("Сохранено: %s (%d строк, %d колонок)", out_path, len(df), len(df.columns))
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        log.info("Временный checkpoint удалён: %s", checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/checkpoints/features"))
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
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
        )

    log.info("=== Готово. Следующий шаг: scripts/train_sentence_model.py ===")


if __name__ == "__main__":
    main()
