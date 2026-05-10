"""
scripts/train_semantic_pca.py

Обучает PCA для semantic difference vectors MiniLM.

Важно:
  - извлечение MiniLM embeddings использует GPU, если доступен CUDA;
  - сам PCA из sklearn считается на CPU;
  - чтобы не выглядеть "зависшим", скрипт идёт потоково по чанкам
    и показывает tqdm progress bars.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import IncrementalPCA
from tqdm.auto import tqdm

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
DEFAULT_EMBED_BATCH_SIZE = 256
DEFAULT_CHUNK_SIZE = 8192


def resolve_input_path(processed_dir: Path) -> Path:
    augmented = processed_dir / "sentence_da_augmented.parquet"
    if augmented.exists():
        return augmented
    return processed_dir / "sentence_da.parquet"


def load_training_frame(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Не найден входной файл: {path}")

    df = pd.read_parquet(path)
    if "split" in df.columns:
        df = df[df["split"] == "train"].copy()
    if max_rows is not None:
        df = df.iloc[:max_rows].copy()

    df = df[["src", "mt"]].copy()
    df["src"] = df["src"].astype(str)
    df["mt"] = df["mt"].astype(str)

    if df.empty:
        raise ValueError("Не найдено обучающих пар для PCA")
    return df.reset_index(drop=True)


def iter_pair_chunks(
    df: pd.DataFrame,
    chunk_size: int,
):
    total = len(df)
    for start in range(0, total, chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        yield start, list(zip(chunk["src"], chunk["mt"], strict=False))


def fit_incremental_pca(
    df: pd.DataFrame,
    encoder,
    n_components: int,
    embed_batch_size: int,
    chunk_size: int,
    checkpoint_path: Path,
    checkpoint_every: int,
) -> tuple[IncrementalPCA, int]:
    resume_state = None
    if checkpoint_path.exists():
        log.info("Найден PCA checkpoint: %s", checkpoint_path)
        resume_state = joblib.load(checkpoint_path)

    if chunk_size < n_components:
        raise ValueError(
            f"chunk_size={chunk_size} должен быть >= n_components={n_components} "
            "для IncrementalPCA.partial_fit"
        )

    if resume_state is not None:
        ipca = resume_state["ipca"]
        resume_start = int(resume_state["next_start"])
        processed_rows = int(resume_state["processed_rows"])
        log.info(
            "Возобновляем PCA: start=%d processed_rows=%d",
            resume_start,
            processed_rows,
        )
    else:
        ipca = IncrementalPCA(n_components=n_components, batch_size=chunk_size)
        resume_start = 0
        processed_rows = 0

    n_chunks = (len(df) + chunk_size - 1) // chunk_size

    progress = tqdm(
        iter_pair_chunks(df.iloc[resume_start:].reset_index(drop=True), chunk_size),
        total=n_chunks,
        desc="PCA fit chunks",
        unit="chunk",
        initial=resume_start // chunk_size,
    )

    for chunk_idx, (relative_start, pairs) in enumerate(progress, start=1):
        absolute_start = resume_start + relative_start
        diff_vectors = neural.build_difference_vectors(
            pairs,
            encoder,
            batch_size=embed_batch_size,
            show_progress_bar=False,
        )
        ipca.partial_fit(diff_vectors)
        processed_rows += len(diff_vectors)

        progress.set_postfix(
            rows=processed_rows,
            last_chunk=len(diff_vectors),
            device=neural.resolve_device(),
        )

        del diff_vectors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if chunk_idx == 1:
            log.info(
                "Первый chunk обработан: %d пар. GPU inference работает, "
                "дальше идёт потоковый partial_fit PCA на CPU.",
                len(pairs),
            )

        if checkpoint_every > 0 and (chunk_idx % checkpoint_every == 0):
            save_checkpoint_atomic(
                checkpoint_path,
                {
                    "ipca": ipca,
                    "next_start": absolute_start + len(pairs),
                    "processed_rows": processed_rows,
                    "chunk_size": chunk_size,
                    "n_components": n_components,
                },
            )
            log.info(
                "PCA checkpoint сохранён: next_start=%d processed_rows=%d",
                absolute_start + len(pairs),
                processed_rows,
            )

    return ipca, processed_rows


def save_checkpoint_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    joblib.dump(payload, tmp_path)
    tmp_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--n-components", type=int, default=neural.SEMANTIC_VECTOR_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--device", type=str, default=None, help='Например "cuda" или "cpu"')
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    models_dir = args.models_dir
    out_path = models_dir / "semantic_pca.pkl"
    meta_path = models_dir / "semantic_pca_meta.json"
    checkpoint_path = models_dir / "semantic_pca.resume.joblib"

    if out_path.exists() and not args.force:
        log.info("%s уже существует - пропускаем. Используй --force для пересчёта.", out_path)
        return

    if args.force and checkpoint_path.exists():
        checkpoint_path.unlink()

    processed_dir = args.data_dir / "processed"
    input_path = resolve_input_path(processed_dir)
    device = neural.resolve_device(args.device)

    log.info("=== train_semantic_pca.py ===")
    log.info("Источник пар: %s", input_path)
    log.info("Device для MiniLM: %s", device)
    log.info(
        "Параметры: n_components=%d  embed_batch_size=%d  chunk_size=%d",
        args.n_components,
        args.batch_size,
        args.chunk_size,
    )
    log.info("Checkpoint каждые %d chunk(ов)", args.checkpoint_every)

    df = load_training_frame(input_path, max_rows=args.max_rows)
    log.info("Пар для PCA: %d", len(df))

    input_dim = 384
    n_components = min(args.n_components, len(df), input_dim)
    if n_components != args.n_components:
        log.warning(
            "Снижаем число компонент PCA с %d до %d из-за размера данных",
            args.n_components,
            n_components,
        )

    encoder = neural.load_encoder(device=device)

    if torch.cuda.is_available() and device.startswith("cuda"):
        gpu_name = torch.cuda.get_device_name(0)
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info("CUDA устройство: %s (%.1f GB VRAM)", gpu_name, total_mem_gb)

    pca, processed_rows = fit_incremental_pca(
        df=df,
        encoder=encoder,
        n_components=n_components,
        embed_batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        checkpoint_path=checkpoint_path,
        checkpoint_every=args.checkpoint_every,
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, out_path)

    explained = getattr(pca, "explained_variance_ratio_", None)
    explained_sum = float(np.sum(explained)) if explained is not None else None

    meta = {
        "input_path": str(input_path),
        "num_pairs": int(len(df)),
        "processed_rows": int(processed_rows),
        "n_components": int(n_components),
        "device": device,
        "embed_batch_size": int(args.batch_size),
        "chunk_size": int(args.chunk_size),
        "explained_variance_ratio_sum": explained_sum,
        "pca_type": "IncrementalPCA",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("PCA сохранён: %s", out_path)
    if explained_sum is not None:
        log.info("Explained variance ratio sum: %.4f", explained_sum)
    log.info("Метаданные сохранены: %s", meta_path)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        log.info("Временный checkpoint удалён: %s", checkpoint_path)


if __name__ == "__main__":
    main()
