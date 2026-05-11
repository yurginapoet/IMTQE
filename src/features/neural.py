"""Нейронные sentence-level признаки: MiniLM embeddings + PCA."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np
import torch

from src.config import Config
from src.features.schema import SEMANTIC_FEATURE_NAMES

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

SEMANTIC_VECTOR_SIZE = len(SEMANTIC_FEATURE_NAMES)
log = logging.getLogger(__name__)


def resolve_device(device: str | None = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_encoder(
    model_name: str = Config.SEMANTIC_ENCODER_NAME,
    device: str | None = None,
) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    resolved_device = resolve_device(device)
    log.info("Загрузка semantic encoder на device=%s", resolved_device)

    if resolved_device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model_path = Config.resolve_hf_model_path(model_name)
    model = SentenceTransformer(
        model_path,
        device=resolved_device,
        local_files_only=Config.hf_local_files_only(),
    )
    if resolved_device.startswith("cuda"):
        try:
            model.half()
            log.info("MiniLM переведён в float16 для CUDA inference")
        except Exception as exc:
            log.warning("Не удалось перевести MiniLM в float16: %s", exc)
    return model


def load_pca(path: str | Path = Config.SEMANTIC_PCA_PATH):
    pca = joblib.load(path)
    n_components = getattr(pca, "n_components_", getattr(pca, "n_components", None))
    if int(n_components) != SEMANTIC_VECTOR_SIZE:
        raise ValueError(
            f"Ожидалось {SEMANTIC_VECTOR_SIZE} PCA-компонент, получено {n_components}"
        )
    return pca


def build_difference_vectors(
    pairs: list[tuple[str, str]],
    encoder: "SentenceTransformer",
    batch_size: int = 64,
    show_progress_bar: bool = False,
) -> np.ndarray:
    if not pairs:
        return np.zeros((0, 384), dtype=np.float32)

    src_texts = [src for src, _ in pairs]
    mt_texts = [mt for _, mt in pairs]

    src_embs = encoder.encode(
        src_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
    )
    mt_embs = encoder.encode(
        mt_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=show_progress_bar,
    )
    return np.abs(src_embs - mt_embs).astype(np.float32)


def project_vectors(diff_vectors: np.ndarray, pca) -> np.ndarray:
    if diff_vectors.size == 0:
        return np.zeros((0, SEMANTIC_VECTOR_SIZE), dtype=np.float32)
    return pca.transform(diff_vectors).astype(np.float32)


def vector_to_feature_dict(vector: np.ndarray) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(SEMANTIC_FEATURE_NAMES, vector, strict=False)
    }


def extract(
    src: str,
    mt: str,
    encoder: "SentenceTransformer",
    pca,
) -> dict[str, float]:
    projected = extract_batch([(src, mt)], encoder, pca, batch_size=1)
    return projected[0] if projected else vector_to_feature_dict(np.zeros(SEMANTIC_VECTOR_SIZE))


def extract_batch(
    pairs: list[tuple[str, str]],
    encoder: "SentenceTransformer",
    pca,
    batch_size: int = 64,
) -> list[dict[str, float]]:
    diff_vectors = build_difference_vectors(pairs, encoder, batch_size=batch_size)
    reduced = project_vectors(diff_vectors, pca)
    return [vector_to_feature_dict(row) for row in reduced]
