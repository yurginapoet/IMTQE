"""Константы HF-моделей и пути. Значения каталогов — из окружения (см. src.settings)."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from src.settings import get_settings


class Config:
    ROOT_DIR = Path(__file__).resolve().parent.parent
    RANDOM_SEED = 42

    LABSE_MODEL_NAME = "sentence-transformers/LaBSE"
    RUGPT_MODEL_NAME = "sberbank-ai/rugpt3small_based_on_gpt2"

    @staticmethod
    def models_dir() -> Path:
        return get_settings().models_dir

    @staticmethod
    def data_dir() -> Path:
        return get_settings().data_dir

    @staticmethod
    def processed_dir() -> Path:
        return get_settings().processed_dir

    @staticmethod
    def hf_local_files_only() -> bool:
        return get_settings().hf_hub_offline

    @staticmethod
    def resolve_hf_model_path(model_name: str) -> str:
        return snapshot_download(
            repo_id=model_name,
            local_files_only=Config.hf_local_files_only(),
        )


Config.models_dir().mkdir(parents=True, exist_ok=True)
