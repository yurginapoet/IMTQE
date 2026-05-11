"""Пути и флаги окружения. Без скрытых импортов внутри методов."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    data_dir: Path
    models_dir: Path
    log_dir: Path
    random_seed: int
    colab: bool
    hf_hub_offline: bool

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    def default_feature_batch_size(self) -> int:
        return 128 if self.colab else 64


def get_settings() -> Settings:
    root = Path(__file__).resolve().parent.parent
    data_dir = Path(os.environ.get("IMTQE_DATA_DIR", root / "data")).resolve()
    models_dir = Path(os.environ.get("IMTQE_MODELS_DIR", root / "models")).resolve()
    log_dir = Path(os.environ.get("IMTQE_LOG_DIR", root / "logs")).resolve()
    seed = int(os.environ.get("IMTQE_SEED", "42"))
    colab = os.environ.get("IMTQE_COLAB", "").lower() in {"1", "true", "yes"}
    hf_off = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"}
    return Settings(
        root_dir=root,
        data_dir=data_dir,
        models_dir=models_dir,
        log_dir=log_dir,
        random_seed=seed,
        colab=colab,
        hf_hub_offline=hf_off,
    )
