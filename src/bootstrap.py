"""Инициализация логирования и seed для CLI-скриптов (отделено от бизнес-логики)."""

from __future__ import annotations

from src.determinism import seed_everything
from src.logging_config import configure_logging
from src.settings import Settings, get_settings


def init_script_runtime(*, seed: int | None = None) -> Settings:
    s = get_settings()
    configure_logging(s)
    seed_everything(seed if seed is not None else s.random_seed)
    return s
