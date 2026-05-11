"""Единая установка seed для воспроизводимых признаков и обучения."""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:
    torch = None

try:
    from transformers import set_seed as hf_set_seed
except ImportError:
    hf_set_seed = None


def seed_everything(seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)

    if hf_set_seed is not None:
        hf_set_seed(seed)
