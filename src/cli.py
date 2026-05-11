"""Единая точка входа CLI: шаги пайплайна или полный прогон."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STEP_SCRIPTS: dict[str, str] = {
    "prepare-data": "prepare_data.py",
    "build-wordlevel": "build_wordlevel.py",
    "dedup-mqm": "dedup_mqm.py",
    "build-synthetic-negatives": "build_synthetic_negatives.py",
    "train-semantic-pca": "train_semantic_pca.py",
    "extract-features": "extract_features.py",
    "train-sentence": "train_sentence_model.py",
    "train-span": "train_span_model.py",
    "train-neural-head": "train_neural_head.py",
    "warmup-inference": "warmup_inference_models.py",
}


def _run_script(name: str, forwarded: list[str]) -> int:
    rel = STEP_SCRIPTS.get(name)
    if rel is None:
        raise KeyError(name)
    path = ROOT / "scripts" / rel
    cmd = [sys.executable, str(path), *forwarded]
    return subprocess.call(cmd)


def _cmd_pipeline(argv: list[str]) -> int:
    path = ROOT / "scripts" / "run_full_pipeline.py"
    return subprocess.call([sys.executable, str(path), *argv])


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in {"-h", "--help"}:
        steps = "\n  ".join(sorted(STEP_SCRIPTS))
        print(
            "imtqe <команда> [аргументы скрипта]\n\n"
            "Команды шагов:\n  "
            + steps
            + "\n\n"
            "  pipeline   — полный прогон (те же флаги, что run_full_pipeline.py)\n"
            "Переменные окружения: IMTQE_DATA_DIR, IMTQE_MODELS_DIR, IMTQE_LOG_DIR, "
            "IMTQE_SEED, IMTQE_COLAB=1 (Colab: удобнее batch по умолчанию в скриптах), "
            "HF_HUB_OFFLINE\n"
        )
        raise SystemExit(0 if argv else 1)

    cmd, *rest = argv
    if cmd == "pipeline":
        raise SystemExit(_cmd_pipeline(rest))

    if cmd not in STEP_SCRIPTS:
        print(f"Неизвестная команда: {cmd}", file=sys.stderr)
        raise SystemExit(1)

    raise SystemExit(_run_script(cmd, rest))


if __name__ == "__main__":
    main()
