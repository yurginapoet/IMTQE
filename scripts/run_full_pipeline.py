"""Оркестратор полного пайплайна (локально или Colab: IMTQE_COLAB=1)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.bootstrap import init_script_runtime
from src.settings import get_settings


def run_step(name: str, command: list[str]) -> None:
    print(name)
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    init_script_runtime()
    s = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=s.data_dir)
    parser.add_argument("--models-dir", type=Path, default=s.models_dir)
    parser.add_argument("--seed", type=int, default=s.random_seed)
    parser.add_argument("--batch-size", type=int, default=s.default_feature_batch_size())
    parser.add_argument("--skip-span", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-synth-rows", type=int, default=None)
    args = parser.parse_args()

    py = sys.executable
    force_flag = ["--force"] if args.force else []

    steps: list[tuple[str, list[str]]] = [
        (
            "Prepare Data",
            [
                py, "scripts/prepare_data.py",
                "--data-dir", str(args.data_dir),
                "--seed", str(args.seed),
                *force_flag,
            ],
        ),
        (
            "Build Wordlevel",
            [
                py, "scripts/build_wordlevel.py",
                "--data-dir", str(args.data_dir),
                "--seed", str(args.seed),
                *force_flag,
            ],
        ),
        (
            "Dedup MQM",
            [
                py, "scripts/dedup_mqm.py",
                "--data-dir", str(args.data_dir),
                *force_flag,
            ],
        ),
        (
            "Build Synthetic Negatives",
            [
                py, "scripts/build_synthetic_negatives.py",
                "--data-dir", str(args.data_dir),
                "--seed", str(args.seed),
                *([ "--max-rows", str(args.max_synth_rows) ] if args.max_synth_rows is not None else []),
                *force_flag,
            ],
        ),
        (
            "Extract Features",
            [
                py, "scripts/extract_features.py",
                "--data-dir", str(args.data_dir),
                "--batch-size", str(args.batch_size),
                *force_flag,
            ],
        ),
        (
            "Train Sentence Model",
            [
                py, "scripts/train_sentence_model.py",
                "--data-dir", str(args.data_dir),
                "--models-dir", str(args.models_dir),
                "--seed", str(args.seed),
            ],
        ),
    ]

    if not args.skip_span:
        steps.append(
            (
                "Train Span Model",
                [
                    py, "scripts/train_span_model.py",
                    "--data-dir", str(args.data_dir),
                    "--models-dir", str(args.models_dir),
                ],
            )
        )

    for name, command in steps:
        run_step(name, command)

    print("pipeline: completed")


if __name__ == "__main__":
    main()
