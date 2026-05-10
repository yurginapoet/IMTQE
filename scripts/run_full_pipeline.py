"""
scripts/run_full_pipeline.py

Оркестратор полного пайплайна для Colab/локального запуска.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(name: str, command: list[str]) -> None:
    print(f"\n=== {name} ===")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sentence-model", choices=["xgboost", "ngboost"], default="xgboost")
    parser.add_argument("--skip-span", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-synth-rows", type=int, default=None)
    parser.add_argument("--max-pca-rows", type=int, default=None)
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
            "Train Semantic PCA",
            [
                py, "scripts/train_semantic_pca.py",
                "--data-dir", str(args.data_dir),
                "--models-dir", str(args.models_dir),
                "--seed", str(args.seed),
                "--batch-size", str(max(args.batch_size, 128)),
                *([ "--max-rows", str(args.max_pca_rows) ] if args.max_pca_rows is not None else []),
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
                "--model", args.sentence_model,
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

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
