"""
Прогревает и проверяет все артефакты, нужные для полного инференса.

По умолчанию работает в offline-режиме и проверяет, что модели уже есть в локальном кэше.
С флагом --download разрешает Hugging Face загрузить недостающие sentence-модели.

Примеры:
  poetry run python scripts/warmup_inference_models.py
  poetry run python scripts/warmup_inference_models.py --download
  poetry run python scripts/warmup_inference_models.py --download --predict
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.bootstrap import init_script_runtime
from src.config import Config
from src.features.extractor import FeatureExtractor
from src.models.sentence_model import SentenceModel
from src.models.span_model import SpanModel
from src.predict import Predictor

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download",
        action="store_true",
        help="Разрешить скачивание недостающих HF-моделей в локальный кэш",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="После прогрева сделать один реальный predict_sentence",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Устройство для тяжёлых моделей и span-модели",
    )
    return parser.parse_args()


def configure_env(download: bool) -> None:
    if download:
        os.environ["HF_HUB_OFFLINE"] = "0"
        os.environ["TRANSFORMERS_OFFLINE"] = "0"
    else:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def main() -> None:
    init_script_runtime()
    args = parse_args()
    configure_env(download=args.download)

    t0 = time.time()
    log.info("Проверка sentence-level модели...")
    sentence_model = SentenceModel(
        Config.models_dir() / "xgboost_sentence.model",
        Config.models_dir() / "shap_explainer.pkl",
    )
    log.info(
        "SentenceModel OK: expected_feature_count=%d",
        sentence_model.expected_feature_count,
    )

    log.info("Проверка FeatureExtractor и тяжёлых sentence-моделей...")
    extractor = FeatureExtractor(device=args.device)
    extractor.load_heavy_models(require_neural=True)
    log.info(
        "FeatureExtractor OK: active_feature_count=%d, semantic_augmented=%s",
        len(extractor.active_feature_names),
        extractor.semantic_augmented_loaded,
    )

    log.info("Проверка span-модели...")
    span_model = SpanModel(Config.models_dir() / "xlm_roberta_span", device=args.device)
    log.info("SpanModel OK: tokenizer=%s", type(span_model.tokenizer).__name__)

    if args.predict:
        log.info("Полный прогон Predictor...")
        predictor = Predictor(device=args.device)
        result = predictor.predict_sentence(
            "The bank raised interest rates by 0.5 percent.",
            "Банк повысил процентные ставки на 0,5 процента.",
        )
        log.info(
            "Predictor OK: score=%.4f mqm=%.4f features=%d errors=%d",
            result.score,
            result.mqm_score,
            len(result.debug.get("features", {})),
            len(result.errors),
        )

    elapsed = time.time() - t0
    log.info("Готово за %.1f сек", elapsed)


if __name__ == "__main__":
    main()
