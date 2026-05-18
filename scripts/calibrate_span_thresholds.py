"""
scripts/calibrate_span_thresholds.py

Подбор оптимальных порогов bad_threshold и major_threshold для SpanModel
на val-сплите wordlevel_train.parquet.

Метрика оптимизации: F1(BAD-major) — та же что и при early stopping обучения.
Дополнительно логируется F1(BAD-minor) и F1(BAD) = macro среднее по двум классам.

Результат сохраняется в models/span_thresholds.json.
SpanModel читает этот файл при инициализации если он существует.

Запуск:
    python scripts/calibrate_span_thresholds.py
    python scripts/calibrate_span_thresholds.py --data-dir data --models-dir models
    python scripts/calibrate_span_thresholds.py --metric f1_bad_macro
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.bootstrap import init_script_runtime
from src.settings import get_settings

log = logging.getLogger(__name__)

LABEL2ID = {"OK": 0, "BAD-minor": 1, "BAD-major": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
IGNORE_INDEX = -100
MAX_LENGTH = 512

# Сетка поиска порогов
BAD_THRESHOLD_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
MAJOR_THRESHOLD_GRID = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]

THRESHOLDS_FILENAME = "span_thresholds.json"


# ---------------------------------------------------------------------------
# Сбор сырых вероятностей с val-сплита
# ---------------------------------------------------------------------------

def collect_val_probs(
    model_dir: Path,
    val_df: pd.DataFrame,
    batch_size: int,
    device: torch.device,
) -> tuple[list[list[float]], list[list[float]], list[list[int]]]:
    """
    Прогоняет val-сплит через модель и возвращает для каждого предложения:
        p_bad_list   — p(BAD-minor) + p(BAD-major) для каждого слова
        p_major_list — p(BAD-major) для каждого слова
        true_list    — истинные метки (0/1/2) для каждого слова

    Слова с IGNORE_INDEX в метках (субтокены, src-часть) исключаются.
    Используется тот же SpanDataset что и в train_span_model.py.
    """
    from scripts.train_span_model import SpanDataset

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        str(model_dir),
        local_files_only=True,
    ).to(device).eval()

    dataset = SpanDataset(val_df, tokenizer, max_length=MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    p_bad_list:   list[list[float]] = []
    p_major_list: list[list[float]] = []
    true_list:    list[list[int]]   = []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]  # (B, L)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs   = F.softmax(outputs.logits, dim=-1).cpu()  # (B, L, 3)

            for b in range(labels.shape[0]):
                mask = labels[b] != IGNORE_INDEX  # (L,)
                p_bad   = (probs[b, mask, LABEL2ID["BAD-minor"]]
                           + probs[b, mask, LABEL2ID["BAD-major"]]).tolist()
                p_major = probs[b, mask, LABEL2ID["BAD-major"]].tolist()
                true    = labels[b, mask].tolist()

                p_bad_list.append(p_bad)
                p_major_list.append(p_major)
                true_list.append(true)

    log.info(
        "Val прогон завершён: %d предложений, %d слов",
        len(p_bad_list),
        sum(len(x) for x in true_list),
    )
    return p_bad_list, p_major_list, true_list


# ---------------------------------------------------------------------------
# Применение порогов к сырым вероятностям
# ---------------------------------------------------------------------------

def apply_thresholds(
    p_bad_list: list[list[float]],
    p_major_list: list[list[float]],
    bad_threshold: float,
    major_threshold: float,
) -> tuple[list[int], list[int]]:
    """
    Применяет пороги к сырым вероятностям и возвращает плоские списки
    предсказанных и истинных меток для sklearn.metrics.
    """
    preds: list[int] = []
    for p_bad, p_major in zip(p_bad_list, p_major_list):
        for pb, pm in zip(p_bad, p_major):
            if pb >= bad_threshold:
                pred = LABEL2ID["BAD-major"] if pm >= major_threshold else LABEL2ID["BAD-minor"]
            else:
                pred = LABEL2ID["OK"]
            preds.append(pred)
    return preds


def compute_f1_scores(
    preds: list[int],
    trues: list[int],
) -> dict[str, float]:
    labels_order = [LABEL2ID["OK"], LABEL2ID["BAD-minor"], LABEL2ID["BAD-major"]]
    f1_per_class = f1_score(
        trues, preds,
        labels=labels_order,
        average=None,
        zero_division=0,
    )
    return {
        "f1_ok":        float(f1_per_class[0]),
        "f1_bad_minor": float(f1_per_class[1]),
        "f1_bad_major": float(f1_per_class[2]),
        "f1_bad_macro": float((f1_per_class[1] + f1_per_class[2]) / 2),
    }


# ---------------------------------------------------------------------------
# Поиск по сетке
# ---------------------------------------------------------------------------

def grid_search(
    p_bad_list: list[list[float]],
    p_major_list: list[list[float]],
    true_list: list[list[int]],
    metric: str,
) -> tuple[float, float, dict[str, float]]:
    """
    Перебирает все комбинации порогов и возвращает лучшую пару
    (bad_threshold, major_threshold) по заданной метрике.
    """
    trues_flat = [label for sentence in true_list for label in sentence]

    best_score  = -1.0
    best_bad    = BAD_THRESHOLD_GRID[0]
    best_major  = MAJOR_THRESHOLD_GRID[0]
    best_metrics: dict[str, float] = {}

    log.info(
        "Поиск по сетке: %d x %d = %d комбинаций, метрика=%s",
        len(BAD_THRESHOLD_GRID),
        len(MAJOR_THRESHOLD_GRID),
        len(BAD_THRESHOLD_GRID) * len(MAJOR_THRESHOLD_GRID),
        metric,
    )

    results: list[dict] = []

    for bad_t in BAD_THRESHOLD_GRID:
        for major_t in MAJOR_THRESHOLD_GRID:
            # major_threshold не может быть ниже bad_threshold
            if major_t < bad_t:
                continue

            preds = apply_thresholds(p_bad_list, p_major_list, bad_t, major_t)
            scores = compute_f1_scores(preds, trues_flat)
            score = scores[metric]

            results.append({
                "bad_threshold":   bad_t,
                "major_threshold": major_t,
                **scores,
            })

            if score > best_score:
                best_score  = score
                best_bad    = bad_t
                best_major  = major_t
                best_metrics = scores

    # Логируем топ-5
    results.sort(key=lambda x: x[metric], reverse=True)
    log.info("Топ-5 комбинаций по %s:", metric)
    for r in results[:5]:
        log.info(
            "  bad=%.2f major=%.2f  F1_major=%.4f  F1_minor=%.4f  F1_macro=%.4f",
            r["bad_threshold"],
            r["major_threshold"],
            r["f1_bad_major"],
            r["f1_bad_minor"],
            r["f1_bad_macro"],
        )

    return best_bad, best_major, best_metrics


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    init_script_runtime()
    s = get_settings()

    parser = argparse.ArgumentParser(
        description="Калибровка порогов классификации span-модели на val-сплите."
    )
    parser.add_argument("--data-dir",   type=Path, default=s.data_dir)
    parser.add_argument("--models-dir", type=Path, default=s.models_dir)
    parser.add_argument("--batch-size", type=int,  default=32)
    parser.add_argument(
        "--metric",
        choices=["f1_bad_major", "f1_bad_minor", "f1_bad_macro"],
        default="f1_bad_major",
        help="Метрика для выбора лучших порогов.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Перезаписать существующий span_thresholds.json.",
    )
    args = parser.parse_args()

    out_path = args.models_dir / THRESHOLDS_FILENAME
    if out_path.exists() and not args.force:
        log.info(
            "%s уже существует. Используй --force для перекалибровки.",
            out_path,
        )
        with open(out_path, encoding="utf-8") as f:
            saved = json.load(f)
        log.info("Текущие пороги: %s", saved)
        return

    model_dir = args.models_dir / "xlm_roberta_span"
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Не найдена span-модель: {model_dir}\n"
            "Сначала обучи: python scripts/train_span_model.py"
        )

    wl_path = args.data_dir / "processed" / "wordlevel_train.parquet"
    if not wl_path.exists():
        raise FileNotFoundError(
            f"Не найден файл: {wl_path}\n"
            "Сначала запусти: python scripts/build_wordlevel.py"
        )

    df = pd.read_parquet(wl_path)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    log.info("Val-сплит: %d предложений", len(val_df))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Устройство: %s", device)

    p_bad_list, p_major_list, true_list = collect_val_probs(
        model_dir, val_df, args.batch_size, device
    )

    best_bad, best_major, best_metrics = grid_search(
        p_bad_list, p_major_list, true_list, metric=args.metric
    )

    log.info(
        "Лучшие пороги: bad_threshold=%.2f  major_threshold=%.2f",
        best_bad,
        best_major,
    )
    log.info(
        "Метрики: F1(BAD-major)=%.4f  F1(BAD-minor)=%.4f  F1(BAD-macro)=%.4f  F1(OK)=%.4f",
        best_metrics["f1_bad_major"],
        best_metrics["f1_bad_minor"],
        best_metrics["f1_bad_macro"],
        best_metrics["f1_ok"],
    )

    # Логируем метрики с дефолтными порогами для сравнения
    default_preds = apply_thresholds(p_bad_list, p_major_list, 0.45, 0.60)
    trues_flat = [label for sentence in true_list for label in sentence]
    default_metrics = compute_f1_scores(default_preds, trues_flat)
    log.info(
        "Дефолтные пороги (bad=0.45 major=0.60): "
        "F1(BAD-major)=%.4f  F1(BAD-minor)=%.4f  F1(BAD-macro)=%.4f",
        default_metrics["f1_bad_major"],
        default_metrics["f1_bad_minor"],
        default_metrics["f1_bad_macro"],
    )

    payload = {
        "bad_threshold":   best_bad,
        "major_threshold": best_major,
        "metric":          args.metric,
        "val_metrics":     best_metrics,
        "default_metrics": default_metrics,
    }
    args.models_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    log.info("Пороги сохранены: %s", out_path)


if __name__ == "__main__":
    main()
