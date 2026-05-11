"""
scripts/train_span_model.py

Шаг 6 из пайплайна MTQE.
Fine-tuning xlm-roberta-base на WMT21 word-level данных.
Задача: token classification — для каждого слова mt предсказать
OK / BAD-minor / BAD-major.

Вход:
  data/processed/wordlevel_train.parquet
    колонки: src, mt, word_labels (List[str]), split, n_words, domain

Выход:
  models/xlm_roberta_span/   — HuggingFace формат (config + weights)

Ключевые особенности:
  - Вход модели: "[CLS] src [SEP] mt [SEP]"
  - Предсказание ТОЛЬКО для токенов mt части
  - Маппинг SentencePiece → word через first-subtoken стратегию
  - Weighted loss: BAD-major=5, BAD-minor=2, OK=1
  - Early stopping по val F1(BAD-major)
  - RANDOM_SEED = 42

Запуск (рекомендуется Colab T4):
  python scripts/train_span_model.py
  python scripts/train_span_model.py --epochs 3 --batch-size 8   # меньше памяти
  python scripts/train_span_model.py --eval-only                  # только тест
  python scripts/train_span_model.py --data-dir data --models-dir models
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.bootstrap import init_script_runtime
from src.settings import get_settings
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from sklearn.metrics import f1_score, classification_report

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

RANDOM_SEED   = 42
MODEL_NAME    = "xlm-roberta-base"
MAX_LENGTH    = 512
LABEL2ID      = {"OK": 0, "BAD-minor": 1, "BAD-major": 2}
ID2LABEL      = {v: k for k, v in LABEL2ID.items()}
CLASS_WEIGHTS = [1.0, 2.0, 5.0]   # OK, BAD-minor, BAD-major
IGNORE_INDEX  = -100


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SpanDataset(Dataset):
    """
    Каждый элемент — одна пара (src, mt) с пословными метками mt.

    Токенизация:
      "[CLS] src_tokens [SEP] mt_tokens [SEP]"
      Метки назначаются только токенам mt части.
      Для каждого spaCy слова: первый субтокен → настоящая метка,
      остальные субтокены → IGNORE_INDEX (-100).
      Все токены src части и специальные токены → IGNORE_INDEX.

    word_labels в parquet — это List[str] уровня spaCy слов mt.
    Число элементов = n_words = число пробел-разделённых токенов mt
    (именно так разбивает build_wordlevel.py через tags_str.split()).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = MAX_LENGTH,
    ) -> None:
        self.records    = df.reset_index(drop=True)
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row        = self.records.iloc[idx]
        src        = str(row["src"])
        mt         = str(row["mt"])
        word_labels: list[str] = row["word_labels"]

        return self._encode(src, mt, word_labels)

    def _encode(
        self,
        src: str,
        mt: str,
        word_labels: list[str],
    ) -> dict:
        """
        Возвращает словарь с ключами:
          input_ids, attention_mask, labels
        Все тензоры имеют длину max_length (padding/truncation).

        Схема: [CLS] src [SEP] mt [SEP]
        Метки только на mt-токенах (first-subtoken стратегия).
        src получает 1/3 бюджета, mt — 2/3 (mt важнее, на нём метки).
        """
        tokenizer = self.tokenizer

        src_enc = tokenizer(src, add_special_tokens=False, return_offsets_mapping=False)
        mt_enc  = tokenizer(mt,  add_special_tokens=False, return_offsets_mapping=True)

        src_ids    = src_enc["input_ids"]
        mt_ids     = mt_enc["input_ids"]
        mt_offsets = mt_enc["offset_mapping"]

        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id

        max_content = self.max_length - 3   # [CLS], [SEP], [SEP]
        max_src = max_content // 3
        max_mt  = max_content - max_src

        src_ids    = src_ids[:max_src]
        mt_ids     = mt_ids[:max_mt]
        mt_offsets = mt_offsets[:max_mt]

        input_ids = [cls_id] + src_ids + [sep_id] + mt_ids + [sep_id]

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length - 1] + [sep_id]

        mt_start_pos = 1 + len(src_ids) + 1  # [CLS] + src + [SEP]

        labels = [IGNORE_INDEX] * len(input_ids)

        word_subtoken_assigned = self._map_subtokens_to_words(
            mt, mt_offsets, len(word_labels)
        )

        for subtoken_local_idx, word_idx in enumerate(word_subtoken_assigned):
            if word_idx is None:
                continue
            if word_idx >= len(word_labels):
                continue
            pos_in_full = mt_start_pos + subtoken_local_idx
            if pos_in_full >= len(labels):
                continue
            label_str = word_labels[word_idx]
            labels[pos_in_full] = LABEL2ID.get(label_str, IGNORE_INDEX)

        seq_len = len(input_ids)
        padding = self.max_length - seq_len

        attention_mask = [1] * seq_len + [0] * padding
        input_ids      = input_ids    + [pad_id]       * padding
        labels         = labels       + [IGNORE_INDEX] * padding

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }

    @staticmethod
    def _map_subtokens_to_words(
        mt_text: str,
        offsets: list[tuple[int, int]],
        n_words: int,
    ) -> list[Optional[int]]:
        """
        Для каждого субтокена возвращает индекс spaCy слова (0-indexed) или None.
        Первый субтокен слова → его индекс, остальные → None (→ IGNORE_INDEX).
        """
        word_spans: list[tuple[int, int]] = []
        pos = 0
        for word in mt_text.split():
            start = mt_text.index(word, pos)
            end   = start + len(word)
            word_spans.append((start, end))
            pos = end

        result: list[Optional[int]] = []
        last_word_idx_assigned: dict[int, bool] = {}

        for (tok_start, tok_end) in offsets:
            if tok_start == tok_end:
                result.append(None)
                continue

            assigned = None
            for word_idx, (w_start, w_end) in enumerate(word_spans):
                if tok_start >= w_start and tok_end <= w_end:
                    if word_idx not in last_word_idx_assigned:
                        last_word_idx_assigned[word_idx] = True
                        assigned = word_idx
                    else:
                        assigned = None
                    break

            result.append(assigned)

        return result


# ---------------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------------

class WeightedSpanModel(nn.Module):
    """
    xlm-roberta-base с weighted cross-entropy loss.
    Weights: OK=1, BAD-minor=2, BAD-major=5.
    Используется при обучении.
    """

    def __init__(self, model_name: str, num_labels: int, class_weights: list[float]) -> None:
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self.register_buffer(
            "weight",
            torch.tensor(class_weights, dtype=torch.float),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits   # (B, L, num_labels)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=IGNORE_INDEX)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss, logits


class SimpleEvalWrapper(nn.Module):
    """
    Лёгкая обёртка для инференса/eval на уже сохранённой HF модели.
    Использует встроенный weighted loss через передачу labels в HF модель,
    но class_weights применяем вручную — так же как в WeightedSpanModel.

    Используется в eval_only и для финального теста после обучения.
    Это правильный способ загрузки без хака через __new__.
    """

    def __init__(
        self,
        model_dir: str,
        class_weights: list[float],
        device: torch.device,
    ) -> None:
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir).to(device)
        self.register_buffer(
            "weight",
            torch.tensor(class_weights, dtype=torch.float).to(device),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> tuple:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits  = outputs.logits

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=IGNORE_INDEX)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss, logits


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def compute_metrics(all_preds: list[int], all_labels: list[int]) -> dict:
    labels_order = [LABEL2ID["OK"], LABEL2ID["BAD-minor"], LABEL2ID["BAD-major"]]

    f1_per_class = f1_score(
        all_labels, all_preds,
        labels=labels_order,
        average=None,
        zero_division=0,
    )
    report = classification_report(
        all_labels, all_preds,
        labels=labels_order,
        target_names=["OK", "BAD-minor", "BAD-major"],
        zero_division=0,
    )
    return {
        "f1_ok":        f1_per_class[0],
        "f1_bad_minor": f1_per_class[1],
        "f1_bad_major": f1_per_class[2],
        "report":       report,
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    model.eval()
    all_preds:  list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            loss, logits = model(input_ids, attention_mask, labels)
            total_loss  += loss.item()
            n_batches   += 1

            preds = logits.argmax(dim=-1)
            mask  = labels != IGNORE_INDEX
            all_preds.extend(preds[mask].cpu().numpy().tolist())
            all_labels.extend(labels[mask].cpu().numpy().tolist())

    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------

def train(
    df: pd.DataFrame,
    models_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: torch.device,
) -> None:
    set_seed(RANDOM_SEED)

    log.info("Загрузка токенизатора: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)
    log.info("train=%d  val=%d  test=%d", len(train_df), len(val_df), len(test_df))

    train_ds = SpanDataset(train_df, tokenizer)
    val_ds   = SpanDataset(val_df,   tokenizer)
    test_ds  = SpanDataset(test_df,  tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,     shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,   batch_size=batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,  batch_size=batch_size * 2, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    log.info("Загрузка модели: %s", MODEL_NAME)
    model = WeightedSpanModel(
        model_name=MODEL_NAME,
        num_labels=len(LABEL2ID),
        class_weights=CLASS_WEIGHTS,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log.info("Параметров модели: %d (%.1fM)", total_params, total_params / 1e6)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps  = len(train_loader) * epochs
    warmup_steps = total_steps // 10
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )

    log.info(
        "Обучение: epochs=%d  batch=%d  lr=%s  warmup=%d  total_steps=%d",
        epochs, batch_size, lr, warmup_steps, total_steps,
    )

    best_f1_major = -1.0
    best_epoch    = -1
    no_improve    = 0
    models_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = models_dir / "xlm_roberta_span"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_steps    = 0

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            optimizer.zero_grad()
            loss, _ = model(input_ids, attention_mask, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_steps    += 1

            if step % 50 == 0:
                log.info(
                    "Epoch %d/%d  step %d/%d  loss=%.4f",
                    epoch, epochs, step, len(train_loader),
                    epoch_loss / n_steps,
                )

        avg_train_loss = epoch_loss / max(n_steps, 1)
        val_metrics    = evaluate(model, val_loader, device)

        log.info(
            "Epoch %d — train_loss=%.4f  val_loss=%.4f  "
            "F1(OK)=%.4f  F1(BAD-minor)=%.4f  F1(BAD-major)=%.4f",
            epoch, avg_train_loss, val_metrics["loss"],
            val_metrics["f1_ok"], val_metrics["f1_bad_minor"], val_metrics["f1_bad_major"],
        )
        log.info("Val classification report:\n%s", val_metrics["report"])

        f1_major = val_metrics["f1_bad_major"]
        if f1_major > best_f1_major + 1e-4:
            best_f1_major = f1_major
            best_epoch    = epoch
            no_improve    = 0
            model.model.save_pretrained(str(best_model_dir))
            tokenizer.save_pretrained(str(best_model_dir))
            log.info("  ✓ Новый лучший F1(BAD-major)=%.4f — сохранено в %s",
                     best_f1_major, best_model_dir)
        else:
            no_improve += 1
            log.info("  F1(BAD-major) не улучшился (%d/%d). Лучший: %.4f @ epoch %d",
                     no_improve, patience, best_f1_major, best_epoch)
            if no_improve >= patience:
                log.info("Early stopping на эпохе %d.", epoch)
                break

    log.info("Обучение завершено. Лучший F1(BAD-major)=%.4f @ epoch %d",
             best_f1_major, best_epoch)

    # Финальный тест — загружаем лучшую модель через SimpleEvalWrapper
    log.info("Загружаем лучшую модель для финального теста...")
    best_wrapper = SimpleEvalWrapper(
        model_dir=str(best_model_dir),
        class_weights=CLASS_WEIGHTS,
        device=device,
    )
    _run_test(best_wrapper, test_loader, device)


def _run_test(model: nn.Module, test_loader: DataLoader, device: torch.device) -> None:
    test_metrics = evaluate(model, test_loader, device)
    log.info(
        "TEST — F1(OK)=%.4f  F1(BAD-minor)=%.4f  F1(BAD-major)=%.4f",
        test_metrics["f1_ok"], test_metrics["f1_bad_minor"], test_metrics["f1_bad_major"],
    )
    log.info("Test classification report:\n%s", test_metrics["report"])


# ---------------------------------------------------------------------------
# eval-only режим
# ---------------------------------------------------------------------------

def eval_only(
    df: pd.DataFrame,
    models_dir: Path,
    batch_size: int,
    device: torch.device,
) -> None:
    model_dir = models_dir / "xlm_roberta_span"
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Не найдена модель: {model_dir}\n"
            "Сначала обучи: python scripts/train_span_model.py"
        )

    log.info("Загрузка сохранённой модели из %s", model_dir)
    # SimpleEvalWrapper — чистая загрузка без хаков с __new__
    wrapper = SimpleEvalWrapper(
        model_dir=str(model_dir),
        class_weights=CLASS_WEIGHTS,
        device=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    test_df  = df[df["split"] == "test"].reset_index(drop=True)
    test_ds  = SpanDataset(test_df, tokenizer)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=2,
    )

    _run_test(wrapper, test_loader, device)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    init_script_runtime()
    s = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=s.data_dir)
    parser.add_argument("--models-dir", type=Path, default=s.models_dir)
    parser.add_argument("--epochs",     type=int,  default=5)
    parser.add_argument("--batch-size", type=int,  default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--patience",   type=int,  default=3,
                        help="Early stopping patience (в эпохах)")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Только тест, без переобучения")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Устройство: %s", device)
    if device.type == "cuda":
        log.info("GPU: %s  (%.1f GB)", torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    wl_path = args.data_dir / "processed" / "wordlevel_train.parquet"
    if not wl_path.exists():
        raise FileNotFoundError(
            f"Не найден файл: {wl_path}\n"
            "Запусти: python scripts/build_wordlevel.py"
        )

    log.info("Загрузка данных: %s", wl_path)
    df = pd.read_parquet(wl_path)
    log.info("Строк: %d  колонок: %s", len(df), list(df.columns))

    assert "word_labels" in df.columns
    assert "split"       in df.columns
    assert "src"         in df.columns
    assert "mt"          in df.columns

    all_labels = [l for labels in df["word_labels"] for l in labels]
    total = len(all_labels)
    for label in ["OK", "BAD-minor", "BAD-major"]:
        cnt = all_labels.count(label)
        log.info("  %-12s %6d слов  (%.1f%%)", label, cnt, 100 * cnt / total)

    if args.eval_only:
        eval_only(df, args.models_dir, args.batch_size * 2, device)
    else:
        train(
            df=df, models_dir=args.models_dir,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, patience=args.patience, device=device,
        )

    log.info("train_span_model: finished")


if __name__ == "__main__":
    main()