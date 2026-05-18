"""
src/models/span_model.py

Инференс-обёртка над обученной XLM-RoBERTa для span-level предсказания.
Используется из src/predict.py — загружается один раз при старте сервера.

Интерфейс:
    model = SpanModel("models/xlm_roberta_span")
    result = model.predict(src, mt, word_logprobs)
    # result — SpanPrediction

SpanResult:
    start_idx:          int   — индекс первого слова спана (0-based, spaCy-слова mt)
    end_idx:            int   — индекс последнего слова спана (включительно)
    severity:           str   — "BAD-minor" | "BAD-major"
    confidence:         float — p(BAD) первого слова спана
    word_logprobs_span: list  — logprob слов спана (из fluency.py), для rules.py

Пороги классификации:
    Загружаются из models/span_thresholds.json если файл существует
    (создаётся scripts/calibrate_span_thresholds.py).
    Иначе читаются из переменных окружения IMTQE_SPAN_BAD_THRESHOLD /
    IMTQE_SPAN_MAJOR_THRESHOLD. Иначе дефолты 0.45 / 0.60.

Фильтрация по word_logprobs:
    Спаны с низкой уверенностью модели на флюентных словах убираются.
    Порог регулируется IMTQE_SPAN_LOGPROB_FILTER_THRESHOLD (по умолчанию -3.5).
    Фильтр применяется только если word_logprobs переданы.

Токенизация:
    "[CLS] src [SEP] mt [SEP]"
    first-subtoken стратегия для маппинга SentencePiece на spaCy слова.
    src — 1/3 бюджета (169 токенов), mt — 2/3 (340 токенов).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer

log = logging.getLogger(__name__)

LABEL2ID = {"OK": 0, "BAD-minor": 1, "BAD-major": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MAX_LENGTH = 512
_OK_ID = LABEL2ID["OK"]

DEFAULT_BAD_THRESHOLD = 0.45
DEFAULT_MAJOR_THRESHOLD = 0.60

# Порог logprob для фильтрации: слова с mean_logprob выше этого значения
# считаются флюентными по ruGPT-3. Совпадает с границей в rules.py
# (_looks_like_agreement_error использует -6.0 < mean_lp < -3.5).
DEFAULT_LOGPROB_FILTER_THRESHOLD = -3.5

# Если уверенность span-модели ниже этого значения И слова флюентные,
# спан удаляется как вероятно ложный.
DEFAULT_LOGPROB_CONFIDENCE_CAP = 0.65

THRESHOLDS_FILENAME = "span_thresholds.json"


# ---------------------------------------------------------------------------
# Выходные типы
# ---------------------------------------------------------------------------

@dataclass
class SpanResult:
    """Один непрерывный BAD-спан в тексте mt."""
    start_idx:          int
    end_idx:            int
    severity:           str
    confidence:         float
    word_logprobs_span: List[float] = field(default_factory=list)


@dataclass
class SpanPrediction:
    """Полный результат span-модели для одной пары (src, mt)."""
    word_labels: List[str]
    word_probs:  List[float]
    spans:       List[SpanResult]


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class SpanModel:
    """
    Загружает сохранённую HF-модель и токенизатор из model_dir.

    Параметры:
        model_dir   — путь к директории models/xlm_roberta_span/
        device      — "cpu", "cuda" или None (авто)
        thresholds_path — путь к span_thresholds.json; None = искать рядом с models_dir
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: Optional[str] = None,
        thresholds_path: Optional[str | Path] = None,
    ) -> None:
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Не найдена директория модели: {model_dir}\n"
                "Запусти: python scripts/train_span_model.py"
            )
        _validate_model_dir(model_dir)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.bad_threshold, self.major_threshold = _load_thresholds(
            model_dir, thresholds_path
        )
        if self.major_threshold < self.bad_threshold:
            log.warning(
                "major_threshold=%.3f ниже bad_threshold=%.3f; поднимаем до bad_threshold",
                self.major_threshold,
                self.bad_threshold,
            )
            self.major_threshold = self.bad_threshold

        self.logprob_filter_threshold = _read_float_env(
            "IMTQE_SPAN_LOGPROB_FILTER_THRESHOLD",
            DEFAULT_LOGPROB_FILTER_THRESHOLD,
            min_val=-20.0,
            max_val=0.0,
        )
        self.logprob_confidence_cap = _read_float_env(
            "IMTQE_SPAN_LOGPROB_CONFIDENCE_CAP",
            DEFAULT_LOGPROB_CONFIDENCE_CAP,
            min_val=0.0,
            max_val=1.0,
        )

        log.info("SpanModel: загрузка из %s на %s", model_dir, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            str(model_dir),
            local_files_only=True,
        ).to(self.device).eval()

        total = sum(p.numel() for p in self.model.parameters())
        log.info(
            "SpanModel загружена: %.1fM параметров; "
            "bad_threshold=%.2f major_threshold=%.2f logprob_filter=%.1f",
            total / 1e6,
            self.bad_threshold,
            self.major_threshold,
            self.logprob_filter_threshold,
        )

    # ------------------------------------------------------------------
    # Публичный метод
    # ------------------------------------------------------------------

    def predict(
        self,
        src: str,
        mt: str,
        word_logprobs: Optional[List[float]] = None,
        mt_words: Optional[Sequence[str]] = None,
    ) -> SpanPrediction:
        """
        Предсказывает severity для каждого слова mt и группирует BAD-слова в спаны.

        Параметры:
            src           — исходное предложение (EN)
            mt            — машинный перевод (RU)
            word_logprobs — logprob каждого слова mt из fluency.py (может быть None)
            mt_words      — токены mt из spaCy (если None, используется mt.split())
        """
        token_words = list(mt_words) if mt_words is not None else mt.split()
        n_words = len(token_words)

        if n_words == 0:
            return SpanPrediction(word_labels=[], word_probs=[], spans=[])

        input_ids, attention_mask, mt_start_pos, word_to_first_subtoken = (
            self._tokenize(src, mt, token_words)
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
            )
        logits = outputs.logits[0]        # (seq_len, 3)
        probs  = F.softmax(logits, dim=-1)  # (seq_len, 3)

        word_labels, word_probs_bad = self._decode_words(
            probs, mt_start_pos, word_to_first_subtoken, n_words
        )

        spans = self._build_spans(word_labels, word_probs_bad, word_logprobs, n_words)

        return SpanPrediction(
            word_labels=word_labels,
            word_probs=word_probs_bad,
            spans=spans,
        )

    # ------------------------------------------------------------------
    # Токенизация
    # ------------------------------------------------------------------

    def _tokenize(
        self,
        src: str,
        mt: str,
        mt_words: Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor, int, List[Optional[int]]]:
        tok = self.tokenizer

        src_enc = tok(src, add_special_tokens=False, return_offsets_mapping=False)
        mt_enc  = tok(mt,  add_special_tokens=False, return_offsets_mapping=True)

        src_ids    = src_enc["input_ids"]
        mt_ids     = mt_enc["input_ids"]
        mt_offsets = mt_enc["offset_mapping"]

        cls_id = tok.cls_token_id
        sep_id = tok.sep_token_id
        pad_id = tok.pad_token_id

        max_content = MAX_LENGTH - 3
        max_src = max_content // 3
        max_mt  = max_content - max_src

        src_ids    = src_ids[:max_src]
        mt_ids     = mt_ids[:max_mt]
        mt_offsets = mt_offsets[:max_mt]

        input_ids_list = [cls_id] + src_ids + [sep_id] + mt_ids + [sep_id]
        if len(input_ids_list) > MAX_LENGTH:
            input_ids_list = input_ids_list[:MAX_LENGTH - 1] + [sep_id]

        mt_start_pos = 1 + len(src_ids) + 1

        word_to_first_subtoken = _map_subtokens_to_words(
            mt,
            mt_offsets,
            len(mt_words),
            mt_words=mt_words,
        )

        seq_len = len(input_ids_list)
        padding = MAX_LENGTH - seq_len
        attention_mask_list = [1] * seq_len + [0] * padding
        input_ids_list      = input_ids_list + [pad_id] * padding

        input_ids      = torch.tensor([input_ids_list],      dtype=torch.long)
        attention_mask = torch.tensor([attention_mask_list], dtype=torch.long)

        return input_ids, attention_mask, mt_start_pos, word_to_first_subtoken

    # ------------------------------------------------------------------
    # Декодирование: probs -> пословные метки
    # ------------------------------------------------------------------

    def _decode_words(
        self,
        probs: torch.Tensor,
        mt_start_pos: int,
        word_to_first_subtoken: List[Optional[int]],
        n_words: int,
    ) -> tuple[List[str], List[float]]:
        word_probs_tensor: dict[int, torch.Tensor] = {}

        for local_idx, word_idx in enumerate(word_to_first_subtoken):
            if word_idx is None:
                continue
            if word_idx in word_probs_tensor:
                continue
            pos = mt_start_pos + local_idx
            if pos >= probs.shape[0]:
                continue
            word_probs_tensor[word_idx] = probs[pos]

        word_labels: List[str]    = []
        word_probs_bad: List[float] = []

        for i in range(n_words):
            if i in word_probs_tensor:
                p = word_probs_tensor[i]
                p_major = float(p[LABEL2ID["BAD-major"]].item())
                p_bad   = float((p[LABEL2ID["BAD-minor"]] + p[LABEL2ID["BAD-major"]]).item())
                if p_bad >= self.bad_threshold:
                    label_id = (
                        LABEL2ID["BAD-major"]
                        if p_major >= self.major_threshold
                        else LABEL2ID["BAD-minor"]
                    )
                else:
                    label_id = _OK_ID
            else:
                label_id = _OK_ID
                p_bad    = 0.0

            word_labels.append(ID2LABEL[label_id])
            word_probs_bad.append(p_bad)

        return word_labels, word_probs_bad

    # ------------------------------------------------------------------
    # Группировка BAD-слов в спаны с logprob-фильтром
    # ------------------------------------------------------------------

    def _build_spans(
        self,
        word_labels: List[str],
        word_probs_bad: List[float],
        word_logprobs: Optional[List[float]],
        n_words: int,
    ) -> List[SpanResult]:
        """
        Смежные BAD-слова объединяются в SpanResult.
        Severity спана = максимальный severity среди его слов.
        Confidence = p(BAD) первого слова спана.

        Logprob-фильтр: если word_logprobs переданы и mean(logprob) спана
        выше logprob_filter_threshold (слова флюентны по ruGPT-3),
        а confidence ниже logprob_confidence_cap (модель неуверена),
        спан отбрасывается как вероятно ложный.
        """
        logprobs_available = (
            word_logprobs is not None
            and len(word_logprobs) == n_words
        )

        spans: List[SpanResult] = []
        i = 0
        while i < n_words:
            if word_labels[i] == "OK":
                i += 1
                continue

            span_start    = i
            span_severity = word_labels[i]

            while i < n_words and word_labels[i] != "OK":
                if word_labels[i] == "BAD-major":
                    span_severity = "BAD-major"
                i += 1

            span_end   = i - 1
            confidence = word_probs_bad[span_start]

            if logprobs_available:
                lp_span = word_logprobs[span_start : span_end + 1]
            else:
                lp_span = []

            # Logprob-фильтр
            if lp_span and confidence < self.logprob_confidence_cap:
                mean_lp = float(np.mean(lp_span))
                if mean_lp > self.logprob_filter_threshold:
                    log.debug(
                        "Logprob-фильтр: спан [%d:%d] удалён "
                        "(confidence=%.3f mean_lp=%.2f)",
                        span_start, span_end, confidence, mean_lp,
                    )
                    continue

            spans.append(SpanResult(
                start_idx=span_start,
                end_idx=span_end,
                severity=span_severity,
                confidence=confidence,
                word_logprobs_span=lp_span,
            ))

        return spans


# ---------------------------------------------------------------------------
# Загрузка порогов
# ---------------------------------------------------------------------------

def _load_thresholds(
    model_dir: Path,
    thresholds_path: Optional[str | Path],
) -> tuple[float, float]:
    """
    Приоритет:
    1. Явно переданный thresholds_path
    2. models/span_thresholds.json рядом с model_dir (../span_thresholds.json)
    3. Переменные окружения IMTQE_SPAN_BAD_THRESHOLD / IMTQE_SPAN_MAJOR_THRESHOLD
    4. Дефолты 0.45 / 0.60
    """
    candidates: list[Path] = []
    if thresholds_path is not None:
        candidates.append(Path(thresholds_path))
    # models/ — родительская папка model_dir
    candidates.append(model_dir.parent / THRESHOLDS_FILENAME)

    for path in candidates:
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                bad   = float(data["bad_threshold"])
                major = float(data["major_threshold"])
                log.info(
                    "Пороги загружены из %s: bad=%.2f major=%.2f",
                    path, bad, major,
                )
                return bad, major
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                log.warning(
                    "Не удалось прочитать пороги из %s: %s. Используем env/дефолты.",
                    path, exc,
                )

    # Читаем из env
    bad   = _read_probability_threshold("IMTQE_SPAN_BAD_THRESHOLD",   DEFAULT_BAD_THRESHOLD)
    major = _read_probability_threshold("IMTQE_SPAN_MAJOR_THRESHOLD", DEFAULT_MAJOR_THRESHOLD)
    return bad, major


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

def _map_subtokens_to_words(
    mt_text: str,
    offsets: list[tuple[int, int]],
    n_words: int,
    mt_words: Optional[Sequence[str]] = None,
) -> List[Optional[int]]:
    """
    Для каждого субтокена возвращает индекс spaCy слова (0-indexed) или None.
    Первый субтокен слова — его индекс, последующие — None (IGNORE).
    """
    word_spans = _build_word_spans(mt_text, mt_words)
    if len(word_spans) != n_words:
        raise ValueError(
            f"Ожидалось {n_words} слов для mt, но построено {len(word_spans)} спанов"
        )

    result: List[Optional[int]] = []
    assigned: dict[int, bool]   = {}

    for (tok_start, tok_end) in offsets:
        if tok_start == tok_end:
            result.append(None)
            continue

        found = None
        for word_idx, (w_start, w_end) in enumerate(word_spans):
            if tok_start >= w_start and tok_end <= w_end:
                if word_idx not in assigned:
                    assigned[word_idx] = True
                    found = word_idx
                break

        result.append(found)

    return result


def _build_word_spans(
    mt_text: str,
    mt_words: Optional[Sequence[str]] = None,
) -> List[tuple[int, int]]:
    """
    Строит char-span каждого слова mt в согласованной токенизации.
    Если mt_words передан, используем его; иначе mt_text.split().
    """
    words = list(mt_words) if mt_words is not None else mt_text.split()
    word_spans: List[tuple[int, int]] = []
    pos = 0
    for word in words:
        if not word:
            continue
        start = mt_text.find(word, pos)
        if start < 0:
            raise ValueError(
                f"Не удалось сопоставить токен '{word}' с текстом mt при pos={pos}"
            )
        end = start + len(word)
        word_spans.append((start, end))
        pos = end
    return word_spans


def _validate_model_dir(model_dir: Path) -> None:
    required = [model_dir / "config.json"]
    weight_candidates = [
        model_dir / "model.safetensors",
        model_dir / "pytorch_model.bin",
    ]
    tokenizer_candidates = [
        model_dir / "tokenizer.json",
        model_dir / "sentencepiece.bpe.model",
        model_dir / "spiece.model",
    ]

    missing = [path.name for path in required if not path.exists()]
    if not any(path.exists() for path in weight_candidates):
        missing.append("model.safetensors|pytorch_model.bin")
    if not any(path.exists() for path in tokenizer_candidates):
        missing.append("tokenizer.json|sentencepiece.bpe.model|spiece.model")

    if missing:
        raise FileNotFoundError(
            f"Неполный HF-артефакт span-модели в {model_dir}. "
            f"Не найдены: {', '.join(missing)}. "
            "Переобучи/пересохрани модель через scripts/train_span_model.py."
        )


def _read_probability_threshold(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warning(
            "%s=%r не число; используем значение по умолчанию %.2f",
            name, raw, default,
        )
        return default
    if not 0.0 <= value <= 1.0:
        log.warning(
            "%s=%.3f вне диапазона [0,1]; используем значение по умолчанию %.2f",
            name, value, default,
        )
        return default
    return value


def _read_float_env(name: str, default: float, min_val: float, max_val: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        log.warning(
            "%s=%r не число; используем значение по умолчанию %.2f",
            name, raw, default,
        )
        return default
    if not min_val <= value <= max_val:
        log.warning(
            "%s=%.3f вне диапазона [%.1f, %.1f]; используем %.2f",
            name, value, min_val, max_val, default,
        )
        return default
    return value