"""
src/models/span_model.py

Инференс-обёртка над обученной XLM-RoBERTa для span-level предсказания.
Используется из src/predict.py — загружается один раз при старте сервера.

Интерфейс:
    model = SpanModel("models/xlm_roberta_span")
    result = model.predict(src, mt, word_logprobs)
    # result — список SpanResult (датакласс)

SpanResult:
    start_idx:  int          — индекс первого слова спана (0-based, spaCy-слова mt)
    end_idx:    int          — индекс последнего слова спана (включительно)
    severity:   str          — "BAD-minor" | "BAD-major"
    confidence: float        — p(BAD) = p(BAD-minor) + p(BAD-major) для первого слова спана
    word_labels: List[str]   — пословные метки всех слов mt ("OK"/"BAD-minor"/"BAD-major")

word_logprobs (из fluency.py / Блока 1) передаётся для будущей интеграции с rules.py —
сейчас сохраняется в SpanResult.word_logprobs_span для передачи в rules.py.

Токенизация:
    Схема идентична SpanDataset из train_span_model.py:
    "[CLS] src [SEP] mt [SEP]"
    first-subtoken стратегия для маппинга SentencePiece → spaCy слова.
    src получает 1/3 бюджета (169 токенов), mt — 2/3 (340 токенов).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForTokenClassification, AutoTokenizer

log = logging.getLogger(__name__)

# Должно совпадать с train_span_model.py
LABEL2ID = {"OK": 0, "BAD-minor": 1, "BAD-major": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MAX_LENGTH = 512
_OK_ID = LABEL2ID["OK"]


# ---------------------------------------------------------------------------
# Выходной тип
# ---------------------------------------------------------------------------

@dataclass
class SpanResult:
    """Один непрерывный BAD-спан в тексте mt."""
    start_idx:         int          # индекс первого слова спана
    end_idx:           int          # индекс последнего слова спана (включительно)
    severity:          str          # "BAD-minor" | "BAD-major"
    confidence:        float        # p(BAD) первого слова спана
    word_logprobs_span: List[float] = field(default_factory=list)
    # пословные logprob для слов спана (из fluency.py) — для rules.py


@dataclass
class SpanPrediction:
    """Полный результат span-модели для одной пары (src, mt)."""
    word_labels:  List[str]         # метка каждого слова mt
    word_probs:   List[float]       # p(BAD) каждого слова mt
    spans:        List[SpanResult]  # только BAD-спаны


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

class SpanModel:
    """
    Загружает сохранённую HF-модель и токенизатор из model_dir.
    Работает на CPU (device=-1) или GPU если доступен.

    Параметры:
        model_dir   — путь к директории models/xlm_roberta_span/
        device      — "cpu", "cuda", или None (авто)
    """

    def __init__(self, model_dir: str | Path, device: Optional[str] = None) -> None:
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Не найдена директория модели: {model_dir}\n"
                "Запусти: python scripts/train_span_model.py"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        log.info("SpanModel: загрузка из %s на %s", model_dir, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(
            str(model_dir)
        ).to(self.device).eval()

        total = sum(p.numel() for p in self.model.parameters())
        log.info("SpanModel загружена: %.1fM параметров", total / 1e6)

    # ------------------------------------------------------------------
    # Публичный метод
    # ------------------------------------------------------------------

    def predict(
        self,
        src: str,
        mt: str,
        word_logprobs: Optional[List[float]] = None,
    ) -> SpanPrediction:
        """
        Предсказывает severity для каждого слова mt и группирует BAD-слова в спаны.

        Параметры:
            src           — исходное предложение (EN)
            mt            — машинный перевод (RU)
            word_logprobs — logprob каждого слова mt из fluency.py (может быть None)

        Возвращает SpanPrediction.
        """
        mt_words = mt.split()
        n_words  = len(mt_words)

        if n_words == 0:
            return SpanPrediction(word_labels=[], word_probs=[], spans=[])

        # Токенизация и инференс
        input_ids, attention_mask, mt_start_pos, word_to_first_subtoken = (
            self._tokenize(src, mt, n_words)
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
            )
        logits = outputs.logits[0]  # (seq_len, 3)
        probs  = F.softmax(logits, dim=-1)  # (seq_len, 3)

        # Декодируем метки для каждого spaCy слова mt
        word_labels, word_probs_bad = self._decode_words(
            probs, mt_start_pos, word_to_first_subtoken, n_words
        )

        # Группируем смежные BAD-слова в спаны
        spans = self._build_spans(word_labels, word_probs_bad, word_logprobs, n_words)

        return SpanPrediction(
            word_labels=word_labels,
            word_probs=word_probs_bad,
            spans=spans,
        )

    # ------------------------------------------------------------------
    # Токенизация (идентична SpanDataset._encode из train_span_model.py)
    # ------------------------------------------------------------------

    def _tokenize(
        self,
        src: str,
        mt: str,
        n_words: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int, List[Optional[int]]]:
        """
        Возвращает:
            input_ids        — (1, MAX_LENGTH)
            attention_mask   — (1, MAX_LENGTH)
            mt_start_pos     — позиция первого mt-токена в полной последовательности
            word_to_first_subtoken — List длиной len(mt_subtokens_used),
                                     элемент i = индекс spaCy слова или None
        """
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

        mt_start_pos = 1 + len(src_ids) + 1  # [CLS] + src + [SEP]

        # Маппинг субтокенов → spaCy слова
        word_to_first_subtoken = _map_subtokens_to_words(mt, mt_offsets, n_words)

        # Padding
        seq_len = len(input_ids_list)
        padding = MAX_LENGTH - seq_len
        attention_mask_list = [1] * seq_len + [0] * padding
        input_ids_list      = input_ids_list + [pad_id] * padding

        input_ids      = torch.tensor([input_ids_list],      dtype=torch.long)
        attention_mask = torch.tensor([attention_mask_list], dtype=torch.long)

        return input_ids, attention_mask, mt_start_pos, word_to_first_subtoken

    # ------------------------------------------------------------------
    # Декодирование: probs → пословные метки
    # ------------------------------------------------------------------

    def _decode_words(
        self,
        probs: torch.Tensor,
        mt_start_pos: int,
        word_to_first_subtoken: List[Optional[int]],
        n_words: int,
    ) -> tuple[List[str], List[float]]:
        """
        Для каждого spaCy слова mt берём вероятности первого субтокена.
        Возвращает:
            word_labels    — List[str] длиной n_words
            word_probs_bad — List[float] p(BAD) = p(BAD-minor) + p(BAD-major)
        """
        # word_idx → вероятности (3,)
        word_probs_tensor: dict[int, torch.Tensor] = {}

        for local_idx, word_idx in enumerate(word_to_first_subtoken):
            if word_idx is None:
                continue
            if word_idx in word_probs_tensor:
                # уже взяли первый субтокен этого слова
                continue
            pos = mt_start_pos + local_idx
            if pos >= probs.shape[0]:
                continue
            word_probs_tensor[word_idx] = probs[pos]  # (3,)

        word_labels: List[str]   = []
        word_probs_bad: List[float] = []

        for i in range(n_words):
            if i in word_probs_tensor:
                p = word_probs_tensor[i]
                label_id = int(p.argmax().item())
                p_bad    = float((p[LABEL2ID["BAD-minor"]] + p[LABEL2ID["BAD-major"]]).item())
            else:
                # токен не попал в окно — консервативно OK
                label_id = _OK_ID
                p_bad    = 0.0

            word_labels.append(ID2LABEL[label_id])
            word_probs_bad.append(p_bad)

        return word_labels, word_probs_bad

    # ------------------------------------------------------------------
    # Группировка BAD-слов в спаны
    # ------------------------------------------------------------------

    def _build_spans(
        self,
        word_labels: List[str],
        word_probs_bad: List[float],
        word_logprobs: Optional[List[float]],
        n_words: int,
    ) -> List[SpanResult]:
        """
        Смежные BAD-слова (BAD-minor или BAD-major) объединяются в один SpanResult.
        Severity спана = максимальный severity среди его слов (BAD-major важнее).
        Confidence = p(BAD) первого слова спана.
        """
        spans: List[SpanResult] = []
        i = 0
        while i < n_words:
            if word_labels[i] == "OK":
                i += 1
                continue

            # Начало нового спана
            span_start = i
            span_severity = word_labels[i]

            while i < n_words and word_labels[i] != "OK":
                # Повышаем severity если встретили BAD-major внутри спана
                if word_labels[i] == "BAD-major":
                    span_severity = "BAD-major"
                i += 1

            span_end = i - 1  # включительно

            # logprob слов спана (если переданы)
            if word_logprobs is not None and len(word_logprobs) == n_words:
                lp_span = word_logprobs[span_start : span_end + 1]
            else:
                lp_span = []

            spans.append(SpanResult(
                start_idx=span_start,
                end_idx=span_end,
                severity=span_severity,
                confidence=word_probs_bad[span_start],
                word_logprobs_span=lp_span,
            ))

        return spans


# ---------------------------------------------------------------------------
# Вспомогательная функция маппинга (копия из train_span_model.py)
# Вынесена на уровень модуля чтобы не дублировать в классе
# ---------------------------------------------------------------------------

def _map_subtokens_to_words(
    mt_text: str,
    offsets: list[tuple[int, int]],
    n_words: int,
) -> List[Optional[int]]:
    """
    Для каждого субтокена (по char offset в mt_text) возвращает
    индекс spaCy слова (0-indexed) или None.

    Первый субтокен слова → его индекс.
    Последующие субтокены того же слова → None (IGNORE).
    """
    word_spans: List[tuple[int, int]] = []
    pos = 0
    for word in mt_text.split():
        start = mt_text.index(word, pos)
        end   = start + len(word)
        word_spans.append((start, end))
        pos = end

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
                # else: не первый субтокен → None
                break

        result.append(found)

    return result