# src/features/structural.py
# 8 признаков на основе длин src и mt с учётом EN→RU специфики.

import spacy
import re

_RE_SENTENCE_END = re.compile(r'[.!?]+')

# Ожидаемое среднее соотношение длин EN→RU
EXPECTED_LENGTH_RATIO = 1.30


def extract(src_doc: spacy.tokens.Doc, mt_doc: spacy.tokens.Doc) -> dict:
    src_len = len(src_doc)
    mt_len  = len(mt_doc)

    src_text = src_doc.text
    mt_text  = mt_doc.text

    if src_len == 0:
        length_ratio = 0.0
        length_ratio_dev = 0.0
        length_ok = 1.0
    else:
        length_ratio = mt_len / src_len
        length_ratio_dev = length_ratio - EXPECTED_LENGTH_RATIO
        # Мягкая функция: до |dev| < 0.25 — почти идеально
        length_ok = max(0.0, 1.0 - abs(length_ratio_dev) / 0.35)

    # Количество символов
    src_chars = len(src_text)
    mt_chars = len(mt_text)
    compression_ratio = mt_chars / src_chars if src_chars > 0 else 1.0

    # Количество предложений
    src_sentences = len(_RE_SENTENCE_END.findall(src_text))
    mt_sentences = len(_RE_SENTENCE_END.findall(mt_text))
    sentence_count_diff = mt_sentences - src_sentences

    return {
        "length_ratio":        length_ratio,
        "length_ratio_dev":    length_ratio_dev,      # новое
        "length_ok":           length_ok,             # новое
        "abs_length_diff":     abs(mt_len - src_len),
        "token_count_diff":    mt_len - src_len,
        "src_length":          src_len,
        "mt_length":           mt_len,
        "compression_ratio":   compression_ratio,
        "sentence_count_diff": sentence_count_diff,
    }