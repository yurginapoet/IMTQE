# src/features/formatting.py
# 7 признаков: числа, пунктуация, кавычки, даты + новые

import re
import spacy

# числа (целые и дробные)
RE_DIGITS = re.compile(r"\b\d+(?:[.,]\d+)?\b")
# знаки препинания
RE_PUNCT  = re.compile(r"[^\w\s]")
# даты
RE_DATE   = re.compile(
    r"\b\d{1,2}/\d{1,2}/\d{4}\b"
    r"|"
    r"\b(?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
)
# прямые кавычки
RE_STRAIGHT_QUOTES = re.compile(r'["\']')
# заглавные буквы в середине слова/предложения (имена)
RE_CAPITAL_MID = re.compile(r'\b[A-ZА-ЯЁ][a-zа-яё]*[A-ZА-ЯЁ][a-zа-яё]*\b')
# символы валют
RE_CURRENCY = re.compile(r"[$€£¥₽]")


def extract(src_doc: spacy.tokens.Doc, mt_doc: spacy.tokens.Doc) -> dict:
    src_text = src_doc.text
    mt_text  = mt_doc.text

    # старые признаки
    src_digits = set(RE_DIGITS.findall(src_text))
    mt_digits  = set(RE_DIGITS.findall(mt_text))
    digit_match_ratio = (
        len(src_digits & mt_digits) / len(src_digits)
        if src_digits else 1.0
    )

    src_punct = len(RE_PUNCT.findall(src_text))
    mt_punct  = len(RE_PUNCT.findall(mt_text))
    punct_ratio = mt_punct / src_punct if src_punct > 0 else 1.0

    src_has_quotes    = bool(RE_STRAIGHT_QUOTES.search(src_text))
    mt_has_straight   = bool(RE_STRAIGHT_QUOTES.search(mt_text))
    quotes_mismatch   = int(src_has_quotes and mt_has_straight)

    src_dates = RE_DATE.findall(src_text)
    date_format_error = int(
        bool(src_dates) and any(d in mt_text for d in src_dates)
    )

    # === новые признаки ===
    src_digits_count = len(RE_DIGITS.findall(src_text))
    mt_digits_count = len(RE_DIGITS.findall(mt_text))
    number_count_diff = mt_digits_count - src_digits_count

    src_cap_mismatch = bool(RE_CAPITAL_MID.search(src_text)) and not bool(RE_CAPITAL_MID.search(mt_text))
    capitalization_mismatch = int(src_cap_mismatch)

    src_curr = bool(RE_CURRENCY.search(src_text))
    mt_curr = bool(RE_CURRENCY.search(mt_text))
    currency_symbol_mismatch = int(src_curr and not mt_curr)

    return {
        "digit_match_ratio": digit_match_ratio,
        "punct_ratio":       punct_ratio,
        "quotes_mismatch":   quotes_mismatch,
        "date_format_error": date_format_error,
        # новые
        "number_count_diff": number_count_diff,
        "capitalization_mismatch": capitalization_mismatch,
        "currency_symbol_mismatch": currency_symbol_mismatch,
    }