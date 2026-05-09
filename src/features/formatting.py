# src/features/formatting.py
# 4 признака: числа, пунктуация, кавычки, даты. Только regex
import re
import spacy

# числа (целые и дробные)
RE_DIGITS = re.compile(r"\b\d+(?:[.,]\d+)?\b")
# знаки препинания
RE_PUNCT  = re.compile(r"[^\w\s]")
# даты вида MM/DD/YYYY или Month DD, YYYY
RE_DATE   = re.compile(
    r"\b\d{1,2}/\d{1,2}/\d{4}\b"
    r"|"
    r"\b(?:January|February|March|April|May|June|July|August"
    r"|September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
)
# английские прямые кавычки
RE_STRAIGHT_QUOTES = re.compile(r'["\']')


def extract(src_doc: spacy.tokens.Doc, mt_doc: spacy.tokens.Doc) -> dict:
    src_text = src_doc.text
    mt_text  = mt_doc.text

    # доля чисел из src которые есть в mt
    src_digits = set(RE_DIGITS.findall(src_text))
    mt_digits  = set(RE_DIGITS.findall(mt_text))
    digit_match_ratio = (
        len(src_digits & mt_digits) / len(src_digits)
        if src_digits else 1.0
    )

    # соотношение знаков препинания
    src_punct = len(RE_PUNCT.findall(src_text))
    mt_punct  = len(RE_PUNCT.findall(mt_text))
    punct_ratio = mt_punct / src_punct if src_punct > 0 else 1.0

    # в src есть кавычки, а в mt прямые английские (должны быть «ёлочки»)
    src_has_quotes    = bool(RE_STRAIGHT_QUOTES.search(src_text))
    mt_has_straight   = bool(RE_STRAIGHT_QUOTES.search(mt_text))
    quotes_mismatch   = int(src_has_quotes and mt_has_straight)

    # дата скопирована без адаптации формата
    src_dates = RE_DATE.findall(src_text)
    date_format_error = int(
        bool(src_dates) and any(d in mt_text for d in src_dates)
    )

    return {
        "digit_match_ratio": digit_match_ratio,
        "punct_ratio":       punct_ratio,
        "quotes_mismatch":   quotes_mismatch,
        "date_format_error": date_format_error,
    }