# 5 признаков на основе длин src и mt.

import spacy


def extract(src_doc: spacy.tokens.Doc, mt_doc: spacy.tokens.Doc) -> dict:
    src_len = len(src_doc)
    mt_len  = len(mt_doc)

    return {
        "length_ratio":      mt_len / src_len if src_len > 0 else 0.0,
        "abs_length_diff":   abs(mt_len - src_len),
        "token_count_diff":  mt_len - src_len,
        "src_length":        src_len,
        "mt_length":         mt_len,
    }