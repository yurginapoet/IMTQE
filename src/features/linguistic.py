# src/features/linguistic.py
# 7 признаков через spaCy: OOV, TTR, длина токена, NER, согласование,
# глубина дерева, регистр речи. ~0.1 сек на предложение.

import spacy

# Словарь формальной лексики для русского языка.
# Используется для признака formal_ratio и межпредложенческого анализа (Блок 4).
# Расширяй по необходимости.
FORMAL_VOCAB = {
    "осуществить", "осуществлять", "являться", "представлять", "обеспечивать",
    "реализовать", "функционировать", "обозначить", "уведомить", "предоставить",
    "рассмотреть", "утвердить", "согласовать", "направить", "установить",
    "подтвердить", "сообщить", "проинформировать", "ходатайствовать",
    "вследствие", "посредством", "относительно", "соответственно", "согласно",
    "настоящий", "данный", "указанный", "вышеуказанный", "нижеследующий",
    "надлежащий", "соответствующий", "установленный", "обязательный",
}


def _agreement_errors(mt_doc: spacy.tokens.Doc) -> int:
    """
    Считает нарушения согласования по роду/числу/падежу
    для пар nsubj-verb и adj-noun через dependency parse.
    """
    errors = 0
    for token in mt_doc:
        if token.dep_ not in ("nsubj", "amod"):
            continue
        head = token.head
        # сравниваем морфологические признаки
        tok_morph  = token.morph.to_dict()
        head_morph = head.morph.to_dict()
        for feat in ("Number", "Gender", "Case"):
            t_val = tok_morph.get(feat)
            h_val = head_morph.get(feat)
            if t_val and h_val and t_val != h_val:
                errors += 1
                break  # один раз на пару
    return errors


def _tree_depth(doc: spacy.tokens.Doc) -> int:
    """Максимальная глубина дерева зависимостей."""
    def depth(token: spacy.tokens.Token) -> int:
        if token == token.head:  # корень
            return 0
        return 1 + depth(token.head)

    return max((depth(t) for t in doc), default=0)


def extract(
    src_doc: spacy.tokens.Doc,
    mt_doc: spacy.tokens.Doc,
    src_en_doc: spacy.tokens.Doc,  # spaCy en модель для NER
) -> dict:
    mt_words = [t.text.lower() for t in mt_doc if not t.is_space]
    n        = len(mt_words) or 1

    # доля слов вне словаря spaCy
    oov_ratio = sum(1 for t in mt_doc if t.is_oov and not t.is_space) / n

    # лексическое разнообразие
    type_token_ratio = len(set(mt_words)) / n

    # средняя длина слова в символах
    avg_token_length = sum(len(w) for w in mt_words) / n

    # доля именованных сущностей из src которые есть в mt
    src_ents = {ent.text for ent in src_en_doc.ents}
    if src_ents:
        mt_text = mt_doc.text
        entity_overlap_ratio = sum(1 for e in src_ents if e in mt_text) / len(src_ents)
    else:
        entity_overlap_ratio = 1.0

    # нарушения согласования
    agreement_errors = _agreement_errors(mt_doc)

    # глубина синтаксического дерева
    syntax_depth = _tree_depth(mt_doc)

    # доля формальной лексики (лемматизированные слова)
    mt_lemmas   = [t.lemma_.lower() for t in mt_doc if not t.is_space]
    formal_ratio = sum(1 for l in mt_lemmas if l in FORMAL_VOCAB) / (len(mt_lemmas) or 1)

    return {
        "oov_ratio":            oov_ratio,
        "type_token_ratio":     type_token_ratio,
        "avg_token_length":     avg_token_length,
        "entity_overlap_ratio": entity_overlap_ratio,
        "agreement_errors":     agreement_errors,
        "syntax_depth":         syntax_depth,
        "formal_ratio":         formal_ratio,
    }