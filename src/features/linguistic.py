# src/features/linguistic.py
# Лингвистические признаки через spaCy + pymorphy(3/2) + regex

import math
import re
import spacy

try:
    import pymorphy3 as _pymorphy
except Exception:
    try:
        import pymorphy2 as _pymorphy
    except Exception:
        _pymorphy = None

try:
    _morph = _pymorphy.MorphAnalyzer() if _pymorphy is not None else None
except Exception:
    _morph = None

_RE_LATIN = re.compile(r"[A-Za-z]")
_RE_CYRILLIC = re.compile(r"[а-яёА-ЯЁ]")

def _formal_ratio(mt_doc: spacy.tokens.Doc) -> float:
    """
    Доля слов с формальными морфологическими признаками через pymorphy2:
    - отглагольные существительные (суффиксы -ние, -ция, -ость, -ение)
    - причастия и деепричастия (PRTF, PRTS, GRND)
    - нет личных местоимений (они снижают формальность)
    Не требует словаря — только морфология.
    """
    if _morph is None:
        return 0.0
    words = [t.text for t in mt_doc if not t.is_space and not t.is_punct
             and _RE_CYRILLIC.search(t.text)]
    if not words:
        return 0.0
    formal_count = 0
    for w in words:
        parses = _morph.parse(w)
        if not parses:
            continue
        tag = parses[0].tag
        pos = tag.POS
        if pos in ("PRTF", "PRTS", "GRND"):
            formal_count += 1
            continue
        if pos == "NOUN":
            lemma = parses[0].normal_form
            if any(lemma.endswith(suf) for suf in ("ние", "ция", "ость", "ение", "ание")):
                formal_count += 1
    return formal_count / len(words)



def _agreement_errors(mt_doc: spacy.tokens.Doc) -> int:
    errors = 0
    for token in mt_doc:
        if token.dep_ not in ("nsubj", "amod"):
            continue
        head = token.head
        tok_morph = token.morph.to_dict()
        head_morph = head.morph.to_dict()
        for feat in ("Number", "Gender", "Case"):
            t_val = tok_morph.get(feat)
            h_val = head_morph.get(feat)
            if t_val and h_val and t_val != h_val:
                errors += 1
                break
    return errors


def _tree_depth(doc: spacy.tokens.Doc) -> int:
    def depth(token: spacy.tokens.Token) -> int:
        if token == token.head:
            return 0
        return 1 + depth(token.head)
    return max((depth(t) for t in doc), default=0)


def _morphology_error_rate(mt_doc: spacy.tokens.Doc) -> float:
    """
    Доля слов mt для которых pymorphy2 не нашёл ни одного разбора
    с известной частью речи. Сигнал опечаток и несуществующих форм.
    """
    if _morph is None:
        return 0.0
    words = [t.text for t in mt_doc if not t.is_space and _RE_CYRILLIC.search(t.text)]
    if not words:
        return 0.0
    errors = 0
    for w in words:
        parses = _morph.parse(w)
        if not parses or parses[0].tag.POS is None:
            errors += 1
    return errors / len(words)


def _repetition_ratio(mt_doc: spacy.tokens.Doc) -> float:
    """
    Доля слов которые встречаются более одного раза (кальки, тавтология).
    Нормализована на длину.
    """
    words = [t.lemma_.lower() for t in mt_doc if not t.is_space and not t.is_punct]
    if len(words) < 2:
        return 0.0
    from collections import Counter
    counts = Counter(words)
    repeated = sum(1 for w, c in counts.items() if c > 1)
    return repeated / len(words)


def _named_entity_missing_ratio(src_en_doc: spacy.tokens.Doc, mt_doc: spacy.tokens.Doc) -> float:
    """
    Доля именованных сущностей из src которые НЕ найдены в mt.
    Дополняет entity_overlap_ratio — смотрит на пропуски а не на совпадения.
    """
    src_ents = {ent.text.lower() for ent in src_en_doc.ents}
    if not src_ents:
        return 0.0
    mt_text = mt_doc.text.lower()
    missing = sum(1 for e in src_ents if e not in mt_text)
    return missing / len(src_ents)


def _latin_ratio(mt_doc: spacy.tokens.Doc) -> float:
    """
    Доля символов в mt которые латинские.
    Сигнал непереведённых фрагментов.
    """
    text = mt_doc.text
    if not text:
        return 0.0
    latin = sum(1 for c in text if _RE_LATIN.match(c))
    return latin / len(text)


def _avg_word_rank(mt_doc: spacy.tokens.Doc) -> float:
    """
    Псевдо-ранг редкости слов.

    Используем отрицательный log(score) лучшего морфологического разбора:
    - частые/ожидаемые слова обычно дают score ближе к 1 -> вклад ближе к 0;
    - более редкие/сомнительные/сложные формы дают меньший score -> вклад выше.

    Это не корпусный Zipf/rank, но устойчивый лёгкий сигнал редкости без внешних словарей.
    """
    words = [
        t.text
        for t in mt_doc
        if not t.is_space and not t.is_punct and _RE_CYRILLIC.search(t.text)
    ]
    if not words or _morph is None:
        return 0.0

    rarity_scores: list[float] = []
    for word in words:
        parses = _morph.parse(word)
        if not parses:
            rarity_scores.append(12.0)
            continue
        best_score = max(float(getattr(parse, "score", 0.0) or 0.0) for parse in parses)
        rarity_scores.append(-math.log(max(best_score, 1e-6)))

    return sum(rarity_scores) / len(rarity_scores)


def _untranslated_ratio(mt_doc: spacy.tokens.Doc) -> float:
    """
    Доля явно непереведённых слов (латиница + почти нет кириллицы).
    Фильтруем короткие слова и вероятные имена собственные.
    """
    words = [t for t in mt_doc if not t.is_space and not t.is_punct and len(t.text) > 2]
    if not words:
        return 0.0

    bad_count = 0
    for token in words:
        text = token.text
        has_latin = bool(_RE_LATIN.search(text))
        has_cyr = bool(_RE_CYRILLIC.search(text))

        if has_latin and not has_cyr:
            # Пропускаем вероятные имена собственные и короткие токены
            if (token.text[0].isupper() and len(text) <= 12) or len(text) <= 3:
                continue
            bad_count += 1

    return bad_count / len(words)


def extract(
    src_doc: spacy.tokens.Doc,
    mt_doc: spacy.tokens.Doc,
    src_en_doc: spacy.tokens.Doc,
) -> dict:
    mt_words = [t.text.lower() for t in mt_doc if not t.is_space]
    n = len(mt_words) or 1

    oov_ratio = sum(1 for t in mt_doc if t.is_oov and not t.is_space) / n
    type_token_ratio = len(set(mt_words)) / n
    avg_token_length = sum(len(w) for w in mt_words) / n

    src_ents = {ent.text for ent in src_en_doc.ents}
    if src_ents:
        mt_text = mt_doc.text
        entity_overlap_ratio = sum(1 for e in src_ents if e in mt_text) / len(src_ents)
    else:
        entity_overlap_ratio = 1.0

    agreement_errors = _agreement_errors(mt_doc)
    syntax_depth = _tree_depth(mt_doc)
    formal_ratio = _formal_ratio(mt_doc)

    morphology_error_rate = _morphology_error_rate(mt_doc)
    repetition_ratio = _repetition_ratio(mt_doc)
    named_entity_missing_ratio = _named_entity_missing_ratio(src_en_doc, mt_doc)
    latin_ratio = _latin_ratio(mt_doc)
    avg_word_rank = _avg_word_rank(mt_doc)

    untranslated_ratio = _untranslated_ratio(mt_doc)

    return {
        "oov_ratio":                    oov_ratio,
        "type_token_ratio":             type_token_ratio,
        "avg_token_length":             avg_token_length,
        "entity_overlap_ratio":         entity_overlap_ratio,
        "agreement_errors":             agreement_errors,
        "syntax_depth":                 syntax_depth,
        "formal_ratio":                 formal_ratio,
        "morphology_error_rate":        morphology_error_rate,
        "repetition_ratio":             repetition_ratio,
        "named_entity_missing_ratio":   named_entity_missing_ratio,
        "latin_ratio":                  latin_ratio,
        "avg_word_rank":                avg_word_rank,
        "untranslated_ratio":           untranslated_ratio,
    }
