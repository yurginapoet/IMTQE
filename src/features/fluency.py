# src/features/fluency.py
# 4 sentence-level признака + пословные logprobs через ruGPT-3 Small.
# Модель и токенизатор загружаются один раз снаружи.
# ~0.3 сек на предложение на CPU.
#
# Использование:
#   from transformers import AutoTokenizer, AutoModelForCausalLM
#   tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
#   model     = AutoModelForCausalLM.from_pretrained("sberbank-ai/rugpt3small_based_on_gpt2")
#   feats = extract(mt, tokenizer, model, nlp_ru)
#
# word_logprobs не входит в вектор 22 признаков NGBoost —
# передаётся напрямую в span-модель (Блок 3).

import numpy as np
import spacy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _get_word_logprobs(
    mt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    nlp_ru: spacy.language.Language,
) -> list[float]:
    """
    Считает logprob для каждого spaCy слова mt.
    BPE субтокены одного слова суммируются (log вероятности складываются).
    Маппинг BPE -> spaCy слово строится через char offsets.
    """
    doc = nlp_ru(mt)
    spacy_words = [t for t in doc if not t.is_space]

    if not spacy_words:
        return []

    # токенизируем с offset mapping чтобы знать позиции субтокенов
    encoding = tokenizer(
        mt,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids     = encoding["input_ids"]       # [1, seq_len]
    offset_mapping = encoding["offset_mapping"][0].tolist()  # [(start, end), ...]

    if input_ids.shape[1] == 0:
        return [0.0] * len(spacy_words)

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # logprobs всех токенов через авторегрессионное разложение
    with torch.no_grad():
        outputs = model(input_ids)
        logits  = outputs.logits[0]  # [seq_len, vocab_size]

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # logprob токена t = log_probs[t-1, input_ids[t]] (предсказание следующего)
    token_logprobs = [0.0]  # первый токен не имеет предыдущего контекста
    for i in range(1, input_ids.shape[1]):
        lp = log_probs[i - 1, input_ids[0, i]].item()
        token_logprobs.append(lp)

    # агрегация BPE субтокенов -> spaCy слова через char offsets
    word_logprobs = []
    for word in spacy_words:
        w_start, w_end = word.idx, word.idx + len(word.text)
        lp_sum = 0.0
        found  = False
        for tok_idx, (t_start, t_end) in enumerate(offset_mapping):
            # субтокен попадает в границы слова
            if t_start >= w_start and t_end <= w_end:
                lp_sum += token_logprobs[tok_idx]
                found = True
        word_logprobs.append(lp_sum if found else 0.0)

    return word_logprobs


def extract(
    mt: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    nlp_ru: spacy.language.Language,
) -> dict:
    """
    Возвращает 4 sentence-level признака и word_logprobs.
    word_logprobs — список float, один на spaCy слово mt.
    """
    word_logprobs = _get_word_logprobs(mt, tokenizer, model, nlp_ru)

    if not word_logprobs:
        return {
            "perplexity":         0.0,
            "mean_log_prob":      0.0,
            "token_ppl_variance": 0.0,
            "min_token_log_prob": 0.0,
            "word_logprobs":      [],
        }

    lps  = np.array(word_logprobs)
    mean = float(lps.mean())

    # perplexity = exp(-mean_log_prob)
    perplexity = float(np.exp(-mean))

    # разброс logprobs — высокий std при нормальном mean сигнализирует
    # о локальных аномалиях (кандидаты на ошибки)
    variance = float(np.mean((lps - mean) ** 2))

    return {
        "perplexity":         perplexity,
        "mean_log_prob":      mean,
        "token_ppl_variance": variance,
        "min_token_log_prob": float(lps.min()),
        "word_logprobs":      word_logprobs,
    }
