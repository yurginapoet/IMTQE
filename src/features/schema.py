"""Лёгкий модуль со схемой sentence-level признаков без тяжёлых импортов."""

SEMANTIC_EXPLICIT = ["cosine_similarity", "embedding_distance"]

FEATURE_NAMES_LIGHT = [
    # structural (7)
    "length_ratio", "abs_length_diff", "token_count_diff",
    "src_length", "mt_length", "compression_ratio", "sentence_count_diff",

    # formatting (7)
    "digit_match_ratio", "punct_ratio", "quotes_mismatch", "date_format_error",
    "number_count_diff", "capitalization_mismatch", "currency_symbol_mismatch",

    # linguistic (13)
    "oov_ratio", "type_token_ratio", "avg_token_length",
    "entity_overlap_ratio", "agreement_errors", "syntax_depth", "formal_ratio",
    "morphology_error_rate", "repetition_ratio",
    "named_entity_missing_ratio", "latin_ratio", "avg_word_rank",
    "untranslated_ratio",                    # новый
]

FEATURE_NAMES_HEAVY = [
    "perplexity", "mean_log_prob", "token_ppl_variance", "min_token_log_prob",
]

# Основной вектор: 7 + 7 + 13 + 2 + 4 = 33 базовых признака
FEATURE_NAMES_CLASSIC = FEATURE_NAMES_LIGHT + SEMANTIC_EXPLICIT + FEATURE_NAMES_HEAVY

# Производные признаки
INTERACTION_FEATURE_NAMES = [
    "cosine_x_length_ok",
    "log_perplexity",
    "cosine_per_logppl",
    "entity_x_cosine",
    "oov_x_bad_cosine",
    "logprob_spike",
    "variance_x_bad_cosine",
    "normed_length_diff",
    "digit_x_entity",
    "formal_x_cosine",
]

# Полный вектор
SENTENCE_FEATURE_NAMES = FEATURE_NAMES_CLASSIC + INTERACTION_FEATURE_NAMES
SENTENCE_FEATURE_NAMES_CLASSIC = SENTENCE_FEATURE_NAMES