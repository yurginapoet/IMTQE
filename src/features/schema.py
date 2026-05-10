"""Лёгкий модуль со схемой sentence-level признаков без тяжёлых импортов."""

FEATURE_NAMES_LIGHT = [
    "length_ratio", "abs_length_diff", "token_count_diff",
    "src_length", "mt_length",
    "digit_match_ratio", "punct_ratio", "quotes_mismatch", "date_format_error",
    "oov_ratio", "type_token_ratio", "avg_token_length",
    "entity_overlap_ratio", "agreement_errors", "syntax_depth", "formal_ratio",
]

FEATURE_NAMES_HEAVY = [
    "cosine_similarity", "embedding_distance",
    "perplexity", "mean_log_prob", "token_ppl_variance", "min_token_log_prob",
]

FEATURE_NAMES_CLASSIC = FEATURE_NAMES_LIGHT + FEATURE_NAMES_HEAVY

SEMANTIC_FEATURE_NAMES = [
    f"semantic_{idx:02d}" for idx in range(64)
]

FEATURE_NAMES = FEATURE_NAMES_CLASSIC + SEMANTIC_FEATURE_NAMES
