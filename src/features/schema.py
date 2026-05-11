"""Лёгкий модуль со схемой sentence-level признаков без тяжёлых импортов."""

SEMANTIC_EXPLICIT = ["cosine_similarity", "embedding_distance"]

FEATURE_NAMES_LIGHT = [
    "length_ratio", "abs_length_diff", "token_count_diff",
    "src_length", "mt_length",
    "digit_match_ratio", "punct_ratio", "quotes_mismatch", "date_format_error",
    "oov_ratio", "type_token_ratio", "avg_token_length",
    "entity_overlap_ratio", "agreement_errors", "syntax_depth", "formal_ratio",
]

FEATURE_NAMES_HEAVY = [
    "perplexity", "mean_log_prob", "token_ppl_variance", "min_token_log_prob",
]

FEATURE_NAMES_CLASSIC = FEATURE_NAMES_LIGHT + SEMANTIC_EXPLICIT + FEATURE_NAMES_HEAVY 

SEMANTIC_FEATURE_NAMES = [
    f"semantic_{idx:02d}" for idx in range(64)
]

FEATURE_NAMES = FEATURE_NAMES_CLASSIC + SEMANTIC_FEATURE_NAMES

# Производные признаки (тот же порядок, что в add_interaction_features / FeatureExtractor).
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
    "dist_x_logppl",
]

# Полный вектор sentence-модели: базовые 86 + 11 interaction = 97.
SENTENCE_FEATURE_NAMES = FEATURE_NAMES + INTERACTION_FEATURE_NAMES

# Классический тяжёлый режим без semantic PCA: 22 + 11 = 33.
SENTENCE_FEATURE_NAMES_CLASSIC = FEATURE_NAMES_CLASSIC + INTERACTION_FEATURE_NAMES
