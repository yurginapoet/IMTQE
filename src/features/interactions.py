"""
Производные (interaction) признаки для sentence-модели.

Единая реализация для обучения (pandas) и инференса (dict в FeatureExtractor).
Порядок имён — INTERACTION_FEATURE_NAMES в schema.py.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import pandas as pd


def interaction_features(feats: Mapping[str, float]) -> dict[str, float]:
    """
    Считает все interaction-признаки из уже собранного dict базовых фич.
    Отсутствующие базовые ключи трактуются как 0.0 (кроме length_ratio=1.0).
    """
    cos = float(feats.get("cosine_similarity", 0.0))
    lr = float(feats.get("length_ratio", 1.0))
    ppl = float(feats.get("perplexity", 0.0))
    ent = float(feats.get("entity_overlap_ratio", 0.0))
    oov = float(feats.get("oov_ratio", 0.0))
    min_lp = float(feats.get("min_token_log_prob", 0.0))
    mean_lp = float(feats.get("mean_log_prob", 0.0))
    var_ppl = float(feats.get("token_ppl_variance", 0.0))
    abs_ld = float(feats.get("abs_length_diff", 0.0))
    src_len = float(feats.get("src_length", 0.0))
    digit = float(feats.get("digit_match_ratio", 0.0))
    formal = float(feats.get("formal_ratio", 0.0))
    formal = float(feats.get("formal_ratio", 0.0))

    log_ppl = math.log1p(ppl)

    return {
        "cosine_x_length_ok":     cos * (1.0 - min(abs(lr - 1.0), 1.0)),
        "log_perplexity":         log_ppl,
        "cosine_per_logppl":      cos / (log_ppl + 1e-6),
        "entity_x_cosine":        ent * cos,
        "oov_x_bad_cosine":       oov * (1.0 - cos),
        "logprob_spike":          mean_lp - min_lp,
        "variance_x_bad_cosine":  var_ppl * (1.0 - cos),
        "normed_length_diff":     abs_ld / (src_len + 1e-6),
        "digit_x_entity":         digit * ent,
        "formal_x_cosine":        formal * cos,
    }


def _series(df: pd.DataFrame, name: str, default: float) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series(default, index=df.index, dtype=np.float64)


def add_interaction_columns_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Векторно добавляет колонки INTERACTION_FEATURE_NAMES (idempotent)."""
    df = df.copy()
    cos = _series(df, "cosine_similarity", 0.0)
    lr = _series(df, "length_ratio", 1.0)
    ppl = _series(df, "perplexity", 0.0)
    ent = _series(df, "entity_overlap_ratio", 0.0)
    oov = _series(df, "oov_ratio", 0.0)
    min_lp = _series(df, "min_token_log_prob", 0.0)
    mean_lp = _series(df, "mean_log_prob", 0.0)
    var_ppl = _series(df, "token_ppl_variance", 0.0)
    abs_ld = _series(df, "abs_length_diff", 0.0)
    src_len = _series(df, "src_length", 0.0)
    digit = _series(df, "digit_match_ratio", 0.0)
    formal = _series(df, "formal_ratio", 0.0)

    log_ppl = np.log1p(ppl)

    df["cosine_x_length_ok"]     = cos * (1.0 - (lr - 1.0).abs().clip(0, 1))
    df["log_perplexity"]         = log_ppl
    df["cosine_per_logppl"]      = cos / (log_ppl + 1e-6)
    df["entity_x_cosine"]        = ent * cos
    df["oov_x_bad_cosine"]       = oov * (1.0 - cos)
    df["logprob_spike"]          = mean_lp - min_lp
    df["variance_x_bad_cosine"]  = var_ppl * (1.0 - cos)
    df["normed_length_diff"]     = abs_ld / (src_len + 1e-6)
    df["digit_x_entity"]         = digit * ent
    df["formal_x_cosine"]        = formal * cos
    return df


__all__ = [
    "interaction_features",
    "add_interaction_columns_to_dataframe",
]
