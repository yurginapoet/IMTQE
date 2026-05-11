"""Имена колонок и файлов между шагами пайплайна (явный контракт)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Сплиты и целевая переменная
SPLIT_COL = "split"
SCORE_NORM_COL = "score_norm"
SRC_COL = "src"
MT_COL = "mt"

# Имена parquet (processed/)
SENTENCE_DA = "sentence_da.parquet"
SENTENCE_DA_AUGMENTED = "sentence_da_augmented.parquet"
SENTENCE_DA_FEATURES = "sentence_da_features.parquet"
HF_MQM_RAW = "hf_mqm_raw.parquet"
HF_MQM_DEDUP = "hf_mqm_dedup.parquet"
HF_MQM_FEATURES = "hf_mqm_features.parquet"
WORDLEVEL_TRAIN = "wordlevel_train.parquet"
WORDLEVEL_FEATURES = "wordlevel_features.parquet"


@dataclass(frozen=True)
class DataLayout:
    """Пути артефактов данных относительно корня данных."""

    data_dir: Path

    @property
    def processed(self) -> Path:
        return self.data_dir / "processed"

    def sentence_da_path(self) -> Path:
        return self.processed / SENTENCE_DA

    def sentence_da_features_path(self) -> Path:
        return self.processed / SENTENCE_DA_FEATURES

    def hf_mqm_features_path(self) -> Path:
        return self.processed / HF_MQM_FEATURES


@dataclass(frozen=True)
class ModelArtifacts:
    """Пути sentence/span моделей."""

    models_dir: Path

    def xgboost_sentence(self) -> Path:
        return self.models_dir / "xgboost_sentence.model"

    def shap_explainer(self) -> Path:
        return self.models_dir / "shap_explainer.pkl"

    def span_model_dir(self) -> Path:
        return self.models_dir / "xlm_roberta_span"
