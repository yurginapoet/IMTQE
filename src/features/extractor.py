"""
Собирает sentence-level признаки в единый numpy-вектор.

Два режима:
  1. Лёгкий: structural + formatting + linguistic (spaCy + pymorphy2).
  2. Тяжёлый: + LaBSE + ruGPT-3 + interaction признаки.
"""

from __future__ import annotations

import logging

import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import Config
from src.features import fluency, formatting, linguistic, semantic, structural
from src.features.interactions import interaction_features
from src.features.schema import FEATURE_NAMES_LIGHT, SENTENCE_FEATURE_NAMES


class FeatureExtractor:
    def __init__(self, device: str | None = None) -> None:
        self._log = logging.getLogger(__name__)
        self.nlp_ru = spacy.load("ru_core_news_sm")
        self.nlp_en = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.labse_model = None
        self.gpt_tokenizer = None
        self.gpt_model = None

    def load_heavy_models(self, require_neural: bool = False) -> None:
        """Загружает LaBSE и ruGPT-3."""
        self._log.info("Device=%s", self.device)
        local_files_only = Config.hf_local_files_only()

        if self.labse_model is None:
            self._log.info("Загрузка LaBSE...")
            try:
                labse_path = Config.resolve_hf_model_path(Config.LABSE_MODEL_NAME)
                self.labse_model = SentenceTransformer(
                    labse_path,
                    device=self.device,
                    local_files_only=local_files_only,
                )
                self._log.info("LaBSE загружена")
            except Exception as exc:
                self._log.error("Не удалось загрузить LaBSE: %s", exc)
                if require_neural:
                    raise

        if self.gpt_model is None or self.gpt_tokenizer is None:
            self._log.info("Загрузка ruGPT-3 Small...")
            try:
                rugpt_path = Config.resolve_hf_model_path(Config.RUGPT_MODEL_NAME)
                self.gpt_tokenizer = AutoTokenizer.from_pretrained(
                    rugpt_path,
                    local_files_only=local_files_only,
                )
                if self.gpt_tokenizer.pad_token is None:
                    self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
                self.gpt_model = AutoModelForCausalLM.from_pretrained(
                    rugpt_path,
                    local_files_only=local_files_only,
                )
                self.gpt_model.to(self.device)
                self.gpt_model.eval()
                self._log.info("ruGPT-3 загружен")
            except Exception as exc:
                self._log.error("Не удалось загрузить ruGPT-3: %s", exc)
                if require_neural:
                    raise

        self._log.info(
            "Тяжёлые модели готовы. active_feature_count=%d",
            len(self.active_feature_names),
        )

    @property
    def heavy_loaded(self) -> bool:
        return (
            self.labse_model is not None
            and self.gpt_model is not None
            and self.gpt_tokenizer is not None
        )

    @property
    def active_feature_names(self) -> list[str]:
        if self.heavy_loaded:
            return list(SENTENCE_FEATURE_NAMES)
        return list(FEATURE_NAMES_LIGHT)

    @staticmethod
    def _vectorize_features(feats: dict, feature_names: list[str]) -> np.ndarray:
        missing = [name for name in feature_names if name not in feats]
        if missing:
            raise KeyError(
                "Missing extracted features: "
                + ", ".join(missing)
            )
        return np.array([feats[name] for name in feature_names], dtype=np.float32)

    def extract(self, src: str, mt: str) -> dict:
        src_en_doc = self.nlp_en(src)
        mt_doc = self.nlp_ru(mt)
        src_doc = self.nlp_ru(src)

        feats = {}
        feats.update(structural.extract(src_doc, mt_doc))
        feats.update(formatting.extract(src_doc, mt_doc))
        feats.update(linguistic.extract(src_doc, mt_doc, src_en_doc))

        mt_words = [token.text for token in mt_doc if not token.is_space]
        word_logprobs: list[float] = []

        if self.heavy_loaded:
            feats.update(semantic.extract(src, mt, self.labse_model))
            fluency_result = fluency.extract(
                mt,
                self.gpt_tokenizer,
                self.gpt_model,
                self.nlp_ru,
            )
            word_logprobs = fluency_result.pop("word_logprobs")
            feats.update(fluency_result)
            feats.update(interaction_features(feats))

        vector = self._vectorize_features(feats, self.active_feature_names)

        return {
            "vector": vector,
            "word_logprobs": word_logprobs,
            "mt_words": mt_words,
            "raw": feats,
        }

    def extract_batch(self, pairs: list[tuple[str, str]]) -> list[dict]:
        srcs = [src for src, _ in pairs]
        mts = [mt for _, mt in pairs]

        src_en_docs = list(self.nlp_en.pipe(srcs, batch_size=64))
        mt_docs = list(self.nlp_ru.pipe(mts, batch_size=64))
        src_docs = list(self.nlp_ru.pipe(srcs, batch_size=64))

        light_results: list[dict] = []
        mt_words_list: list[list[str]] = []

        for src_doc, mt_doc, src_en_doc in zip(src_docs, mt_docs, src_en_docs):
            feats = {}
            feats.update(structural.extract(src_doc, mt_doc))
            feats.update(formatting.extract(src_doc, mt_doc))
            feats.update(linguistic.extract(src_doc, mt_doc, src_en_doc))
            light_results.append(feats)
            mt_words_list.append([token.text for token in mt_doc if not token.is_space])

        if not self.heavy_loaded:
            return [
                {
                    "vector": self._vectorize_features(feats, FEATURE_NAMES_LIGHT),
                    "word_logprobs": [],
                    "mt_words": mt_words,
                    "raw": feats,
                }
                for feats, mt_words in zip(light_results, mt_words_list)
            ]

        semantic_results = semantic.extract_batch(list(zip(srcs, mts)), self.labse_model)
        out_names = self.active_feature_names

        results = []
        for idx, mt in enumerate(mts):
            feats = light_results[idx]
            feats.update(semantic_results[idx])

            fluency_result = fluency.extract(
                mt,
                self.gpt_tokenizer,
                self.gpt_model,
                self.nlp_ru,
            )
            word_logprobs = fluency_result.pop("word_logprobs")
            feats.update(fluency_result)
            feats.update(interaction_features(feats))

            vector = self._vectorize_features(feats, out_names)
            results.append(
                {
                    "vector": vector,
                    "word_logprobs": word_logprobs,
                    "mt_words": mt_words_list[idx],
                    "raw": feats,
                }
            )

        return results
