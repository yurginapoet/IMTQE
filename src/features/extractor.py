"""
Собирает sentence-level признаки в единый numpy-вектор.

Доступны три режима:
  1. Лёгкий: structural + formatting + linguistic.
  2. Классический тяжёлый: + LaBSE + ruGPT-3.
  3. Расширенный: + MiniLM semantic PCA признаки.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import spacy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import Config
from src.features import fluency, formatting, linguistic, neural, semantic, structural
from src.features.schema import (
    FEATURE_NAMES,
    FEATURE_NAMES_CLASSIC,
    FEATURE_NAMES_LIGHT,
    SEMANTIC_FEATURE_NAMES,
)


class FeatureExtractor:
    def __init__(
        self,
        semantic_pca_path: str | Path | None = None,
        device: str | None = None,
    ) -> None:
        self._log = logging.getLogger(__name__)
        self.nlp_ru = spacy.load("ru_core_news_sm")
        self.nlp_en = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.semantic_pca_path = Path(semantic_pca_path or Config.SEMANTIC_PCA_PATH)

        self.labse_model = None
        self.gpt_tokenizer = None
        self.gpt_model = None
        self.semantic_encoder = None
        self.semantic_pca = None

    def load_heavy_models(self, require_neural: bool = False) -> None:
        """Загружает LaBSE, ruGPT-3 и при наличии MiniLM + PCA."""
        from sentence_transformers import SentenceTransformer

        self._log.info("Используем device=%s для тяжёлых моделей", self.device)
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

        self._load_semantic_augmentation(require_neural=require_neural)
        self._log.info(
            "Тяжёлые модели готовы. active_feature_count=%d",
            len(self.active_feature_names),
        )

    def _load_semantic_augmentation(self, require_neural: bool) -> None:
        if self.semantic_encoder is None:
            self._log.info("Загрузка MiniLM encoder для semantic PCA...")
            try:
                self.semantic_encoder = neural.load_encoder(device=self.device)
                self._log.info("MiniLM encoder загружен")
            except Exception as exc:
                self._log.error("Не удалось загрузить MiniLM encoder: %s", exc)
                if require_neural:
                    raise

        if self.semantic_pca is None:
            self._log.info("Загрузка PCA артефакта: %s", self.semantic_pca_path)
            try:
                self.semantic_pca = neural.load_pca(self.semantic_pca_path)
                self._log.info("Semantic PCA загружен")
            except FileNotFoundError as exc:
                self._log.warning(
                    "PCA артефакт не найден: %s. "
                    "Для 86-мерных признаков сначала запусти scripts/train_semantic_pca.py",
                    exc,
                )
                if require_neural:
                    raise
            except Exception as exc:
                self._log.error("Не удалось загрузить semantic PCA: %s", exc)
                if require_neural:
                    raise

    @property
    def heavy_loaded(self) -> bool:
        return (
            self.labse_model is not None
            and self.gpt_model is not None
            and self.gpt_tokenizer is not None
        )

    @property
    def semantic_augmented_loaded(self) -> bool:
        return self.semantic_encoder is not None and self.semantic_pca is not None

    @property
    def active_feature_names(self) -> list[str]:
        if self.heavy_loaded and self.semantic_augmented_loaded:
            return FEATURE_NAMES
        if self.heavy_loaded:
            return FEATURE_NAMES_CLASSIC
        return FEATURE_NAMES_LIGHT

    def extract(self, src: str, mt: str) -> dict:
        """
        Возвращает:
          vector         np.array с активными признаками
          formal_ratio   float для paragraph-level анализа
          word_logprobs  list[float] для span-модели
          mt_words       list[str] для согласованной токенизации UI/span-model
          raw            dict всех рассчитанных признаков по имени
        """
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

            if self.semantic_augmented_loaded:
                feats.update(
                    neural.extract(
                        src,
                        mt,
                        self.semantic_encoder,
                        self.semantic_pca,
                    )
                )

        vector = np.array(
            [feats[name] for name in self.active_feature_names],
            dtype=np.float32,
        )

        return {
            "vector": vector,
            "formal_ratio": feats["formal_ratio"],
            "word_logprobs": word_logprobs,
            "mt_words": mt_words,
            "raw": feats,
        }

    def extract_batch(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """Батчевое извлечение через spaCy pipe + batched encoders."""
        srcs = [src for src, _ in pairs]
        mts = [mt for _, mt in pairs]

        src_en_docs = list(self.nlp_en.pipe(srcs, batch_size=64))
        mt_docs = list(self.nlp_ru.pipe(mts, batch_size=64))
        src_docs = list(self.nlp_ru.pipe(srcs, batch_size=64))

        light_results: list[dict[str, float]] = []
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
                    "vector": np.array(
                        [feats[name] for name in FEATURE_NAMES_LIGHT],
                        dtype=np.float32,
                    ),
                    "formal_ratio": feats["formal_ratio"],
                    "word_logprobs": [],
                    "mt_words": mt_words,
                    "raw": feats,
                }
                for feats, mt_words in zip(light_results, mt_words_list)
            ]

        semantic_results = semantic.extract_batch(list(zip(srcs, mts)), self.labse_model)
        neural_results = (
            neural.extract_batch(
                list(zip(srcs, mts)),
                self.semantic_encoder,
                self.semantic_pca,
            )
            if self.semantic_augmented_loaded
            else None
        )
        feature_names = self.active_feature_names

        results = []
        for idx, mt in enumerate(mts):
            feats = light_results[idx]
            feats.update(semantic_results[idx])
            if neural_results is not None:
                feats.update(neural_results[idx])

            fluency_result = fluency.extract(
                mt,
                self.gpt_tokenizer,
                self.gpt_model,
                self.nlp_ru,
            )
            word_logprobs = fluency_result.pop("word_logprobs")
            feats.update(fluency_result)

            vector = np.array([feats[name] for name in feature_names], dtype=np.float32)
            results.append(
                {
                    "vector": vector,
                    "formal_ratio": feats["formal_ratio"],
                    "word_logprobs": word_logprobs,
                    "mt_words": mt_words_list[idx],
                    "raw": feats,
                }
            )

        return results
