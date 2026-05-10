# src/features/extractor.py
# Собирает все признаки в вектор numpy.
# Лёгкие (15): structural + formatting + linguistic — всегда.
# Тяжёлые (7): semantic + fluency — только если переданы модели.
#
# Использование (только лёгкие):
#   ex = FeatureExtractor()
#   result = ex.extract(src, mt)
#
# Использование (все 22 признака):
#   ex = FeatureExtractor()
#   ex.load_heavy_models()   # загружает LaBSE и ruGPT-3
#   result = ex.extract(src, mt)

import logging

import numpy as np
import spacy

from src.features import fluency, formatting, linguistic, semantic, structural

# порядок признаков — должен совпадать с разделом 5.1 архитектуры
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

FEATURE_NAMES = FEATURE_NAMES_LIGHT + FEATURE_NAMES_HEAVY  # все 22


class FeatureExtractor:
    def __init__(self) -> None:
        self._log = logging.getLogger(__name__)
        self.nlp_ru = spacy.load("ru_core_news_sm")
        self.nlp_en = spacy.load("en_core_web_sm", disable=["tagger", "parser", "lemmatizer"])

        # тяжёлые модели — None пока не вызван load_heavy_models()
        self.labse_model  = None
        self.gpt_tokenizer = None
        self.gpt_model    = None

    def load_heavy_models(self) -> None:
        """Загружает LaBSE и ruGPT-3. Вызывать один раз при старте."""
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._log.info("Загрузка LaBSE...")
        self.labse_model = SentenceTransformer("sentence-transformers/LaBSE")

        self._log.info("Загрузка ruGPT-3 Small...")
        gpt_name = "sberbank-ai/rugpt3small_based_on_gpt2"
        self.gpt_tokenizer = AutoTokenizer.from_pretrained(gpt_name)
        self.gpt_model     = AutoModelForCausalLM.from_pretrained(gpt_name)
        self.gpt_model.eval()

        self._log.info("Тяжёлые модели загружены.")

    @property
    def heavy_loaded(self) -> bool:
        return self.labse_model is not None

    def extract(self, src: str, mt: str) -> dict:
        """
        Извлекает признаки для одной пары (src, mt).
        Возвращает dict:
          vector       — np.array[15 или 22]
          formal_ratio — для межпредложенческого анализа (Блок 4)
          word_logprobs — list[float] для span-модели (только если heavy загружены)
          raw          — все признаки по имени
        """
        src_en_doc = self.nlp_en(src)
        mt_doc     = self.nlp_ru(mt)
        src_doc    = self.nlp_ru(src)

        feats = {}
        feats.update(structural.extract(src_doc, mt_doc))
        feats.update(formatting.extract(src_doc, mt_doc))
        feats.update(linguistic.extract(src_doc, mt_doc, src_en_doc))

        word_logprobs = []

        if self.heavy_loaded:
            feats.update(semantic.extract(src, mt, self.labse_model))
            fluency_result = fluency.extract(
                mt, self.gpt_tokenizer, self.gpt_model, self.nlp_ru
            )
            word_logprobs = fluency_result.pop("word_logprobs")
            feats.update(fluency_result)
            names = FEATURE_NAMES
        else:
            names = FEATURE_NAMES_LIGHT

        vector = np.array([feats[n] for n in names], dtype=np.float32)

        return {
            "vector":       vector,
            "formal_ratio": feats["formal_ratio"],
            "word_logprobs": word_logprobs,
            "raw":          feats,
        }

    def extract_batch(self, pairs: list[tuple[str, str]]) -> list[dict]:
        """Батчевое извлечение через nlp.pipe."""
        srcs = [p[0] for p in pairs]
        mts  = [p[1] for p in pairs]

        src_en_docs = list(self.nlp_en.pipe(srcs, batch_size=64))
        mt_docs     = list(self.nlp_ru.pipe(mts,  batch_size=64))
        src_docs    = list(self.nlp_ru.pipe(srcs, batch_size=64))

        light_results = []
        for src_doc, mt_doc, src_en_doc in zip(src_docs, mt_docs, src_en_docs):
            feats = {}
            feats.update(structural.extract(src_doc, mt_doc))
            feats.update(formatting.extract(src_doc, mt_doc))
            feats.update(linguistic.extract(src_doc, mt_doc, src_en_doc))
            light_results.append(feats)

        if not self.heavy_loaded:
            return [
                {
                    "vector":        np.array([f[n] for n in FEATURE_NAMES_LIGHT], dtype=np.float32),
                    "formal_ratio":  f["formal_ratio"],
                    "word_logprobs": [],
                    "raw":           f,
                }
                for f in light_results
            ]

        # тяжёлые батчем
        semantic_results = semantic.extract_batch(list(zip(srcs, mts)), self.labse_model)

        results = []
        for i, (src, mt) in enumerate(zip(srcs, mts)):
            feats = light_results[i]
            feats.update(semantic_results[i])

            fluency_result = fluency.extract(
                mt, self.gpt_tokenizer, self.gpt_model, self.nlp_ru
            )
            word_logprobs = fluency_result.pop("word_logprobs")
            feats.update(fluency_result)

            vector = np.array([feats[n] for n in FEATURE_NAMES], dtype=np.float32)
            results.append({
                "vector":        vector,
                "formal_ratio":  feats["formal_ratio"],
                "word_logprobs": word_logprobs,
                "raw":           feats,
            })

        return results
