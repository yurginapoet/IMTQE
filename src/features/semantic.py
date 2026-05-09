# src/features/semantic.py
# 2 признака семантического сходства через LaBSE.
# Модель загружается один раз снаружи и передаётся в extract().
# ~0.5 сек на предложение на CPU.
#
# Использование:
#   from sentence_transformers import SentenceTransformer
#   model = SentenceTransformer("sentence-transformers/LaBSE")
#   feats = extract(src, mt, model)

import numpy as np
from sentence_transformers import SentenceTransformer


def extract(src: str, mt: str, model: SentenceTransformer) -> dict:
    emb_src, emb_mt = model.encode([src, mt], convert_to_numpy=True)

    # косинусное сходство
    norm_src = np.linalg.norm(emb_src)
    norm_mt  = np.linalg.norm(emb_mt)
    cosine   = float(np.dot(emb_src, emb_mt) / (norm_src * norm_mt + 1e-9))

    # евклидово расстояние — дополняет косинус, учитывает норму вектора
    distance = float(np.linalg.norm(emb_src - emb_mt))

    return {
        "cosine_similarity":  cosine,
        "embedding_distance": distance,
    }


def extract_batch(
    pairs: list[tuple[str, str]],
    model: SentenceTransformer,
    batch_size: int = 64,
) -> list[dict]:
    """Батчевое извлечение — эффективнее на GPU/Colab."""
    srcs = [p[0] for p in pairs]
    mts  = [p[1] for p in pairs]

    all_texts = srcs + mts
    embs = model.encode(all_texts, batch_size=batch_size, convert_to_numpy=True)

    embs_src = embs[:len(srcs)]
    embs_mt  = embs[len(srcs):]

    results = []
    for esrc, emt in zip(embs_src, embs_mt):
        norm_src = np.linalg.norm(esrc)
        norm_mt  = np.linalg.norm(emt)
        cosine   = float(np.dot(esrc, emt) / (norm_src * norm_mt + 1e-9))
        distance = float(np.linalg.norm(esrc - emt))
        results.append({
            "cosine_similarity":  cosine,
            "embedding_distance": distance,
        })

    return results