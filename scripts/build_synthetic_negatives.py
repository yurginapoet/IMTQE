"""
scripts/build_synthetic_negatives.py

Добавляет синтетические низкокачественные переводы в train split sentence DA.

Вход:
  data/processed/sentence_da.parquet

Выход:
  data/processed/sentence_da_augmented.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RANDOM_SEED = 42

SCORE_RANGES = {
    "shuffle": (0.10, 0.30),
    "untranslated": (0.00, 0.20),
    "deletion": (0.10, 0.40),
    "entity_corruption": (0.20, 0.50),
}

ENTITY_RE = re.compile(r"^[A-ZА-ЯЁ][\w.-]*$", re.UNICODE)
ENTITY_POOL = [
    "Google",
    "Amazon",
    "Apple",
    "Microsoft",
    "Москва",
    "Сбербанк",
    "Газпром",
    "2027",
]


def make_hash(src: str, mt: str) -> str:
    key = src.strip() + "|||" + mt.strip()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def sample_score(kind: str, rng: np.random.Generator) -> float:
    low, high = SCORE_RANGES[kind]
    return float(rng.uniform(low, high))


def tokenize(text: str) -> list[str]:
    return [token for token in str(text).split() if token]


def corrupt_shuffle(mt: str, rng: np.random.Generator) -> str | None:
    tokens = tokenize(mt)
    if len(tokens) < 3:
        return None
    shuffled = tokens.copy()
    rng.shuffle(shuffled)
    if shuffled == tokens:
        shuffled = list(reversed(tokens))
    return " ".join(shuffled)


def corrupt_partial_untranslation(src: str, mt: str, rng: np.random.Generator) -> str | None:
    src_tokens = tokenize(src)
    mt_tokens = tokenize(mt)
    if not src_tokens or len(mt_tokens) < 2:
        return None

    n_replace = max(1, int(round(len(mt_tokens) * 0.3)))
    positions = rng.choice(len(mt_tokens), size=min(n_replace, len(mt_tokens)), replace=False)
    replacements = rng.choice(src_tokens, size=len(positions), replace=len(src_tokens) < len(positions))

    corrupted = mt_tokens.copy()
    for pos, replacement in zip(positions, replacements, strict=False):
        corrupted[int(pos)] = str(replacement)
    return " ".join(corrupted)


def corrupt_deletion(mt: str, rng: np.random.Generator) -> str | None:
    tokens = tokenize(mt)
    if len(tokens) < 3:
        return None

    n_delete = max(1, int(round(len(tokens) * 0.3)))
    delete_positions = set(
        int(pos) for pos in rng.choice(len(tokens), size=min(n_delete, len(tokens) - 1), replace=False)
    )
    corrupted = [token for idx, token in enumerate(tokens) if idx not in delete_positions]
    if len(corrupted) == len(tokens):
        corrupted = tokens[:-1]
    return " ".join(corrupted) if corrupted else None


def corrupt_entity(mt: str, rng: np.random.Generator) -> str | None:
    tokens = tokenize(mt)
    if not tokens:
        return None

    entity_positions = [
        idx for idx, token in enumerate(tokens)
        if ENTITY_RE.match(token) or any(ch.isdigit() for ch in token)
    ]
    if not entity_positions:
        return None

    pos = int(rng.choice(entity_positions))
    current = tokens[pos]
    candidates = [candidate for candidate in ENTITY_POOL if candidate != current]
    if not candidates:
        return None

    corrupted = tokens.copy()
    corrupted[pos] = str(rng.choice(candidates))
    return " ".join(corrupted)


CORRUPTION_FUNCS = {
    "shuffle": lambda src, mt, rng: corrupt_shuffle(mt, rng),
    "untranslated": corrupt_partial_untranslation,
    "deletion": lambda src, mt, rng: corrupt_deletion(mt, rng),
    "entity_corruption": lambda src, mt, rng: corrupt_entity(mt, rng),
}


def build_synthetic_rows(
    df: pd.DataFrame,
    seed: int,
    max_rows: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    train_df = df[df["split"] == "train"].copy()
    if max_rows is not None:
        train_df = train_df.iloc[:max_rows].copy()

    seen_hashes = set(df["pair_hash"].astype(str)) if "pair_hash" in df.columns else set()
    rows: list[dict] = []

    for _, row in train_df.iterrows():
        src = str(row["src"])
        mt = str(row["mt"])
        parent_hash = str(row.get("pair_hash", make_hash(src, mt)))

        for corruption_type, build_fn in CORRUPTION_FUNCS.items():
            corrupted_mt = build_fn(src, mt, rng)
            if not corrupted_mt or corrupted_mt.strip() == mt.strip():
                continue

            pair_hash = make_hash(src, corrupted_mt)
            if pair_hash in seen_hashes:
                continue
            seen_hashes.add(pair_hash)

            new_row = row.to_dict()
            score_norm = sample_score(corruption_type, rng)
            new_row.update(
                {
                    "mt": corrupted_mt,
                    "score_norm": score_norm,
                    "score": float(score_norm * 100.0),
                    "pair_hash": pair_hash,
                    "split": "train",
                    "is_synthetic": True,
                    "synthetic_type": corruption_type,
                    "synthetic_parent_hash": parent_hash,
                }
            )
            rows.append(new_row)

    synth_df = pd.DataFrame(rows)
    if synth_df.empty:
        return synth_df

    return synth_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    processed_dir = args.data_dir / "processed"
    in_path = processed_dir / "sentence_da.parquet"
    out_path = processed_dir / "sentence_da_augmented.parquet"

    if out_path.exists() and not args.force:
        log.info("%s уже существует - пропускаем. Используй --force для пересчёта.", out_path)
        return
    if not in_path.exists():
        raise FileNotFoundError(f"Не найден входной файл: {in_path}")

    log.info("=== build_synthetic_negatives.py ===")
    df = pd.read_parquet(in_path)
    if "is_synthetic" not in df.columns:
        df["is_synthetic"] = False
    if "synthetic_type" not in df.columns:
        df["synthetic_type"] = "original"
    if "synthetic_parent_hash" not in df.columns:
        df["synthetic_parent_hash"] = pd.NA

    synth_df = build_synthetic_rows(df, seed=args.seed, max_rows=args.max_rows)
    if synth_df.empty:
        log.warning("Синтетические примеры не были созданы")
        augmented = df.copy()
    else:
        augmented = pd.concat([df, synth_df], ignore_index=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_parquet(out_path, index=False)

    log.info("Оригинальных строк: %d", len(df))
    log.info("Синтетических строк: %d", len(synth_df))
    if not synth_df.empty:
        type_counts = synth_df["synthetic_type"].value_counts().to_dict()
        log.info("Распределение synthetic_type: %s", type_counts)
    log.info(
        "Итого сохранено: %s (%d строк, train=%d, val=%d, test=%d)",
        out_path,
        len(augmented),
        (augmented["split"] == "train").sum(),
        (augmented["split"] == "val").sum(),
        (augmented["split"] == "test").sum(),
    )


if __name__ == "__main__":
    main()
