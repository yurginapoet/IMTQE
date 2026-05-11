"""
scripts/build_wordlevel.py

Шаг 2 из пайплайна MTQE.
Собирает WMT21 word-level датасет для обучения span-модели (XLM-RoBERTa).

Источники:
  .src   - исходные предложения EN
  .mt    - машинные переводы RU (содержат токен <EOS> в конце)
  .tags  - пословные метки OK/BAD
  .tsv   - severity аннотации (major/minor/critical)

seg_id в TSV везде -1 и бесполезен.
Связываем TSV с .src/.tags через совпадение текста колонки source с .src файлом.

Логика меток:
  .tags даёт OK/BAD для каждого слова
  TSV даёт severity на уровне предложения (берём максимальный)
  major или critical -> BAD-major, minor -> BAD-minor
  если предложение не найдено в TSV -> BAD-minor (консервативно)

Выход:
  data/processed/wordlevel_train.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.bootstrap import init_script_runtime
from src.settings import get_settings

log = logging.getLogger(__name__)

RANDOM_SEED = 42
SPLIT_TRAIN = 0.85
SPLIT_VAL   = 0.10
DOMAINS     = ["news", "ted"]


def read_lines(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_domain(wordlevel_dir: Path, domain: str) -> pd.DataFrame:
    src_path  = wordlevel_dir / f"mqm_dev2021_enru.{domain}.src"
    mt_path   = wordlevel_dir / f"mqm_dev2021_enru.{domain}.mt"
    tags_path = wordlevel_dir / f"mqm_dev2021_enru.{domain}.tags"
    tsv_path  = wordlevel_dir / f"mqm_dev2021_enru.{domain}.tsv"

    for p in [src_path, mt_path, tags_path, tsv_path]:
        if not p.exists():
            raise FileNotFoundError(f"Не найден файл: {p}")

    src_lines  = read_lines(src_path)
    mt_lines   = read_lines(mt_path)
    tags_lines = read_lines(tags_path)

    if not len(src_lines) == len(mt_lines) == len(tags_lines):
        raise ValueError(
            f"Домен {domain}: число строк не совпадает "
            f"src={len(src_lines)} mt={len(mt_lines)} tags={len(tags_lines)}"
        )

    # читаем TSV, убираем No-error строки
    tsv = pd.read_csv(tsv_path, sep="\t")
    tsv.columns = tsv.columns.str.strip().str.lower()
    tsv = tsv[tsv["severity"].str.lower() != "no-error"].reset_index(drop=True)

    # severity по тексту source: major/critical -> BAD-major, иначе BAD-minor
    severity_map = {}
    for src_text, group in tsv.groupby("source"):
        sevs = group["severity"].str.lower().tolist()
        severity_map[src_text.strip()] = (
            "BAD-major" if any(s in ("major", "critical") for s in sevs)
            else "BAD-minor"
        )

    log.info("Домен %s: %d предложений, %d с ошибками в TSV",
             domain, len(src_lines), len(severity_map))

    rows = []
    for src, mt, tags_str in zip(src_lines, mt_lines, tags_lines):
        # mt содержит <EOS> в конце - убираем его
        mt_clean = mt.replace("<EOS>", "").strip()

        seg_severity = severity_map.get(src.strip(), "BAD-minor")
        tags = tags_str.split()

        word_labels = [
            "OK" if t == "OK" else seg_severity
            for t in tags
        ]

        has_error = any(l != "OK" for l in word_labels)

        rows.append({
            "domain":       domain,
            "src":          src.strip(),
            "mt":           mt_clean,
            "word_labels":  word_labels,
            "n_words":      len(tags),
            "has_error":    has_error,
            "max_severity": seg_severity if has_error else "OK",
        })

    return pd.DataFrame(rows)


def stratified_split(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Split 85/10/5 стратифицированный по max_severity."""
    rng = np.random.default_rng(seed)
    splits = np.full(len(df), "test", dtype=object)

    for _, group in df.groupby("max_severity"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n_train = int(len(idx) * SPLIT_TRAIN)
        n_val   = int(len(idx) * SPLIT_VAL)
        splits[idx[:n_train]]              = "train"
        splits[idx[n_train:n_train+n_val]] = "val"

    df = df.copy()
    df["split"] = splits
    return df


def print_stats(df: pd.DataFrame) -> None:
    all_labels  = [l for labels in df["word_labels"] for l in labels]
    total_words = len(all_labels)
    for label in ["OK", "BAD-minor", "BAD-major"]:
        count = all_labels.count(label)
        log.info("  %-12s %6d слов  (%.1f%%)", label, count, 100 * count / total_words)
    for split in ["train", "val", "test"]:
        log.info("  %-6s %d предложений", split, (df["split"] == split).sum())


def main() -> None:
    init_script_runtime()
    s = get_settings()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=s.data_dir)
    parser.add_argument("--seed", type=int, default=s.random_seed)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_path = args.data_dir / "processed" / "wordlevel_train.parquet"
    if out_path.exists() and not args.force:
        log.info("Файл уже существует: %s - пропускаем", out_path)
        return

    df = pd.concat(
        [load_domain(args.data_dir / "raw" / "wordlevel", d) for d in DOMAINS],
        ignore_index=True,
    )
    log.info("Итого: %d предложений", len(df))

    df = stratified_split(df, seed=args.seed)

    log.info("Распределение меток:")
    print_stats(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("Сохранено: %s", out_path)


if __name__ == "__main__":
    main()
