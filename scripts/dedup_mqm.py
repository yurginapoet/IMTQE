"""
scripts/dedup_mqm.py

Шаг 3 из пайплайна MTQE.
Удаляет из HF MQM пары (src, mt) которые уже есть в HF DA train.
Это предотвращает утечку данных при внешнем тесте.

Алгоритм:
  1. Загружаем pair_hash из DA train
  2. Фильтруем MQM по этим хэшам
  3. Если удалено >5% - выводим предупреждение

Выход:
  data/processed/hf_mqm_dedup.parquet
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    processed_dir = args.data_dir / "processed"
    out_path      = processed_dir / "hf_mqm_dedup.parquet"

    if out_path.exists():
        log.info("Файл уже существует: %s - пропускаем", out_path)
        return

    log.info("=== dedup_mqm.py ===")

    da  = pd.read_parquet(processed_dir / "sentence_da.parquet")
    mqm = pd.read_parquet(processed_dir / "hf_mqm_raw.parquet")

    # хэши пар из DA train
    da_train_hashes = set(da.loc[da["split"] == "train", "pair_hash"])
    log.info("DA train: %d уникальных хэшей", len(da_train_hashes))
    log.info("MQM до дедупликации: %d строк", len(mqm))

    mask    = ~mqm["pair_hash"].isin(da_train_hashes)
    removed = (~mask).sum()
    pct     = 100 * removed / len(mqm)

    if pct > 5:
        log.warning(
            "Удалено %.1f%% строк MQM (%d) - возможна системная утечка данных",
            pct, removed,
        )
    else:
        log.info("Удалено %d строк (%.1f%%)", removed, pct)

    mqm_dedup = mqm[mask].reset_index(drop=True)
    log.info("MQM после дедупликации: %d строк", len(mqm_dedup))

    mqm_dedup.to_parquet(out_path, index=False)
    log.info("Сохранено: %s", out_path)
    log.info("Следующий шаг: scripts/extract_features.py (запускать на Colab T4)")


if __name__ == "__main__":
    main()