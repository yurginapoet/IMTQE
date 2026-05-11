"""
scripts/extract_features.py

Шаг feature extraction для sentence- и word-level датасетов.

Формат sentence-колонок соответствует schema.SENTENCE_FEATURE_NAMES
(базовые признаки + interaction), см. FeatureExtractor.active_feature_names.

Режим ``--append-light``: дописывает в уже существующий *_features.parquet
только колонки из schema.FEATURE_NAMES_LIGHT (spaCy), без GPU-тяжёлых моделей;
остальные колонки не удаляются и не пересчитываются.
"""

from __future__ import annotations

import argparse
import joblib
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.features.extractor import FeatureExtractor
from src.features.schema import FEATURE_NAMES_LIGHT, SENTENCE_FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_LIGHT_INDEX = {name: idx for idx, name in enumerate(FEATURE_NAMES_LIGHT)}


def _input_matches_existing_rows(
    input_df: pd.DataFrame,
    existing_df: pd.DataFrame,
    src_col: str,
    mt_col: str,
) -> bool:
    if len(input_df) != len(existing_df):
        return False
    if src_col not in existing_df.columns or mt_col not in existing_df.columns:
        return False
    a_src = input_df[src_col].astype(str).str.strip().reset_index(drop=True)
    a_mt = input_df[mt_col].astype(str).str.strip().reset_index(drop=True)
    b_src = existing_df[src_col].astype(str).str.strip().reset_index(drop=True)
    b_mt = existing_df[mt_col].astype(str).str.strip().reset_index(drop=True)
    return bool(a_src.equals(b_src) and a_mt.equals(b_mt))


def extract_light_columns_for_df(
    name: str,
    df: pd.DataFrame,
    src_col: str,
    mt_col: str,
    extractor: FeatureExtractor,
    batch_size: int,
    progress_desc: str,
    columns_to_write: list[str],
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 100,
) -> np.ndarray:
    """
    Считает только лёгкие признаки (без LaBSE/ruGPT/MiniLM). Возвращает (n, len(columns_to_write)).
    """
    n = len(df)
    out_dim = len(columns_to_write)
    write_pos = {name: i for i, name in enumerate(columns_to_write)}
    resume_state = None
    if checkpoint_path is not None and checkpoint_path.exists():
        log.info("Найден checkpoint (append-light): %s", checkpoint_path)
        resume_state = joblib.load(checkpoint_path)

    ckpt_key = (name, "append_light", tuple(columns_to_write), n, batch_size)
    stored_key = resume_state.get("ckpt_key") if resume_state is not None else None
    if stored_key is not None and tuple(stored_key) == ckpt_key:
        all_vectors = resume_state["light_partial"]
        resume_start = int(resume_state["next_start"])
        log.info("Возобновляем append-light с позиции %d / %d", resume_start, n)
    else:
        if resume_state is not None:
            log.warning("Checkpoint append-light несовместим, начинаем заново")
        all_vectors = np.zeros((n, out_dim), dtype=np.float32)
        resume_start = 0

    pairs = list(zip(df[src_col].astype(str), df[mt_col].astype(str), strict=False))

    for batch_idx, start in enumerate(
        tqdm(
            range(resume_start, n, batch_size),
            desc=progress_desc,
            unit="batch",
            initial=resume_start // batch_size,
            total=(n + batch_size - 1) // batch_size,
        ),
        start=1,
    ):
        batch = pairs[start : start + batch_size]
        results = extractor.extract_batch(batch)
        for idx, result in enumerate(results):
            row = start + idx
            vec16 = result["vector"]
            for name in columns_to_write:
                j = write_pos[name]
                k = _LIGHT_INDEX[name]
                all_vectors[row, j] = float(vec16[k])

        if checkpoint_path is not None and checkpoint_every > 0 and (batch_idx % checkpoint_every == 0):
            save_checkpoint_atomic(
                checkpoint_path,
                {
                    "ckpt_key": ckpt_key,
                    "next_start": start + len(batch),
                    "light_partial": all_vectors,
                },
            )
            log.info("Checkpoint append-light: %s next_start=%d", checkpoint_path, start + len(batch))

    return all_vectors


def resolve_sentence_input(processed_dir: Path) -> Path:
    augmented = processed_dir / "sentence_da_augmented.parquet"
    if augmented.exists():
        return augmented
    return processed_dir / "sentence_da.parquet"


def extract_for_df(
    df: pd.DataFrame,
    src_col: str,
    mt_col: str,
    extractor: FeatureExtractor,
    batch_size: int,
    progress_desc: str,
    checkpoint_path: Path | None = None,
    checkpoint_every: int = 100,
) -> pd.DataFrame:
    """
    Извлекает все активные признаки для всех строк df батчами.
    Возвращает df с добавленными числовыми колонками и word_logprobs.
    """
    feature_names = extractor.active_feature_names
    n = len(df)
    resume_state = None
    if checkpoint_path is not None and checkpoint_path.exists():
        log.info("Найден checkpoint: %s", checkpoint_path)
        resume_state = joblib.load(checkpoint_path)

    if resume_state is not None:
        if (
            int(resume_state["total_rows"]) == n
            and list(resume_state["feature_names"]) == list(feature_names)
            and int(resume_state["batch_size"]) == batch_size
        ):
            all_vectors = resume_state["all_vectors"]
            all_word_logprobs = resume_state["all_word_logprobs"]
            resume_start = int(resume_state["next_start"])
            log.info("Возобновляем с позиции %d / %d", resume_start, n)
        else:
            log.warning("Checkpoint несовместим с текущим запуском, начинаем заново")
            all_vectors = np.zeros((n, len(feature_names)), dtype=np.float32)
            all_word_logprobs = [None] * n
            resume_start = 0
    else:
        all_vectors = np.zeros((n, len(feature_names)), dtype=np.float32)
        all_word_logprobs = [None] * n
        resume_start = 0

    pairs = list(zip(df[src_col].astype(str), df[mt_col].astype(str), strict=False))

    for batch_idx, start in enumerate(
        tqdm(
            range(resume_start, n, batch_size),
            desc=progress_desc,
            unit="batch",
            initial=resume_start // batch_size,
            total=(n + batch_size - 1) // batch_size,
        ),
        start=1,
    ):
        batch = pairs[start : start + batch_size]
        results = extractor.extract_batch(batch)
        for idx, result in enumerate(results):
            all_vectors[start + idx] = result["vector"]
            all_word_logprobs[start + idx] = result["word_logprobs"]

        if checkpoint_path is not None and checkpoint_every > 0 and (batch_idx % checkpoint_every == 0):
            save_checkpoint_atomic(
                checkpoint_path,
                {
                    "feature_names": list(feature_names),
                    "batch_size": batch_size,
                    "total_rows": n,
                    "next_start": start + len(batch),
                    "all_vectors": all_vectors,
                    "all_word_logprobs": all_word_logprobs,
                },
            )
            log.info("Checkpoint сохранён: %s (next_start=%d)", checkpoint_path, start + len(batch))

    df = df.copy()
    for feat_idx, name in enumerate(feature_names):
        df[name] = all_vectors[:, feat_idx]
    df["word_logprobs"] = all_word_logprobs

    log.info(
        "Признаки добавлены: %d числовых + word_logprobs. Итого колонок: %d",
        len(feature_names),
        len(df.columns),
    )
    return df


def save_checkpoint_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    joblib.dump(payload, tmp_path)
    tmp_path.replace(path)


def process_dataset(
    name: str,
    in_path: Path,
    out_path: Path,
    src_col: str,
    mt_col: str,
    extractor: FeatureExtractor,
    batch_size: int,
    force: bool,
    checkpoint_dir: Path,
    checkpoint_every: int,
    append_light: bool,
) -> None:
    if not in_path.exists():
        log.warning("Входной файл не найден: %s - пропускаем %s", in_path, name)
        return

    if append_light:
        if not out_path.exists():
            log.error(
                "%s: --append-light требует существующий %s. Сначала полный прогон без этого флага.",
                name,
                out_path.name,
            )
            return

        log.info("--- %s (append-light: только %s) ---", name, ", ".join(FEATURE_NAMES_LIGHT[:3]) + ", …")
        input_df = pd.read_parquet(in_path)
        existing_df = pd.read_parquet(out_path)
        log.info("Вход: %d строк; существующий parquet: %d строк, %d колонок", len(input_df), len(existing_df), len(existing_df.columns))

        if len(input_df) != len(existing_df):
            log.error(
                "Число строк не совпадает (вход %d vs выход %d). Append-light отменён.",
                len(input_df),
                len(existing_df),
            )
            return

        if not _input_matches_existing_rows(input_df, existing_df, src_col, mt_col):
            log.error(
                "Колонки %s / %s не совпадают с существующим parquet по строкам. "
                "Append-light отменён (нужен тот же порядок и те же пары, что при полном извлечении).",
                src_col,
                mt_col,
            )
            return

        missing = [c for c in FEATURE_NAMES_LIGHT if c not in existing_df.columns]
        if force:
            columns_to_write = list(FEATURE_NAMES_LIGHT)
            log.info("--force: пересчитываем все лёгкие колонки (%d шт.), тяжёлые не трогаем.", len(columns_to_write))
        elif missing:
            columns_to_write = missing
            log.info("Дописываем недостающие лёгкие колонки (%d): %s", len(missing), missing)
        else:
            log.info("Все лёгкие колонки уже есть в %s — пропуск.", out_path.name)
            return

        ck_append = checkpoint_dir / f"{name}_append_light.resume.joblib"
        if force and ck_append.exists():
            ck_append.unlink()

        light_block = extract_light_columns_for_df(
            name,
            input_df,
            src_col=src_col,
            mt_col=mt_col,
            extractor=extractor,
            batch_size=batch_size,
            progress_desc=f"{name} append-light",
            columns_to_write=columns_to_write,
            checkpoint_path=ck_append,
            checkpoint_every=checkpoint_every,
        )

        merged = existing_df.copy()
        for j, col in enumerate(columns_to_write):
            merged[col] = light_block[:, j]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out_path, index=False)
        log.info("Обновлено (append-light): %s (%d строк, %d колонок)", out_path, len(merged), len(merged.columns))
        if ck_append.exists():
            ck_append.unlink()
            log.info("Временный checkpoint удалён: %s", ck_append)
        return

    if out_path.exists() and not force:
        log.info("%s уже существует - пропускаем. Используй --force для пересчёта.", out_path.name)
        return

    log.info("--- %s ---", name)
    log.info("Загрузка: %s", in_path)
    df = pd.read_parquet(in_path)
    log.info("Строк: %d", len(df))
    checkpoint_path = checkpoint_dir / f"{name}_features.resume.joblib"
    if force and checkpoint_path.exists():
        checkpoint_path.unlink()

    df = extract_for_df(
        df,
        src_col=src_col,
        mt_col=mt_col,
        extractor=extractor,
        batch_size=batch_size,
        progress_desc=f"{name} feature batches",
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info("Сохранено: %s (%d строк, %d колонок)", out_path, len(df), len(df.columns))
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        log.info("Временный checkpoint удалён: %s", checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("data/checkpoints/features"))
    parser.add_argument("--only", choices=["da", "wl", "mqm"], help="Обработать только один датасет")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Пересчитать признаки даже если выходной файл уже существует",
    )
    parser.add_argument(
        "--append-light",
        action="store_true",
        help=(
            "Дописать только лёгкие признаки (spaCy structural/formatting/linguistic) в уже "
            "существующий parquet, без LaBSE/ruGPT/MiniLM. Не удаляет и не пересчитывает "
            "тяжёлые колонки. Требует совпадения строк и пар src/mt с входным файлом. "
            "С --force пересчитывает все лёгкие колонки поверх существующих."
        ),
    )
    args = parser.parse_args()

    processed_dir = args.data_dir / "processed"
    sentence_input = resolve_sentence_input(processed_dir)

    log.info("=== extract_features.py ===")
    log.info("Sentence-level источник: %s", sentence_input.name)

    extractor = FeatureExtractor()
    if args.append_light:
        log.info("Режим --append-light: тяжёлые модели не загружаются.")
    else:
        extractor.load_heavy_models(require_neural=True)
        if list(extractor.active_feature_names) != list(SENTENCE_FEATURE_NAMES):
            raise RuntimeError(
                "FeatureExtractor не активировал полный sentence-вектор. "
                "Проверь наличие models/semantic_pca.pkl и доступность MiniLM."
            )
        log.info("Все модели загружены. Считаем %d признаков.", len(SENTENCE_FEATURE_NAMES))

    datasets = {
        "da": (
            sentence_input,
            processed_dir / "sentence_da_features.parquet",
            "src",
            "mt",
        ),
        "wl": (
            processed_dir / "wordlevel_train.parquet",
            processed_dir / "wordlevel_features.parquet",
            "src",
            "mt",
        ),
        "mqm": (
            processed_dir / "hf_mqm_dedup.parquet",
            processed_dir / "hf_mqm_features.parquet",
            "src",
            "mt",
        ),
    }

    for key, (in_path, out_path, src_col, mt_col) in datasets.items():
        if args.only and args.only != key:
            continue
        process_dataset(
            name=key,
            in_path=in_path,
            out_path=out_path,
            src_col=src_col,
            mt_col=mt_col,
            extractor=extractor,
            batch_size=args.batch_size,
            force=args.force,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_every=args.checkpoint_every,
            append_light=args.append_light,
        )

    log.info("=== Готово. Следующий шаг: scripts/train_sentence_model.py ===")


if __name__ == "__main__":
    main()
