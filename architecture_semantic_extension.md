# Semantic Embedding Augmentation

Это краткое дополнение к `architecture_req_v2.md` для нового sentence-level блока.

## Что меняется

- `FeatureExtractor` теперь может собирать `86` признаков:
  - `22` текущих handcrafted/classic
  - `64` semantic PCA-компоненты
- новый модуль: `src/features/neural.py`
- новый артефакт: `models/semantic_pca.pkl`
- новый датасет перед извлечением признаков: `data/processed/sentence_da_augmented.parquet`

## Новый training flow

```text
prepare_data.py
    ↓
build_wordlevel.py
    ↓
dedup_mqm.py
    ↓
build_synthetic_negatives.py
    ↓
train_semantic_pca.py
    ↓
extract_features.py
    ↓
train_sentence_model.py
    ↓
train_span_model.py
```

## Семантический блок

Для каждой пары `(src, mt)`:

1. `MiniLM` строит два sentence embedding вектора.
2. Считается `abs(src_emb - mt_emb)`.
3. Разность проецируется через `PCA(n_components=64)`.
4. Результат конкатенируется с текущими 22 признаками.

## Новые скрипты

- `scripts/build_synthetic_negatives.py`
  - создаёт synthetic negatives только для `train` split
  - добавляет `shuffle`, `untranslated`, `deletion`, `entity_corruption`

- `scripts/train_semantic_pca.py`
  - обучает PCA по train-парам из `sentence_da_augmented.parquet`
  - сохраняет `models/semantic_pca.pkl`

- `scripts/run_full_pipeline.py`
  - запускает весь pipeline по порядку
  - подходит для Colab

## Colab-запуск

```bash
python scripts/run_full_pipeline.py --force --sentence-model xgboost
```

Если span-модель не нужна в конкретном прогоне:

```bash
python scripts/run_full_pipeline.py --force --skip-span
```
