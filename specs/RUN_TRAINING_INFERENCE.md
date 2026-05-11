# Запуск обучения и инференса IMTQE

Корень проекта: каталог с `pyproject.toml`. Рекомендуется Poetry.

## Переменные окружения

| Переменная | Назначение |
|------------|------------|
| `IMTQE_DATA_DIR` | Каталог данных (по умолчанию `./data`) |
| `IMTQE_MODELS_DIR` | Каталог артефактов моделей (по умолчанию `./models`) |
| `IMTQE_LOG_DIR` | Каталог логов (по умолчанию `./logs`, файл `imtqe.log`) |
| `IMTQE_SEED` | Seed для NumPy/torch/transformers (по умолчанию `42`) |
| `IMTQE_COLAB` | `1` / `true` — режим Colab: для `extract_features` умолчанию больший batch |
| `HF_HUB_OFFLINE` | `1` — только локальный кэш Hugging Face |

## Установка

```bash
cd /path/to/IMTQE
poetry install
```

## Единая точка входа CLI (`imtqe`)

После `poetry install` команда доступна как `poetry run imtqe …`. Если скрипт не подхватился, используйте `poetry run python -m src.cli …`.

Список шагов:

```bash
poetry run imtqe --help
```

Полный пайплайн (как `scripts/run_full_pipeline.py`):

```bash
poetry run imtqe pipeline --data-dir data --models-dir models --seed 42
poetry run imtqe pipeline --skip-span
```

Отдельные шаги (аргументы те же, что у соответствующего скрипта в `scripts/`):

```bash
poetry run imtqe prepare-data --seed 42
poetry run imtqe build-wordlevel
poetry run imtqe dedup-mqm
poetry run imtqe build-synthetic-negatives
poetry run imtqe train-semantic-pca
poetry run imtqe extract-features --batch-size 64
poetry run imtqe train-sentence
poetry run imtqe train-span
poetry run imtqe train-neural-head
poetry run imtqe warmup-inference --download
```

На **Google Colab** задайте пути и флаг Colab, затем те же команды через `!poetry run` или `!python scripts/...`:

```bash
export IMTQE_COLAB=1
export IMTQE_DATA_DIR=/content/drive/MyDrive/imtqe/data
export IMTQE_MODELS_DIR=/content/drive/MyDrive/imtqe/models
poetry run imtqe pipeline
```

Тяжёлые шаги (`train-semantic-pca`, `extract-features`, `train-span`) удобно гонять на Colab с GPU; локально всё то же самое без `IMTQE_COLAB`.

## Альтернатива: прямые вызовы скриптов

```bash
poetry run python scripts/run_full_pipeline.py
poetry run python scripts/prepare_data.py
poetry run python scripts/extract_features.py
poetry run python scripts/train_sentence_model.py
poetry run python scripts/train_span_model.py
```

## Опционально: нейронная голова объяснений

После обучения XGBoost:

```bash
poetry run python scripts/train_neural_head.py
```

## Инференс (Python API)

```python
from pathlib import Path
from src.predict import Predictor

p = Predictor(models_dir=Path("models"))
r = p.predict_sentence("Hello.", "Привет.")
print(r.score, r.mqm_score)
```

## Веб-приложение

```bash
poetry run uvicorn src.app.server:app --host 0.0.0.0 --port 8000
```

## Тесты

```bash
poetry run pytest tests/ -q
```
