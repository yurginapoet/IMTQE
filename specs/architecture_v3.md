# IMTQE — архитектура системы (v3)

Документ описывает **фактическую** реализацию репозитория IMTQE на момент составления: назначение модулей, контракты данных, обучение, инференс, веб-API и эксплуатацию. За основу взяты требования и идеи из `architecture_req_v2.md`; там, где реализация расходится с v2, это явно отмечено.

**Связанные материалы в репозитории:** `README.md`, `RUN_TRAINING_INFERENCE.md`, `architecture_req_v2.md`, `architecture_semantic_extension.md`, `ui_spec.md`, `report.md` (контекст исследования, не спецификация кода).

---

## 1. Назначение и область применения

**IMTQE (Interpretable Machine Translation Quality Estimation)** — reference-free оценка качества машинного перевода для пары **исходный текст (EN) — машинный перевод (RU)** без эталонного человеческого перевода.

Система одновременно:

- выдаёт **скалярную оценку качества** на уровне предложения (непрерывная шкала, по смыслу [0, 1]);
- даёт **псевдо-доверительный интервал** и меру разброса (через эвристику на базе Beta-статистики поверх точечной регрессии);
- **локализует** проблемные фрагменты в переводе (пословно, индексы spaCy-слов);
- **классифицирует** каждый BAD-спан по типологии MQM с приоритетом **детерминированных правил** поверх признаков;
- строит **MQM-style штрафной score** и **объяснение** в терминах агрегированных категорий MQM (доли «потери» для UI).

**Языковая пара:** EN→RU (жёстко заложена в признаки, spaCy-модели и обучающие датасеты).

**Не входит в текущий код `Predictor`:** абзацный режим с `pysbd`/`vecalign`, отдельный paragraph score и межпредложенческая терминология из раздела 4 блока 4 v2 — в репозитории нет соответствующего конвейера; UI и API работают **по одному сегменту (предложению)** за запрос (см. `src/predict.py`, `src/app/api.py`).

---

## 2. Соответствие `architecture_req_v2.md` и отличия

| Аспект | v2 (требование) | v3 (факт в коде) |
|--------|-----------------|------------------|
| Sentence-модель | NGBoost + Beta, нативные α, β | **XGBoost** регрессия в [0, 1]; SHAP через `shap`; **uncertainty и CI** через фиксированную Beta-аппроксимацию от точечного score (`src/models/sentence_model.py`, `_xgboost_uncertainty`) |
| Размерность признаков | 22 | До **97**: 22 базовых + 64 semantic PCA + 11 interaction (`src/features/schema.py`, `FeatureExtractor`) |
| Обучающий поток | notebooks + минимум скриптов | **Poetry CLI** `imtqe`, оркестратор `scripts/run_full_pipeline.py`, полный набор `scripts/*.py` |
| Доп. данные | — | `sentence_da_augmented.parquet` после **synthetic negatives**; PCA на MiniLM (`scripts/build_synthetic_negatives.py`, `scripts/train_semantic_pca.py`) |
| Объяснение UI | SHAP → категории | По умолчанию **доли потери** из SHAP (`src/interpretation/explanation_loss.py`); опционально **нейронная голова** внимания над признаками + `xgb_score` (`src/models/neural_head.py`, артефакты `neural_head.pt` / `neural_head_config.json`, обучение `scripts/train_neural_head.py`) |
| Веб-стек | Gradio (в старых описаниях) | **FastAPI** + Jinja2 + статика (`src/app/server.py`, `src/app/api.py`); зависимость `gradio` в `pyproject.toml` **не используется** кодом приложения |
| Управление путями | относительные пути | **`src/settings.py`**: `IMTQE_DATA_DIR`, `IMTQE_MODELS_DIR`, `IMTQE_LOG_DIR`, `IMTQE_SEED`, `IMTQE_COLAB`, `HF_HUB_OFFLINE` |

---

## 3. Функциональные возможности (актуальные)

**ФТ-A.** Вход: строки `src`, `mt` (одно предложение или короткий сегмент).

**ФТ-B.** Выход: `score`, `ci_low`, `ci_high`, `uncertainty`, `mqm_score`, список ошибок со `severity`, `error_type` (MQM), русские подписи, HTML-подсветка mt, словарь `explanation` (категории → доли для панели «потерь»), поле `debug` (сырые признаки, word_logprobs, SHAP по признакам) для feedback API.

**ФТ-C.** Span-severity: **XLM-RoBERTa** token classification (3 класса: OK / BAD-minor / BAD-major), смежные BAD объединяются в спаны (`src/models/span_model.py`).

**ФТ-D.** Тип MQM для спана: **только** `assign_mqm_types` в `src/interpretation/rules.py` (иерархия правил: валюта, кавычки, даты, числа, латиница, низкий logprob, agreement из признаков, дефолт Mistranslation и т.д.).

**ФТ-E.** REST: `GET /`, `GET /api/status`, `POST /api/evaluate`, `POST /api/evaluate_batch` (≤50 пар), `POST /api/feedback`, `POST /api/reload_models` (`src/app/api.py`).

**ФТ-F.** Горячая перезагрузка лёгких моделей (XGBoost, span, neural head) без перезагрузки LaBSE/ruGPT/MiniLM: `Predictor.reload_light_models()` (`src/predict.py`).

---

## 4. Нефункциональные требования и эксплуатация

- **Интерпретируемость:** явные признаки + SHAP по деревьям XGBoost; тип ошибки не предсказывается отдельной нейросетью.
- **CPU-инференс:** целевой режим `device=cpu` для тяжёлых трансформеров в `FeatureExtractor` и для span-модели в проде; GPU опционален (Colab, локальный CUDA).
- **Память:** одновременно в RAM удерживаются LaBSE, ruGPT-3 Small, MiniLM (если загружен semantic блок), spaCy (en/ru), XLM-R span, XGBoost booster — порядок величины многие ГБ; планирование под конкретный хост обязательно.
- **Воспроизводимость:** `src/determinism.py` + `IMTQE_SEED`; скрипты вызывают `init_script_runtime()` из `src/bootstrap.py`.
- **Логи:** `src/logging_config.py` — stdout + `logs/imtqe.log` (каталог из `IMTQE_LOG_DIR`).
- **HF кэш:** на старте сервера выставляются offline-переменные по умолчанию (`src/app/server.py`); модели span грузятся с `local_files_only=True`.

---

## 5. Логическая архитектура инференса

Последовательность для одной пары `(src, mt)` (`Predictor.predict_sentence`):

1. **`FeatureExtractor.extract`** (`src/features/extractor.py`): spaCy-документы EN (облегчённый пайплайн) и RU; признаки structural, formatting, linguistic; при загруженных тяжёлых моделях — LaBSE, ruGPT (sentence + **word_logprobs** по spaCy-словам), при наличии PCA — блок `neural.extract`, затем **`interaction_features`** (`src/features/interactions.py`); итоговый `vector` длины `len(active_feature_names)`; вспомогательно `mt_words`, `raw`, `formal_ratio`.
2. **`SentenceModel.predict`** (`src/models/sentence_model.py`): предсказание XGBoost, SHAP (или нули при отсутствии explainer), агрегация SHAP в верхнеуровневые MQM-категории (`FEATURE_TO_MQM`), Beta-подобные `uncertainty`, `ci_low`, `ci_high`.
3. **`SpanModel.predict`**: токенизация `[CLS] src [SEP] mt [SEP]`, маппинг на spaCy-слова, спаны BAD, `word_logprobs` прокидываются в `SpanResult` для правил.
4. **`OverallSentenceEvaluator.evaluate`** (`src/interpretation/overall.py`): `assign_mqm_types` → `aggregate_sentence_mqm` с весами из `models/weights_mqm.npy` (или единицы).
5. **`_display_explanation_en`**: нейронная голова или `shap_categories_to_loss_shares`; затем ключи explanation переводятся для UI через `MQM_CATEGORY_RU`.
6. **`_build_ui_result`**: HTML-подсветка, список `SentenceErrorItem`, клиппинг score.

Батч: `extract_batch` + `predict_batch` по векторам sentence; span пока **последовательно** по парам (в коде отмечено как возможное улучшение).

---

## 6. Признаки: схема и источники

Имена и порядок задаются **`src/features/schema.py`**:

| Режим | Состав | Длина вектора |
|-------|--------|----------------|
| Light | `FEATURE_NAMES_LIGHT` | 16 |
| Classic heavy | light + LaBSE (`cosine_similarity`, `embedding_distance`) + ruGPT (`FEATURE_NAMES_HEAVY`) + `INTERACTION_FEATURE_NAMES` | 16+2+4+11 = **33** |
| Full (обучение/прод по умолчанию для semantic pipeline) | classic base **22** (без interaction в имени `FEATURE_NAMES` — см. ниже) + 64 `semantic_00…` + 11 interaction | **97** |

Уточнение по именованию в коде:

- `FEATURE_NAMES_CLASSIC` = light + semantic_explicit + heavy (**22**).
- `FEATURE_NAMES` = **22 + 64** без interaction.
- `SENTENCE_FEATURE_NAMES` = `FEATURE_NAMES` + **11** interaction = **97** — это ожидаемый вектор для полного пайплайна после `extract_features` с тяжёлыми моделями и PCA.

**Источники по файлам:**

- `src/features/structural.py` — длины, ratio, diff.
- `src/features/formatting.py` — числа, пунктуация, кавычки, даты.
- `src/features/linguistic.py` — OOV, TTR, длина токена, NER overlap, agreement heuristics, syntax depth, formal_ratio (частотный стиль).
- `src/features/semantic.py` — LaBSE encode пары.
- `src/features/fluency.py` — ruGPT logprobs, perplexity, агрегация на слова mt.
- `src/features/neural.py` — MiniLM, |emb_src − emb_mt|, PCA → `semantic_*`.
- `src/features/interactions.py` — нелинейные комбинации (логарифм perplexity, произведения cosine × entity, и т.д.).

`Config` (`src/config.py`): имена HF-моделей (`LABSE_MODEL_NAME`, `RUGPT_MODEL_NAME`, `SEMANTIC_ENCODER_NAME`), разрешение путей через `huggingface_hub.snapshot_download` с учётом `HF_HUB_OFFLINE`.

---

## 7. Модели и артефакты

| Артефакт | Назначение |
|----------|------------|
| `models/xgboost_sentence.model` | XGBoost regressor (формат native `.model` или pickle — поддерживается в `SentenceModel`) |
| `models/shap_explainer.pkl` | `TreeExplainer` + опционально словарь с `feature_names` |
| `models/semantic_pca.pkl` | Обученный sklearn/joblib PCA на разностях эмбеддингов |
| `models/xlm_roberta_span/` | HuggingFace-сохранение `AutoModelForTokenClassification` + tokenizer |
| `models/weights_mqm.npy` | Вектор весов длины `len(MQM_ERROR_TYPES)` |
| `models/neural_head.pt`, `models/neural_head_config.json` | Опционально: объяснения через attention-голову (`FeatureAttentionHead` в `src/models/neural_head.py`) |

**Согласование размерности:** при создании `Predictor` вызывается `_validate_extractor_features(expected_feature_count)` — число активных имён признаков `FeatureExtractor` должно быть ≥ числа признаков бустинга; иначе `RuntimeError` с подсказкой про PCA и тяжёлые модели.

---

## 8. Данные и контракт файлов

Явные имена в **`src/data_contract.py`**:

- `data/processed/sentence_da.parquet` — HF DA после `prepare_data`.
- `sentence_da_augmented.parquet` — DA + синтетика (train-only синтетика помечается колонками `is_synthetic`, `synthetic_type`, …).
- `wordlevel_train.parquet` — WMT21 word-level сборка (`scripts/build_wordlevel.py`).
- `hf_mqm_raw.parquet` / `hf_mqm_dedup.parquet` — MQM с HuggingFace и после дедупликации к DA train по `pair_hash`.
- `sentence_da_features.parquet`, `wordlevel_features.parquet`, `hf_mqm_features.parquet` — результаты `extract_features.py`.

Ключевые колонки: `src`, `mt`, `split`, `score_norm` (и производные для word-level: списки меток и т.д. — см. соответствующие скрипты).

**Дедупликация MQM:** `scripts/dedup_mqm.py` — исключение строк MQM, чей `pair_hash` встречается в DA train; предупреждение при удалении >5% строк.

---

## 9. Пайплайн обучения (порядок)

Рекомендуемая последовательность (см. `RUN_TRAINING_INFERENCE.md`, `scripts/run_full_pipeline.py`, `architecture_semantic_extension.md`):

1. `prepare_data.py` — загрузка HF DA / HF MQM, нормализация score, хэши пар.
2. `build_wordlevel.py` — объединение raw WMT21 news/ted.
3. `dedup_mqm.py`.
4. `build_synthetic_negatives.py` — аугментация train для устойчивости.
5. `train_semantic_pca.py` — PCA на train-парах augmented.
6. `extract_features.py` — требует полной цепочки тяжёлых моделей + `semantic_pca.pkl` для 97 признаков (режим по умолчанию с `require_neural=True`).
7. `train_sentence_model.py` — XGBoost + построение SHAP explainer + внешняя метрика на HF MQM features при наличии.
8. `train_span_model.py` — fine-tuning XLM-R.
9. Опционально: `train_neural_head.py` после sentence.

Оркестрация: `poetry run imtqe pipeline` или прямой вызов `run_full_pipeline.py` с флагами `--skip-span`, `--force`, лимитами строк для ускорения экспериментов.

**Прогрев кэша HF для сервера:** `scripts/warmup_inference_models.py` (через CLI `imtqe warmup-inference`).

---

## 10. Интерпретация и MQM-агрегация

**Правила:** `src/interpretation/rules.py` — константа `MQM_ERROR_TYPES` задаёт фиксированный порядок для `weights_mqm.npy`; функции классификации спана и русские описания для UI.

**Агрегация:** `src/interpretation/aggregation.py` — для предложения: сумма сырого штрафа `wt * ps * confidence` по спанам, деление на число слов mt, пересчёт в `mqm_score` ∈ [0,1] (1 = лучше). Штрафы severity: BAD-minor=1, BAD-major=5.

**Overall:** `OverallSentenceResult` связывает sentence score, CI, explanation, `MQMAggregation`, типизированные спаны.

---

## 11. Веб-приложение и API

- **Точка входа:** `uvicorn src.app.server:app`.
- **Жизненный цикл:** `lifespan` загружает `ModelsState` → `Predictor` один раз; при ошибке сервер поднимается, `GET /api/status` отдаёт `ready=false` и текст ошибки.
- **Шаблоны и статика:** каталоги рядом с `src/app/` (см. `ui_spec.md` для макета CAT-стиля, debounce, последовательных запросов по сегментам).
- **Feedback:** `src/app/feedback.py` — append-only JSONL (путь по умолчанию в модуле), без повторного тяжёлого инференса за счёт `debug` с фронта.

---

## 12. Структура каталогов (фактическая)

```
IMTQE/
├── pyproject.toml          # Poetry, scripts entrypoint imtqe
├── data/
│   ├── raw/wordlevel/      # WMT21 *.src / *.mt / *.tags / *.tsv
│   └── processed/          # parquet по контракту
├── models/                 # артефакты обучения (не все обязаны в git)
├── scripts/                # обучение и пайплайн
├── src/
│   ├── cli.py              # диспетчер subprocess на scripts
│   ├── bootstrap.py        # логирование + seed для скриптов
│   ├── settings.py         # env
│   ├── config.py           # HF имена, пути через settings
│   ├── data_contract.py
│   ├── determinism.py
│   ├── logging_config.py
│   ├── predict.py          # Predictor, UI dataclasses
│   ├── features/           # извлечение признаков
│   ├── models/             # sentence_model, span_model; neural_head — по сценарию обучения
│   ├── interpretation/   # rules, aggregation, overall, explanation_loss
│   └── app/                # FastAPI, models_state, api, feedback, templates, static
└── tests/                  # pytest
```

---

## 13. Зависимости (уровень пакетов)

Из `pyproject.toml` (сжато): Python 3.10–3.12, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`, `pyarrow`, `scipy`, `joblib`, `sentence-transformers`, `transformers`, `datasets`, `pysbd`, `fastapi`, `uvicorn`, `jinja2`, `torch` (CPU index), `spacy`, `pymorphy2/3`, `huggingface-hub`, `pytest` / dev-утилиты. **Gradio** объявлен, но **не импортируется** приложением.

Spacy-модели **`ru_core_news_sm`** и **`en_core_web_sm`** должны быть установлены в окружении (`python -m spacy download …`).

---

## 14. Тестирование

Каталог `tests/`: юнит-тесты на компоненты (в т.ч. моки тяжёлых зависимостей при импорте `predict`). Запуск: `poetry run pytest tests/ -q` (см. `RUN_TRAINING_INFERENCE.md`).

---

## 15. Известные ограничения и расхождения с «идеальной» спецификацией

- Абзацный уровень из v2 **не реализован** в `Predictor`.
- **NGBoost** из v2 **заменён** на XGBoost; интерпретация uncertainty — **эвристика**, не параметрическая Beta-модель на выходе бустинга.
- WMT21 word-level: как в v2, возможно грубое выравнивание severity по предложению при обогащении BAD-токенов (см. описание в v2 §4.5 — логика сборки в `build_wordlevel.py`).
- Зависимость объяснений от **нейронной головы** требует синхронизации списка признаков с `SENTENCE_FEATURE_NAMES` и совместимости `neural_head_config.json` с текущим XGBoost.
- Качество и latency сильно зависят от железа и включения GPU на этапе извлечения признаков.

---

## 16. Краткая карта модулей `src/`

| Модуль | Роль |
|--------|------|
| `predict.py` | Оркестрация инференса, `SentenceUIResult`, батч, reload |
| `features/extractor.py` | Сборка вектора, загрузка тяжёлых моделей |
| `features/schema.py` | Имена и порядок признаков |
| `src/models/sentence_model.py` | XGBoost + SHAP + маппинг в MQM-категории |
| `src/models/span_model.py` | Инференс XLM-R |
| `src/models/neural_head.py` | Лёгкая голова для альтернативного разложения объяснения по MQM-категориям |
| `interpretation/rules.py` | MQM типы для спанов |
| `interpretation/aggregation.py` | MQM score |
| `interpretation/overall.py` | Слой сборки результата |
| `interpretation/explanation_loss.py` | SHAP → доли потерь для UI |
| `app/server.py` | FastAPI app + lifespan |
| `app/api.py` | HTTP контракт |
| `app/models_state.py` | Состояние загрузки Predictor |

---

## 17. HTTP API (контракты)

**`GET /api/status`** → JSON:

- `ready: bool` — Predictor успешно создан;
- `status: str` — значение enum `LoadStatus` (`not_started` | `loading` | `ready` | `error`);
- `models_loaded_at: float | null` — timestamp после успешной загрузки;
- `error: str | null` — текст исключения при ошибке загрузки;
- `feedback_count: int` — число записей feedback (см. `src/app/feedback.py`).

**`POST /api/evaluate`** — тело JSON `{ "src": string, "mt": string }`. Оба поля не могут быть пустыми после trim. Ответ: словарь `SentenceUIResult.to_dict()` плюс `elapsed_sec`; поля включают `score`, `ci_low`, `ci_high`, `uncertainty`, `mqm_score`, `highlighted_mt_html`, `errors` (массив объектов с `severity`, `error_type`, `error_label`, `confidence`, `span_text`, `start_idx`, `end_idx`), `explanation` (русские ключи категорий), `debug` (признаки, `word_logprobs`, `shap_values`).

**`POST /api/evaluate_batch`** — `{ "pairs": [ {"src": "...", "mt": "..."}, ... ] }`, максимум **50** пар. Возвращает массив тех же словарей без гарантии `elapsed_sec` на каждый элемент (см. реализацию).

**`POST /api/feedback`** — см. `FeedbackRequest` в `api.py`: координаты ошибки в символах (`start_char`, `end_char`), `error_type`, `severity` ∈ `{BAD-minor, BAD-major}`, опционально `features` и `word_logprobs` с фронта. Ответ `{ "saved": true, "feedback_id": ... }`.

**`POST /api/reload_models`** — без тела; требует `ready=true`. Вызывает `Predictor.reload_light_models()`. Ответ `{ "reloaded": true }`.

**`GET /`** — HTML из Jinja (`templates/index.html`).

---

## 18. Перечень скриптов `scripts/`

| Скрипт | Назначение |
|--------|------------|
| `prepare_data.py` | HF DA + HF MQM → parquet, нормализация, `pair_hash`, сплиты |
| `build_wordlevel.py` | Сборка `wordlevel_train.parquet` из WMT21 raw |
| `dedup_mqm.py` | `hf_mqm_dedup.parquet` |
| `build_synthetic_negatives.py` | `sentence_da_augmented.parquet` |
| `train_semantic_pca.py` | `models/semantic_pca.pkl` по train augmented |
| `extract_features.py` | Признаки для DA / wordlevel / MQM dedup → `*_features.parquet` |
| `train_sentence_model.py` | XGBoost + SHAP + внешний тест на MQM features |
| `train_span_model.py` | Fine-tune XLM-R → `models/xlm_roberta_span/` |
| `train_neural_head.py` | Опциональная голова объяснений → `neural_head.pt` |
| `run_full_pipeline.py` | Последовательный запуск шагов (см. `--skip-span`, `--force`) |
| `warmup_inference_models.py` | Прогрев/скачивание HF в кэш для офлайн-сервера |

---

## 19. Обучение sentence-модели (детали реализации)

- Вход: `sentence_da_features.parquet`, колонка цели `score_norm`.
- Признаки: пересечение с `SENTENCE_FEATURE_NAMES` или база `FEATURE_NAMES` + дозаполнение interaction в RAM (`add_interaction_columns_to_dataframe`).
- **Синтетические примеры:** вес при обучении задаётся параметром `--synthetic-weight` (по умолчанию 0.1); семантические PCA-колонки могут иметь отдельный вес (`--semantic-feature-weight`).
- **SHAP:** `TreeExplainer`, в pickle сохраняется dict `explainer` + `feature_names` (важно при несовпадении автоматического вывода имён с фактическими колонками).
- **Валидация:** Pearson/Spearman на сплитах; внешний набор — признаки из `hf_mqm_features.parquet` при наличии (нормализация z-score таргета для метрик — см. код `external_test`).

Точные гиперпараметры XGBoost (depth, eta, …) заданы в теле `train_xgboost` в `train_sentence_model.py` и могут меняться версиями — источник истины: код скрипта.

---

## 20. Обучение span-модели (зафиксированные константы)

- База: **`xlm-roberta-base`** (`MODEL_NAME` в `train_span_model.py`).
- Классы: OK / BAD-minor / BAD-major; веса loss **1 / 2 / 5**.
- `MAX_LENGTH = 512`; распределение длины контекста между src и mt согласовано с `SpanModel` (см. комментарии в `src/models/span_model.py`).
- Early stopping по **val F1(BAD-major)**.
- `set_seed(RANDOM_SEED)` из transformers.

---

## 21. Типология MQM в коде (`MQM_ERROR_TYPES`)

Фиксированный кортеж в `rules.py` (порядок = индексы `weights_mqm.npy`):

1. Accuracy: Mistranslation, Omission, Addition, Untranslated, Hallucination  
2. Fluency: Morphology, Agreement, Spelling, LexicalChoice  
3. Terminology: WrongTerm, Inconsistency  
4. Locale: NumberFormat, DateFormat, Quotes, Currency  
5. Style: Register  

Русские пользовательские строки: `MQM_ERROR_TYPE_RU`, маппинг категорий sentence-SHAP: `MQM_CATEGORY_RU` в `sentence_model.py`.

---

## 22. CLI `imtqe`

Реализация: `src/cli.py` — для каждой команды запускается `python scripts/<script>.py` с переданными аргументами. Команды: `prepare-data`, `build-wordlevel`, `dedup-mqm`, `build-synthetic-negatives`, `train-semantic-pca`, `extract-features`, `train-sentence`, `train-span`, `train-neural-head`, `warmup-inference`, а также **`pipeline`** → `run_full_pipeline.py`.

---

## 23. Ошибки загрузки и деградация

- Если SHAP explainer не загрузился (например, несовместимость pickle), `SentenceModel` продолжает работу с **нулевыми** SHAP и пустой/нейтральной агрегацией вкладов (см. логирование предупреждения).
- Если `neural_head` отсутствует, объяснение строится только из SHAP через `shap_categories_to_loss_shares`.
- Если веса MQM отсутствуют, используются **единицы** (`load_mqm_weights`).

---

Конец документа v3. При изменении пайплайна или контрактов данных обновляйте разделы **6–9**, **17–18** и таблицу в разделе **2** в первую очередь.
