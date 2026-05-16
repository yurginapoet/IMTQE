# Архитектура системы оценки качества машинного перевода (MTQE)

## 1. Общее описание

Система MTQE (Machine Translation Quality Estimation) предназначена для автоматической оценки качества машинного перевода с английского на русский без доступа к эталонному переводу (reference-free QE). На вход подаётся пара (src, mt) — исходное предложение на английском и машинный перевод на русском. На выходе формируется:

- числовая оценка качества перевода в диапазоне [0, 1];
- доверительный интервал и мера неопределённости;
- список ошибочных span-ов на уровне слов с типами ошибок по таксономии MQM;
- объяснение оценки через SHAP-вклады признаков, агрегированные по MQM-категориям.

Архитектура является гибридной: она объединяет лингвистически интерпретируемые признаки, семантические представления на основе трансформеров, градиентный бустинг и тонко настроенную модель классификации на уровне токенов.

---

## 2. Остатки после удаления MiniLM/PCA/neural head

После удаления блока семантического расширения (MiniLM + IncrementalPCA) и нейронной головы (FeatureAttentionHead) в кодовой базе остались артефакты, которые необходимо удалить или скорректировать:

### Файлы-скрипты (мёртвый код):
- `scripts/train_semantic_pca.py` — обучение IncrementalPCA поверх MiniLM-эмбеддингов; больше не нужен, `SEMANTIC_FEATURE_NAMES` отсутствует в `schema.py`.
- `scripts/train_neural_head.py` — обучение `FeatureAttentionHead`; модуль `src/models/neural_head.py` отсутствует.

### Файлы исходного кода с нерабочими ссылками:
- `src/features/neural.py` — ссылается на `Config.SEMANTIC_ENCODER_NAME`, `Config.semantic_pca_path()` и `SEMANTIC_FEATURE_NAMES`, которых нет в `config.py` и `schema.py`. Файл нерабочий.

### Артефакты модели (можно удалить):
- `models/semantic_pca.pkl` — сохранённый IncrementalPCA.
- `models/semantic_pca_meta.json` — метаданные PCA.
- `models/neural_head.pt` — веса FeatureAttentionHead.
- `models/neural_head_config.json` — конфиг нейронной головы.
- `models/hybrid_meta.json` — метаданные гибридной модели.

### Что осталось и продолжает работать:
Семантические признаки LaBSE (`cosine_similarity`, `embedding_distance`) продолжают использоваться через `src/features/semantic.py` и `src/features/extractor.py`. Они входят в `FEATURE_NAMES_CLASSIC` и `SENTENCE_FEATURE_NAMES`. Этот путь не затронут удалением MiniLM.

---

## 3. Пайплайн подготовки данных

### Шаг 1. Загрузка и нормализация (prepare_data.py)

Система использует два публичных датасета с Hugging Face:

**HF DA (Direct Assessment):** `RicardoRei/wmt-da-human-evaluation`, языковая пара EN-RU. Оценка качества в диапазоне [0, 100] (Direct Assessment). Нормализуется методом min-max по обучающему сплиту в диапазон [ε, 1-ε], где ε = 0.0001 (клиппинг для совместимости с Beta-распределением). Стратифицированное разбиение 85/10/5 по квантилям оценки.

**HF MQM:** `RicardoRei/wmt-mqm-human-evaluation`, языковая пара EN-RU. Штрафная оценка качества (чем ниже — тем хуже). Используется только для внешней валидации sentence-модели.

### Шаг 2. Сборка word-level датасета (build_wordlevel.py)

Источник: WMT21 word-level данные (`mqm_dev2021_enru`), домены `news` и `ted`. Файлы `.src`, `.mt`, `.tags`, `.tsv`. Метки на уровне слов: OK, BAD-minor, BAD-major. Стратифицированное разбиение 85/10/5 по `max_severity`. Используется для fine-tuning XLM-RoBERTa.

### Шаг 3. Дедупликация MQM (dedup_mqm.py)

Из MQM-датасета удаляются пары (src, mt), совпадающие по sha256-хэшу с DA train. Предотвращает утечку данных при внешней валидации.

### Шаг 4. Синтетические негативы (build_synthetic_negatives.py)

Генерация низкокачественных переводов четырьмя методами:
- **shuffle** — перестановка токенов mt, score ∈ [0.10, 0.30];
- **untranslated** — замена случайных токенов mt на токены из src, score ∈ [0.00, 0.20];
- **deletion** — удаление ~30% токенов mt, score ∈ [0.10, 0.40];
- **entity_corruption** — замена именованных сущностей и чисел на случайные из пула, score ∈ [0.20, 0.50].

Дедупликация по хэшу пары. Синтетические строки добавляются только в train-сплит.

### Шаг 5. Извлечение признаков (extract_features.py)

Признаки извлекаются батчами с checkpoint-возобновлением. Поддерживается режим `--append-light` для дозаписи только лёгких признаков без перезапуска тяжёлых моделей. Обрабатываются три датасета: DA, word-level, MQM.

---

## 4. Признаковое пространство

Полный вектор признаков содержит **43 признака** и формируется в три слоя.

### Слой 1. Лёгкие признаки (27 штук, без GPU)

Вычисляются через spaCy (модели `ru_core_news_sm`, `en_core_web_sm`) и pymorphy2/pymorphy3.

**Структурные (7):** `length_ratio`, `abs_length_diff`, `token_count_diff`, `src_length`, `mt_length`, `compression_ratio`, `sentence_count_diff`.

**Форматные (7):** `digit_match_ratio`, `punct_ratio`, `quotes_mismatch`, `date_format_error`, `number_count_diff`, `capitalization_mismatch`, `currency_symbol_mismatch`.

**Лингвистические (13):** `oov_ratio`, `type_token_ratio`, `avg_token_length`, `entity_overlap_ratio`, `agreement_errors`, `syntax_depth`, `formal_ratio`, `morphology_error_rate`, `repetition_ratio`, `named_entity_missing_ratio`, `latin_ratio`, `avg_word_rank`, `untranslated_ratio`.

### Слой 2. Тяжёлые признаки (6 штук, требуют GPU/CPU с моделями)

**Семантические (2, LaBSE):** `cosine_similarity`, `embedding_distance` — косинусное сходство и евклидово расстояние эмбеддингов src и mt в пространстве LaBSE (multilingual sentence encoder).

**Fluency (4, ruGPT-3 Small):** `perplexity`, `mean_log_prob`, `token_ppl_variance`, `min_token_log_prob` — оценка языковой вероятности mt через авторегрессионную модель ruGPT-3 Small.

### Слой 3. Производные (interaction) признаки (10 штук)

Нелинейные комбинации базовых признаков, вычисляются аналитически:

`cosine_x_length_ok`, `log_perplexity`, `cosine_per_logppl`, `entity_x_cosine`, `oov_x_bad_cosine`, `logprob_spike`, `variance_x_bad_cosine`, `normed_length_diff`, `digit_x_entity`, `formal_x_cosine`.

---

## 5. Sentence-level оценка

### Модели

Обучаются три независимые регрессионные модели на `sentence_da_features.parquet`:

**XGBoost** (`sentence_xgboost.model`) — градиентный бустинг на деревьях. Обучение с early stopping по Pearson r на val-сплите (patience=120 итераций из 4000 максимальных). Параметры: learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.7. Синтетические строки получают пониженный вес (0.12), downsampling до 30% от объёма. Метрика качества — Pearson r и Spearman ρ на DA test и MQM external test.

**Ridge регрессия** (`sentence_ridge.pkl`) — линейная модель (alpha=2.0) для интерпретируемого baseline.

**Random Forest** (`sentence_rf.pkl`) — 400 деревьев, max_depth=14. Дополнительно даёт оценку неопределённости через дисперсию предсказаний деревьев.

### Объяснение (SHAP)

Для XGBoost и RF используется `shap.TreeExplainer`. Для Ridge — произведение коэффициентов на значения признаков. SHAP-вклады агрегируются по MQM-категориям (Accuracy, Fluency, Terminology, Locale, Style) через словарь `FEATURE_TO_MQM`.

### Ансамбль

На инференсе три модели объединяются взвешенным усреднением:
- XGBoost: 0.45
- RF: 0.35
- Ridge: 0.20

Веса нормализуются при неполном составе ансамбля. SHAP-вклады ансамбля — взвешенная сумма SHAP-вкладов компонент.

### Неопределённость и доверительный интервал

Для XGBoost и Ridge: Beta-аппроксимация через псевдоконцентрацию (concentration=10), CI₉₅ через `scipy.stats.beta.ppf`. Для RF: стандартное отклонение предсказаний деревьев, CI₉₅ = score ± 1.96 * std.

### Штраф за span-ошибки

После получения ансамблевой оценки она корректируется вниз на основе результатов span-модели:
- BAD-major span: -0.06
- BAD-minor span (нет major): -0.03
- Каждый дополнительный span сверх первого: -0.01
- Максимальный суммарный штраф: -0.12

---

## 6. Word-level / Span-level оценка

### Модель

**XLM-RoBERTa-base** (`xlm_roberta_span/`) fine-tuned на WMT21 word-level данных. Задача: token classification с тремя классами — OK (0), BAD-minor (1), BAD-major (2).

**Схема токенизации:** `[CLS] src [SEP] mt [SEP]`. src получает 1/3 токенового бюджета (169 токенов), mt — 2/3 (340 токенов). Маппинг SentencePiece субтокенов на слова: стратегия первого субтокена (first-subtoken). Остальные субтокены слова получают метку IGNORE_INDEX = -100 и не учитываются в loss.

**Взвешенный loss:** OK=1.0, BAD-minor=2.0, BAD-major=5.0.

**Обучение:** AdamW, lr=2e-5, линейный warmup (10% от total_steps), early stopping по F1(BAD-major) на val-сплите (patience=3 эпохи).

### Порог классификации

Слово признаётся BAD если p(BAD-minor) + p(BAD-major) >= 0.45 (IMTQE_SPAN_BAD_THRESHOLD). Из BAD-слов класс BAD-major присваивается если p(BAD-major) >= 0.60 (IMTQE_SPAN_MAJOR_THRESHOLD). Пороги настраиваются через переменные окружения.

### Группировка в спаны

Смежные BAD-слова объединяются в один SpanResult. Severity спана — максимальный severity среди его слов. Confidence спана — p(BAD) первого слова.

---

## 7. MQM-интерпретация и агрегация

### Классификация типов ошибок (rules.py)

XLM-RoBERTa определяет только severity. Тип ошибки по таксономии MQM назначается детерминированными правилами на основе признаков предложения и содержимого span-а:

Таксономия MQM (17 типов):
- Accuracy: Mistranslation, Omission, Addition, Untranslated, Hallucination
- Fluency: Morphology, Agreement, Spelling, LexicalChoice, Repetition
- Terminology: WrongTerm, Inconsistency
- Locale: NumberFormat, DateFormat, Quotes, Currency
- Style: Register

Приоритет правил (от высшего к низшему): Untranslated → Locale/Currency → Locale/Quotes → Locale/DateFormat → Locale/NumberFormat → Fluency/Morphology → Fluency/Spelling → Fluency/Repetition → Fluency/Agreement → Accuracy/Omission → Fluency/LexicalChoice → Accuracy/Mistranslation (дефолт).

### MQM-агрегация (aggregation.py)

Формула штрафа на уровне предложения:

```
penalty = sum_i(w_t_i * p_s_i * c_i)
mqm_score = clip((100 - penalty / Z) / 100, 0, 1)
```

где `w_t` — вес типа ошибки (по умолчанию 1.0, загружается из `weights_mqm.npy` при наличии), `p_s` — severity penalty (BAD-major=5.0, BAD-minor=1.0), `c_i` — confidence span-а, `Z` — число слов mt.

### Объяснение через SHAP (explanation_loss.py)

Отрицательные SHAP-вклады (снижающие оценку) нормируются до суммы = 1 и масштабируются на `loss_budget = 1 - score`. Доли менее 0.5% отбрасываются. Результат отображается как "разбор штрафа" в интерфейсе — распределение недостающего до 100% по MQM-категориям.

---

## 8. Инференс и сервис

### Predictor (predict.py)

Единая точка инференса. При инициализации загружает все три sentence-модели, FeatureExtractor (LaBSE + ruGPT-3), SpanModel, OverallSentenceEvaluator.

Метод `predict_sentence(src, mt)`:
1. Извлечение 43 признаков через FeatureExtractor.
2. Предсказание score всеми тремя sentence-моделями + SHAP.
3. Предсказание word-level меток XLM-RoBERTa.
4. Построение ансамблевого score.
5. Применение span-штрафа.
6. Назначение типов ошибок через rules.py.
7. MQM-агрегация.
8. Формирование объяснения через SHAP.
9. Рендеринг highlighted HTML.

Метод `predict_batch(pairs)` — аналогично, без батчевого инференса SpanModel (последовательно).

### FastAPI сервис (server.py, api.py)

Lifespan-загрузка моделей при старте. Эндпоинты:
- `POST /api/evaluate` — оценка одного сегмента.
- `POST /api/evaluate_batch` — до 50 пар.
- `POST /api/feedback` — сохранение ручной разметки в `data/feedback/feedback.jsonl`.
- `POST /api/reload_models` — горячая перезагрузка sentence/span моделей без перезагрузки LaBSE и ruGPT-3.
- `GET /api/status` — статус готовности моделей.

### Веб-интерфейс (index.html, app.js, style.css)

Таблица сегментов с inline-редактированием. Подсветка ошибочных span-ов непосредственно в тексте перевода через `contenteditable`-overlay. Панель деталей с разбором штрафа по признакам. Переключение между моделями ансамбля. Тултипы с типом и severity ошибки при наведении.

---

## 9. Воспроизводимость

- Единый seed (42) через `src/determinism.py`: Python random, numpy, torch, HuggingFace transformers.
- Конфигурация через переменные окружения: `IMTQE_DATA_DIR`, `IMTQE_MODELS_DIR`, `IMTQE_LOG_DIR`, `IMTQE_SEED`, `HF_HUB_OFFLINE`, `IMTQE_SPAN_BAD_THRESHOLD`, `IMTQE_SPAN_MAJOR_THRESHOLD`.
- Checkpoint-возобновление для extract_features.py и train_semantic_pca.py (атомарная запись через `tempfile` + `replace`).

---

## 10. Технологический стек

| Компонент | Библиотека/модель |
|---|---|
| Морфологический анализ | spaCy ru_core_news_sm, en_core_web_sm; pymorphy2/pymorphy3 |
| Семантические признаки | sentence-transformers/LaBSE |
| Fluency признаки | sberbank-ai/rugpt3small_based_on_gpt2 |
| Gradient boosting | XGBoost 2.x |
| Интерпретируемость | SHAP (TreeExplainer) |
| Span classification | xlm-roberta-base (HuggingFace Transformers) |
| Сервис | FastAPI + uvicorn |
| Хранение данных | Apache Parquet (pandas) |
| Статистика | scipy (Beta distribution, Pearson, Spearman) |
