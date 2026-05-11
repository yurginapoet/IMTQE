# IMTQE — Архитектура системы v3

Документ описывает **фактическую реализацию** репозитория IMTQE: блочную структуру системы, контракты данных между блоками, модели, обучение, инференс, API и эксплуатацию. Структура изложения следует формату `architecture_req_v2.md`; там, где реализация расходится с v2 — это явно отмечено.

**Языковая пара:** EN → RU (жёстко заложена в признаки, spaCy-модели и датасеты).

**Связанные материалы:** `RUN_TRAINING_INFERENCE.md`, `architecture_semantic_extension.md`, `ui_spec.md`.

---

## §1. Назначение и область применения

**IMTQE (Interpretable Machine Translation Quality Estimation)** — reference-free оценка качества машинного перевода без эталонного перевода. Система работает **на уровне одного сегмента (предложения)** за запрос.

Система одновременно:

- выдаёт **скалярный score качества** на уровне предложения (∈ [0, 1], 1 = идеально);
- строит **псевдо-доверительный интервал** (CI₉₅) и меру неопределённости через Beta-аппроксимацию поверх точечной XGBoost-регрессии;
- **локализует** проблемные фрагменты в переводе (пословно, по spaCy-индексам слов mt);
- **классифицирует** каждый BAD-спан по типологии MQM через **детерминированные правила** (не нейросеть);
- строит **MQM-style штрафной score** и **объяснение** в терминах MQM-категорий для UI (доли «потери» из SHAP или нейронной головы).

**Не входит в текущую реализацию:** абзацный режим, межпредложенческая терминология.

---

## §2. Блочная структура системы

Система состоит из **пяти последовательных блоков** инференса и **одного блока сборки**:

```
Блок 1: FeatureExtractor       → вектор признаков (до 97-мерный), word_logprobs
Блок 2: SentenceModel          → score, CI, SHAP, explanation по MQM-категориям
Блок 3: SpanModel              → пословные метки OK/BAD-minor/BAD-major, спаны
Блок 4: MQM Rules              → типы ошибок для спанов (детерминировано)
Блок 5: MQM Aggregation        → mqm_score ∈ [0,1], штрафы по типам
Блок 6: ResultBuilder          → SentenceUIResult: HTML, errors[], explanation
```

Блоки 4–6 не требуют дополнительных тяжёлых моделей: только правила и арифметика.

---

## §3. Блок 1 — FeatureExtractor

### 3.1 Назначение

Строит числовой вектор признаков пары `(src_en, mt_ru)` для подачи в Блок 2 (SentenceModel). Дополнительно производит `word_logprobs` (per-word logprobs mt) для Блока 3 и `mt_words` (список spaCy-слов mt) для согласованной адресации в UI.

### 3.2 Вход / Выход

| | Тип | Описание |
|---|---|---|
| **Вход** | `str` | `src` — исходное предложение (EN) |
| **Вход** | `str` | `mt` — машинный перевод (RU) |
| **Выход** | `np.ndarray` | `vector` — вектор признаков (см. §3.5) |
| **Выход** | `list[float]` | `word_logprobs` — logprob каждого spaCy-слова mt |
| **Выход** | `list[str]` | `mt_words` — spaCy-токены mt (согласованы со SpanModel) |
| **Выход** | `dict` | `raw` — словарь всех признаков по имени (для debug и rules) |
| **Выход** | `float` | `formal_ratio` — доля формальной лексики |

### 3.3 Требуемые модели

| Модель | Назначение | Загрузка |
|---|---|---|
| `spacy ru_core_news_sm` | Токенизация, морфология, NER, синтаксис RU | всегда |
| `spacy en_core_web_sm` | Токенизация, NER EN (лёгкий пайплайн) | всегда |
| `sentence-transformers/LaBSE` | Билингвальные эмбеддинги для семантики | тяжёлый режим |
| `sberbank-ai/rugpt3small_based_on_gpt2` | Логарифмические вероятности токенов RU | тяжёлый режим |
| `paraphrase-multilingual-MiniLM-L12-v2` | Эмбеддинги для semantic PCA | semantic режим |
| `models/semantic_pca.pkl` | PCA 384→64 по разностям MiniLM-эмбеддингов | semantic режим |

Режимы загрузки:
- **Light** — только spaCy (16 признаков, без тяжёлых моделей)
- **Classic** — Light + LaBSE + ruGPT-3 + interaction (33 признака)
- **Full / Semantic** — Classic + MiniLM + PCA (97 признаков) — **режим по умолчанию в prod**

### 3.4 Группы признаков

**Структурные** (`src/features/structural.py`) — 5 признаков:

| Признак | Описание |
|---|---|
| `length_ratio` | mt_len / src_len |
| `abs_length_diff` | |mt_len − src_len| |
| `token_count_diff` | mt_len − src_len (со знаком) |
| `src_length` | число токенов src |
| `mt_length` | число токенов mt |

**Форматные** (`src/features/formatting.py`) — 4 признака:

| Признак | Описание |
|---|---|
| `digit_match_ratio` | доля чисел src, найденных в mt |
| `punct_ratio` | mt_punct / src_punct |
| `quotes_mismatch` | 1 если в src кавычки, а в mt прямые `"'` |
| `date_format_error` | 1 если дата из src скопирована без адаптации |

**Лингвистические** (`src/features/linguistic.py`) — 7 признаков:

| Признак | Описание |
|---|---|
| `oov_ratio` | доля OOV-слов mt по spaCy |
| `type_token_ratio` | лексическое разнообразие mt |
| `avg_token_length` | средняя длина слова mt в символах |
| `entity_overlap_ratio` | доля NER-сущностей src, найденных в mt |
| `agreement_errors` | число нарушений согласования (род/число/падеж) |
| `syntax_depth` | максимальная глубина дерева зависимостей mt |
| `formal_ratio` | доля формальной лексики mt (словарь FORMAL_VOCAB) |

**Семантические** (`src/features/semantic.py`) — 2 признака (тяжёлый режим):

| Признак | Описание |
|---|---|
| `cosine_similarity` | косинусное сходство LaBSE(src) и LaBSE(mt) |
| `embedding_distance` | евклидово расстояние LaBSE-эмбеддингов |

**Fluency** (`src/features/fluency.py`) — 4 признака + word_logprobs (тяжёлый режим):

| Признак | Описание |
|---|---|
| `perplexity` | exp(−mean_log_prob) по ruGPT-3 |
| `mean_log_prob` | среднее logprob spaCy-слов mt |
| `token_ppl_variance` | дисперсия logprob |
| `min_token_log_prob` | минимальный logprob (наихудшее слово) |
| `word_logprobs` | список logprob по каждому spaCy-слову (→ SpanModel) |

**Semantic PCA** (`src/features/neural.py`) — 64 признака (semantic режим):

Для пары `(src, mt)`:
1. MiniLM кодирует оба текста в вектора размерности 384.
2. Считается `abs(emb_src − emb_mt)` — разностный вектор 384.
3. PCA(n_components=64) проецирует разность → `semantic_00` … `semantic_63`.

**Interaction** (`src/features/interactions.py`) — 11 признаков (тяжёлый режим):

Нелинейные комбинации базовых признаков: `cosine_x_length_ok`, `log_perplexity`, `cosine_per_logppl`, `entity_x_cosine`, `oov_x_bad_cosine`, `logprob_spike`, `variance_x_bad_cosine`, `normed_length_diff`, `digit_x_entity`, `formal_x_cosine`, `dist_x_logppl`.

### 3.5 Схема признаков (итог)

Порядок и имена задаются `src/features/schema.py`:

| Режим | Состав | Длина |
|---|---|---|
| Light | structural + formatting + linguistic | **16** |
| Classic (heavy без PCA) | Light + semantic_explicit + fluency heavy + interaction | **33** |
| Full / Semantic (prod) | 22 базовых + 64 semantic PCA + 11 interaction | **97** |

Точные имена: `FEATURE_NAMES_LIGHT` (16), `FEATURE_NAMES_CLASSIC` (22), `FEATURE_NAMES` (86), `SENTENCE_FEATURE_NAMES` (97).

### 3.6 Реализация

`src/features/extractor.py` — класс `FeatureExtractor`:
- `load_heavy_models(require_neural: bool)` — загрузка LaBSE / ruGPT / MiniLM + PCA; вызывается один раз при старте `Predictor`.
- `extract(src, mt) → dict` — одиночный инференс.
- `extract_batch(pairs) → list[dict]` — батчевый инференс через spaCy pipe + batched encode.
- `active_feature_names` — актуальный список в зависимости от загруженных моделей.

---

## §4. Блок 2 — SentenceModel (XGBoost + SHAP)

### 4.1 Назначение

Предсказывает скалярный **score качества** предложения ∈ [0, 1] и строит объяснение через SHAP.

**Отличие от v2:** вместо NGBoost с параметрической Beta на выходе используется XGBoost-регрессия + Beta-аппроксимация неопределённости как эвристика поверх точечного предсказания.

### 4.2 Вход / Выход

| | Тип | Описание |
|---|---|---|
| **Вход** | `np.ndarray` | вектор признаков из Блока 1 (длина = `expected_feature_count`) |
| **Выход** | `float` | `score` ∈ [0, 1] |
| **Выход** | `float` | `uncertainty` — Var[Beta]-аппроксимация |
| **Выход** | `float`, `float` | `ci_low`, `ci_high` — CI₉₅ через Beta.ppf(0.025/0.975) |
| **Выход** | `np.ndarray` | `shap_values` — SHAP-вклады по признакам |
| **Выход** | `dict[str, float]` | `explanation` — SHAP агрегированный по MQM-категориям |

### 4.3 Требуемые артефакты

| Файл | Назначение |
|---|---|
| `models/xgboost_sentence.model` | XGBoost booster (native format) |
| `models/shap_explainer.pkl` | `TreeExplainer` + список `feature_names` |

### 4.4 Неопределённость (Beta-аппроксимация)

```
concentration = 10.0
alpha   = score * concentration
beta_p  = (1 − score) * concentration
CI₉₅    = [Beta.ppf(0.025, alpha, beta_p), Beta.ppf(0.975, alpha, beta_p)]
Var     = alpha * beta_p / (total² * (total + 1))
```

### 4.5 Агрегация SHAP → MQM-категории

Маппинг `FEATURE_TO_MQM` (`src/models/sentence_model.py`) назначает каждому признаку одну из категорий: `Accuracy`, `Fluency`, `Terminology`, `Locale`, `Style`, `Semantic`. Значения SHAP суммируются по категориям → `explanation`.

### 4.6 Согласование размерностей

При инициализации `Predictor` вызывается `_validate_extractor_features(expected_feature_count)`. Если `FeatureExtractor` возвращает меньше признаков, чем ожидает модель — `RuntimeError` с подсказкой.

### 4.7 Реализация

`src/models/sentence_model.py` — класс `SentenceModel`:
- `predict(features) → SentencePrediction`
- `predict_batch(features) → list[SentencePrediction]`
- `_xgboost_uncertainty(score)` — Beta-аппроксимация
- `_aggregate_shap(shap_vals, feature_names)` → MQM-категории

### 4.8 Опциональная нейронная голова (`neural_head`)

Если присутствуют `models/neural_head.pt` + `models/neural_head_config.json`, вместо SHAP для построения `explanation` используется `FeatureAttentionHead` (`src/models/neural_head.py`):

- Вход головы: `[вектор_97_признаков | xgb_score]` (98 значений).
- Attention-веса softmax по входам → распределение `(1 − score_head)` по MQM-категориям.
- Если файлы отсутствуют — fallback на `shap_categories_to_loss_shares`.

---

## §5. Блок 3 — SpanModel (XLM-RoBERTa token classification)

### 5.1 Назначение

Определяет **severity** каждого spaCy-слова mt: `OK` / `BAD-minor` / `BAD-major`. Смежные BAD-слова объединяются в спаны `SpanResult`. **Тип ошибки MQM в этом блоке не определяется** — это задача Блока 4.

### 5.2 Вход / Выход

| | Тип | Описание |
|---|---|---|
| **Вход** | `str` | `src` (EN) |
| **Вход** | `str` | `mt` (RU) |
| **Вход** | `list[float]` | `word_logprobs` из Блока 1 (передаётся в спаны для Блока 4) |
| **Вход** | `list[str]` | `mt_words` — согласованный список слов mt |
| **Выход** | `list[str]` | `word_labels` — метка каждого слова mt |
| **Выход** | `list[float]` | `word_probs` — p(BAD) = p(BAD-minor) + p(BAD-major) для каждого слова |
| **Выход** | `list[SpanResult]` | BAD-спаны с `start_idx`, `end_idx`, `severity`, `confidence`, `word_logprobs_span` |

### 5.3 Требуемые артефакты

| Файл | Описание |
|---|---|
| `models/xlm_roberta_span/config.json` | Конфигурация HF-модели |
| `models/xlm_roberta_span/model.safetensors` | Веса fine-tuned XLM-RoBERTa |
| `models/xlm_roberta_span/tokenizer.json` | Токенизатор |

### 5.4 Схема токенизации

```
[CLS]  src_tokens  [SEP]  mt_tokens  [SEP]
```

Бюджет: src получает 1/3 от 509 контентных позиций (≈169), mt — 2/3 (≈340). Маппинг SentencePiece-субтоков → spaCy-слова mt: **first-subtoken** стратегия через char offsets.

### 5.5 Метки классов

| Класс | ID | Описание |
|---|---|---|
| OK | 0 | Слово корректно |
| BAD-minor | 1 | Незначительная ошибка |
| BAD-major | 2 | Серьёзная ошибка |

### 5.6 Группировка в спаны

Смежные BAD-слова (любого severity) объединяются в один `SpanResult`. Severity спана = максимальный severity среди его слов (BAD-major поглощает BAD-minor). Confidence = p(BAD) первого слова спана.

### 5.7 Реализация

`src/models/span_model.py` — класс `SpanModel`.

---

## §6. Блок 4 — MQM Rules (детерминированная типизация спанов)

### 6.1 Назначение

Назначает **тип ошибки MQM** каждому BAD-спану из Блока 3. Правила применяются **детерминированно** — никакой нейросети для классификации типа.

### 6.2 Вход / Выход

| | Тип | Описание |
|---|---|---|
| **Вход** | `list[str]` | `mt_words` |
| **Вход** | `list[SpanResult]` | BAD-спаны из Блока 3 |
| **Вход** | `dict` | `sentence_features` — признаки из Блока 1 (`raw`) |
| **Выход** | `list[TypedSpan]` | Спаны с полем `error_type` |

### 6.3 Иерархия правил

Правила применяются в порядке убывания специфичности (первое совпавшее — победитель):

| Приоритет | Тип | Условие |
|---|---|---|
| 1 | `Locale/Currency` | Спан содержит `$€£¥₽` |
| 2 | `Locale/Quotes` | `quotes_mismatch=1` И спан содержит прямые `"'` |
| 3 | `Locale/DateFormat` | `date_format_error=1` И спан содержит дату |
| 4 | `Locale/NumberFormat` | `digit_match_ratio < 1.0` И спан содержит цифры |
| 5 | `Accuracy/Untranslated` | Спан содержит латиницу, кириллицы нет |
| 6 | `Fluency/Spelling` | `oov_ratio > 0.3` И спан выглядит как опечатка (слово > 20 символов) |
| 7 | `Fluency/Agreement` | `agreement_errors > 0` И `−6.0 < mean_logprob_span < −4.0` |
| 8 | `Fluency/LexicalChoice` | `mean_logprob_span < −6.0` (порог `_LOGPROB_FLUENCY_THRESHOLD`) |
| 9 | `Accuracy/Mistranslation` | Дефолт |

**Примечание по порогу:** `_LOGPROB_FLUENCY_THRESHOLD = −6.0` (исправлено с −8.0 — более реалистичный диапазон для ruGPT-3 Small).

### 6.4 Реализация

`src/interpretation/rules.py` — функция `assign_mqm_types(mt_words, spans, sentence_features)`.

---

## §7. Блок 5 — MQM Aggregation

### 7.1 Назначение

Агрегирует штрафы за спаны в единый **MQM score** ∈ [0, 1] уровня предложения.

### 7.2 Вход / Выход

| | Тип | Описание |
|---|---|---|
| **Вход** | `list[TypedSpan]` | Типизированные спаны из Блока 4 |
| **Вход** | `int` | `mt_word_count` — число слов mt (нормализатор Z) |
| **Вход** | `np.ndarray` | `weights` — веса типов ошибок из `models/weights_mqm.npy` |
| **Выход** | `MQMAggregation` | `mqm_score`, `penalty`, `z`, `per_type_penalty` |

### 7.3 Формула

```
penalty   = Σᵢ ( w_type_i · ps_i · confidence_i )
mqm_score = clip( (100 − penalty / Z) / 100, 0, 1 )
```

Штрафы severity: `BAD-minor = 1.0`, `BAD-major = 5.0`. Деление на Z применяется **один раз** к сумме (не к каждому слагаемому). `per_type_penalty` хранит ненормализованные вклады по типам для отладки.

### 7.4 Требуемые артефакты

| Файл | Описание |
|---|---|
| `models/weights_mqm.npy` | Вектор весов длины 16 (по числу `MQM_ERROR_TYPES`); при отсутствии — единичные |

### 7.5 Типология MQM (фиксированный порядок = индексы в `weights_mqm.npy`)

```
Accuracy:    Mistranslation, Omission, Addition, Untranslated, Hallucination
Fluency:     Morphology, Agreement, Spelling, LexicalChoice
Terminology: WrongTerm, Inconsistency
Locale:      NumberFormat, DateFormat, Quotes, Currency
Style:       Register
```

### 7.6 Реализация

`src/interpretation/aggregation.py` — функция `aggregate_sentence_mqm(...)`, класс `MQMAggregation`.

---

## §8. Блок 6 — ResultBuilder

### 8.1 Назначение

Собирает финальный `SentenceUIResult` из результатов всех предыдущих блоков.

### 8.2 Вход / Выход

| | Тип | Описание |
|---|---|---|
| **Вход** | `OverallSentenceResult` | Агрегат блоков 2–5 |
| **Вход** | `dict` | `debug_info` (признаки, word_logprobs, SHAP) |
| **Выход** | `SentenceUIResult` | score, CI, mqm_score, HTML, errors[], explanation, debug |

### 8.3 Формирование выходных полей

**`highlighted_mt_html`** — HTML-строка с подсветкой BAD-слов:
- `BAD-major` → фон `#ffb3b3`
- `BAD-minor` → фон `#ffe3a3`
- OK → без разметки

**`errors[]`** — список `SentenceErrorItem` по спанам: `severity`, `error_type`, `error_label` (русское описание из `MQM_ERROR_TYPE_RU`), `confidence`, `span_text`, `start_idx`, `end_idx`.

**`explanation`** — доли «потери» по MQM-категориям с русскими ключами (`MQM_CATEGORY_RU`). Строится из `shap_categories_to_loss_shares` или `neural_head.explain_mqm_loss_shares`. Берутся только отрицательные SHAP-вклады (что тянет оценку вниз), нормируются к сумме 1, отфильтровываются доли < 0.5%.

**`debug`** — словарь `{ features: dict, word_logprobs: list, shap_values: dict }` — кэшируется на фронтенде для последующей отправки feedback без пересчёта признаков.

### 8.4 Реализация

`src/predict.py` — функции `_build_ui_result`, `_render_highlighted_mt`, `_build_explanation_ru`, `build_sentence_debug_payload`.

---

## §9. Инференс: последовательность вызовов

```
Predictor.predict_sentence(src, mt)
  │
  ├─ Блок 1: FeatureExtractor.extract(src, mt)
  │       → vector, mt_words, word_logprobs, raw, formal_ratio
  │
  ├─ Блок 2: SentenceModel.predict(vector)
  │       → score, uncertainty, ci_low, ci_high, shap_values, explanation
  │
  ├─ Блок 3: SpanModel.predict(src, mt, word_logprobs, mt_words)
  │       → word_labels, word_probs, spans: list[SpanResult]
  │
  ├─ OverallSentenceEvaluator.evaluate(sentence_pred, span_pred, mt_words, raw)
  │   ├─ Блок 4: assign_mqm_types(mt_words, spans, raw)
  │   │       → list[TypedSpan]
  │   └─ Блок 5: aggregate_sentence_mqm(typed_spans, mt_word_count, weights)
  │           → MQMAggregation
  │
  ├─ _display_explanation_en(feats, sentence_pred)   ← neural_head или SHAP
  │
  └─ Блок 6: _build_ui_result(src, mt, mt_words, overall, debug_info)
          → SentenceUIResult
```

**Батч:** `extract_batch` + `predict_batch` по матрице векторов; SpanModel вызывается последовательно.

---

## §10. Пайплайн обучения

### 10.1 Порядок шагов

```
Шаг 1: prepare_data.py
Шаг 2: build_wordlevel.py
Шаг 3: dedup_mqm.py
Шаг 4: build_synthetic_negatives.py
Шаг 5: train_semantic_pca.py
Шаг 6: extract_features.py
Шаг 7: train_sentence_model.py
Шаг 8: train_span_model.py
Шаг 9: train_neural_head.py          ← опционально
```

Полный прогон: `poetry run imtqe pipeline` или `python scripts/run_full_pipeline.py`.

### 10.2 Шаг 1 — prepare_data.py

**Источники:**
- HF DA: `RicardoRei/wmt-da-human-evaluation` (Direct Assessment, EN-RU, raw score 0–100)
- HF MQM: `RicardoRei/wmt-mqm-human-evaluation` (EN-RU)

**Действия:**
- Фильтрация по `lp = "en-ru"`.
- Стратифицированный split **85/10/5** по квантилям score (5 бинов, `pd.qcut`).
- Min-max нормализация DA score → `score_norm ∈ [EPS, 1−EPS]` **только по train**, clip для Beta.
- SHA-256 хэш пары `(src, mt)` → `pair_hash` (для дедупликации).

**Выходные файлы:**
- `data/processed/sentence_da.parquet` — DA с `score_norm`, `split`, `pair_hash`
- `data/processed/hf_mqm_raw.parquet` — MQM с `pair_hash`

### 10.3 Шаг 2 — build_wordlevel.py

**Источник:** WMT21 word-level данные `data/raw/wordlevel/` (домены `news`, `ted`):
- `.src` — исходные предложения EN
- `.mt` — переводы RU (содержат `<EOS>` в конце, удаляется)
- `.tags` — пословные метки `OK/BAD`
- `.tsv` — severity аннотации на уровне предложения

**Логика меток:** `.tags` даёт OK/BAD; из TSV берётся максимальный severity предложения (`major/critical → BAD-major`, иначе `BAD-minor`). Если предложение не в TSV → `BAD-minor` (консервативно).

**Split:** стратифицированный **85/10/5** по `max_severity`.

**Выходной файл:** `data/processed/wordlevel_train.parquet` с колонками `src`, `mt`, `word_labels`, `split`, `domain`.

### 10.4 Шаг 3 — dedup_mqm.py

Исключает из HF MQM строки, чей `pair_hash` встречается в DA train. Предотвращает утечку данных при внешнем тесте. При удалении > 5% строк — предупреждение.

**Выходной файл:** `data/processed/hf_mqm_dedup.parquet`

### 10.5 Шаг 4 — build_synthetic_negatives.py

Создаёт синтетические негативные примеры **только для train split** DA. Четыре типа корруптации:

| Тип | Метод | Диапазон score_norm |
|---|---|---|
| `shuffle` | Случайная перестановка слов mt | 0.10–0.30 |
| `untranslated` | Замена 30% слов mt словами из src | 0.00–0.20 |
| `deletion` | Удаление 30% слов mt | 0.10–0.40 |
| `entity_corruption` | Замена сущностей mt на случайные из пула | 0.20–0.50 |

Дублирование по `pair_hash` исключено. Синтетические строки помечаются `is_synthetic=True`, `synthetic_type`, `synthetic_parent_hash`.

**Выходной файл:** `data/processed/sentence_da_augmented.parquet`

### 10.6 Шаг 5 — train_semantic_pca.py

Обучает `IncrementalPCA(n_components=64)` на разностях MiniLM-эмбеддингов `abs(emb_src − emb_mt)` по train-парам из `sentence_da_augmented.parquet`.

**Технические детали:**
- Потоковая обработка чанками по 8192 пар → `partial_fit`; прогресс через `tqdm`.
- Checkpoint каждые 5 чанков (атомарная запись через `tmp` файл).
- На CUDA: MiniLM переводится в float16.
- Сохраняются `models/semantic_pca.pkl` и `models/semantic_pca_meta.json`.

**Выходной файл:** `models/semantic_pca.pkl`

### 10.7 Шаг 6 — extract_features.py

Извлекает признаки для трёх датасетов:

| Датасет | Вход | Выход |
|---|---|---|
| DA | `sentence_da_augmented.parquet` | `sentence_da_features.parquet` |
| WordLevel | `wordlevel_train.parquet` | `wordlevel_features.parquet` |
| MQM | `hf_mqm_dedup.parquet` | `hf_mqm_features.parquet` |

Требует полной цепочки тяжёлых моделей + `semantic_pca.pkl`. Вектор = 97 признаков (`SENTENCE_FEATURE_NAMES`). Checkpoint каждые 100 батчей.

Режим `--append-light`: дозаписывает только 16 лёгких spaCy-признаков в уже существующий parquet без пересчёта тяжёлых — для быстрого обновления после изменений в `structural/formatting/linguistic`.

### 10.8 Шаг 7 — train_sentence_model.py

**Вход:** `sentence_da_features.parquet`, цель: `score_norm`.

**Признаки:** `SENTENCE_FEATURE_NAMES` (97) или база + interaction в RAM через `add_interaction_columns_to_dataframe`.

**Веса обучения:**
- Синтетические строки: downsampling до 20% от их числа, затем вес `--synthetic-weight` (дефолт 0.10).
- Лёгкий upweighting низких score: `tau=0.15` (отключён по умолчанию: `low_score_weight=1.0`).
- Опционально: `--semantic-feature-weight` (дефолт 1.0) — снижает частоту попадания `semantic_*` признаков в colsample.

**XGBoost гиперпараметры (зафиксированы в коде):**

| Параметр | Значение |
|---|---|
| `objective` | `reg:squarederror` |
| `learning_rate` | 0.03 |
| `max_depth` | 5 |
| `min_child_weight` | 5 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.7 |
| `reg_lambda` | 1.0 |
| `reg_alpha` | 0.05 |
| `gamma` | 0.05 |
| `num_boost_round` | до 4000 |

**Early stopping:** кастомный `_PearsonCallback` — по val Pearson r, patience=120 итераций.

**Валидация:** Pearson/Spearman на val-сплите DA; внешний тест на `hf_mqm_features.parquet` — Spearman ρ по zscore-нормализованному MQM score (по системе или глобальный).

**Выходные файлы:** `models/xgboost_sentence.model`, `models/shap_explainer.pkl`.

`shap_explainer.pkl` содержит dict `{ "explainer": TreeExplainer, "feature_names": list[str] }` — имена признаков включены для корректной агрегации на инференсе.

### 10.9 Шаг 8 — train_span_model.py

**Базовая модель:** `xlm-roberta-base`.

**Вход:** `wordlevel_train.parquet`, колонка `word_labels: list[str]`.

**Схема токенизации:** `[CLS] src [SEP] mt [SEP]`, `MAX_LENGTH=512`. Метки только на mt-части, first-subtoken стратегия.

**Loss:** `CrossEntropyLoss` с весами `OK=1, BAD-minor=2, BAD-major=5`.

**Гиперпараметры:**

| Параметр | Значение |
|---|---|
| Base model | `xlm-roberta-base` |
| Epochs | до 5 (early stopping) |
| Batch size | 16 |
| Learning rate | 2e-5 |
| Warmup | 10% total steps |
| Early stopping metric | val F1(BAD-major) |
| Patience | 3 эпохи |
| `set_seed` | 42 (transformers) |

**Выходной файл:** `models/xlm_roberta_span/` (HuggingFace format: config.json, model.safetensors, tokenizer.json).

### 10.10 Шаг 9 — train_neural_head.py (опционально)

**Назначение:** альтернативный механизм объяснений через attention поверх XGBoost.

**Вход:** `sentence_da_features.parquet` + предсказания XGBoost (`xgb_score`).

**Архитектура `FeatureAttentionHead`:**
- Вход: `[97 признаков | xgb_score]` = 98 значений.
- LayerNorm → instance-level attention (Linear → Tanh → Linear → Softmax по признакам) → взвешенный вектор.
- Head: Linear(98→64) → GELU → Dropout(0.1) → Linear(64→32) → GELU → Dropout(0.1) → Linear(32→1) → Sigmoid.
- ~12k параметров; быстрый CPU-инференс.

**Обучение:**
- Loss: `HuberLoss(delta=0.1)`.
- Optimizer: AdamW, lr=3e-4, weight_decay=1e-4.
- Scheduler: CosineAnnealingLR.
- Early stopping по val Pearson, patience=20 эпох.

**Выходные файлы:** `models/neural_head.pt`, `models/neural_head_config.json` (содержит `input_dim` и `feature_names`).

---

## §11. Данные: контракт файлов

Явные имена в `src/data_contract.py`:

| Файл | Кто создаёт | Кто читает | Ключевые колонки |
|---|---|---|---|
| `sentence_da.parquet` | prepare_data | build_synthetic_negatives, dedup_mqm | src, mt, score_norm, split, pair_hash |
| `sentence_da_augmented.parquet` | build_synthetic_negatives | train_semantic_pca, extract_features | + is_synthetic, synthetic_type |
| `hf_mqm_raw.parquet` | prepare_data | dedup_mqm | src, mt, score, pair_hash |
| `hf_mqm_dedup.parquet` | dedup_mqm | extract_features | src, mt, score |
| `wordlevel_train.parquet` | build_wordlevel | extract_features, train_span_model | src, mt, word_labels, split |
| `sentence_da_features.parquet` | extract_features | train_sentence_model, train_neural_head | src, mt, score_norm, split, + 97 признаков, word_logprobs |
| `wordlevel_features.parquet` | extract_features | (резерв) | src, mt, + признаки |
| `hf_mqm_features.parquet` | extract_features | train_sentence_model (external test) | src, mt, score, + признаки |

---

## §12. Веб-приложение и API

### 12.1 Стек

```
Backend:  FastAPI + Uvicorn
          src/app/server.py    — точка запуска, lifespan загрузки моделей
          src/app/api.py       — HTTP роутер
          src/app/models_state.py — синглтон состояния Predictor

Frontend: Jinja2 + Vanilla JS (без фреймворков)
          src/app/templates/index.html
          src/app/static/app.js
          src/app/static/style.css
```

### 12.2 Загрузка моделей

Модели загружаются **один раз** при старте `uvicorn` через `lifespan`. `ModelsState` хранит `Predictor` в `app.state.models_state`. Перезагрузка страницы браузером не вызывает повторную загрузку.

Переменные окружения при старте: `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` (офлайн-режим по умолчанию).

### 12.3 HTTP API

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/` | HTML интерфейс (Jinja2) |
| GET | `/api/status` | `{ready, status, models_loaded_at, error, feedback_count}` |
| POST | `/api/evaluate` | Оценка одного сегмента `{src, mt}` → `SentenceUIResult` + `elapsed_sec` |
| POST | `/api/evaluate_batch` | Батч до 50 пар → `list[SentenceUIResult]` |
| POST | `/api/feedback` | Сохранение ручной разметки → `{saved, feedback_id}` |
| POST | `/api/reload_models` | Горячая перезагрузка SentenceModel + SpanModel (без LaBSE/ruGPT) |

### 12.4 Горячая перезагрузка

`Predictor.reload_light_models()` — перезагружает XGBoost + SHAP explainer + SpanModel из файлов. LaBSE, ruGPT-3, MiniLM остаются в RAM. Вызывается через `POST /api/reload_models` после `finetune_from_feedback.py`.

### 12.5 Feedback

`src/app/feedback.py` — append-only JSONL `data/feedback/feedback.jsonl`. Каждая запись содержит `id`, `timestamp`, `src`, `mt`, `start_char`, `end_char`, `error_type`, `severity`, `features` (из кэша фронтенда), `word_logprobs`. Тяжёлые признаки **не пересчитываются** при сохранении feedback.

---

## §13. UI: поведение интерфейса

CAT-стиль таблица сегментов:
- Строки с `textarea` SRC и MT, score-badge с цветовой кодировкой.
- Оценка запускается автоматически при `blur` или через debounce 1.5 сек.
- Запросы **последовательны** (EvalQueue в JS) — не параллельные.
- MT переключается в режим div с HTML-подсветкой при готовой оценке; клик → снова textarea.

Панель деталей (sticky bottom):
- Score + CI + метка качества (5 уровней: Хороший / Приемлемый / Требует правки / Плохой / Очень плохой).
- Вклад по MQM-категориям (из `explanation`).
- Список найденных ошибок с tooltip.
- Кнопка «Отметить ошибку вручную» → mini-форма выбора типа и severity.

---

## §14. Эксплуатация и воспроизводимость

### 14.1 Переменные окружения

| Переменная | Назначение | Дефолт |
|---|---|---|
| `IMTQE_DATA_DIR` | Каталог данных | `./data` |
| `IMTQE_MODELS_DIR` | Каталог артефактов | `./models` |
| `IMTQE_LOG_DIR` | Каталог логов | `./logs` |
| `IMTQE_SEED` | Random seed | `42` |
| `IMTQE_COLAB` | Режим Colab (batch_size×2) | off |
| `HF_HUB_OFFLINE` | Только локальный HF-кэш | off |

### 14.2 Воспроизводимость

`src/determinism.py` — `seed_everything(seed)` устанавливает seed для Python, NumPy, PyTorch, transformers. Все скрипты вызывают `init_script_runtime()` из `src/bootstrap.py`.

### 14.3 Логирование

`src/logging_config.py` — stdout + `logs/imtqe.log`. Формат: `HH:MM:SS  LEVEL  logger  message`.

### 14.4 Деградация при ошибках загрузки

- Нет SHAP explainer → нулевые SHAP, нейтральная агрегация.
- Нет neural_head → explanation из SHAP.
- Нет `weights_mqm.npy` → единичные веса всех типов.
- Ошибка загрузки моделей → сервер стартует, `GET /api/status` возвращает `ready=false`.

### 14.5 CLI

`src/cli.py` — команда `poetry run imtqe <шаг>` запускает соответствующий скрипт из `scripts/` через `subprocess.call`. Доступные команды: `prepare-data`, `build-wordlevel`, `dedup-mqm`, `build-synthetic-negatives`, `train-semantic-pca`, `extract-features`, `train-sentence`, `train-span`, `train-neural-head`, `warmup-inference`, `pipeline`.

---

## §15. Структура каталогов

```
IMTQE/
├── pyproject.toml
├── data/
│   ├── raw/wordlevel/          # WMT21 *.src / *.mt / *.tags / *.tsv
│   └── processed/              # parquet по контракту §11
├── models/                     # артефакты обучения
│   ├── xgboost_sentence.model
│   ├── shap_explainer.pkl
│   ├── semantic_pca.pkl
│   ├── weights_mqm.npy
│   ├── neural_head.pt          # опционально
│   ├── neural_head_config.json # опционально
│   └── xlm_roberta_span/       # HF format
├── scripts/                    # шаги обучения и пайплайн
├── src/
│   ├── cli.py
│   ├── bootstrap.py
│   ├── settings.py
│   ├── config.py
│   ├── data_contract.py
│   ├── determinism.py
│   ├── logging_config.py
│   ├── predict.py              # Predictor, SentenceUIResult
│   ├── features/               # Блок 1
│   │   ├── extractor.py
│   │   ├── schema.py
│   │   ├── structural.py
│   │   ├── formatting.py
│   │   ├── linguistic.py
│   │   ├── semantic.py
│   │   ├── fluency.py
│   │   ├── neural.py
│   │   └── interactions.py
│   ├── models/                 # Блоки 2, 3, (нейронная голова)
│   │   ├── sentence_model.py
│   │   ├── span_model.py
│   │   └── neural_head.py
│   ├── interpretation/         # Блоки 4, 5, 6
│   │   ├── rules.py
│   │   ├── aggregation.py
│   │   ├── overall.py
│   │   └── explanation_loss.py
│   └── app/                    # FastAPI
│       ├── server.py
│       ├── api.py
│       ├── models_state.py
│       ├── feedback.py
│       ├── templates/
│       └── static/
└── tests/
```

---

## §16. Известные ограничения и расхождения с v2

| Аспект | v2 (требование) | v3 (факт) |
|---|---|---|
| Sentence-модель | NGBoost + параметрическая Beta | XGBoost + Beta-эвристика |
| Uncertainty | Var[NGBoost Beta] | Var[Beta(score·10, (1-score)·10)] |
| Признаки | 22 базовых | До 97: 22 + 64 semantic PCA + 11 interaction |
| Абзацный режим | Планировался | Не реализован |
| Paragraph score | Планировался | Нет |
| Межпредложенческая терминология | Планировалась | Нет |
| UI стек | Gradio (упоминался) | FastAPI + Jinja2 + Vanilla JS |
| Тип MQM | Нейросеть классификатор | Только детерминированные правила |

---

*Конец документа v3. При изменении пайплайна или контрактов данных обновляйте §§ 3–11 в первую очередь.*
