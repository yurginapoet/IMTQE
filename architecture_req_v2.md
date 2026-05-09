# Техническое описание системы MTQE
# Разделы 3, 4, 5, 6 — рабочий документ для реализации

---

# 3. Требования к системе

## 3.1 Функциональные требования

```
ФТ-1  Система принимает на вход пару (src, mt) где
      src — предложение или абзац на английском,
      mt  — его машинный перевод на русском.

ФТ-2  Система выдаёт скалярную оценку качества
      перевода q ∈ [0,1] на уровне предложения.

ФТ-3  Система выдаёт меру неопределённости оценки
      через параметры Beta распределения (α, β),
      позволяющую вычислить доверительный интервал.

ФТ-4  Система локализует ошибочные спаны в тексте mt
      с указанием позиции (start_idx, end_idx).

ФТ-5  Система классифицирует каждый найденный спан
      по severity: OK / BAD-minor / BAD-major.

ФТ-6  Система определяет тип каждой ошибки по
      типологии MQM из следующего списка:
        Accuracy:    Mistranslation, Omission,
                     Addition, Untranslated,
                     Hallucination
        Fluency:     Morphology, Agreement,
                     Spelling, LexicalChoice
        Terminology: WrongTerm, Inconsistency
        Locale:      NumberFormat, DateFormat,
                     Quotes, Currency
        Style:       Register

ФТ-7  Система предоставляет объяснение итоговой
      оценки через вклады признаков (SHAP),
      сопоставленные с категориями MQM.

ФТ-8  При подаче абзаца система разбивает его на
      предложения и анализирует межпредложенческую
      согласованность: Terminology/Inconsistency
      и Style/RegisterShift.

ФТ-9  Система предоставляет итоговую оценку абзаца
      по формуле MQM-style агрегации с обучаемыми
      весами типов ошибок.

ФТ-10 Система предоставляет веб-интерфейс в стиле
      CAT с подсветкой ошибочных токенов,
      цветовой кодировкой по severity и типу.
      Каждое предложение абзаца отображается
      в отдельной строке, что позволяет наглядно
      отслеживать терминологическое постоянство
      по всему тексту.
```

## 3.2 Нефункциональные требования

```
НФТ-1  Latency
       Режим sentence: ≤ 3 сек на предложение на CPU.
       Режим paragraph: ≤ 3 сек × N предложений.

НФТ-2  Интерпретируемость
       Каждое предсказание сопровождается:
       - SHAP вкладами для sentence score (NGBoost)
       - детерминированным правилом для типа ошибки
       Ни один тип ошибки не определяется нейросетью
       без явного лингвистического обоснования.
       Span модель (XLM-RoBERTa) определяет только
       severity (OK/BAD-minor/BAD-major) — тип MQM
       всегда назначается через rules.py.

НФТ-3  Портативность
       Инференс работает на CPU без GPU.
       Все зависимости устанавливаются через pip.
       Поддерживаемые ОС: Linux, macOS, Windows.

НФТ-4  Воспроизводимость
       Все правила типизации ошибок детерминированы.
       Зафиксированы random seed для всех моделей.
       Результаты воспроизводятся при повторном
       запуске на тех же данных.

НФТ-5  Расширяемость
       Добавление нового признака требует изменений
       только в feature_extractor.py.
       Добавление нового типа ошибки требует
       изменений только в rules.py.

НФТ-6  Память
       Суммарный объём моделей в RAM ≤ 4 GB.
       LaBSE:              1.8 GB
       ruGPT-3 Small:      0.5 GB
       XLM-RoBERTa base:   1.1 GB  (span model)
       spaCy + остальное:  0.2 GB
       Итого:             ~3.6 GB

НФТ-7  Языковая пара
       Система оптимизирована для EN→RU.
       Расширение на другие пары не предусмотрено
       в текущей версии.
```

## 3.3 Ограничения

```
ОГР-1  Языковая пара только EN→RU.

ОГР-2  Оценка без эталонного перевода (reference-free).
       Система не использует human reference.

ОГР-3  Источники данных (только публично доступные):

       Sentence model (обучение):
         HF DA — RicardoRei/wmt-da-human-evaluation
         EN-RU, 72 062 пары, split 85/10/5
         Шкала: Direct Assessment [0, 100] → нормализация
         в [0,1] min-max по train-сету.

       Span model (обучение):
         WMT21 word-level EN-RU, ~10k предложений
         (два домена: news + ted, ~20k строк итого)
         Источник: файлы *.src, *.mt, *.tags + *.tsv
         Пословные метки: OK / BAD-minor / BAD-major
         split 85/10/5

       Sentence model (внешний тест):
         HF MQM — RicardoRei/wmt-mqm-human-evaluation
         EN-RU, zscore колонка.
         Используется ТОЛЬКО как внешний тест после
         обучения — не входит ни в train, ни в val.
         Метрика: ранговая корреляция (Spearman ρ),
         не MSE, т.к. шкалы несовместимы.
         Перед использованием: удалить строки где
         пара (src, mt) совпадает с парами из
         HF DA train-сета (дедупликация по хэшу
         конкатенации src+mt).

ОГР-4  GPU не требуется на инференсе.
       GPU используется только при предвычислении
       признаков LaBSE и ruGPT-3 на этапе
       обучения (Google Colab T4).
       XLM-RoBERTa fine-tuning также на Colab T4.

ОГР-5  Система работает на уровне предложения и
       абзаца. Документный уровень не поддерживается.

ОГР-6  Register определяется через частотный словарь
       без дообучения классификатора формальности.

ОГР-7  Train/val/test split: 85/10/5 для всех датасетов.
       Splits стратифицированы по квантилям score
       чтобы распределение оценок было одинаковым
       во всех частях.
```

---

# 4. Проектирование архитектуры

## 4.1 Архитектурные драйверы

Три требования определяют все ключевые архитектурные решения:

**Интерпретируемость (НФТ-2)** — sentence score строится на явных признаках через NGBoost с SHAP объяснением. Типы ошибок определяются детерминированными правилами. XLM-RoBERTa используется только для локализации ошибок (severity), но не для их классификации.

**Работа на CPU (НФТ-3)** — тяжёлые модели (LaBSE, ruGPT-3) используются для вычисления признаков и кэшируются при обучении. На инференсе все три модели (LaBSE, ruGPT-3, XLM-RoBERTa) загружаются в RAM один раз при старте.

**Типизация ошибок по MQM (ФТ-6)** — требует модульной архитектуры где каждый тип ошибки имеет независимый детерминированный детектор в rules.py.

## 4.2 Общая архитектура

```
┌─────────────────────────────────────────────────────────┐
│                       ВХОД                              │
│         src (EN) + mt (RU) — предложение или абзац      │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│               БЛОК 0: ПРЕПРОЦЕССИНГ                     │
│  Если абзац: pysbd → список предложений src и mt        │
│  Выравнивание по количеству: если число предложений     │
│  совпадает — zip. Если не совпадает — vecalign.         │
│  Выход: список пар [(src₁,mt₁), (src₂,mt₂), ...]        │
└───────────────────────────┬─────────────────────────────┘
                            │  для каждой пары
                            ▼
┌─────────────────────────────────────────────────────────┐
│            БЛОК 1: ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ                 │
│                                                         │
│  structural.py  → 5 признаков  (без моделей, ~0 сек)    │
│  formatting.py  → 4 признака   (regex, ~0 сек)          │
│  linguistic.py  → 7 признаков  (spaCy, ~0.1 сек)        │
│  semantic.py    → 2 признака   (LaBSE, ~0.5 сек CPU)    │
│  fluency.py     → 4 признака   (ruGPT-3, ~0.3 сек CPU)  │
│                  ──────────────                         │
│  Итого: вектор 22 sentence-level признака               │
│                                                         │
│  Побочно из fluency.py:                                 │
│    word_logprobs[i] — logprob каждого слова mt          │
│    (агрегация BPE субтокенов → spaCy слова)             │
└──────────┬────────────────────────────────┬─────────────┘
           │ 22 признака                    │ word_logprobs
           ▼                                ▼
┌──────────────────────┐      ┌─────────────────────────────┐
│  БЛОК 2: SENTENCE    │      │  БЛОК 3: SPAN-LEVEL         │
│  MODEL               │      │  MODEL                      │
│                      │      │                             │
│  NGBoost             │      │  XLM-RoBERTa fine-tuned     │
│  распределение Beta  │      │  token classification       │
│  → (α̂, β̂)            │      │  вход: src + mt текст       │
│                      │      │  выход: метка каждого слова │
│  E[q]   = α̂/(α̂+β̂)    │      │  OK / BAD-minor / BAD-major │
│  Var[q] = uncertainty│      │                             │
│  CI₉₅ = [q_lo, q_hi] │      │  → смежные BAD → спаны      │
│                      │      │  → rules.py → тип MQM       │
│  SHAP → объяснение   │      │  → confidence = p(BAD)      │
│  по категориям MQM   │      │                             │
└──────────┬───────────┘      └─────────────┬───────────────┘
           │                                │
           └───────────────┬────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│              БЛОК 4: АБЗАЦНЫЙ АНАЛИЗ                    │
│  (только если входной текст — абзац)                    │
│                                                         │
│  Terminology/Inconsistency:                             │
│    src_lemma → Set[mt_lemma] через spaCy лемматизацию   │
│    по всем предложениям абзаца                          │
│    если |Set| > 1 → ошибка, указываем варианты          │
│                                                         │
│  Style/RegisterShift:                                   │
│    formal_ratio по каждому предложению                  │
│    если std(formal_ratios) > порог → ошибка             │
│                                                         │
│  Paragraph score:                                       │
│    Q = 100 - Σ(wₜ · pₛ · cᵢ) / Z                          │
└───────────────────────────┬─────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                       ВЫХОД                             │
│  score:               float [0,1]                       │
│  uncertainty:         float (Var из Beta)               │
│  confidence_interval: [q_low, q_high]                   │
│  explanation:         {mqm_category: shap_value}        │
│  spans: [{start_idx, end_idx, severity,                 │
│           error_type, confidence}]                      │
│  paragraph_errors: [{type, description}]  если абзац    │
│  paragraph_score:  float                  если абзац    │
└─────────────────────────────────────────────────────────┘
```

## 4.3 Описание подсистем

### Блок 0 — Препроцессинг

```
Назначение:  разбить абзац на предложения и выровнять
             пары src/mt по предложениям.

Вход:        src_text: str, mt_text: str

Выход:       List[Tuple[str, str]] — выровненные пары

Логика:
  1. Разбить src и mt на предложения через pysbd.
  2. Если число предложений совпадает — zip по индексу.
  3. Если не совпадает — vecalign для выравнивания
     (обрабатывает дробление и слияние предложений).

Если вход — одно предложение (не абзац):
  список содержит одну пару [(src, mt)].
```

### Блок 1 — Извлечение признаков

```
Назначение:  преобразовать текстовую пару (src, mt)
             в числовой вектор для Блока 2,
             и в пословные logprobs для Блока 3.

Вход:        src: str, mt: str

Выход:       features: np.array[22]  — sentence-level
             word_logprobs: List[float]  — per-word mt

Единица токенизации: spaCy слово (word).
  Все признаки считаются на уровне spaCy слов.
  ruGPT-3 работает с BPE субтокенами — агрегация
  word_logprob[i] = сумма logprob субтокенов слова i.
  Это стандартный подход (log вероятностей складываются
  = вероятности перемножаются для одного слова).
```

#### structural.py — 5 признаков, без моделей, ~0 сек

```
length_ratio
  = len(mt_words) / len(src_words)
  Что значит: насколько mt длиннее или короче src.
  Сильное отклонение от 1.0 сигнализирует об Omission
  (слишком короткий) или Addition (слишком длинный).

abs_length_diff
  = |len(mt_words) - len(src_words)|
  Что значит: абсолютная разница длин в словах.
  Дополняет length_ratio — важно для коротких предложений
  где ratio нестабилен.

token_count_diff
  = len(mt_words) - len(src_words)
  Что значит: знаковая разница. Отрицательное значение
  = перевод короче (возможное Omission), положительное
  = длиннее (возможное Addition).

src_length
  = len(src_words)
  Что значит: абсолютная длина источника. Нужен как
  контекст — короткие и длинные предложения имеют
  разные нормальные значения length_ratio.

mt_length
  = len(mt_words)
  Что значит: абсолютная длина перевода.
```

#### formatting.py — 4 признака, regex, ~0 сек

```
digit_match_ratio
  = |{числа src} ∩ {числа mt}| / max(|{числа src}|, 1)
  Что значит: доля чисел из src которые встречаются в mt.
  Важно: пересечение множеств, не разность.
  Низкое значение при наличии чисел в src → NumberFormat
  или пропущенное число.

punct_ratio
  = count_punct(mt) / max(count_punct(src), 1)
  Что значит: соотношение знаков препинания.
  Сильное отклонение от 1.0 → структурная проблема.

quotes_mismatch
  = 1 если в src есть кавычки и в mt английские кавычки,
    0 иначе.
  Что значит: в русском тексте должны быть «ёлочки»,
  а не "прямые". Бинарный флаг → Locale/Quotes.

date_format_error
  = 1 если в src найден паттерн даты (MM/DD/YYYY или
    Month DD, YYYY) и в mt та же строка без перевода,
    0 иначе.
  Что значит: дата скопирована без адаптации формата
  → Locale/DateFormat.
```

#### linguistic.py — 7 признаков, spaCy ru_core_news_sm, ~0.1 сек

```
oov_ratio
  = count(token.is_oov) / len(mt_words)
  Что значит: доля слов вне словаря spaCy.
  Высокое значение → возможные орфографические ошибки,
  транслитерации, нераспознанные термины.

type_token_ratio (TTR)
  = len(unique_words) / len(mt_words)
  Что значит: лексическое разнообразие. Очень низкое TTR
  при длинном тексте → монотонность или повторы.

avg_token_length
  = mean(len(word) for word in mt_words)
  Что значит: средняя длина слова в символах.
  Косвенный индикатор сложности лексики.

entity_overlap_ratio
  = |NER(src) ∩ mt_text| / max(|NER(src)|, 1)
  Что значит: доля именованных сущностей из src
  (персоны, организации, локации) которые встречаются
  в mt. Низкое значение → Accuracy/Untranslated или
  Terminology/WrongTerm.
  NER применяется к src через spaCy en_core_web_sm,
  поиск в mt — строковое вхождение.

agreement_errors
  = count нарушений согласования в mt через
    dependency parse (spaCy ru_core_news_sm).
  Что значит: число пар токенов где грамматическая
  связь (nsubj→verb, adj→noun) нарушает согласование
  по роду/числу/падежу → Fluency/Agreement.

syntax_depth
  = max глубина дерева зависимостей mt.
  Что значит: сложность синтаксической структуры.
  Аномально малая глубина при длинном предложении →
  возможная проблема парсинга или фрагментарный перевод.

formal_ratio
  = count(word in formal_vocab) / len(mt_words)
  Что значит: доля слов из частотного словаря
  формальной лексики. Используется для:
  (1) признак для NGBoost → Style/Register
  (2) межпредложенческий анализ в Блоке 4 (RegisterShift)
  formal_vocab — предопределённый словарь, не обучается.
```

#### semantic.py — 2 признака, LaBSE, ~0.5 сек CPU

```
Модель: sentence-transformers/LaBSE
Статус на инференсе: ЗАГРУЖАЕТСЯ при старте, не обучается.
LaBSE обучена на 109 языках включая EN и RU —
даёт сопоставимые эмбеддинги для src и mt.

cosine_similarity
  = cos(emb(src), emb(mt))
  Что значит: семантическое сходство src и mt в общем
  мультиязычном пространстве. Главный признак точности
  перевода. Значение близкое к 1.0 → смысл сохранён.
  Низкое значение → Accuracy/Mistranslation или
  Hallucination.

embedding_distance
  = ||emb(src) - emb(mt)||₂
  Что значит: евклидово расстояние. Дополняет косинус —
  косинус инвариантен к норме вектора, евклидово
  расстояние учитывает и угол и длину. Оба признака
  вместе дают NGBoost более полную картину.
```

#### fluency.py — 4 sentence-level признака + word_logprobs, ruGPT-3 Small, ~0.3 сек CPU

```
Модель: sberbank-ai/rugpt3small_based_on_gpt2
Статус на инференсе: ЗАГРУЖАЕТСЯ при старте, не обучается.
Авторегрессионная языковая модель русского языка.
Оценивает насколько "по-русски" звучит mt,
не зная ничего о src (чисто fluency).

ТОКЕНИЗАЦИЯ И АГРЕГАЦИЯ:
  ruGPT-3 использует BPE — одно слово может быть
  разбито на несколько субтокенов.
  Например: "извиняться" → ["из", "вин", "яться"]
  Агрегация к spaCy слову:
    word_logprob[i] = Σ logP(subtoken_j) для всех j
                      принадлежащих слову i.
  Это корректно т.к. logP(слово) = Σ logP(субтокенов)
  при авторегрессионном разложении.
  Маппинг BPE → spaCy слова строится через
  сравнение char offsets.

perplexity
  = exp(-(1/n) · Σᵢ log P(tᵢ | t₁,...,tᵢ₋₁))
  Что значит: насколько ruGPT-3 "удивляется" тексту mt.
  Низкая perplexity → естественный русский текст.
  Высокая perplexity → неестественные конструкции,
  кальки с английского → Fluency/LexicalChoice или
  Fluency/Morphology.

mean_log_prob
  = (1/n) · Σᵢ log P(tᵢ | t₁,...,tᵢ₋₁)
  Что значит: среднее логарифмическое правдоподобие
  токенов. Отрицательное число, чем ближе к 0 — тем
  лучше. Дополняет perplexity (perplexity = exp(-mean)).

token_ppl_variance
  = (1/n) · Σᵢ (log P(tᵢ|...) - mean_log_prob)²
  Что значит: разброс вероятностей по токенам.
  Высокая дисперсия при среднем mean_log_prob → в тексте
  есть отдельные очень неожиданные токены на фоне
  нормального текста. Именно такие места — кандидаты
  на ошибки. Используется и как sentence-level признак
  и косвенно указывает на проблемные слова для Блока 3.

min_token_log_prob
  = min logP(tᵢ) по всем токенам mt
  Что значит: самый "удивительный" токен в предложении.
  Если min очень низкий при нормальном mean → в тексте
  есть один аномальный токен → локальная ошибка.

word_logprobs (побочный выход, не sentence-level признак)
  = List[float], len = len(mt_words)
  word_logprobs[i] = агрегированный logprob слова i.
  Передаётся напрямую в Блок 3 (XLM-RoBERTa).
  НЕ входит в вектор 22 признаков для NGBoost.
```

### Блок 2 — Sentence-level модель

```
Назначение:  предсказать качество перевода на уровне
             предложения с мерой неопределённости.

Вход:        features: np.array[22]

Выход:       alpha:    float  > 0
             beta:     float  > 0
             score:    float = alpha/(alpha+beta) ∈ [0,1]
             uncertainty: float = Var[Beta(alpha,beta)]
             CI₉₅:    [q_low, q_high]
             shap_values: np.array[22]

Модель:      NGBoost (ngboost.NGBRegressor)
             с Beta распределением (ngboost.distns.Beta)
             NGBoost нативно предсказывает параметры
             распределения — в отличие от XGBoost,
             не требует двух отдельных моделей для (α, β).

Обучение:
  Данные: HF DA EN-RU, 72 062 пары
  Целевая переменная: score нормализованный в [0,1]
    min-max нормализация только по train-сету:
    score_norm = (score - min_train) / (max_train - min_train)
    eps = 1e-4, клиппинг в [eps, 1-eps] (Beta требует (0,1))
  Split: 85% train / 10% val / 5% test
  Стратификация: по квантилям score (5 бинов)
  RANDOM_SEED = 42

Asymmetric loss (кастомный scoring в NGBoost):
  Ошибки на плохих переводах важнее.
  weight_i = w_high если score_i < τ (плохой перевод)
             w_low  если score_i ≥ τ (хороший перевод)
  τ подбирается на val по метрике Pearson.
  w_high = 3.0, w_low = 1.0 (начальные значения).

Валидация:
  Основная метрика: Pearson r на HF DA test (5%)
  Внешний тест: Spearman ρ на HF MQM zscore
    (после дедупликации пересечений с DA train)

SHAP:
  shap.TreeExplainer(ngboost_model) — точные SHAP
  значения (не аппроксимация) для tree-based моделей.
  shap_values[i] → категория MQM через таблицу маппинга
  (см. раздел 5.5).
  Инференс: pkl файл, CPU, <10 мс.
```

### Блок 3 — Span-level модель

```
Назначение:  для каждого слова mt определить severity
             (OK/BAD-minor/BAD-major), объединить
             смежные BAD слова в спаны, определить
             тип ошибки через rules.py.

Вход:        src: str
             mt:  str
             word_logprobs: List[float]  из Блока 1

Выход:       List[SpanError]:
               start_idx:  int  (индекс spaCy слова)
               end_idx:    int
               severity:   OK | BAD-minor | BAD-major
               error_type: str (из типологии MQM)
               confidence: float = p(BAD)

Модель:      xlm-roberta-base, fine-tuned
             задача: token classification (3 класса)
             вход: "[CLS] src [SEP] mt [SEP]"
             предсказание только для токенов mt части.
             XLM-RoBERTa видит полный контекст src+mt
             для каждого токена mt — это ключевое
             преимущество перед ручными признаками.

Токенизация в Блоке 3:
  XLM-RoBERTa использует SentencePiece токенизатор.
  Один spaCy слово → один или несколько SentencePiece
  субтокенов. Маппинг строится через char offsets.
  Предсказание слова = предсказание первого субтокена
  (стандартный подход для NER/token classification).

Обучение XLM-RoBERTa:

  ДАННЫЕ — WMT21 word-level (детально в разделе 4.6):
    Итоговый датасет после сборки: ~10k предложений
    с пословными метками OK/BAD-minor/BAD-major.
    Split: 85% train / 10% val / 5% test.
    RANDOM_SEED = 42.

  Параметры fine-tuning:
    lr = 2e-5, epochs = 5, batch_size = 16
    early stopping по val F1(BAD-major)
    weighted loss: BAD-major=5, BAD-minor=2, OK=1
    (класс BAD-major редкий но критичный)

  Метрики:
    F1(BAD-major) — основная
    F1(BAD-minor)
    Accuracy (OK)

После предсказания severity:
  1. Смежные BAD слова объединяются в спан.
     Граница спана: первое и последнее BAD слово
     в непрерывной последовательности.
  2. rules.py определяет тип MQM для каждого спана.
     Вход rules.py: span текст + признаки слов спана
     из Блока 1 (is_number, is_entity, oov и т.д.)
     Выход: строка типа из типологии MQM.

Инференс: xlm-roberta-base на CPU ~1-2 сек/предложение.
```

### Блок 4 — Абзацный анализ

```
Назначение:  обнаружить межпредложенческие проблемы
             невидимые на уровне одного предложения.

Вход:        List[SentenceResult] — результаты Блоков
             2 и 3 для каждого предложения абзаца.
             Также: List[(src_i, mt_i)] — сами тексты.

Выход:       paragraph_score: float
             paragraph_errors: List[ParagraphError]

Terminology/Inconsistency (детерминировано):
  Для каждой пары (src_i, mt_i):
    Лемматизировать src_i через spaCy en_core_web_sm.
    Лемматизировать mt_i через spaCy ru_core_news_sm.
    Для каждой леммы src_lemma найти все mt_lemma
    которые встречались рядом с ней по всем предложениям.
  Словарь: src_lemma → Set[mt_lemma] по всему абзацу.
  Если |Set| > 1 → ошибка Terminology/Inconsistency.
  Сообщение: "слово X переведено как Y и Z в разных
  предложениях".
  Примечание: spaCy уже загружена в Блоке 1 → 0 доп. памяти.

Style/RegisterShift (детерминировано):
  formal_ratio по каждому предложению (из Блока 1).
  Если std(formal_ratios) > threshold → RegisterShift.
  threshold подбирается вручную (рекомендуется 0.15).

Paragraph score:
  Q = 100 - Σᵢ (wₜᵢ · pₛᵢ · cᵢ) / Z

  wₜ  — вес типа ошибки, вектор ∈ ℝ^|T|
  pₛ  — штраф severity: BAD-major=5, BAD-minor=1
  cᵢ  — confidence XLM-RoBERTa для спана i
  Z   = суммарное число слов mt в абзаце

  Оптимизация wₜ:
    argmax_{wₜ} Spearman(Q(wₜ), Q_human)
    Q_human — из WMT21 TSV (sentence-level zscore)
    метод: scipy.optimize.minimize (L-BFGS-B)
    ограничение: wₜ ≥ 0
    сохраняется в: models/weights_mqm.npy
```

## 4.4 Архитектурные решения

**Почему NGBoost а не XGBoost для sentence score:**
XGBoost не предсказывает параметры распределения нативно — потребовались бы два отдельных дерева что усложняет архитектуру и ломает SHAP. NGBoost напрямую обучает параметры Beta распределения через натуральный градиентный бустинг. SHAP через TreeExplainer работает с NGBoost так же как с XGBoost.

**Почему Beta distribution а не точечная оценка:**
скалярная оценка не различает случаи когда признаки согласованы (высокая уверенность) и когда противоречат друг другу (низкая уверенность). Beta параметризует именно это различие естественным образом для q ∈ [0,1].

**Почему XLM-RoBERTa для span detection:**
XGBoost на пословных признаках без контекста не видит что слово "Bank" переведено неверно потому что соседние слова указывают на финансовый а не географический контекст. XLM-RoBERTa обрабатывает src+mt совместно и использует full attention по всей паре предложений — это стандарт для word-level QE задачи (WMT22 winning systems).

**Почему правила для типов ошибок а не классификатор:**
классификатор типов требует размеченных данных по каждому типу отдельно. WMT21 содержит severity (OK/BAD) но категории MQM распределены неравномерно и некоторые типы встречаются единицами. Детерминированные правила поверх признаков воспроизводимы, объяснимы и не требуют доразметки.

**Почему SimAlign убран:**
SimAlign использует XLM-R внутри и не предоставляет публичный доступ к модели. Использование shared weights невозможно без переписывания библиотеки. Задачи для которых нужен alignment (Terminology/Inconsistency в Блоке 4) решаются через spaCy лемматизацию без явного word alignment. Убирает ~1GB из бюджета памяти.

**Почему HF MQM только для внешнего теста:**
DA и MQM шкалы математически несовместимы. DA — прямая оценка [0,100] нормализованная по аннотатору. MQM — штрафная система (0 = идеально, уходит в минус). Смешение при обучении ломает целевую переменную. Ранговая корреляция (Spearman ρ) позволяет сравнивать предсказания с MQM без преобразования шкал.

**Почему модели загружаются один раз:**
LaBSE + ruGPT-3 + XLM-RoBERTa суммарно ~3.4 GB.
Загрузка при каждом запросе заняла бы 15-20 секунд.
При старте сервера модели загружаются в RAM и остаются.

## 4.5 Предобработка WMT21 word-level датасета

WMT21 word-level данные поставляются в виде отдельных файлов которые нужно аккуратно объединить. Это критически важный этап — ошибки здесь ведут к неверным меткам.

### Структура файлов

```
Для каждого домена (news, ted):
  mqm_dev2021_enru.<domain>.src   — исходные предложения EN
  mqm_dev2021_enru.<domain>.mt    — переводы RU
  mqm_dev2021_enru.<domain>.tags  — пословные теги

  Формат .tags файла (одна строка = одно предложение):
  "OK OK BAD OK BAD BAD OK"
  Теги разделены пробелами, число тегов = число слов mt.
  Индекс строки в .tags = индекс строки в .mt = seg_id.

  Формат .tsv файла:
  system | doc | doc_id | seg_id | rater | source | target |
  category | severity
  
  ВАЖНО: seg_id в TSV = (номер строки в .tags файле) + 1
  (TSV 1-indexed, файлы 0-indexed).
  Одному seg_id может соответствовать:
  - 0 строк в TSV (нет ошибок, все теги OK)
  - 1 строка (одна ошибка)
  - N строк (N ошибок в разных местах предложения)
  Строка "No-error / No-error" в TSV = предложение без ошибок.
```

### Алгоритм объединения

```
Шаг 1: Загрузка файлов для каждого домена.
  src_lines  = read_lines(*.src)   # List[str]
  mt_lines   = read_lines(*.mt)    # List[str]
  tags_lines = read_lines(*.tags)  # List[str]

  Проверка: len(src_lines) == len(mt_lines) == len(tags_lines)
  Если нет → ошибка на этапе загрузки данных.

Шаг 2: Построить DataFrame уровня предложения.
  df = pd.DataFrame({
    "seg_id": range(len(src_lines)),  # 0-indexed
    "src": src_lines,
    "mt":  mt_lines,
    "tags": tags_lines,  # "OK OK BAD OK" как строка
    "domain": "news" | "ted"
  })

Шаг 3: Загрузить TSV и нормализовать seg_id.
  tsv = pd.read_csv(*.tsv, sep="\t")
  tsv["seg_id_0"] = tsv["seg_id"] - 1  # конвертация в 0-indexed
  # Убрать строки "No-error" — они не несут информации об ошибках
  tsv = tsv[tsv["category"] != "No-error"]

Шаг 4: Агрегировать severity по предложению.
  Для каждого seg_id взять максимальный severity:
    "Major" → BAD-major
    "Minor" → BAD-minor
    (если несколько ошибок — берём наибольший severity)
  
  sentence_severity = tsv.groupby("seg_id_0")["severity"]\
    .apply(lambda x: "BAD-major" if "Major" in x.values
                     else "BAD-minor")

Шаг 5: Развернуть теги в пословные метки.
  Для каждого предложения i:
    word_tags = df.loc[i, "tags"].split()
    # word_tags[j] ∈ {"OK", "BAD"}
    
  ПРОБЛЕМА: .tags файл содержит только OK/BAD,
  без разбивки на minor/major.
  
  РЕШЕНИЕ: обогащение через TSV.
  Если seg_id есть в sentence_severity:
    все BAD токены получают метку из sentence_severity
    (т.е. если в предложении есть Major ошибка —
    все BAD токены помечаются BAD-major)
  Если seg_id нет в TSV:
    все BAD токены → BAD-minor (консервативная оценка)
  OK токены всегда остаются OK.

  Ограничение: точная пословная severity неизвестна
  (TSV не всегда указывает конкретные токены).
  Это сознательное упрощение — приемлемо для обучения.

Шаг 6: Собрать итоговую схему.
  Итоговый DataFrame строка = одно предложение:
  {
    seg_id:         int
    domain:         str  ("news" | "ted")
    src:            str
    mt:             str
    word_labels:    List[str]  ["OK","OK","BAD-major",...]
    n_words:        int  (число слов mt)
    has_error:      bool
    max_severity:   str  ("OK" | "BAD-minor" | "BAD-major")
  }

Шаг 7: Объединить домены.
  df_final = pd.concat([df_news, df_ted], ignore_index=True)
  
  Проверки перед сохранением:
  - Для каждой строки: len(word_labels) == n_words
  - Нет NaN в word_labels
  - Распределение классов: вывести % OK/BAD-minor/BAD-major

Шаг 8: Сохранить.
  df_final.to_parquet("data/processed/wordlevel_train.parquet")
```

### Баланс классов

```
Ожидаемое распределение (типично для WMT QE данных):
  OK:        ~75-80% слов
  BAD-minor: ~15-20% слов
  BAD-major: ~3-5%  слов

BAD-major редкий но критичный — используем weighted loss
при обучении XLM-RoBERTa (BAD-major weight=5).
```

## 4.6 Дедупликация перед внешним тестом

```
Цель: убедиться что HF MQM не содержит пар (src, mt)
которые уже есть в HF DA train-сете.

Алгоритм:
  1. Для каждой строки HF DA train вычислить:
     key = sha256(src.strip() + "|||" + mt.strip())
  2. Построить множество: da_train_keys = Set[key]
  3. Для каждой строки HF MQM вычислить тот же key.
  4. Удалить строки HF MQM где key ∈ da_train_keys.
  5. Логировать: сколько строк удалено.

Порог: если удалено >5% HF MQM — предупреждение,
возможна системная утечка, требует анализа.

Сохранить очищенный датасет:
  hf_mqm_dedup.parquet — используется ТОЛЬКО для
  финального внешнего теста.
```

## 4.7 Ограничения архитектуры

```
- Качество sentence score ниже CometKiwi (~0.65-0.72 vs
  ~0.85 Pearson) из-за отказа от нейросетевых регрессоров.
  Сознательный компромисс в пользу интерпретируемости.

- Severity обогащение WMT21 (шаг 5 в 4.5) неточно:
  все BAD токены предложения получают один severity.
  Реальная пословная severity неизвестна.

- Register определяется через словарь без учёта
  контекста — возможны ложные срабатывания.

- Терминологическая согласованность через лемматизацию
  без word alignment — возможны ложные срабатывания
  для многозначных слов.

- Система не обновляется онлайн — требует переобучения
  при добавлении новых данных.
```

---

# 5. Математическая модель

## 5.1 Вектор признаков

**f** = (f₁, f₂, ..., f₂₂) ∈ ℝ²²

```
Группа         Признаки (5)              Индексы
Structural:    length_ratio              f₁
               abs_length_diff           f₂
               token_count_diff          f₃
               src_length                f₄
               mt_length                 f₅

Formatting:    digit_match_ratio         f₆
               punct_ratio               f₇
               quotes_mismatch           f₈
               date_format_error         f₉

Linguistic:    oov_ratio                 f₁₀
               type_token_ratio          f₁₁
               avg_token_length          f₁₂
               entity_overlap_ratio      f₁₃
               agreement_errors          f₁₄
               syntax_depth              f₁₅
               formal_ratio              f₁₆

Semantic:      cosine_similarity         f₁₇
               embedding_distance        f₁₈

Fluency:       perplexity                f₁₉
               mean_log_prob             f₂₀
               token_ppl_variance        f₂₁
               min_token_log_prob        f₂₂
```

## 5.2 Sentence-level модель — Beta prediction

```
NGBoost обучает два натуральных параметра:
  η = (η₁, η₂) = NGBoost(f)

Параметры Beta распределения:
  α = softplus(η₁) > 0
  β = softplus(η₂) > 0
  (softplus гарантирует положительность)

Предсказания:
  E[q]   = α / (α + β)                    — оценка качества

  Var[q] = αβ / ((α+β)²(α+β+1))           — неопределённость

  CI₉₅  = [B⁻¹(0.025; α,β), B⁻¹(0.975; α,β)]
           где B⁻¹ — квантильная функция Beta

Интерпретация uncertainty:
  Низкий Var при высоком score → уверенно хороший перевод
  Низкий Var при низком score  → уверенно плохой перевод
  Высокий Var                  → признаки противоречат
                                 друг другу, нужна проверка
```

## 5.3 Функция потерь NGBoost с асимметричными весами

```
NGBoost минимизирует отрицательное log-правдоподобие
Beta распределения с весами примеров:

L = -Σᵢ wᵢ · log Beta(qᵢ; αᵢ, βᵢ)

wᵢ = w_high  если qᵢ < τ    (плохой перевод, важнее)
     w_low   если qᵢ ≥ τ    (хороший перевод)

w_high = 3.0, w_low = 1.0 (стартовые значения)
τ — порог, подбирается на val по Pearson r

Смысл: ошибки на плохих переводах важнее для задачи
постредактирования — редактор тратит время именно
на плохие переводы.
```

## 5.4 MQM-style агрегация с обучаемыми весами

```
Q = 100 − Σᵢ (wₜᵢ · pₛᵢ · cᵢ) / Z

wₜ ∈ ℝ^|T|  — вектор весов типов ошибок (обучается)
pₛ           — штраф severity: BAD-major=5, BAD-minor=1
cᵢ ∈ [0,1]  — confidence XLM-RoBERTa для спана i
Z            = суммарное число слов mt в абзаце

Оптимизация wₜ:
  argmax_{wₜ≥0} Spearman(Q(wₜ), Q_human)
  Q_human — zscore из WMT21 TSV (val часть)
  метод: scipy.optimize.minimize, L-BFGS-B
  сохраняется в: models/weights_mqm.npy
```

## 5.5 SHAP интерпретация

```
q̂ = E[q̂] + Σᵢ φᵢ
где Σᵢ φᵢ = q̂ − E[q̂]  — условие эффективности SHAP

Маппинг φᵢ → категория MQM:

Признак               → Категория MQM
──────────────────────────────────────
cosine_similarity     → Accuracy
embedding_distance    → Accuracy
length_ratio          → Accuracy (Omission/Addition)
abs_length_diff       → Accuracy (Omission/Addition)
token_count_diff      → Accuracy (Omission/Addition)
entity_overlap_ratio  → Terminology
perplexity            → Fluency
mean_log_prob         → Fluency
token_ppl_variance    → Fluency
min_token_log_prob    → Fluency
agreement_errors      → Fluency/Morphology
syntax_depth          → Fluency/Syntax
oov_ratio             → Fluency/Spelling
type_token_ratio      → Fluency/LexicalChoice
avg_token_length      → Fluency/LexicalChoice
formal_ratio          → Style/Register
digit_match_ratio     → Locale
quotes_mismatch       → Locale
date_format_error     → Locale
punct_ratio           → Locale
src_length            → (контекстный, без MQM категории)
mt_length             → (контекстный, без MQM категории)

Агрегация по категориям:
  category_impact[c] = Σ φᵢ для всех признаков с категорией c
  Выводится пользователю как объяснение score.
```

---

# 6. Реализация

## 6.1 Технологический стек

```
Язык:              Python 3.12

Признаки:
  sentence-splitter: pysbd
  sentence-align:    vecalign  (только для абзацного режима)
  морфология:        pymorphy3
  синтаксис/NER RU:  spaCy (ru_core_news_sm)
  синтаксис/NER EN:  spaCy (en_core_web_sm)
  семантика:         sentence-transformers (LaBSE)
  fluency:           transformers (ruGPT-3 Small)
                       sberbank-ai/rugpt3small_based_on_gpt2

Модели:
  sentence-level:    ngboost (NGBRegressor + Beta)
  span-level:        transformers (xlm-roberta-base,
                       fine-tuned token classification)
  SHAP:              shap (TreeExplainer)
  оптимизация wₜ:    scipy.optimize

Данные:
  загрузка HF:       datasets (HuggingFace)
  хранение:          pandas + parquet (pyarrow)
  дедупликация:      hashlib (sha256)

API:                 fastapi + uvicorn
Интерфейс:          gradio (CAT-style, предложение в строке)
Окружение:           conda / venv
Обучение (тяжёлое): Google Colab T4
  - предвычисление признаков LaBSE, ruGPT-3
  - fine-tuning XLM-RoBERTa
```

## 6.2 Структура проекта

```
qe_system/
│
├── data/
│   ├── raw/
│   │   ├── wmt_da/                 ← HF DA (скачивается)
│   │   ├── wmt_mqm/                ← HF MQM (скачивается)
│   │   └── wordlevel/              ← WMT21 файлы
│   │       ├── mqm_dev2021_enru.news.src
│   │       ├── mqm_dev2021_enru.news.mt
│   │       ├── mqm_dev2021_enru.news.tags
│   │       ├── mqm_dev2021_enru.news.tsv
│   │       ├── mqm_dev2021_enru.ted.src
│   │       ├── mqm_dev2021_enru.ted.mt
│   │       ├── mqm_dev2021_enru.ted.tags
│   │       └── mqm_dev2021_enru.ted.tsv
│   └── processed/
│       ├── sentence_da.parquet       ← HF DA, нормализован
│       ├── sentence_da_features.parquet  ← + все 22 признака
│       ├── hf_mqm_dedup.parquet      ← HF MQM без утечек
│       ├── wordlevel_train.parquet   ← WMT21 объединённый
│       └── wordlevel_features.parquet ← + word_logprobs
│
├── scripts/
│   ├── prepare_data.py             ← загрузка + нормализация
│   ├── build_wordlevel.py          ← объединение WMT21 (4.5)
│   ├── dedup_mqm.py                ← дедупликация (4.6)
│   └── extract_features.py         ← Блок 1 в batch режиме
│
├── src/
│   ├── features/
│   │   ├── structural.py           ← 5 признаков
│   │   ├── formatting.py           ← 4 признака
│   │   ├── linguistic.py           ← 7 признаков (spaCy)
│   │   ├── semantic.py             ← 2 признака (LaBSE)
│   │   ├── fluency.py              ← 4 признака + word_logprobs
│   │   └── extractor.py            ← FeatureExtractor, nlp.pipe
│   │
│   ├── models/
│   │   ├── sentence_model.py       ← NGBoost + Beta + SHAP
│   │   └── span_model.py           ← XLM-RoBERTa fine-tune
│   │
│   ├── interpretation/
│   │   ├── shap_explainer.py       ← SHAP → MQM маппинг
│   │   ├── rules.py                ← детерминированные правила
│   │   └── aggregation.py          ← MQM агрегация + wₜ
│   │
│   ├── preprocessing/
│   │   ├── sentence_splitter.py    ← pysbd
│   │   └── sentence_aligner.py     ← vecalign
│   │
│   ├── paragraph.py                ← Блок 4
│   ├── tokenization.py             ← BPE → spaCy агрегация
│   └── predict.py                  ← единая точка входа
│
├── notebooks/
│   ├── 01_prepare_data.ipynb       ← загрузка, нормализация
│   ├── 02_build_wordlevel.ipynb    ← сборка WMT21
│   ├── 03_fast_features.ipynb      ← structural/formatting/ling
│   ├── 04_heavy_features_colab.ipynb  ← LaBSE + ruGPT-3 (GPU)
│   ├── 05_train_sentence_model.ipynb  ← NGBoost
│   ├── 06_train_span_model.ipynb   ← XLM-RoBERTa fine-tuning
│   └── 07_evaluation.ipynb         ← метрики + внешний тест
│
├── app/
│   ├── api.py                      ← FastAPI
│   └── interface.py                ← Gradio CAT demo
│
├── models/                         ← сохранённые модели
│   ├── ngboost_sentence.pkl
│   ├── xlm_roberta_span/           ← директория HF формата
│   ├── shap_explainer.pkl
│   └── weights_mqm.npy
│
├── tests/
│   └── test_predict.py
│
└── requirements.txt
```

## 6.3 Особенности реализации

**Загрузка моделей один раз:**
```python
# api.py — при старте сервера
@app.on_event("startup")
async def load_models():
    app.state.spacy_ru  = spacy.load("ru_core_news_sm")
    app.state.spacy_en  = spacy.load("en_core_web_sm")
    app.state.labse     = SentenceTransformer("LaBSE")
    app.state.gpt_tok   = AutoTokenizer.from_pretrained(
                            "sberbank-ai/rugpt3small_based_on_gpt2")
    app.state.gpt_model = AutoModelForCausalLM.from_pretrained(
                            "sberbank-ai/rugpt3small_based_on_gpt2")
    app.state.xlmr      = pipeline(
                            "token-classification",
                            model="models/xlm_roberta_span",
                            device=-1)  # CPU
    app.state.ngboost   = joblib.load("models/ngboost_sentence.pkl")
    app.state.shap_exp  = joblib.load("models/shap_explainer.pkl")
    app.state.wt        = np.load("models/weights_mqm.npy")
```

**Батчинг при предвычислении признаков (Colab):**
```python
# Векторизация через nlp.pipe вместо iterrows:
docs = list(nlp.pipe(df["mt"].tolist(), batch_size=64))
# LaBSE батчами:
embeddings = labse.encode(texts, batch_size=64, show_progress_bar=True)
# ruGPT-3 батчами:
# batch_size=16 из-за контекстного окна и памяти GPU
```

**Кэширование по этапам:**
каждый этап сохраняется в parquet немедленно после вычисления. Если Colab сессия упала — пересчёт начинается с последнего сохранённого чекпоинта.

```python
if not Path("data/processed/sentence_da_features.parquet").exists():
    # вычислить признаки
    df_features.to_parquet("data/processed/sentence_da_features.parquet")
else:
    df_features = pd.read_parquet("data/processed/sentence_da_features.parquet")
```

**BPE → spaCy агрегация (tokenization.py):**
```python
# Логика: для каждого spaCy слова найти его char span,
# найти все BPE субтокены в этом span,
# просуммировать их logprob.
# Реализация через tokenizer.encode_plus(return_offsets_mapping=True)
```

**Детерминированность:**
```python
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
# NGBoost:
ngb = NGBRegressor(Dist=Beta, random_state=RANDOM_SEED)
# XLM-RoBERTa:
set_seed(RANDOM_SEED)  # transformers.set_seed
```

**Порядок выполнения скриптов:**
```
1. scripts/prepare_data.py        → sentence_da.parquet
                                     hf_mqm_raw.parquet
2. scripts/build_wordlevel.py     → wordlevel_train.parquet
3. scripts/dedup_mqm.py           → hf_mqm_dedup.parquet
4. scripts/extract_features.py    → sentence_da_features.parquet
                                     wordlevel_features.parquet
   (шаг 4 запускать на Colab T4)
5. notebooks/05_train_sentence_model.ipynb → ngboost_sentence.pkl
6. notebooks/06_train_span_model.ipynb     → xlm_roberta_span/
7. notebooks/07_evaluation.ipynb           → финальные метрики
```
