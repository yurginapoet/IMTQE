let segments = [];
let nextId = 1;
let currentSegmentId = null;
let activeErrorHighlight = null;
let modelReady = false;
/** @type {HTMLElement | null} */
let tokenTooltipEl = null;

const tbody = document.getElementById('segments-tbody');
const addBtn = document.getElementById('add-segment-btn');
const evalAllBtn = document.getElementById('eval-all-btn');
const statusDiv = document.getElementById('status');
const detailPanel = document.getElementById('detail-panel');
const detailTitle = document.getElementById('detail-segment-title');
const detailContent = document.getElementById('detail-content');
const closeDetailBtn = document.getElementById('close-detail-btn');

/**
 * Короткие названия и развёрнутые подсказки для переводчика (EN→RU).
 * Ключи точно совпадают с именами признаков из schema.py / FeatureExtractor.
 */
const FEATURE_REGISTRY = {

    // ─── НЕПЕРЕВЕДЁННЫЕ ФРАГМЕНТЫ ─────────────────────────────
    untranslated_ratio: {
        group: 'Непереведённые фрагменты',
        label: 'Доля непереведённых слов',
        hint: 'Какую часть слов в переводе составляют полностью английские слова (только латиница без кириллицы). Один из самых важных сигналов о плохом переводе.',
    },
    latin_ratio: {
        group: 'Непереведённые фрагменты',
        label: 'Доля латинских символов',
        hint: 'Общая доля латинских букв во всём тексте перевода. Высокое значение говорит о том, что перевод содержит много непереведённых кусков.',
    },

    // ─── СМЫСЛОВАЯ ТОЧНОСТЬ ───────────────────────────────────
    cosine_similarity: {
        group: 'Смысловая точность',
        label: 'Сохранность смысла',
        hint: 'Насколько близок смысл перевода к оригиналу по семантической модели LaBSE. Низкое значение — возможна потеря или искажение ключевой информации.',
    },
    embedding_distance: {
        group: 'Смысловая точность',
        label: 'Смысловой сдвиг',
        hint: 'Насколько далеко перевод ушёл от оригинала в смысловом пространстве. Чем больше значение — тем вероятнее серьёзное искажение смысла.',
    },
    named_entity_missing_ratio: {
        group: 'Смысловая точность',
        label: 'Пропуск имён и сущностей',
        hint: 'Доля имён собственных, организаций, дат и ключевых терминов из оригинала, которые не найдены в переводе.',
    },
    entity_overlap_ratio: {
        group: 'Смысловая точность',
        label: 'Сохранность ключевых сущностей',
        hint: 'Какую долю важных именованных сущностей из оригинала удалось сохранить в переводе.',
    },

    // ─── ГРАММАТИКА И МОРФОЛОГИЯ ───────────────────────────────
    morphology_error_rate: {
        group: 'Грамматика',
        label: 'Морфологические ошибки',
        hint: 'Доля слов, которые pymorphy2 не смог нормально разобрать. Сильный сигнал грамматических и орфографических ошибок.',
    },
    agreement_errors: {
        group: 'Грамматика',
        label: 'Ошибки согласования',
        hint: 'Количество нарушений согласования по роду, числу и падежу (эвристика на основе spaCy).',
    },

    // ─── ОРФОГРАФИЯ И ЛЕКСИКА ──────────────────────────────────
    oov_ratio: {
        group: 'Орфография и лексика',
        label: 'Необычные слова',
        hint: 'Доля слов, которые языковая модель считает редкими или нетипичными. Часто указывает на опечатки или неудачные словоформы.',
    },
    avg_word_rank: {
        group: 'Орфография и лексика',
        label: 'Средняя редкость слов',
        hint: 'Средний частотный ранг слов в переводе. Чем выше — тем больше редких и потенциально проблемных слов используется.',
    },

    // ─── СТИЛЬ И ПОВТОРЫ ───────────────────────────────────────
    repetition_ratio: {
        group: 'Стиль и повторы',
        label: 'Повторы слов',
        hint: 'Доля слов, которые повторяются слишком часто. Сигнал тавтологии, калькирования и бедной лексики.',
    },
    formal_ratio: {
        group: 'Стиль и повторы',
        label: 'Формальность стиля',
        hint: 'Насколько формальный или разговорный стиль используется в переводе. Важно для соблюдения регистра.',
    },

    // ─── ЛОКАЛИЗАЦИЯ И ФОРМАТ ──────────────────────────────────
    digit_match_ratio: {
        group: 'Локализация и формат',
        label: 'Совпадение чисел',
        hint: 'Насколько точно числовые значения из оригинала переданы в переводе.',
    },
    number_count_diff: {
        group: 'Локализация и формат',
        label: 'Разница в количестве чисел',
        hint: 'Насколько отличается количество чисел между оригиналом и переводом.',
    },
    date_format_error: {
        group: 'Локализация и формат',
        label: 'Формат дат',
        hint: 'Дата из оригинала скопирована без адаптации под русский формат.',
    },
    quotes_mismatch: {
        group: 'Локализация и формат',
        label: 'Кавычки',
        hint: 'В переводе остались прямые английские кавычки вместо русских «ёлочек».',
    },
    currency_symbol_mismatch: {
        group: 'Локализация и формат',
        label: 'Символы валют',
        hint: 'В оригинале есть символ валюты, который не был адаптирован или пропущен в переводе.',
    },
    capitalization_mismatch: {
        group: 'Локализация и формат',
        label: 'Регистр имён',
        hint: 'Имена собственные в оригинале написаны с заглавной буквы, а в переводе — со строчной.',
    },

    // ─── СТРУКТУРА И ДЛИНА ─────────────────────────────────────
    compression_ratio: {
        group: 'Структура и длина',
        label: 'Сжатие по символам',
        hint: 'Соотношение количества символов в переводе и оригинале. Сильное отклонение может говорить о пропуске или добавлении текста.',
    },
    sentence_count_diff: {
        group: 'Структура и длина',
        label: 'Разбиение на предложения',
        hint: 'Насколько количество предложений в переводе отличается от оригинала (слияние или разбиение).',
    },
    length_ratio: {
        group: 'Структура и длина',
        label: 'Соотношение длины (слов)',
        hint: 'Насколько перевод длиннее или короче оригинала. Для EN→RU нормальное значение — около 1.25–1.40.',
    },
    abs_length_diff: {
        group: 'Структура и длина',
        label: 'Абсолютная разница в словах',
        hint: 'На сколько слов перевод отличается от оригинала.',
    },

    // ─── FLUENCY (тяжёлые признаки) ───────────────────────────
    perplexity: {
        group: 'Грамотность',
        label: 'Естественность текста',
        hint: 'Насколько естественно и «по-русски» звучит перевод по оценке языковой модели ruGPT-3. Высокая перплексия — текст звучит странно.',
    },
    mean_log_prob: {
        group: 'Грамотность',
        label: 'Средняя предсказуемость слов',
        hint: 'Насколько «ожидаемы» слова в переводе для языковой модели. Низкое значение — много неудачных формулировок.',
    },
    min_token_log_prob: {
        group: 'Грамотность',
        label: 'Самое странное слово',
        hint: 'Самое «неожиданное» слово в переводе. Часто совпадает с местом реальной ошибки.',
    },
    token_ppl_variance: {
        group: 'Грамотность',
        label: 'Неравномерность качества',
        hint: 'Насколько неровно распределена естественность по словам. Большой разброс — в целом нормальный текст с одним-двумя очень странными местами.',
    },

    // ─── КОНТЕКСТНЫЕ ПРИЗНАКИ ──────────────────────────────────
    src_length: {
        group: 'Контекст',
        label: 'Длина оригинала',
        hint: 'Количество слов в английском предложении. Используется моделью как контекст для оценки других признаков.',
    },
    mt_length: {
        group: 'Контекст',
        label: 'Длина перевода',
        hint: 'Количество слов в русском переводе.',
    },

    // ─── СОСТАВНЫЕ ПРИЗНАКИ (INTERACTION) ──────────────────────
    cosine_x_length_ok: {
        group: 'Смысловая точность',
        label: 'Смысл при нормальной длине',
        hint: 'Комбинация: хороший смысл при нормальной длине перевода. Помогает отличить реальную смысловую ошибку от просто короткого/длинного текста.',
    },
    logprob_spike: {
        group: 'Грамотность',
        label: 'Всплеск неуверенности',
        hint: 'Разница между средней и минимальной предсказуемостью слов. Большой всплеск почти всегда указывает на конкретное проблемное место.',
    },

    // Добавь остальные interaction-признаки по аналогии, если нужно
};


/** Отображаемые русские заголовки блоков (ключ — внутренний group из FEATURE_REGISTRY). */
const FEATURE_GROUP_LABEL_RU = {
    Accuracy: 'Точность (смысл)',
    'Accuracy · Omission': 'Точность · пропуски сущностей',
    'Accuracy · Mistranslation': 'Точность · искажения фактов',
    'Accuracy · Omission / Addition': 'Точность · объём текста',
    Fluency: 'Грамотность',
    'Fluency · Morphology': 'Грамотность · морфология',
    'Fluency · Syntax': 'Грамотность · синтаксис',
    'Fluency · Spelling': 'Грамотность · орфография и OOV',
    'Fluency · LexicalChoice': 'Грамотность · лексика',
    'Locale · Quotes': 'Локаль · кавычки',
    'Locale · DateFormat': 'Локаль · даты',
    'Locale · Punctuation': 'Локаль · пунктуация',
    'Style · Register': 'Стиль · регистр',
    Контекст: 'Контекст',
    'Accuracy + Fluency': 'Точность и грамотность',
    'Fluency + Accuracy': 'Грамотность и точность',
    'Style + Accuracy': 'Стиль и точность',
};

function featureGroupTitleRu(group) {
    if (!group) return '';
    return FEATURE_GROUP_LABEL_RU[group] || group;
}

/** При равной сумме |SHAP| по блокам — порядок как в FEATURE_REGISTRY (стабильный tie-break). */
const FEATURE_GROUP_ORDER = (() => {
    const out = [];
    const seen = new Set();
    for (const meta of Object.values(FEATURE_REGISTRY)) {
        if (meta.group && !seen.has(meta.group)) {
            seen.add(meta.group);
            out.push(meta.group);
        }
    }
    return out;
})();

function getFeatureRegistryMeta(key) {
    return FEATURE_REGISTRY[key] || null;
}

function getFeatureGroup(key) {
    const m = getFeatureRegistryMeta(key);
    if (m?.group) return m.group;
    return 'Прочее';
}

function getFeatureLabel(key) {
    if (FEATURE_REGISTRY[key]) return FEATURE_REGISTRY[key].label;
    return key;
}

function getFeatureHint(key) {
    if (FEATURE_REGISTRY[key]) return FEATURE_REGISTRY[key].hint;
    return 'Признак участвует в sentence-модели; точный смысл вклада задаётся обучением и SHAP.';
}

/** Мин. доля от суммарного штрафа (только отрицательный SHAP), ниже — не показываем. */
const MIN_LOSS_SHARE = 0.005;

/**
 * Только «потери» относительно идеала: берём признаки с SHAP<0,
 * доля i = |SHAP_i| / sum_j|SHAP_j^-|, отбрасываем <0.5%, перенормируем остаток к 100%.
 * Возвращает строки с полем lossFraction (0..1) для отображения.
 */
function buildLossOnlyFeatureRows(combined) {
    const neg = combined.filter((x) => (Number(x.shap) || 0) < 0);
    const totalAbs = neg.reduce((s, x) => s + Math.abs(Number(x.shap) || 0), 0);
    if (totalAbs <= 1e-12) return [];
    const withPart = neg.map((x) => ({
        ...x,
        part: Math.abs(Number(x.shap) || 0) / totalAbs,
    }));
    const kept = withPart.filter((x) => x.part >= MIN_LOSS_SHARE);
    const s2 = kept.reduce((t, x) => t + x.part, 0);
    if (s2 <= 1e-12) return [];
    return kept
        .map((x) => ({
            ...x,
            lossFraction: x.part / s2,
        }))
        .sort((a, b) => (Number(b.lossFraction) || 0) - (Number(a.lossFraction) || 0));
}

function init() {
    tokenTooltipEl = document.createElement('div');
    tokenTooltipEl.id = 'qe-token-tooltip';
    tokenTooltipEl.className = 'qe-token-tooltip';
    tokenTooltipEl.setAttribute('role', 'tooltip');
    document.body.appendChild(tokenTooltipEl);

    addSegment();
    pollStatus();

    addBtn.addEventListener('click', addSegment);
    evalAllBtn.addEventListener('click', evalAllSegments);
    closeDetailBtn.addEventListener('click', () => {
        detailPanel.classList.toggle('collapsed');
    });
    document.querySelector('.detail-header').addEventListener('click', (e) => {
        if (e.target !== closeDetailBtn) detailPanel.classList.toggle('collapsed');
    });

    tbody.addEventListener('input', handleTableInput);
    tbody.addEventListener('click', handleTableClick);
    tbody.addEventListener('scroll', handleEditorScroll, true);
    tbody.addEventListener('mousemove', handleEditorShellMouseMove);
    tbody.addEventListener('mouseleave', hideTokenTooltip, true);
}

init();

async function pollStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        if (data.ready) {
            modelReady = true;
            statusDiv.textContent = 'Модели готовы';
            statusDiv.className = 'status ready';
        } else {
            modelReady = false;
            statusDiv.textContent = 'Загрузка моделей...';
            statusDiv.className = 'status loading';
        }
    } catch (e) {
        modelReady = false;
        statusDiv.textContent = 'Ошибка соединения';
        statusDiv.className = 'status error';
    }
}

setInterval(pollStatus, 3000);

function addSegment() {
    const segment = {
        id: nextId++,
        src: '',
        mt: '',
        result: null,
        cache: null,
        status: 'idle',
    };
    segments.push(segment);
    currentSegmentId = segment.id;
    renderTable();
    renderDetail(currentSegmentId);
    setTimeout(() => {
        const mtArea = document.querySelector(`textarea.mt-area[data-id="${segment.id}"]`);
        if (mtArea) {
            mtArea.focus();
            autoResize(mtArea);
        }
    }, 0);
}

function deleteSegment(id) {
    if (segments.length === 1) return;

    const idx = segments.findIndex((seg) => seg.id === id);
    if (idx === -1) return;

    segments.splice(idx, 1);
    if (activeErrorHighlight && activeErrorHighlight.segmentId === id) {
        activeErrorHighlight = null;
    }

    if (currentSegmentId === id) {
        currentSegmentId = segments[0]?.id || null;
    }

    renderTable();
    renderDetail(currentSegmentId);
}

async function evaluateSegment(id) {
    const seg = getSegment(id);
    if (!seg || !seg.src.trim() || !seg.mt.trim()) {
        alert('Заполните оба текстовых поля.');
        return;
    }
    if (!modelReady) {
        alert('Модели ещё загружаются, подождите.');
        return;
    }
    if (seg.status === 'loading') return;

    seg.status = 'loading';
    refreshRow(id);

    try {
        const res = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ src: seg.src, mt: seg.mt }),
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const data = await res.json();
        seg.result = data;
        seg.cache = data.debug || null;
        seg.status = 'done';

        refreshRow(id);
        if (currentSegmentId === id) renderDetail(id);
    } catch (err) {
        console.error(err);
        seg.status = 'error';
        refreshRow(id);
        alert(`Ошибка при оценке: ${err.message}`);
    }
}

async function evalAllSegments() {
    for (const seg of segments) {
        if (seg.src.trim() && seg.mt.trim() && seg.status !== 'done') {
            await evaluateSegment(seg.id);
            await new Promise((resolve) => setTimeout(resolve, 200));
        }
    }
}

function renderTable() {
    tbody.innerHTML = segments
        .map((seg, index) => buildRowMarkup(seg, index + 1))
        .join('');

    segments.forEach((seg, index) => {
        const row = getRow(seg.id);
        if (!row) return;
        const srcArea = row.querySelector('.src-area');
        const mtArea = row.querySelector('.mt-area');
        if (srcArea) srcArea.value = seg.src;
        if (mtArea) mtArea.value = seg.mt;
        syncRow(row, seg, index + 1);
    });
    autoResizeAll();
}

function buildRowMarkup(seg, displayIndex) {
    return `
        <tr class="segment-row" data-id="${seg.id}">
            <td class="col-num row-number">${displayIndex}</td>
            <td class="col-src">
                <textarea class="editable-text src-area" data-id="${seg.id}" rows="1">${escapeHtml(seg.src)}</textarea>
            </td>
            <td class="col-mt">
                <div class="editor-shell">
                    <div class="editor-highlight" data-role="mt-highlight"></div>
                    <textarea class="editable-text mt-area editable-overlay" data-id="${seg.id}" rows="1">${escapeHtml(seg.mt)}</textarea>
                </div>
                <button class="eval-row-btn" data-id="${seg.id}">Оценить</button>
            </td>
            <td class="col-score score-cell"></td>
            <td class="col-actions">
                <button class="delete-btn" data-id="${seg.id}" title="Удалить строку">✕</button>
            </td>
        </tr>
    `;
}

function syncRow(row, seg, displayIndex) {
    row.classList.toggle('selected', currentSegmentId === seg.id);
    row.querySelector('.row-number').textContent = displayIndex;
    updateScoreCell(row, seg);
    updateMtHighlight(row, seg);
}

function refreshRow(id) {
    const seg = getSegment(id);
    const row = getRow(id);
    if (!seg || !row) return;
    syncRow(row, seg, getDisplayIndex(id));
}

function updateScoreCell(row, seg) {
    const cell = row.querySelector('.score-cell');
    if (!cell) return;

    let scoreHtml = '<span class="score-badge">—</span>';
    if (seg.status === 'loading') {
        scoreHtml = '<span class="score-badge">⏳</span>';
    } else if (seg.status === 'error') {
        scoreHtml = '<span class="score-badge score-verybad">ERR</span>';
    } else if (seg.result && seg.result.score !== undefined) {
        const scorePercent = seg.result.score * 100;
        let cls = 'score-badge';
        if (scorePercent >= 80) cls += ' score-good';
        else if (scorePercent >= 60) cls += ' score-warning';
        else if (scorePercent >= 40) cls += ' score-bad';
        else cls += ' score-verybad';
        scoreHtml = `<span class="${cls}">${Math.round(scorePercent)}%</span>`;
    }
    cell.innerHTML = scoreHtml;
}

function handleTableInput(event) {
    const target = event.target;
    const id = Number(target.dataset.id);
    const seg = getSegment(id);
    if (!seg) return;

    if (target.classList.contains('src-area')) {
        seg.src = target.value;
        autoResize(target);
        clearInference(seg);
        refreshRow(id);
        if (currentSegmentId === id) renderDetail(id);
        return;
    }

    if (target.classList.contains('mt-area')) {
        seg.mt = target.value;
        autoResize(target);
        clearInference(seg);
        refreshRow(id);
        syncEditorScroll(target);
        if (currentSegmentId === id) renderDetail(id);
    }
}

function handleTableClick(event) {
    const target = event.target;
    const id = Number(target.dataset.id || target.closest('tr')?.dataset.id);
    if (!id) return;

    if (target.classList.contains('delete-btn')) {
        deleteSegment(id);
        return;
    }

    if (target.classList.contains('eval-row-btn')) {
        evaluateSegment(id);
        return;
    }

    if (!target.closest('tr')) return;
    currentSegmentId = id;
    renderSelection();
    renderDetail(id);
}

function handleEditorScroll(event) {
    if (event.target.classList?.contains('mt-area')) {
        syncEditorScroll(event.target);
    }
}

function renderSelection() {
    document.querySelectorAll('.segment-row').forEach((row) => {
        row.classList.toggle('selected', Number(row.dataset.id) === currentSegmentId);
    });
}

function clearInference(seg) {
    if (!seg.result && !seg.cache && seg.status === 'idle') return;
    seg.result = null;
    seg.cache = null;
    seg.status = 'idle';
    if (activeErrorHighlight && activeErrorHighlight.segmentId === seg.id) {
        activeErrorHighlight = null;
    }
}

function updateMtHighlight(row, seg) {
    const highlight = row.querySelector('[data-role="mt-highlight"]');
    const mtArea = row.querySelector('.mt-area');
    if (!highlight || !mtArea) return;

    highlight.innerHTML = renderHighlightedMt(seg.mt, seg.result?.errors || [], seg.id);
    highlight.scrollTop = mtArea.scrollTop;
    highlight.scrollLeft = mtArea.scrollLeft;
}

function renderHighlightedMt(text, errors, segmentId) {
    if (!text) return '<span class="editor-empty">Введите перевод...</span>';

    const parts = tokenizeText(text);
    const tokenMeta = buildTokenMeta(errors, segmentId);

    return parts.map((part) => {
        if (part.type === 'space') return escapeHtml(part.value);

        const meta = tokenMeta.get(part.tokenIndex);
        if (!meta) return escapeHtml(part.value);

        const classes = ['token-highlight', meta.severity === 'BAD-major' ? 'token-major' : 'token-minor'];
        if (meta.active) classes.push('token-active');
        const label = meta.error_label || meta.error_type || '';
        const et = escapeAttr(meta.error_type || '');
        const el = escapeAttr(label);
        const conf = meta.confidence != null ? Math.round(Number(meta.confidence) * 100) : '';
        const sev = escapeAttr(meta.severity || '');
        return `<span class="${classes.join(' ')}" data-token-idx="${part.tokenIndex}" data-severity="${sev}" data-error-type="${et}" data-error-label="${el}" data-confidence="${conf}">${escapeHtml(part.value)}</span>`;
    }).join('');
}

function tokenizeText(text) {
    const parts = [];
    const regex = /\s+|[\p{L}\p{N}]+|[^\s\p{L}\p{N}]/gu;
    let match;
    let tokenIndex = 0;

    while ((match = regex.exec(text)) !== null) {
        const value = match[0];
        if (/^\s+$/u.test(value)) {
            parts.push({ type: 'space', value });
        } else {
            parts.push({ type: 'token', value, tokenIndex });
            tokenIndex += 1;
        }
    }
    return parts;
}

function buildTokenMeta(errors, segmentId) {
    const meta = new Map();

    function better(prev, cand) {
        if (!prev) return cand;
        const pMaj = prev.severity === 'BAD-major';
        const cMaj = cand.severity === 'BAD-major';
        if (cMaj && !pMaj) return cand;
        if (pMaj && !cMaj) return prev;
        const pc = Number(prev.confidence) || 0;
        const cc = Number(cand.confidence) || 0;
        return cc >= pc ? cand : prev;
    }

    for (const err of errors || []) {
        const start = Number(err.start_idx);
        const end = Number(err.end_idx);
        if (!Number.isFinite(start) || !Number.isFinite(end)) continue;

        const candBase = {
            severity: err.severity,
            error_type: err.error_type || '',
            error_label: err.error_label || err.error_type || '',
            confidence: err.confidence != null ? Number(err.confidence) : 0,
            active: false,
        };

        for (let tokenIndex = start; tokenIndex <= end; tokenIndex += 1) {
            const prev = meta.get(tokenIndex);
            meta.set(tokenIndex, better(prev, { ...candBase }));
        }
    }

    for (const [tokenIndex, row] of meta) {
        const active = (
            activeErrorHighlight
            && activeErrorHighlight.segmentId === segmentId
            && tokenIndex >= activeErrorHighlight.startIdx
            && tokenIndex <= activeErrorHighlight.endIdx
        );
        row.active = active;
    }
    return meta;
}

function hideTokenTooltip() {
    if (!tokenTooltipEl) return;
    tokenTooltipEl.classList.remove('visible');
    tokenTooltipEl.innerHTML = '';
}

function handleEditorShellMouseMove(event) {
    const shell = event.target.closest('.editor-shell');
    if (!shell || !tokenTooltipEl) {
        hideTokenTooltip();
        return;
    }

    const ta = shell.querySelector('.mt-area');
    const highlight = shell.querySelector('[data-role="mt-highlight"]');
    if (!ta || !highlight) return;

    const prevPe = ta.style.pointerEvents;
    ta.style.pointerEvents = 'none';
    let el = null;
    try {
        el = document.elementFromPoint(event.clientX, event.clientY);
    } finally {
        ta.style.pointerEvents = prevPe || '';
    }

    const span = el && el.closest && el.closest('.token-highlight');
    if (!span || !highlight.contains(span)) {
        hideTokenTooltip();
        return;
    }

    const label = span.dataset.errorLabel || span.dataset.errorType || 'Ошибка';
    const typ = span.dataset.errorType || '';
    const conf = span.dataset.confidence;
    const sev = span.dataset.severity === 'BAD-major' ? 'серьёзная (major)' : 'незначительная (minor)';
    const strength = conf !== '' && conf !== undefined ? `${conf}% уверенности модели` : 'уверенность неизвестна';

    tokenTooltipEl.innerHTML = `
        <div class="tt-title">${escapeHtml(label)}</div>
        <div class="tt-meta">${escapeHtml(typ)} · ${escapeHtml(sev)}</div>
        <div class="tt-meta">${escapeHtml(strength)}</div>
    `;

    const pad = 14;
    let x = event.clientX + pad;
    let y = event.clientY + pad;
    const rect = tokenTooltipEl.getBoundingClientRect();
    if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - pad;
    tokenTooltipEl.style.left = `${Math.max(8, x)}px`;
    tokenTooltipEl.style.top = `${Math.max(8, y)}px`;
    tokenTooltipEl.classList.add('visible');
}

function syncEditorScroll(textarea) {
    const highlight = textarea.parentElement?.querySelector('[data-role="mt-highlight"]');
    if (!highlight) return;
    highlight.scrollTop = textarea.scrollTop;
    highlight.scrollLeft = textarea.scrollLeft;
}

function autoResize(textarea) {
    if (!textarea) return;
    textarea.style.height = 'auto';
    textarea.style.height = `${textarea.scrollHeight}px`;

    const highlight = textarea.parentElement?.querySelector('[data-role="mt-highlight"]');
    if (highlight) {
        highlight.style.height = textarea.style.height;
    }
}

function autoResizeAll() {
    document.querySelectorAll('.editable-text').forEach(autoResize);
}

function renderDetail(segmentId) {
    const seg = getSegment(segmentId);
    if (!seg || !seg.result) {
        detailContent.innerHTML = '<p class="placeholder">Оценка ещё не выполнена. Нажмите «Оценить» для нужной строки.</p>';
        detailTitle.textContent = `Сегмент #${segmentId ? getDisplayIndex(segmentId) : '—'}`;
        return;
    }

    const r = seg.result;
    const scorePercent = (r.score * 100).toFixed(0);
    detailTitle.textContent = `Сегмент #${getDisplayIndex(seg.id)} — Оценка: ${scorePercent}%`;

    const ciLowPct = r.ci_low !== undefined ? (r.ci_low * 100).toFixed(0) : null;
    const ciHighPct = r.ci_high !== undefined ? (r.ci_high * 100).toFixed(0) : null;
    const mqmPct = r.mqm_score !== undefined ? (r.mqm_score * 100).toFixed(0) : null;

    let html = `
        <div class="score-summary">
            <div class="big-score">${scorePercent}%</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width:${scorePercent}%"></div>
            </div>
            <p class="meta-hint">Основная оценка — предсказание sentence-модели: насколько высоко качество перевода на шкале от 0% до 100% (чем выше, тем лучше).</p>
            ${r.mqm_score !== undefined ? `
                <div class="meta">MQM-индекс: ${mqmPct}% (внутренний 0–1: ${r.mqm_score.toFixed(2)})</div>
                <p class="meta-hint">Отдельная шкала по найденным словесным ошибкам (штрафы MQM): 100% означает «нет штрафов за размеченные BAD-спаны»; чем ниже, тем больше суммарный штраф с учётом серьёзности (major/minor) и уверенности span-модели. Это не то же самое, что процент над прогресс-баром, а дополнение к нему.</p>
            ` : ''}
            ${ciLowPct != null && ciHighPct != null ? `
                <div class="meta">Непараметрический доверительный интервал 95% для оценки качества: ${ciLowPct}–${ciHighPct}%</div>
                <p class="meta-hint">По sentence-модели: если бы мы много раз слегка меняли вход, в таком диапазоне с большой вероятностью оказалась бы «истинная» оценка на той же шкале 0–100%. Узкий интервал — модель увереннее; широкий — больше неопределённость.</p>
            ` : ''}
        </div>
    `;

    if (seg.cache && seg.cache.features) {
        const features = seg.cache.features;
        const shapValues = seg.cache.shap_values || {};
        const featureEntries = Object.entries(features).map(([key, value]) => ({
            key,
            value,
            shap: getShapValue(shapValues, key),
        }));

        const lossRows = buildLossOnlyFeatureRows(featureEntries);

        if (lossRows.length) {
            html += renderFeatureGroupsPanel(lossRows);
        } else if (featureEntries.some((x) => (Number(x.shap) || 0) !== 0)) {
            html += '<p class="semantic-note">Нет заметного отрицательного вклада признаков (снижающих оценку от идеала 100%), либо все доли ниже 0.5% от суммарного штрафа.</p>';
        } else {
            html += '<p class="semantic-note">SHAP-вклады для этой пары около нуля.</p>';
        }
    } else {
        html += '<p><em>Debug-признаки недоступны.</em></p>';
    }

    if (r.errors && r.errors.length) {
        html += '<div class="errors-list"><h4>Найденные ошибки</h4>';
        for (const err of r.errors) {
            const spanText = err.span_text || '?';
            html += `
                <div class="error-row" data-segid="${seg.id}" data-start-idx="${err.start_idx}" data-end-idx="${err.end_idx}">
                    <div class="severity-dot severity-${err.severity === 'BAD-major' ? 'major' : 'minor'}"></div>
                    <div class="error-text">«${escapeHtml(spanText)}»</div>
                    <div class="error-type">${escapeHtml(err.error_label || err.error_type)} · ${Math.round((err.confidence || 0) * 100)}%</div>
                </div>
            `;
        }
        html += '</div>';
    } else {
        html += '<p><em>Ошибок не обнаружено.</em></p>';
    }

    html += '<button id="manual-feedback-btn" class="manual-feedback-btn">Отметить ошибку вручную</button>';
    detailContent.innerHTML = html;

    document.querySelectorAll('.error-row').forEach((row) => {
        row.addEventListener('click', () => {
            highlightErrorInSegment(
                Number(row.dataset.segid),
                Number(row.dataset.startIdx),
                Number(row.dataset.endIdx),
            );
        });
    });

    const manualBtn = document.getElementById('manual-feedback-btn');
    if (manualBtn) manualBtn.addEventListener('click', () => startManualFeedback(seg.id));
}

function sumLossFractionForGroup(groupItems) {
    return groupItems.reduce((s, it) => s + (Number(it.lossFraction) || 0), 0);
}

function sortGroupsByLossFraction(groups, byGroup) {
    const registryRank = (g) => {
        const i = FEATURE_GROUP_ORDER.indexOf(g);
        return i === -1 ? 1e6 : i;
    };
    return [...groups].sort((a, b) => {
        const ta = sumLossFractionForGroup(byGroup.get(a));
        const tb = sumLossFractionForGroup(byGroup.get(b));
        if (tb !== ta) return tb - ta;
        const ra = registryRank(a);
        const rb = registryRank(b);
        if (ra !== rb) return ra - rb;
        return a.localeCompare(b, 'ru');
    });
}

function formatLossSharePercent(lossFraction) {
    const pct = (Number(lossFraction) || 0) * 100;
    const decimals = pct >= 1 ? 2 : pct >= 0.01 ? 3 : 4;
    return `${pct.toFixed(decimals)}%`;
}

function renderFeatureGroupsPanel(items) {
    if (!items.length) return '';

    const maxFrac = Math.max(...items.map((item) => Number(item.lossFraction) || 0), 1e-6);
    const byGroup = new Map();
    for (const item of items) {
        const g = getFeatureGroup(item.key);
        if (!byGroup.has(g)) byGroup.set(g, []);
        byGroup.get(g).push(item);
    }
    for (const g of byGroup.keys()) {
        byGroup.set(
            g,
            [...byGroup.get(g)].sort(
                (a, b) => (Number(b.lossFraction) || 0) - (Number(a.lossFraction) || 0),
            ),
        );
    }

    const note =
        'Показаны только признаки, которые по SHAP снижают оценку относительно идеала (100%). Число в строке — доля в суммарном штрафе после отсечения <0.5% и перенормировки к 100%. Карточки блоков — по убыванию суммы долей в блоке.';

    let html = '<h4 class="feature-panel-title">Доли штрафа по признакам (от идеала 100%)</h4>';
    html += `<p class="semantic-note">${escapeHtml(note)}</p>`;
    html += '<div class="feature-groups-grid">';

    for (const group of sortGroupsByLossFraction([...byGroup.keys()], byGroup)) {
        const groupItems = byGroup.get(group);
        const mass = sumLossFractionForGroup(groupItems);
        const titleRu = featureGroupTitleRu(group);
        html += '<section class="feature-group-card">';
        html += `<h5 class="feature-group-title">${escapeHtml(titleRu)}</h5>`;
        html += `<div class="feature-group-mass" title="Сумма долей штрафа по признакам этого блока (после перенормировки, в сумме по всем блокам 100%)">Σ доля ${escapeHtml(formatLossSharePercent(mass))}</div>`;
        html += '<div class="feature-group-rows">';

        for (const item of groupItems) {
            const frac = Number(item.lossFraction) || 0;
            const width = Math.min((frac / maxFrac) * 100, 100);
            const displayValue = typeof item.value === 'number' ? item.value.toFixed(3) : String(item.value);
            const displayShare = formatLossSharePercent(frac);
            const fname = getFeatureLabel(item.key);
            const fhint = getFeatureHint(item.key);

            html += `
            <div class="feature-row-compact">
                <div class="feature-row-metrics">
                    <span class="feature-name">${escapeHtml(fname)}</span>
                    <span class="feature-value">${escapeHtml(displayValue)}</span>
                    <span class="feature-shap negative">${escapeHtml(displayShare)}</span>
                </div>
                <details class="feature-hint-drop">
                    <summary class="feature-hint-summary" aria-label="Описание признака">?</summary>
                    <div class="feature-hint-body">${escapeHtml(fhint)}</div>
                </details>
                <div class="feature-impact feature-impact-compact">
                    <div class="impact-lane impact-negative">
                        <div class="impact-fill negative" style="width:${width}%"></div>
                    </div>
                </div>
            </div>`;
        }

        html += '</div></section>';
    }

    html += '</div>';
    return html;
}

function getShapValue(shapValues, key) {
    if (shapValues && typeof shapValues === 'object' && !Array.isArray(shapValues)) {
        return Number(shapValues[key] || 0);
    }
    return 0;
}

function highlightErrorInSegment(segmentId, startIdx, endIdx) {
    const previousSegmentId = activeErrorHighlight?.segmentId ?? null;
    activeErrorHighlight = { segmentId, startIdx, endIdx };
    currentSegmentId = segmentId;
    renderSelection();
    if (previousSegmentId && previousSegmentId !== segmentId) {
        refreshRow(previousSegmentId);
    }
    refreshRow(segmentId);

    const row = getRow(segmentId);
    const mtArea = row?.querySelector('.mt-area');
    if (mtArea) mtArea.focus();
}

function startManualFeedback(segId) {
    const seg = getSegment(segId);
    if (!seg || !seg.result) {
        alert('Сначала выполните оценку сегмента.');
        return;
    }
    if (!seg.cache || !seg.cache.features) {
        alert('Кэш признаков отсутствует.');
        return;
    }

    const mtText = seg.mt;
    const selection = window.prompt('Введите ошибочный фрагмент (точный текст из перевода):', '');
    if (!selection) return;

    const startIdx = mtText.indexOf(selection);
    if (startIdx === -1) {
        alert('Фрагмент не найден в переводе.');
        return;
    }

    const endIdx = startIdx + selection.length;
    const errorType = window.prompt('Тип ошибки:', 'Fluency/LexicalChoice');
    if (!errorType) return;

    const severity = window.confirm('Сохранить как BAD-minor? Нажми Cancel для BAD-major.') ? 'BAD-minor' : 'BAD-major';

    fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            src: seg.src,
            mt: seg.mt,
            start_char: startIdx,
            end_char: endIdx,
            error_type: errorType,
            severity,
            features: seg.cache.features,
            word_logprobs: seg.cache.word_logprobs || [],
        }),
    })
        .then((res) => res.json())
        .then((data) => alert(`Разметка сохранена (id: ${data.feedback_id})`))
        .catch((err) => alert(`Ошибка: ${err}`));
}

function getSegment(id) {
    return segments.find((seg) => seg.id === id);
}

function getRow(id) {
    return document.querySelector(`tr[data-id="${id}"]`);
}

function getDisplayIndex(id) {
    const index = segments.findIndex((seg) => seg.id === id);
    return index >= 0 ? index + 1 : 0;
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function escapeAttr(str) {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\n/g, ' ');
}
