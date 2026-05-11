let segments = [];
let nextId = 1;
let currentSegmentId = null;
let activeErrorHighlight = null;
let modelReady = false;

const tbody = document.getElementById('segments-tbody');
const addBtn = document.getElementById('add-segment-btn');
const evalAllBtn = document.getElementById('eval-all-btn');
const statusDiv = document.getElementById('status');
const detailPanel = document.getElementById('detail-panel');
const detailTitle = document.getElementById('detail-segment-title');
const detailContent = document.getElementById('detail-content');
const closeDetailBtn = document.getElementById('close-detail-btn');

const FEATURE_LABELS = {
    length_ratio: 'Соотношение длин (mt/src)',
    abs_length_diff: 'Абсолютная разница длин',
    token_count_diff: 'Разница в количестве токенов',
    src_length: 'Длина источника (слова)',
    mt_length: 'Длина перевода (слова)',
    digit_match_ratio: 'Совпадение чисел',
    punct_ratio: 'Соотношение пунктуации',
    quotes_mismatch: 'Несовпадение кавычек',
    date_format_error: 'Ошибка формата даты',
    oov_ratio: 'Доля слов вне словаря',
    type_token_ratio: 'Лексическое разнообразие',
    avg_token_length: 'Средняя длина слова',
    entity_overlap_ratio: 'Пересечение NER',
    agreement_errors: 'Ошибки согласования',
    syntax_depth: 'Глубина синтаксиса',
    formal_ratio: 'Доля формальной лексики',
    cosine_similarity: 'Косинусное сходство (LaBSE)',
    embedding_distance: 'Евклидово расстояние (LaBSE)',
    perplexity: 'Перплексия (ruGPT)',
    mean_log_prob: 'Средний log likelihood',
    token_ppl_variance: 'Дисперсия вероятностей',
    min_token_log_prob: 'Минимальный log prob',
};

function getFeatureDisplayName(key) {
    if (FEATURE_LABELS[key]) return FEATURE_LABELS[key];
    if (key.startsWith('semantic_')) {
        const suffix = key.split('_')[1] || '';
        return `Semantic PCA ${suffix}`;
    }
    return key;
}

function init() {
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
        return `<span class="${classes.join(' ')}">${escapeHtml(part.value)}</span>`;
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
    for (const err of errors || []) {
        const start = Number(err.start_idx);
        const end = Number(err.end_idx);
        if (!Number.isFinite(start) || !Number.isFinite(end)) continue;

        for (let tokenIndex = start; tokenIndex <= end; tokenIndex += 1) {
            const prev = meta.get(tokenIndex);
            const isMajor = err.severity === 'BAD-major';
            const active = (
                activeErrorHighlight
                && activeErrorHighlight.segmentId === segmentId
                && tokenIndex >= activeErrorHighlight.startIdx
                && tokenIndex <= activeErrorHighlight.endIdx
            );

            if (!prev || (isMajor && prev.severity !== 'BAD-major')) {
                meta.set(tokenIndex, { severity: err.severity, active });
            } else if (active) {
                prev.active = true;
            }
        }
    }
    return meta;
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

    let html = `
        <div class="score-summary">
            <div class="big-score">${scorePercent}%</div>
            <div class="progress-bar">
                <div class="progress-fill" style="width:${scorePercent}%"></div>
            </div>
            ${r.mqm_score !== undefined ? `<div class="meta">MQM: ${r.mqm_score.toFixed(2)}</div>` : ''}
            ${r.ci_low !== undefined && r.ci_high !== undefined ? `<div class="meta">CI 95%: ${(r.ci_low * 100).toFixed(0)}–${(r.ci_high * 100).toFixed(0)}%</div>` : ''}
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

        const classicFeatures = featureEntries
            .filter((item) => !item.key.startsWith('semantic_'))
            .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));

        const semanticFeatures = featureEntries
            .filter((item) => item.key.startsWith('semantic_'))
            .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));

        html += renderFeatureSection('Интерпретируемые признаки', classicFeatures);
        html += renderFeatureSection(
            `Semantic PCA компоненты (${semanticFeatures.length})`,
            semanticFeatures,
            'PCA-компоненты латентные: по ним можно видеть силу и знак вклада, но напрямую назвать их “терминологией” или “грамматикой” нельзя.',
        );
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

function renderFeatureSection(title, items, note = '') {
    if (!items.length) return '';

    const maxAbsShap = Math.max(...items.map((item) => Math.abs(item.shap)), 1e-6);
    let html = `<h4>${title}</h4>`;
    if (note) html += `<p class="semantic-note">${note}</p>`;
    html += '<div class="feature-list">';

    for (const item of items) {
        const width = Math.min((Math.abs(item.shap) / maxAbsShap) * 100, 100);
        const displayValue = typeof item.value === 'number' ? item.value.toFixed(3) : String(item.value);
        const displayShap = `${item.shap > 0 ? '+' : ''}${item.shap.toFixed(3)}`;

        html += `
            <div class="feature-item">
                <span class="feature-name">${escapeHtml(getFeatureDisplayName(item.key))}</span>
                <div class="feature-impact">
                    <div class="impact-lane impact-positive">
                        ${item.shap > 0 ? `<div class="impact-fill positive" style="width:${width}%"></div>` : ''}
                    </div>
                    <div class="impact-lane impact-negative">
                        ${item.shap < 0 ? `<div class="impact-fill negative" style="width:${width}%"></div>` : ''}
                    </div>
                </div>
                <span class="feature-value">${escapeHtml(displayValue)}</span>
                <span class="feature-shap ${item.shap >= 0 ? 'positive' : 'negative'}">${displayShap}</span>
            </div>
        `;
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
