// ---------- Состояние ----------
let segments = [];
let nextId = 1;
let currentSegmentId = null;
let modelReady = false;

// DOM элементы
const tbody = document.getElementById('segments-tbody');
const addBtn = document.getElementById('add-segment-btn');
const evalAllBtn = document.getElementById('eval-all-btn');
const statusDiv = document.getElementById('status');
const detailPanel = document.getElementById('detail-panel');
const detailTitle = document.getElementById('detail-segment-title');
const detailContent = document.getElementById('detail-content');
const closeDetailBtn = document.getElementById('close-detail-btn');

// Человеко-читаемые названия признаков
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
    min_token_log_prob: 'Минимальный log prob'
};

function getFeatureDisplayName(key) {
    if (FEATURE_LABELS[key]) return FEATURE_LABELS[key];
    if (key.startsWith('semantic_')) {
        const suffix = key.split('_')[1] || '';
        return `Semantic PCA ${suffix}`;
    }
    return key;
}

// ---------- Инициализация ----------
function init() {
    addSegment(); // первый сегмент
    pollStatus();
    addBtn.addEventListener('click', () => addSegment());
    evalAllBtn.addEventListener('click', () => evalAllSegments());
    closeDetailBtn.addEventListener('click', () => {
        detailPanel.classList.toggle('collapsed');
    });
    document.querySelector('.detail-header').addEventListener('click', (e) => {
        if (e.target !== closeDetailBtn) detailPanel.classList.toggle('collapsed');
    });
}
init();

// Статус моделей
async function pollStatus() {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        if (data.ready) {
            modelReady = true;
            statusDiv.textContent = '✅ Модели готовы';
            statusDiv.classList.remove('loading', 'error');
            statusDiv.classList.add('ready');
        } else {
            statusDiv.textContent = '⏳ Загрузка моделей...';
            statusDiv.classList.add('loading');
            modelReady = false;
        }
    } catch(e) {
        statusDiv.textContent = '❌ Ошибка соединения';
        statusDiv.classList.add('error');
    }
}
setInterval(pollStatus, 3000);

// ---------- Управление сегментами ----------
function addSegment() {
    const id = nextId++;
    segments.push({
        id,
        src: '',
        mt: '',
        result: null,
        cache: null,
        status: 'idle',   // idle, loading, done, error
    });
    renderTable();
    // автофокус на новое поле mt через таймаут
    setTimeout(() => {
        const mtArea = document.querySelector(`textarea.mt-area[data-id="${id}"]`);
        if (mtArea) mtArea.focus();
        autoResizeAll();
    }, 50);
}

function deleteSegment(id) {
    if (segments.length === 1) return;
    const idx = segments.findIndex(s => s.id === id);
    if (idx !== -1) segments.splice(idx, 1);
    if (currentSegmentId === id) {
        currentSegmentId = segments.length ? segments[0].id : null;
        renderDetail(currentSegmentId);
    }
    renderTable();
}

async function evaluateSegment(id) {
    const seg = segments.find(s => s.id === id);
    if (!seg || !seg.src.trim() || !seg.mt.trim()) {
        alert('Заполните оба текстовых поля');
        return;
    }
    if (!modelReady) {
        alert('Модели ещё загружаются, подождите...');
        return;
    }
    if (seg.status === 'loading') return;
    seg.status = 'loading';
    renderTableRow(id);
    try {
        const res = await fetch('/api/evaluate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ src: seg.src, mt: seg.mt })
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        seg.result = data;
        seg.cache = data.debug || null;
        seg.status = 'done';
        renderTableRow(id);
        if (currentSegmentId === id) renderDetail(id);
    } catch (err) {
        console.error(err);
        seg.status = 'error';
        renderTableRow(id);
        alert('Ошибка при оценке: ' + err.message);
    }
}

async function evalAllSegments() {
    for (const seg of segments) {
        if (seg.src.trim() && seg.mt.trim() && seg.status !== 'done') {
            await evaluateSegment(seg.id);
            // небольшая задержка, чтобы не перегружать сервер
            await new Promise(r => setTimeout(r, 200));
        }
    }
}

// ---------- Отрисовка таблицы ----------
function renderTable() {
    tbody.innerHTML = '';
    segments.forEach(seg => renderTableRow(seg.id));
    autoResizeAll();
}

function renderTableRow(id) {
    const seg = segments.find(s => s.id === id);
    if (!seg) return;
    let row = document.querySelector(`tr[data-id="${id}"]`);
    if (!row) {
        row = document.createElement('tr');
        row.setAttribute('data-id', id);
        tbody.appendChild(row);
    }
    // Номер
    const numCell = `<td class="col-num">${id}<\/td>`;
    // Src textarea
    const srcCell = `<td class="col-src"><textarea class="editable-text src-area" data-id="${id}" rows="1">${escapeHtml(seg.src)}<\/textarea><\/td>`;
    // Mt textarea + кнопка Evaluate
    let mtCell = `<td class="col-mt">
        <textarea class="editable-text mt-area" data-id="${id}" rows="1">${escapeHtml(seg.mt)}<\/textarea>
        <button class="eval-row-btn" data-id="${id}">Оценить</button>
    </td>`;
    // Score
    let scoreHtml = '<span class="score-badge">—</span>';
    if (seg.status === 'loading') scoreHtml = '<span class="score-badge">⏳</span>';
    else if (seg.result && seg.result.score !== undefined) {
        const scorePercent = seg.result.score * 100;
        let cls = 'score-badge';
        if (scorePercent >= 80) cls += ' score-good';
        else if (scorePercent >= 60) cls += ' score-warning';
        else if (scorePercent >= 40) cls += ' score-bad';
        else cls += ' score-verybad';
        scoreHtml = `<span class="${cls}">${Math.round(scorePercent)}%</span>`;
    }
    const scoreCell = `<td class="col-score">${scoreHtml}<\/td>`;
    const actionsCell = `<td class="col-actions"><button class="delete-btn" data-id="${id}">✕<\/button><\/td>`;
    row.innerHTML = numCell + srcCell + mtCell + scoreCell + actionsCell;

    // обработчики
    const srcTa = row.querySelector('.src-area');
    const mtTa = row.querySelector('.mt-area');
    const evalBtn = row.querySelector('.eval-row-btn');
    const delBtn = row.querySelector('.delete-btn');

    if (srcTa) {
        srcTa.addEventListener('input', (e) => {
            seg.src = e.target.value;
            autoResize(e.target);
            // сбросить результат при изменении
            if (seg.result) { seg.result = null; seg.cache = null; seg.status = 'idle'; }
            renderTableRow(id); // обновить score badge
            if (currentSegmentId === id) renderDetail(id);
        });
        autoResize(srcTa);
    }
    if (mtTa) {
        mtTa.addEventListener('input', (e) => {
            seg.mt = e.target.value;
            autoResize(e.target);
            if (seg.result) { seg.result = null; seg.cache = null; seg.status = 'idle'; }
            renderTableRow(id);
            if (currentSegmentId === id) renderDetail(id);
        });
        autoResize(mtTa);
    }
    if (evalBtn) evalBtn.addEventListener('click', () => evaluateSegment(id));
    if (delBtn) delBtn.addEventListener('click', () => deleteSegment(id));

    // клик по строке для выбора сегмента
    row.addEventListener('click', (e) => {
        if (e.target.classList && (e.target.classList.contains('delete-btn') || e.target.classList.contains('eval-row-btn'))) return;
        if (e.target.tagName === 'TEXTAREA') return;
        currentSegmentId = id;
        renderDetail(id);
        // подсветка строки
        document.querySelectorAll('tr').forEach(tr => tr.style.background = '');
        row.style.background = '#f0f9ff';
    });
}

// Авто-высота textarea
function autoResize(textarea) {
    if (!textarea) return;
    textarea.style.height = 'auto';
    textarea.style.height = (textarea.scrollHeight) + 'px';
}
function autoResizeAll() {
    document.querySelectorAll('.editable-text').forEach(autoResize);
}

// ---------- Детальная панель (score, признаки, ошибки) ----------
async function renderDetail(segmentId) {
    const seg = segments.find(s => s.id === segmentId);
    if (!seg || !seg.result) {
        detailContent.innerHTML = '<p class="placeholder">Оценка ещё не выполнена. Нажмите «Оценить» для этого сегмента.</p>';
        detailTitle.innerText = `Сегмент #${segmentId || '—'}`;
        return;
    }
    const r = seg.result;
    const scorePercent = (r.score * 100).toFixed(0);
    detailTitle.innerHTML = `Сегмент #${seg.id} — Оценка: ${scorePercent}%`;

    let html = `
        <div class="score-summary">
            <div class="big-score">${scorePercent}%</div>
            <div class="progress-bar"><div style="width: ${scorePercent}%; background: #3b82f6; height: 8px; border-radius: 4px;"></div></div>
            ${r.mqm_score ? `<div class="meta">MQM: ${r.mqm_score.toFixed(2)}</div>` : ''}
            ${r.ci_low ? `<div class="meta">CI 95%: ${(r.ci_low*100).toFixed(0)}–${(r.ci_high*100).toFixed(0)}%</div>` : ''}
        </div>
    `;

    // ---- Признаки (детально) ----
    if (seg.cache && seg.cache.features) {
        const features = seg.cache.features;
        const shapValues = seg.cache.shap_values || {}; // может быть массив или объект
        const featureCount = Object.keys(features).length;
        html += `<h4>Детальные признаки (${featureCount})</h4><div class="feature-list">`;
        for (const [key, value] of Object.entries(features)) {
            const displayName = getFeatureDisplayName(key);
            let shapValue = 0;
            if (shapValues[key] !== undefined) shapValue = shapValues[key];
            else if (Array.isArray(shapValues) && typeof shapValues[0] === 'number') {
                // если shap_values — массив, нужно сопоставить по порядку ключей, но для простоты пропустим
            }
            const shapCls = shapValue >= 0 ? 'positive' : 'negative';
            const barWidth = Math.min(Math.abs(shapValue) * 100, 100);
            html += `
                <div class="feature-item">
                    <span class="feature-name">${displayName}</span>
                    <div class="feature-bar"><div class="feature-fill ${shapCls}" style="width: ${barWidth}%;"></div></div>
                    <span class="feature-value">${typeof value === 'number' ? value.toFixed(3) : value}</span>
                    <span class="feature-shap ${shapCls}">${shapValue > 0 ? '+' : ''}${shapValue.toFixed(3)}</span>
                </div>
            `;
        }
        html += `</div>`;
    } else {
        html += `<p><em>Признаки не доступны (нет debug.features)</em></p>`;
    }

    // ---- Ошибки ----
    if (r.errors && r.errors.length) {
        html += `<div class="errors-list"><h4>Найденные ошибки</h4>`;
        for (const err of r.errors) {
            const spanText = err.span_text || err.text || '?';
            html += `
                <div class="error-row" data-start="${err.start_char}" data-end="${err.end_char}" data-segid="${seg.id}">
                    <div class="severity-dot severity-${err.severity === 'BAD-major' ? 'major' : 'minor'}"></div>
                    <div class="error-text">«${escapeHtml(spanText)}»</div>
                    <div class="error-type">${err.error_type} · ${Math.round((err.confidence||0.5)*100)}%</div>
                </div>
            `;
        }
        html += `</div>`;
    } else {
        html += `<p><em>Ошибок не обнаружено</em></p>`;
    }

    html += `<button id="manual-feedback-btn" class="manual-feedback-btn">✎ Отметить ошибку вручную</button>`;
    detailContent.innerHTML = html;

    // обработчики кликов на ошибках (подсветка)
    document.querySelectorAll('.error-row').forEach(row => {
        row.addEventListener('click', () => {
            const start = parseInt(row.dataset.start);
            const end = parseInt(row.dataset.end);
            highlightErrorInSegment(seg.id, start, end);
        });
    });
    const manualBtn = document.getElementById('manual-feedback-btn');
    if (manualBtn) manualBtn.addEventListener('click', () => startManualFeedback(seg.id));
}

function highlightErrorInSegment(segId, start, end) {
    // для простоты показываем алерт, можно реализовать поиск текста в mt и временную подсветку
    alert(`Ошибка выделена от символа ${start} до ${end} (можно добавить подсветку на слово в UI)`);
}

// ---------- Ручная разметка (упрощённо) ----------
function startManualFeedback(segId) {
    const seg = segments.find(s => s.id === segId);
    if (!seg || !seg.result) {
        alert('Сначала выполните оценку сегмента');
        return;
    }
    if (!seg.cache || !seg.cache.features) {
        alert('Кэш признаков отсутствует');
        return;
    }
    // Получаем текст перевода и просим пользователя выделить фрагмент
    const mtText = seg.mt;
    const selection = window.prompt('Введите ошибочный фрагмент (точный текст из перевода):', '');
    if (!selection) return;
    const startIdx = mtText.indexOf(selection);
    if (startIdx === -1) {
        alert('Фрагмент не найден в переводе. Проверьте точность.');
        return;
    }
    const endIdx = startIdx + selection.length;
    const errorType = prompt('Тип ошибки (например, Accuracy/Mistranslation):', 'Fluency/LexicalChoice');
    if (!errorType) return;
    const severity = confirm('BAD-major? (OK = minor, Cancel = major)') ? 'BAD-minor' : 'BAD-major';
    // Отправить feedback
    fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            src: seg.src,
            mt: seg.mt,
            start_char: startIdx,
            end_char: endIdx,
            error_type: errorType,
            severity: severity,
            features: seg.cache.features,
            word_logprobs: seg.cache.word_logprobs || []
        })
    }).then(res => res.json()).then(data => {
        alert('Разметка сохранена (id: ' + data.feedback_id + ')');
    }).catch(err => alert('Ошибка: ' + err));
}

function escapeHtml(str) {
    if (!str) return '';
    return str.replace(/[&<>]/g, function(m) {
        if (m === '&') return '&amp;';
        if (m === '<') return '&lt;';
        if (m === '>') return '&gt;';
        return m;
    });
}
