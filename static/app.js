// ── Переключение вкладок ────────────────────────────────────────
const tabs = document.querySelectorAll('.tab');
const contents = document.querySelectorAll('.tab-content');
tabs.forEach(tab => {
    tab.addEventListener('click', () => {
        const tabId = tab.getAttribute('data-tab');
        tabs.forEach(t => t.classList.remove('active'));
        contents.forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(tabId).classList.add('active');
    });
});

// ── Утилиты для кнопок ─────────────────────────────────────────
function disableButton(btn, loadingText = '... в процессе') {
    if (!btn) return;
    btn.disabled = true;
    btn.classList.add('loading');
    btn._originalText = btn.innerText;
    btn.innerText = loadingText;
}

function enableButton(btn) {
    if (!btn) return;
    btn.disabled = false;
    btn.classList.remove('loading');
    btn.innerText = btn._originalText || btn.innerText;
}

// ── Копирование / скачивание ────────────────────────────────────
async function copyToClipboard(text, btnElement) {
    const original = btnElement.innerText;
    btnElement.innerText = 'Копирование...';
    try {
        await navigator.clipboard.writeText(text);
        btnElement.innerText = 'Скопировано!';
        setTimeout(() => btnElement.innerText = original, 2000);
    } catch (err) {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
        btnElement.innerText = 'Скопировано!';
        setTimeout(() => btnElement.innerText = original, 2000);
    }
}

function downloadTxt(text, filename = 'summarization.txt') {
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
}

function addCopyDownloadButtons(container, text, btnContainerSelector = null, filename) {
    const btnContainer = btnContainerSelector ? document.querySelector(btnContainerSelector) : container;
    if (!btnContainer) return;
    const existingCopy = btnContainer.querySelector('.copy-summary-btn');
    const existingDownload = btnContainer.querySelector('.download-summary-btn');
    if (existingCopy) existingCopy.remove();
    if (existingDownload) existingDownload.remove();

    const copyBtn = document.createElement('button');
    copyBtn.textContent = 'Копировать';
    copyBtn.className = 'copy-summary-btn copy-btn';
    copyBtn.addEventListener('click', () => copyToClipboard(text, copyBtn));

    const downloadBtn = document.createElement('button');
    downloadBtn.textContent = 'Скачать TXT';
    downloadBtn.className = 'download-summary-btn download-btn';
    downloadBtn.addEventListener('click', () => downloadTxt(text, 'summarization.txt'));

    btnContainer.appendChild(copyBtn);
    btnContainer.appendChild(downloadBtn);
}

// ── Цвета и текст качества ──────────────────────────────────────
function getScoreColor(score, max) {
    const percent = (score / max) * 100;
    if (percent >= 90) return { bg: '#c6f7d0', text: '#0a5c2e' };
    if (percent >= 75) return { bg: '#d1fae5', text: '#065f46' };
    if (percent >= 50) return { bg: '#fef3c7', text: '#92400e' };
    if (percent >= 25) return { bg: '#fed7d7', text: '#9b2c2c' };
    return { bg: '#fecaca', text: '#991b1b' };
}

function getPenaltyColor(penalty) {
    if (penalty >= -5) return { bg: '#c6f7d0', text: '#0a5c2e' };
    if (penalty >= -10) return { bg: '#fef3c7', text: '#92400e' };
    return { bg: '#fecaca', text: '#991b1b' };
}

function getFinalScoreClass(score) {
    if (score >= 80) return 'quality-excellent';
    if (score >= 65) return 'quality-good';
    if (score >= 45) return 'quality-satisfactory';
    if (score >= 25) return 'quality-unsatisfactory';
    return 'quality-dangerous';
}

function getQualityText(score) {
    if (score >= 80) return "отличное";
    if (score >= 65) return "хорошее";
    if (score >= 45) return "удовлетворительное";
    if (score >= 25) return "неудовлетворительное";
    return "опасное";
}

const criteriaDescriptions = {
    complaints: "Жалобы – оценивается полнота описания жалоб пациента (боль, тошнота, слабость и т.д.). Максимум 15 баллов.",
    disease_history: "Анамнез заболевания – длительность, связь с едой/лекарствами, данные осмотра, аллергии. Максимум 15 баллов.",
    comorbidities: "Сопутствующие заболевания – перечень всех упомянутых в ЭМК заболеваний. Максимум 20 баллов.",
    habits: "Вредные привычки – курение, алкоголь и др. Максимум 5 баллов.",
    labs: "Лабораторные данные – полнота перечисления показателей (СРБ, лейкоциты и т.д.). Максимум 20 баллов.",
    imaging: "Инструментальные исследования – описание КТ, МРТ, УЗИ с датами и деталями. Максимум 25 баллов.",
    penalties: "Штрафы за галлюцинации, неверные значения, нерелевантную информацию и т.п. Отрицательные баллы."
};

function showCriteriaModal() {
    let html = '<div class="criteria-description">';
    html += '<h3>Критерии оценки суммаризации</h3>';
    for (const [key, desc] of Object.entries(criteriaDescriptions)) {
        html += `<h4>${key}</h4><p>${desc}</p>`;
    }
    html += '</div>';
    document.getElementById('criteria-modal-body').innerHTML = html;
    document.getElementById('criteriaModal').style.display = 'flex';
}

// ── Результат: контейнер с крестиком ────────────────────────────
function createResultContainer(htmlContent, parentContainer) {
    const wrapper = document.createElement('div');
    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '✖';
    closeBtn.className = 'close-result';
    closeBtn.onclick = () => {
        if (parentContainer) {
            parentContainer.innerHTML = '';
            parentContainer.style.display = 'none';
            parentContainer.classList.remove('error');
        }
    };
    wrapper.appendChild(closeBtn);
    const contentDiv = document.createElement('div');
    contentDiv.innerHTML = htmlContent;
    wrapper.appendChild(contentDiv);
    return wrapper;
}

function displayResultInContainer(container, htmlContent) {
    container.innerHTML = '';
    const contentWrapper = createResultContainer(htmlContent, container);
    container.appendChild(contentWrapper);
    container.style.display = 'block';
    container.classList.remove('error');
}

// ── Словарь красивых названий критериев ──────────────────────────
const criteriaLabels = {
    complaints: "Жалобы",
    disease_history: "Анамнез заболевания",
    comorbidities: "Сопутствующие заболевания",
    habits: "Вредные привычки",
    labs: "Лабораторные данные",
    imaging: "Инструментальные исследования"
};

// ── Карточка оценки (аккордеон) ─────────────────────────────────
function renderEvaluationCard(data, title, isR2 = false) {
    let html = `<details class="detail-group">`;
    html += `<summary>${title} (балл: ${data.final_score})</summary>`;
    html += `<div class="score-badge ${getFinalScoreClass(data.final_score)}">Итоговый балл: ${data.final_score} (${getQualityText(data.final_score)})</div>`;

    // R2: показываем причину изменения баллов
    if (isR2 && data.r2_reason) {
        html += `<div style="margin-top: 8px; font-size: 12px; color: #4a5568; font-style: italic; background: #f7fafc; padding: 6px 10px; border-radius: 4px; border-left: 3px solid #2c7da0;">${data.r2_reason}</div>`;
    }
    html += `<div class="criteria-grid">`;
    const criteria = ['complaints', 'disease_history', 'comorbidities', 'habits', 'labs', 'imaging'];
    const maxMap = { complaints:15, disease_history:15, comorbidities:20, habits:5, labs:20, imaging:25 };
    for (const crit of criteria) {
        const score = data[crit] || 0;
        const max = maxMap[crit];
        const color = getScoreColor(score, max);
        html += `<div class="criteria-card" style="border-left-color: ${color.bg}; background: ${color.bg}10;">`;
        html += `<div class="label criteria-tooltip" title="${criteriaDescriptions[crit]}">${criteriaLabels[crit] || crit}</div>`;
        html += `<div class="value" style="color: ${color.text};">${score} / ${max}</div>`;
        html += `</div>`;
    }
    const penalty = data.penalties || 0;
    const penColor = getPenaltyColor(penalty);
    html += `<div class="criteria-card penalty-card" style="border-left-color: ${penColor.bg};">`;
    html += `<div class="label criteria-tooltip" title="Штрафы за ошибки (галлюцинации, неверные значения)">penalties</div>`;
    html += `<div class="value" style="color: ${penColor.text};">${penalty}</div>`;
    html += `</div>`;
    html += `</div>`;

    // Детализация покрытия (только для R1, где есть coverage_detail)
    const coverageDetail = data.coverage_detail;
    if (coverageDetail && typeof coverageDetail === 'object') {
        html += `<div class="coverage-details" style="margin-top: 15px; border-top: 1px solid #e2e8f0; padding-top: 10px;">`;
        html += `<strong style="font-size: 14px; color: #2c7da0;">Детализация покрытия:</strong>`;
        for (const crit of criteria) {
            const detail = coverageDetail[crit];
            if (!detail) continue;
            const covered = detail.covered || [];
            const missing = detail.missing || [];
            const label = criteriaLabels[crit] || crit;

            let statusHtml;
            if (covered.length === 0 && missing.length === 0) {
                // Нет фактов в ЭМК
                statusHtml = `<span style="color: #999; font-style: italic;">факты отсутствуют в ЭМК</span>`;
            } else if (missing.length === 0 && covered.length > 0) {
                // Всё покрыто
                statusHtml = `<span style="color: #065f46; font-weight: 500;">✓ всё упомянуто</span>`;
            } else {
                // Есть пропуски
                const missingItems = missing.map(m => `<span style="color: #9b2c2c;">${m}</span>`).join('; ');
                statusHtml = `<span style="color: #9b2c2c; font-weight: 500;">пропущено:</span> ${missingItems}`;
            }

            html += `<div style="margin-top: 6px; font-size: 13px; line-height: 1.4;">`;
            html += `<span style="font-weight: 600; color: #4a5568;">${label}:</span> ${statusHtml}`;
            html += `</div>`;
        }
        html += `</div>`;
    }

    // Детализация штрафов — только если есть
    const hasErrors = (data.hallucinations && data.hallucinations.length)
        || (data.wrong_values && data.wrong_values.length)
        || (data.irrelevant && data.irrelevant.length)
        || data.iodine_missing
        || data.wrong_focus;

    if (hasErrors) {
        html += `<div class="error-details" style="margin-top: 15px; border-top: 1px solid #e2e8f0; padding-top: 10px;">`;
        html += `<strong style="font-size: 14px; color: #e53e3e;">Ошибки:</strong>`;

        if (data.wrong_focus) {
            html += `<div style="margin-top: 6px; font-size: 13px;"><span style="color: #9b2c2c; font-weight: 600;">⚠ Неправильный фокус:</span> суммаризация не относится к КТ ОБП</div>`;
        }
        if (data.iodine_missing) {
            html += `<div style="margin-top: 6px; font-size: 13px;"><span style="color: #9b2c2c; font-weight: 600;">⚠ Аллергия на йод:</span> не указана в суммаризации</div>`;
        }
        if (data.safety_flag && data.safety_reason) {
            html += `<div style="margin-top: 6px; font-size: 13px;"><span style="color: #9b2c2c; font-weight: 600;">🚨 Безопасность:</span> ${data.safety_reason}</div>`;
        }
        if (data.hallucinations && data.hallucinations.length) {
            const items = data.hallucinations.map(h => `<span style="color: #9b2c2c;">${h}</span>`).join('; ');
            html += `<div style="margin-top: 6px; font-size: 13px;"><span style="color: #9b2c2c; font-weight: 600;">Галлюцинации (${data.hallucinations.length}):</span> ${items}</div>`;
        }
        if (data.wrong_values && data.wrong_values.length) {
            const items = data.wrong_values.map(v => `<span style="color: #9b2c2c;">${v}</span>`).join('; ');
            html += `<div style="margin-top: 6px; font-size: 13px;"><span style="color: #9b2c2c; font-weight: 600;">Неверные значения (${data.wrong_values.length}):</span> ${items}</div>`;
        }
        if (data.irrelevant && data.irrelevant.length) {
            const items = data.irrelevant.map(i => `<span style="color: #92400e;">${i}</span>`).join('; ');
            html += `<div style="margin-top: 6px; font-size: 13px;"><span style="color: #92400e; font-weight: 600;">Нерелевантные данные (${data.irrelevant.length}):</span> ${items}</div>`;
        }
        html += `</div>`;
    }

    html += `</details>`;
    return html;
}

// ── Модальное окно с деталями ───────────────────────────────────
function fillModal(fullResult) {
    const modalBody = document.getElementById('modal-body');
    modalBody.innerHTML = '';

    const criteriaBtn = document.createElement('button');
    criteriaBtn.textContent = 'Критерии оценки';
    criteriaBtn.className = 'btn-criteria';
    criteriaBtn.onclick = showCriteriaModal;
    modalBody.appendChild(criteriaBtn);
    modalBody.appendChild(document.createElement('hr'));

    const r1Div = document.createElement('div');
    r1Div.innerHTML = '<h3>Результаты первого этапа (R1)</h3>';
    if (fullResult.r1_results && fullResult.r1_results.length) {
        fullResult.r1_results.forEach((res, idx) => {
            r1Div.innerHTML += renderEvaluationCard(res, `Оценка №${idx+1}`);
        });
    } else {
        r1Div.innerHTML += '<p>Нет данных R1</p>';
    }
    modalBody.appendChild(r1Div);

    const r2Div = document.createElement('div');
    r2Div.innerHTML = '<h3>Результаты пересмотра (R2)</h3>';
    if (fullResult.r2_results && fullResult.r2_results.length) {
        fullResult.r2_results.forEach((res, idx) => {
            r2Div.innerHTML += renderEvaluationCard(res, `Оценка №${idx+1}`, true);
        });
    } else {
        r2Div.innerHTML += '<p>Нет данных R2</p>';
    }
    modalBody.appendChild(r2Div);

    const finalDiv = document.createElement('div');
    finalDiv.innerHTML = '<h3>Финальный вердикт (R3)</h3>';
    if (fullResult.final) {
        const final = fullResult.final;
        finalDiv.innerHTML += `<div class="score-badge ${getFinalScoreClass(final.final_score)}">Итоговый балл: ${final.final_score} (${getQualityText(final.final_score)})</div>`;
        if (final.verdict) finalDiv.innerHTML += `<p><strong>Вердикт:</strong> ${final.verdict}</p>`;
        if (final.criteria) {
            finalDiv.innerHTML += `<div class="criteria-grid">`;
            const critMap = final.criteria;
            const maxMap = { complaints:15, disease_history:15, comorbidities:20, habits:5, labs:20, imaging:25 };
            for (const [key, val] of Object.entries(critMap)) {
                if (key === 'penalties') {
                    const color = getPenaltyColor(val);
                    finalDiv.innerHTML += `<div class="criteria-card penalty-card" style="border-left-color: ${color.bg};"><div class="label">${key}</div><div class="value" style="color: ${color.text};">${val}</div></div>`;
                } else {
                    const max = maxMap[key];
                    const color = getScoreColor(val, max);
                    finalDiv.innerHTML += `<div class="criteria-card" style="border-left-color: ${color.bg};"><div class="label">${key}</div><div class="value" style="color: ${color.text};">${val} / ${max}</div></div>`;
                }
            }
            finalDiv.innerHTML += `</div>`;
        }
        if (final.all_hallucinations && final.all_hallucinations.length) {
            finalDiv.innerHTML += `<div><strong>Все галлюцинации:</strong> ${final.all_hallucinations.join(', ')}</div>`;
        }
    } else {
        finalDiv.innerHTML += '<p>Нет данных R3</p>';
    }
    modalBody.appendChild(finalDiv);
}

// ── Отображение результата на странице ──────────────────────────
function displayResult(container, source, summary, fullResult) {
    const final = fullResult.final;
    if (!final) {
        displayResultInContainer(container, '<strong>Ошибка:</strong> Не удалось получить результат оценки');
        container.classList.add('error');
        return;
    }
    const score = final.final_score;
    const quality = final.quality;
    const safety = final.safety_flag ? '🚨' : '✅';
    const iodine = final.iodine_flag ? '⚠' : '✓';
    let html = `<strong>Исходная суммаризация:</strong><br><div id="originalSummaryText">${summary}</div>`;
    html += `<div class="action-buttons" id="originalSummaryButtons"></div><br>`;
    html += `<strong>Финальный результат:</strong> ${score}/100 (${quality}) ${safety} ${iodine}<br>`;
    html += `<button class="btn-details" data-full='${JSON.stringify(fullResult).replace(/'/g, "&apos;")}'>Показать детали</button>`;
    const improveData = { source, summary, r1Results: fullResult.r1_results_full || fullResult.r1_results };
    const improvedKey = `${source}_${summary}`;
    const cachedImproved = window.improvedSummaries ? window.improvedSummaries[improvedKey] : null;
    if (cachedImproved) {
        html += `<button class="btn-improved" data-improved='${cachedImproved.replace(/'/g, "&apos;")}' data-original-summary='${summary.replace(/'/g, "&apos;")}'>Улучшенная суммаризация</button>`;
    } else {
        html += `<button class="btn-improve" data-source='${source.replace(/'/g, "&apos;")}' data-summary='${summary.replace(/'/g, "&apos;")}' data-r1='${JSON.stringify(improveData.r1Results).replace(/'/g, "&apos;")}'>Улучшить суммаризацию</button>`;
    }
    container.innerHTML = '';
    const contentWrapper = createResultContainer(html, container);
    container.appendChild(contentWrapper);
    container.classList.remove('error');
    container.style.display = 'block';

    const originalSummaryText = document.getElementById('originalSummaryText').innerText;
    const originalBtnContainer = document.getElementById('originalSummaryButtons');
    addCopyDownloadButtons(originalBtnContainer, originalSummaryText, '#originalSummaryButtons');

    const detailsBtn = contentWrapper.querySelector('.btn-details');
    if (detailsBtn) {
        detailsBtn.addEventListener('click', () => {
            const full = JSON.parse(detailsBtn.getAttribute('data-full'));
            fillModal(full);
            document.getElementById('modal').style.display = 'flex';
        });
    }
    const improveBtn = contentWrapper.querySelector('.btn-improve');
    const improvedBtn = contentWrapper.querySelector('.btn-improved');
    if (improveBtn) {
        improveBtn.addEventListener('click', async (e) => {
            const btn = e.target;
            const src = btn.getAttribute('data-source');
            const sum = btn.getAttribute('data-summary');
            const r1Results = JSON.parse(btn.getAttribute('data-r1'));
            await improveSummary(src, sum, r1Results, btn, container);
        });
    }
    if (improvedBtn) {
        improvedBtn.addEventListener('click', () => {
            const improvedText = improvedBtn.getAttribute('data-improved');
            const originalSummary = improvedBtn.getAttribute('data-original-summary');
            showCompareModal(originalSummary, improvedText);
        });
    }
}

window.improvedSummaries = window.improvedSummaries || {};

async function improveSummary(source, summary, r1Results, improveBtn, resultContainer) {
    const payload = {
        source: source,
        summary: summary,
        r1_results_full: r1Results
    };
    disableButton(improveBtn, 'Улучшение...');
    try {
        const res = await fetch('/improve_summarization', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (res.ok) {
            const improvedText = data.improved_summary;
            const key = `${source}_${summary}`;
            window.improvedSummaries[key] = improvedText;
            const newBtn = document.createElement('button');
            newBtn.className = 'btn-improved';
            newBtn.textContent = 'Улучшенная суммаризация';
            newBtn.setAttribute('data-improved', improvedText);
            newBtn.setAttribute('data-original-summary', summary);
            newBtn.addEventListener('click', () => showCompareModal(summary, improvedText));
            improveBtn.replaceWith(newBtn);
            showCompareModal(summary, improvedText);
        } else {
            alert(`Ошибка улучшения: ${data.detail || data}`);
        }
    } catch (err) {
        alert(`Ошибка сети: ${err}`);
    } finally {
        enableButton(improveBtn);
    }
}

function showCompareModal(originalSummary, improvedText) {
    const modal = document.getElementById('compareModal');
    const oldSummaryDiv = document.getElementById('oldSummaryText');
    const newSummaryDiv = document.getElementById('newSummaryText');
    oldSummaryDiv.innerText = originalSummary;
    newSummaryDiv.innerText = improvedText;
    const improvedButtonsContainer = document.getElementById('improvedSummaryButtons');
    improvedButtonsContainer.innerHTML = '';
    addCopyDownloadButtons(improvedButtonsContainer, improvedText, '#improvedSummaryButtons');
    modal.style.display = 'flex';
}

// ── Закрытие модальных окон ─────────────────────────────────────
function setupModalClose(modalId, closeSelector) {
    const modal = document.getElementById(modalId);
    const closeBtn = modal.querySelector(closeSelector);
    closeBtn.addEventListener('click', () => modal.style.display = 'none');
    window.addEventListener('click', (e) => {
        if (e.target === modal) modal.style.display = 'none';
    });
}

setupModalClose('modal', '.close');
setupModalClose('criteriaModal', '#criteriaClose');
setupModalClose('compareModal', '.compare-close');

// ── Загрузка файлов ─────────────────────────────────────────────
async function uploadFileToServer(file, targetTextarea, fileNameSpan, fileBadge, clearBtn, fileInput) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch('/extract_text', { method: 'POST', body: formData });
    if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Ошибка извлечения текста');
    }
    const data = await res.json();
    targetTextarea.value = data.text;
    fileNameSpan.textContent = file.name;
    fileBadge.style.display = 'inline-flex';
    clearBtn.onclick = () => {
        targetTextarea.value = '';
        fileBadge.style.display = 'none';
        if (fileInput) fileInput.value = '';
    };
}

function setupManualFileUpload(uploadBtnId, fileInputId, targetTextareaId, fileNameBadgeId, fileNameSpanId, clearBtnId) {
    const uploadBtn = document.getElementById(uploadBtnId);
    const fileInput = document.getElementById(fileInputId);
    const textarea = document.getElementById(targetTextareaId);
    const fileBadge = document.getElementById(fileNameBadgeId);
    const fileNameSpan = document.getElementById(fileNameSpanId);
    const clearBtn = document.getElementById(clearBtnId);
    uploadBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        uploadBtn.disabled = true;
        uploadBtn.innerText = 'Загрузка...';
        try {
            await uploadFileToServer(file, textarea, fileNameSpan, fileBadge, clearBtn, fileInput);
        } catch (err) {
            alert(err.message);
        } finally {
            uploadBtn.disabled = false;
            uploadBtn.innerText = 'Загрузить файл для ' + (targetTextareaId === 'manualSource' ? 'ЭМК' : 'суммаризации');
            fileInput.value = '';
        }
    });
    clearBtn.onclick = () => {
        textarea.value = '';
        fileBadge.style.display = 'none';
        fileInput.value = '';
    };
}

setupManualFileUpload('uploadSourceBtn', 'sourceFileUpload', 'manualSource', 'sourceFileNameBadge', 'sourceFileName', 'sourceClearFileBtn');
setupManualFileUpload('uploadSummaryBtn', 'summaryFileUpload', 'manualSummary', 'summaryFileNameBadge', 'summaryFileName', 'summaryClearFileBtn');

// ── Preview файла на вкладке 1 ──────────────────────────────────
const evalFileInput = document.getElementById('evalFile');
const evalFilePreview = document.getElementById('evalFilePreview');
const evalFileNameSpan = document.getElementById('evalFileName');
const evalClearBtn = document.getElementById('evalClearFileBtn');
const evalFileText = document.getElementById('evalFileText');

evalFileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) {
        evalFilePreview.style.display = 'none';
        evalFileText.value = '';
        return;
    }
    evalFilePreview.style.display = 'block';
    evalFileNameSpan.textContent = file.name;
    try {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('/extract_text', { method: 'POST', body: formData });
        if (!res.ok) throw new Error('Ошибка извлечения текста');
        const data = await res.json();
        evalFileText.value = data.text;
    } catch (err) {
        alert(err.message);
        evalFilePreview.style.display = 'none';
        evalFileInput.value = '';
    }
});

evalClearBtn.addEventListener('click', () => {
    evalFileInput.value = '';
    evalFilePreview.style.display = 'none';
    evalFileText.value = '';
});

// ── Preview файла на вкладке 3 ──────────────────────────────────
const sumFileInput = document.getElementById('sumFile');
const sumFilePreview = document.getElementById('sumFilePreview');
const sumFileNameSpan = document.getElementById('sumFileName');
const sumClearBtn = document.getElementById('sumClearFileBtn');
const sumFileText = document.getElementById('sumFileText');

sumFileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) {
        sumFilePreview.style.display = 'none';
        sumFileText.value = '';
        return;
    }
    sumFilePreview.style.display = 'block';
    sumFileNameSpan.textContent = file.name;
    try {
        const formData = new FormData();
        formData.append('file', file);
        const res = await fetch('/extract_text', { method: 'POST', body: formData });
        if (!res.ok) throw new Error('Ошибка извлечения текста');
        const data = await res.json();
        sumFileText.value = data.text;
    } catch (err) {
        alert(err.message);
        sumFilePreview.style.display = 'none';
        sumFileInput.value = '';
    }
});

sumClearBtn.addEventListener('click', () => {
    sumFileInput.value = '';
    sumFilePreview.style.display = 'none';
    sumFileText.value = '';
});

// ── Валидация ручной формы ──────────────────────────────────────
function validateManualForm(source, summary) {
    if (!source.trim() || !summary.trim()) {
        alert('Оба поля должны быть заполнены.');
        return false;
    }
    const sourcePrefix = source.slice(0, 200).trim();
    const summaryPrefix = summary.slice(0, 200).trim();
    if (sourcePrefix === summaryPrefix) {
        alert('Текст ЭМК и суммаризация не должны совпадать (проверены первые 200 символов).');
        return false;
    }
    return true;
}

// ── Загрузка файла → асинхронная оценка ─────────────────────────
const evalForm = document.getElementById('evaluateForm');
const evalSubmitBtn = document.getElementById('evalSubmitBtn');
evalForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = evalFileInput.files[0];
    if (!file) {
        alert('Пожалуйста, выберите файл.');
        return;
    }
    const formData = new FormData(evalForm);
    disableButton(evalSubmitBtn);
    try {
        const res = await fetch('/upload_for_evaluation', { method: 'POST', body: formData });
        const data = await res.json();
        const resultDiv = document.getElementById('evalResult');
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = `<strong>Суммаризация получена:</strong><br>${data.summary}<br><br><strong>Task ID:</strong> ${data.task_id}<br>Оценка выполняется...`;
        pollStatus(data.task_id, resultDiv, data.summary);
    } catch (err) {
        alert(`Ошибка: ${err}`);
        enableButton(evalSubmitBtn);
    }
});

// ── Ручная оценка (синхронная) ──────────────────────────────────
const manualForm = document.getElementById('manualEvalForm');
const manualEvalBtn = document.getElementById('manualEvalBtn');
manualForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const source = document.getElementById('manualSource').value;
    const summary = document.getElementById('manualSummary').value;
    if (!validateManualForm(source, summary)) return;
    const payload = { source, summary };
    disableButton(manualEvalBtn);
    try {
        const res = await fetch('/evaluate_text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const result = await res.json();
        const resultDiv = document.getElementById('manualResult');
        resultDiv.style.display = 'block';
        if (res.ok) {
            displayResult(resultDiv, source, summary, result);
        } else {
            displayResultInContainer(resultDiv, `<strong>Ошибка:</strong> ${result.detail || result}`);
            resultDiv.classList.add('error');
        }
    } catch (err) {
        alert(`Ошибка: ${err}`);
    } finally {
        enableButton(manualEvalBtn);
    }
});

// ── Суммаризация файлом ─────────────────────────────────────────
const sumForm = document.getElementById('summarizeForm');
const summarizeBtn = document.getElementById('summarizeBtn');
sumForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const file = sumFileInput.files[0];
    if (!file) {
        alert('Пожалуйста, выберите файл.');
        return;
    }
    const formData = new FormData(sumForm);
    disableButton(summarizeBtn);
    try {
        const res = await fetch('/summarize', { method: 'POST', body: formData });
        const data = await res.json();
        const resultDiv = document.getElementById('summarizeResult');
        resultDiv.style.display = 'block';
        if (res.ok) {
            const html = `<strong>Суммаризация:</strong><br><div id="summarizedText">${data.summary}</div><div class="action-buttons" id="summarizeButtons"></div>`;
            displayResultInContainer(resultDiv, html);
            const summaryText = document.getElementById('summarizedText').innerText;
            const btnContainer = document.getElementById('summarizeButtons');
            addCopyDownloadButtons(btnContainer, summaryText, '#summarizeButtons');
        } else {
            displayResultInContainer(resultDiv, `<strong>Ошибка:</strong> ${data.detail || data}`);
            resultDiv.classList.add('error');
        }
    } catch (err) {
        alert(`Ошибка: ${err}`);
    } finally {
        enableButton(summarizeBtn);
    }
});

// ── Ручная суммаризация текста ──────────────────────────────────
const manualSummarizeForm = document.getElementById('manualSummarizeForm');
const manualSummarizeBtn = document.getElementById('manualSummarizeBtn');
manualSummarizeForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const text = document.getElementById('manualSummarizeText').value.trim();
    if (!text) {
        alert('Введите текст для суммаризации');
        return;
    }
    disableButton(manualSummarizeBtn);
    try {
        const res = await fetch('/summarize_text', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        const data = await res.json();
        const resultDiv = document.getElementById('summarizeResult');
        resultDiv.style.display = 'block';
        if (res.ok) {
            const html = `<strong>Суммаризация:</strong><br><div id="summarizedTextManual">${data.summary}</div><div class="action-buttons" id="summarizeButtonsManual"></div>`;
            displayResultInContainer(resultDiv, html);
            const summaryText = document.getElementById('summarizedTextManual').innerText;
            const btnContainer = document.getElementById('summarizeButtonsManual');
            addCopyDownloadButtons(btnContainer, summaryText, '#summarizeButtonsManual');
        } else {
            displayResultInContainer(resultDiv, `<strong>Ошибка:</strong> ${data.detail || data}`);
            resultDiv.classList.add('error');
        }
    } catch (err) {
        alert(`Ошибка: ${err}`);
    } finally {
        enableButton(manualSummarizeBtn);
    }
});

// ── Опрос статуса ───────────────────────────────────────────────
async function pollStatus(taskId, resultDiv, originalSummary) {
    const interval = setInterval(async () => {
        const res = await fetch(`/status/${taskId}`);
        const data = await res.json();
        if (data.status === 'completed') {
            clearInterval(interval);
            displayResult(resultDiv, data.source, data.summary, data.result);
            enableButton(evalSubmitBtn);
        } else if (data.status === 'error') {
            clearInterval(interval);
            resultDiv.innerHTML += `<br><br><strong>Ошибка:</strong> ${data.result?.error || 'Неизвестная ошибка'}`;
            resultDiv.classList.add('error');
            enableButton(evalSubmitBtn);
        } else {
            const statusLine = resultDiv.innerHTML.split('<br>')[0];
            resultDiv.innerHTML = statusLine + `<br>Оценка выполняется... статус: ${data.status}`;
        }
    }, 3000);
}

// ── Кнопки копирования на вкладке примеров ──────────────────────
document.querySelectorAll('.copy-btn').forEach(btn => {
    if (btn.dataset.listener) return;
    btn.dataset.listener = 'true';
    btn.addEventListener('click', () => {
        const targetId = btn.getAttribute('data-copy');
        const textElement = document.getElementById(targetId);
        if (textElement) {
            copyToClipboard(textElement.innerText, btn);
        } else {
            console.error('Элемент с id', targetId, 'не найден');
        }
    });
});
