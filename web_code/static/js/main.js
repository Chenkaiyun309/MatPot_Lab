let currentTaskId = "";
let pollTimer = null;
let selectedPreviewPath = "";
let activeModule = "generate";
let allOutputFiles = [];

const moduleLabels = {
    generate: "Structure Generation",
    relax: "Relaxation",
    phase: "Phase Stability",
    mechanical: "Mechanical Properties",
    stress: "Stress-Strain",
    md: "Molecular Dynamics",
    other: "Other Outputs",
};

const fileTypeOrder = ["plot", "table", "structure", "log", "json", "archive", "other"];

document.addEventListener("DOMContentLoaded", () => {
    // Run each init step independently so a failure in one (e.g. a missing
    // element) cannot prevent later steps — especially loadModels() — from
    // running and leaving the model selector empty.
    const steps = [bindModuleTabs, bindForms, bindActions, refreshFiles, updatePhaseMode, initInstrumentBar, loadModels];
    for (const step of steps) {
        try {
            const result = step();
            if (result && typeof result.catch === "function") {
                result.catch((error) => console.error(`[init] ${step.name} failed:`, error));
            }
        } catch (error) {
            console.error(`[init] ${step.name} failed:`, error);
        }
    }
});

function bindModuleTabs() {
    document.querySelectorAll(".module-tab").forEach((button) => {
        button.addEventListener("click", () => {
            const moduleName = button.dataset.module;
            activeModule = moduleName;
            document.querySelectorAll(".module-tab").forEach((item) => item.classList.toggle("active", item === button));
            document.querySelectorAll(".module-page").forEach((page) => {
                page.classList.toggle("active", page.id === `module-${moduleName}`);
            });
            renderPreviewFileList();
        });
    });
}

function bindForms() {
    document.querySelectorAll(".task-form").forEach((form) => {
        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            await runTask(form.dataset.kind, collectFormData(form));
        });
    });

    document.querySelectorAll(".browse-btn").forEach((button) => {
        button.addEventListener("click", async () => {
            const input = button.closest(".path-line")?.querySelector("input");
            if (!input) return;
            await browseDirectory(input);
        });
    });

    const phaseMode = document.getElementById("phase-mode");
    if (phaseMode) {
        phaseMode.addEventListener("change", updatePhaseMode);
    }
}

function bindActions() {
    document.getElementById("preview-refresh")?.addEventListener("click", refreshFiles);
    document.getElementById("preview-kind-filter")?.addEventListener("change", renderPreviewFileList);
    document.getElementById("open-local-file")?.addEventListener("click", openSelectedFile);
    document.getElementById("clear-log")?.addEventListener("click", () => {
        document.getElementById("run-log").textContent = "Ready.";
    });
    document.getElementById("stop-task")?.addEventListener("click", stopTask);
    document.querySelectorAll("[data-stop]").forEach((button) => {
        button.addEventListener("click", stopTask);
    });
}

function setStopButtonsDisabled(disabled) {
    document.getElementById("stop-task")?.toggleAttribute("disabled", disabled);
    document.querySelectorAll("[data-stop]").forEach((button) => {
        button.toggleAttribute("disabled", disabled);
    });
}

function updatePhaseMode() {
    const phaseMode = document.getElementById("phase-mode");
    document.body.classList.toggle("phase-binary", phaseMode && phaseMode.value === "binary");
}

function collectFormData(form) {
    const data = {};
    const multi = {};
    form.querySelectorAll("input, select").forEach((field) => {
        if (!field.name) return;
        if (field.type === "checkbox") {
            if (field.name === "phases") {
                if (!multi.phases) multi.phases = [];
                if (field.checked) multi.phases.push(field.value);
            } else {
                data[field.name] = field.checked;
            }
            return;
        }
        const value = field.value.trim();
        const fallback = field.dataset.defaultValue || field.dataset.defaultPath || "";
        data[field.name] = value || fallback;
    });
    return { ...data, ...multi };
}

async function runTask(kind, payload) {
    appendLog(`Starting ${kind} Task...`);
    setStatus("running", `${kind} task`);
    const response = await fetch(`/api/run/${kind}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
        appendLog(`[ERROR] ${result.error || "Failed To Start Task"}`);
        setStatus("failed", "Start failed");
        return;
    }
    currentTaskId = result.task_id;
    setStopButtonsDisabled(false);
    appendLog(`Task ID: ${currentTaskId}`);
    startPolling();
}

function startPolling() {
    if (pollTimer) clearInterval(pollTimer);
    pollTask();
    pollTimer = setInterval(pollTask, 1200);
}

async function pollTask() {
    if (!currentTaskId) return;
    const response = await fetch(`/api/task/${currentTaskId}`);
    const task = await response.json();
    if (!response.ok) {
        appendLog(`[ERROR] ${task.error || "Task Not Found"}`);
        clearInterval(pollTimer);
        return;
    }
    renderTask(task);
    if (["completed", "failed", "stopped"].includes(task.status)) {
        clearInterval(pollTimer);
        pollTimer = null;
        setStopButtonsDisabled(true);
        refreshFiles();
    }
}

function renderTask(task) {
    setStatus(task.status, task.id || "Task");
    const logText = (task.logs || []).join("\n");
    const logBox = document.getElementById("run-log");
    logBox.textContent = logText || "Ready.";
    logBox.scrollTop = logBox.scrollHeight;
}

function setStatus(status, title) {
    const pill = document.getElementById("task-status");
    const taskTitle = document.getElementById("task-title");
    pill.className = `status-pill ${status || "idle"}`;
    pill.textContent = sentenceCase(status || "idle");
    taskTitle.textContent = title || "Idle";
    const led = document.getElementById("header-led");
    const headStatus = document.getElementById("header-status");
    if (led) led.dataset.status = status || "idle";
    if (headStatus) headStatus.textContent = sentenceCase(status || "idle");
}

function appendLog(message) {
    const logBox = document.getElementById("run-log");
    const prefix = logBox.textContent && logBox.textContent !== "Ready." ? `${logBox.textContent}\n` : "";
    logBox.textContent = `${prefix}${message}`;
    logBox.scrollTop = logBox.scrollHeight;
}

async function stopTask() {
    if (!currentTaskId) return;
    await fetch(`/api/task/${currentTaskId}/stop`, { method: "POST" });
    await pollTask();
}

let availableModels = [];

// Fallback model list used if /api/models is unreachable, so the selector is
// never left empty and the user can always pick a model.
const FALLBACK_MODELS = [
    { id: "0.3.0", label: "CHGNet 0.3.0 — MPtrj (recommended)", default: true },
    { id: "0.2.0", label: "CHGNet 0.2.0 — legacy MPtrj", default: false },
    { id: "r2scan", label: "CHGNet R2SCAN — transfer-learned", default: false },
];

async function loadModels() {
    const selects = document.querySelectorAll("[data-model-select]");
    if (!selects.length) return;
    try {
        const response = await fetch("/api/models");
        const payload = await response.json();
        availableModels = response.ok && Array.isArray(payload.models) && payload.models.length
            ? payload.models
            : FALLBACK_MODELS;
    } catch (error) {
        console.error("[loadModels] failed to fetch /api/models:", error);
        availableModels = FALLBACK_MODELS;
    }
    const defaultModel = availableModels.find((m) => m.default) || availableModels[0];
    selects.forEach((select) => {
        const previous = select.value;
        select.innerHTML = availableModels.map((model) => (
            `<option value="${escapeHtml(model.id)}">${escapeHtml(model.label)}</option>`
        )).join("");
        const restore = availableModels.some((m) => m.id === previous) ? previous : (defaultModel?.id || "");
        if (restore) select.value = restore;
    });
}

async function refreshFiles() {
    const response = await fetch("/api/files");
    const payload = await response.json();
    if (!response.ok) {
        allOutputFiles = [];
        renderPreviewFileList();
        appendLog(`[ERROR] ${payload.error || "Failed To Load Files"}`);
        return;
    }
    allOutputFiles = payload.files || [];
    renderPreviewFileList();
}

function inferFileKind(path) {
    const lowerPath = path.toLowerCase();
    if (path.startsWith("calculator_runs/generated_structures")) return "generate";
    if (path.startsWith("calculator_runs/relaxed_results")) return "relax";
    if (path.startsWith("calculator_runs/phase_stability_results")) return "phase";
    if (path.startsWith("calculator_runs/mechanical_results")) return "mechanical";
    if (path.startsWith("calculator_runs/stress_strain_results")) return "stress";
    if (path.startsWith("calculator_runs/md_results")) return "md";
    if (path.startsWith("calculator_runs/downloads/")) {
        if (lowerPath.includes("md") || lowerPath.includes("molecular")) return "md";
        if (lowerPath.includes("mechanical")) return "mechanical";
        if (lowerPath.includes("stressstrain") || lowerPath.includes("stress_strain")) return "stress";
        if (lowerPath.includes("phase")) return "phase";
        if (lowerPath.includes("relax")) return "relax";
        return "relax";
    }
    if (path.startsWith("calculator_runs/") && lowerPath.includes("generate")) return "generate";
    return "other";
}

function previewFilterKind() {
    const value = document.getElementById("preview-kind-filter")?.value || "active";
    return value === "active" ? activeModule : value;
}

function filesForPreviewWorkbench() {
    const kind = previewFilterKind();
    const files = kind === "all"
        ? allOutputFiles
        : allOutputFiles.filter((file) => fileKind(file) === kind);
    return files.filter(shouldShowOutputFile);
}

function renderPreviewFileList() {
    const list = document.getElementById("preview-file-list");
    const count = document.getElementById("preview-file-count");
    if (!list) return;
    const files = filesForPreviewWorkbench();
    if (count) count.textContent = `${files.length} ${files.length === 1 ? "File" : "Files"}`;
    if (!files.length) {
        const label = previewFilterKind() === "all" ? "Output" : (moduleLabels[previewFilterKind()] || previewFilterKind());
        const note = previewFilterKind() === "mechanical" ? " Structure files are hidden in this category." : "";
        list.innerHTML = `<div class="preview-empty">No ${escapeHtml(label)} Result Files Found.${escapeHtml(note)}</div>`;
        return;
    }
    list.innerHTML = "";
    const groups = groupedPreviewFiles(files);
    groups.forEach((group) => {
        const details = document.createElement("details");
        details.className = "preview-file-group";
        details.open = groups.length === 1;
        details.innerHTML = `
            <summary class="preview-file-group-head">
                <span class="preview-file-group-name">${escapeHtml(group.label)}</span>
                <span class="preview-file-group-count">${group.files.length}</span>
            </summary>
        `;
        group.files.forEach((file) => {
            const item = document.createElement("button");
            item.type = "button";
            item.className = "preview-file-item";
            item.innerHTML = `
                <span class="preview-file-main">
                    <strong title="${escapeHtml(file.path)}">${escapeHtml(displayFileName(file))}</strong>
                    <span title="${escapeHtml(file.path)}">${escapeHtml(compactFileContext(file))} · ${formatBytes(file.size)}</span>
                </span>
                <span class="preview-file-type">${escapeHtml(fileTypeLabel(file))}</span>
            `;
            item.addEventListener("click", () => previewWorkbenchFile(file.path));
            details.appendChild(item);
        });
        list.appendChild(details);
    });
}

function groupedPreviewFiles(files) {
    const groups = new Map();
    files.forEach((file) => {
        const type = fileTypeGroup(file);
        if (!groups.has(type.key)) groups.set(type.key, { ...type, files: [] });
        groups.get(type.key).files.push(file);
    });
    return fileTypeOrder
        .filter((key) => groups.has(key))
        .map((key) => groups.get(key));
}

async function previewWorkbenchFile(path) {
    selectedPreviewPath = path;
    document.getElementById("visual-title").textContent = path;
    document.getElementById("visual-kind").textContent = previewKindLabel(path);
    const openButton = document.getElementById("open-local-file");
    if (openButton) openButton.disabled = false;
    const stage = document.getElementById("visual-stage");
    stage.innerHTML = `<div class="visual-empty">Loading ${escapeHtml(path)}...</div>`;
    const response = await fetch(`/api/preview?path=${encodeURIComponent(path)}`);
    const payload = await response.json();
    if (!response.ok) {
        stage.innerHTML = `<div class="visual-empty">${escapeHtml(payload.error || "Preview Failed.")}</div>`;
        return;
    }
    renderWorkbenchPayload(stage, path, payload);
}

async function openSelectedFile() {
    if (!selectedPreviewPath) return;
    const button = document.getElementById("open-local-file");
    const originalText = button?.textContent || "Open local file";
    if (button) {
        button.disabled = true;
        button.textContent = "Opening...";
    }
    try {
        const response = await fetch("/api/open_file", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path: selectedPreviewPath }),
        });
        const payload = await response.json();
        if (!response.ok) {
            appendLog(`[ERROR] ${payload.error || "Failed To Open Local File"}`);
            return;
        }
        appendLog(`Opened Local File: ${payload.opened || selectedPreviewPath}`);
    } catch (error) {
        appendLog(`[ERROR] Failed To Open Local File: ${error}`);
    } finally {
        if (button) {
            button.disabled = false;
            button.textContent = originalText;
        }
    }
}

function renderWorkbenchPayload(stage, path, payload) {
    if (payload.type === "image") {
        stage.innerHTML = `<img class="visual-image" src="/api/raw?path=${encodeURIComponent(path)}" alt="${escapeHtml(path)}">`;
        return;
    }
    if (payload.type === "csv") {
        const { columns, rows } = filterResultColumns(payload.columns || [], payload.rows || []);
        stage.innerHTML = `<div class="visual-table-only">${renderTable(columns, rows)}</div>`;
        return;
    }
    if (payload.type === "text" && isStructurePath(path)) {
        stage.innerHTML = `
            <div class="structure-canvas-wrap">
                <div class="structure-stage-row">
                    <canvas id="structure-preview-canvas" width="980" height="560"></canvas>
                    <div class="structure-axes-panel">
                        <span class="eyebrow">Basis Axes</span>
                        <canvas id="structure-axes-canvas" width="150" height="150" aria-hidden="true"></canvas>
                        <div class="axes-legend">
                            <span class="legend-item"><span class="legend-dot" style="background:#d9483f"></span>a</span>
                            <span class="legend-item"><span class="legend-dot" style="background:#15a37f"></span>b</span>
                            <span class="legend-item"><span class="legend-dot" style="background:#2563eb"></span>c</span>
                        </div>
                    </div>
                </div>
                <div class="structure-meta" id="structure-preview-meta"></div>
            </div>
        `;
        renderStructurePreview(document.getElementById("structure-preview-canvas"), document.getElementById("structure-preview-meta"), payload.content || "", path);
        return;
    }
    if (payload.type === "text") {
        stage.innerHTML = `<pre class="visual-text">${escapeHtml(payload.content || "")}</pre>`;
        return;
    }
    stage.innerHTML = `<div class="visual-empty">Binary Preview Is Not Available For This File Type.</div>`;
}

function previewKindLabel(path) {
    const suffix = path.split(".").pop()?.toUpperCase() || "File";
    if (isStructurePath(path)) return "Structure";
    if (["PNG", "JPG", "JPEG", "WEBP", "SVG"].includes(suffix)) return "Image";
    if (suffix === "CSV") return "CSV + Chart";
    if (["LOG", "TXT", "JSON"].includes(suffix)) return "Text";
    return suffix;
}

function displayFileName(file) {
    return file.name || (file.path || "").split("/").pop() || file.path || "Untitled file";
}

function compactFileContext(file) {
    const path = file.path || "";
    const parts = path.split("/").filter(Boolean);
    if (parts.length <= 1) return moduleLabels[file.kind] || file.kind || "Output";
    const parent = parts.at(-2) || "";
    const root = parts.length > 2 ? parts.at(-3) : "";
    if (parent && root && !["calculator_runs", "downloads"].includes(parent)) return `${root}/${parent}`;
    if (parent) return parent;
    return moduleLabels[file.kind] || "Output";
}

function fileKind(file) {
    return file.kind || inferFileKind(file.path || "");
}

function shouldShowOutputFile(file) {
    if (fileKind(file) !== "mechanical") return true;
    return !isStructurePath(file.path || file.name || "");
}

function fileTypeGroup(file) {
    const path = file.path || file.name || "";
    const suffix = (file.suffix || path.split(".").pop() || "").replace(".", "").toLowerCase();
    if (isStructurePath(path)) return { key: "structure", label: "Structure Files" };
    if (["png", "jpg", "jpeg", "webp", "svg"].includes(suffix)) return { key: "plot", label: "Plots And Images" };
    if (suffix === "csv") return { key: "table", label: "Tables And Curves" };
    if (["log", "txt"].includes(suffix)) return { key: "log", label: "Logs And Text" };
    if (suffix === "json") return { key: "json", label: "JSON Metadata" };
    if (suffix === "zip") return { key: "archive", label: "Archives" };
    return { key: "other", label: "Other Files" };
}

function fileTypeLabel(file) {
    const path = file.path || file.name || "";
    const suffix = (file.suffix || path.split(".").pop() || "").replace(".", "").toLowerCase();
    if (isStructurePath(path)) return "Structure";
    if (["png", "jpg", "jpeg", "webp", "svg"].includes(suffix)) return "Plot";
    if (suffix === "csv") return "Table";
    if (suffix === "json") return "JSON";
    if (["log", "txt"].includes(suffix)) return "Log";
    if (suffix === "zip") return "Archive";
    return suffix ? suffix.toUpperCase() : "File";
}

function isStructurePath(path) {
    const lower = path.toLowerCase();
    return lower.endsWith(".poscar")
        || lower.endsWith(".vasp")
        || lower.endsWith(".cif")
        || lower.endsWith(".xyz")
        || lower.endsWith(".traj")
        || lower.includes("poscar")
        || lower.includes("contcar");
}

function renderTable(columns, rows) {
    const head = columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("");
    const body = rows.map((row) => {
        const cells = columns.map((column) => `<td>${escapeHtml(String(row[column] ?? ""))}</td>`).join("");
        return `<tr>${cells}</tr>`;
    }).join("");
    return `<div class="preview-table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

/* Filter a CSV preview down to result columns.
   Hides file-path / internal-reference / bookkeeping columns so the preview
   shows calculation results (plus row-identifying labels), not how the run
   was stored on disk. Pure data tables (no hidden columns) are shown whole. */
const CSV_HIDDEN_COLUMN_PATTERNS = [
    "input_file", "poscar_output", "output_poscar", "trajectory_file",
    "optimizer_log", "step_csv", "curve_csv", "plot_file",
    "elastic_tensor_csv", "response_csv", "properties_json",
];
const CSV_HIDDEN_EXACT = new Set([
    "status", "error", "num_recorded_steps", "num_successful_evaluations",
    "num_successful_points", "relax_atoms", "strain_max", "num_points_per_mode",
    "num_points", "strain_mode", "strain_start", "strain_end",
]);

function filterResultColumns(columns, rows) {
    const kept = columns.filter((column) => {
        const lower = column.toLowerCase();
        if (CSV_HIDDEN_EXACT.has(lower)) return false;
        if (CSV_HIDDEN_COLUMN_PATTERNS.some((pattern) => lower.includes(pattern))) return false;
        return true;
    });
    if (kept.length === 0) return { columns, rows };
    return { columns: kept, rows };
}

function drawCsvChart(canvas, columns, rows) {
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const width = canvas.clientWidth || 720;
    const height = canvas.clientHeight || 280;
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0b1f3a";
    ctx.fillRect(0, 0, width, height);
    const numericColumns = columns.filter((column) => rows.some((row) => Number.isFinite(Number(row[column]))));
    if (!rows.length || !numericColumns.length) {
        drawCanvasMessage(ctx, width, height, "No Numeric Data Available");
        return;
    }
    const xColumn = numericColumns.length > 1 ? numericColumns[0] : null;
    const yColumn = numericColumns.length > 1 ? numericColumns[1] : numericColumns[0];
    const points = rows.map((row, index) => {
        const x = xColumn ? Number(row[xColumn]) : index;
        const y = Number(row[yColumn]);
        return { x, y };
    }).filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
    if (points.length < 2) {
        drawCanvasMessage(ctx, width, height, "Not Enough Numeric Points");
        return;
    }
    const pad = 44;
    const minX = Math.min(...points.map((point) => point.x));
    const maxX = Math.max(...points.map((point) => point.x));
    const minY = Math.min(...points.map((point) => point.y));
    const maxY = Math.max(...points.map((point) => point.y));
    const xSpan = maxX - minX || 1;
    const ySpan = maxY - minY || 1;
    const mapX = (x) => pad + ((x - minX) / xSpan) * (width - pad * 1.6);
    const mapY = (y) => height - pad - ((y - minY) / ySpan) * (height - pad * 1.7);
    ctx.strokeStyle = "rgba(125,211,252,0.16)";
    ctx.lineWidth = 1;
    for (let i = 0; i < 6; i += 1) {
        const gx = pad + i * ((width - pad * 1.6) / 5);
        const gy = pad * 0.7 + i * ((height - pad * 1.7) / 5);
        ctx.beginPath(); ctx.moveTo(gx, pad * 0.7); ctx.lineTo(gx, height - pad); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(pad, gy); ctx.lineTo(width - pad * 0.6, gy); ctx.stroke();
    }
    ctx.strokeStyle = "#22d3ee";
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    points.forEach((point, index) => {
        const x = mapX(point.x);
        const y = mapY(point.y);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#e8f6ff";
    ctx.font = "600 13px 'JetBrains Mono', monospace";
    ctx.fillText(`${yColumn}${xColumn ? ` vs ${xColumn}` : ""}`, pad, 24);
}

function renderStructurePreview(canvas, meta, content, path) {
    const parsed = path.toLowerCase().endsWith(".cif") ? parseCifStructure(content) : parsePoscarStructure(content);
    if (!parsed.atoms.length) {
        const ctx = canvas.getContext("2d");
        drawCanvasMessage(ctx, canvas.width, canvas.height, "Could Not Parse Structure Atoms");
        meta.textContent = "No Atom Coordinates Detected.";
        return;
    }
    setupInteractiveStructureViewer(canvas, parsed.atoms);
    const elements = [...new Set(parsed.atoms.map((atom) => atom.element))];
    const legend = elements.map((element) =>
        `<span class="legend-item"><span class="legend-dot" style="background:${elementColor(element)}"></span>${escapeHtml(element)}</span>`
    ).join("");
    meta.innerHTML = `
        <strong>${escapeHtml(parsed.title || path)}</strong>
        <span class="legend">${legend}</span>
        <span>${parsed.atoms.length} Atoms · Drag / Wheel</span>
    `;
}

function parsePoscarStructure(text) {
    const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
    if (lines.length < 8) return { title: "", atoms: [] };
    const title = lines[0];
    const elements = lines[5].split(/\s+/);
    const counts = lines[6].split(/\s+/).map(Number);
    let coordIndex = 7;
    if (/^s/i.test(lines[coordIndex])) coordIndex += 1;
    const direct = /^d/i.test(lines[coordIndex]);
    coordIndex += 1;
    const atoms = [];
    elements.forEach((element, elementIndex) => {
        const count = counts[elementIndex] || 0;
        for (let i = 0; i < count; i += 1) {
            const parts = (lines[coordIndex++] || "").split(/\s+/).slice(0, 3).map(Number);
            if (parts.every(Number.isFinite)) atoms.push({ element, x: parts[0], y: parts[1], z: parts[2], direct });
        }
    });
    return { title, atoms: normalizeAtoms(atoms) };
}

function parseCifStructure(text) {
    const lines = text.split(/\r?\n/);
    let title = "CIF structure";
    const titleMatch = text.match(/^data_(\S+)/m);
    if (titleMatch) title = titleMatch[1];

    // Locate the atom-site loop, read its column headers, then map the data
    // rows by column name — robust to extra columns (multiplicity, occupancy,
    // label) that previously shifted the x/y/z readout.
    let atoms = [];
    let i = lines.findIndex((line) => line.trim().toLowerCase() === "loop_");
    while (i !== -1) {
        const headers = [];
        let j = i + 1;
        while (j < lines.length && lines[j].trim().startsWith("_")) {
            headers.push(lines[j].trim().toLowerCase());
            j += 1;
        }
        const hasSymbol = headers.some((h) => h.includes("type_symbol"));
        const hasLabel = headers.some((h) => h.includes("label"));
        const hasX = headers.some((h) => h.includes("fract_x"));
        const hasY = headers.some((h) => h.includes("fract_y"));
        const hasZ = headers.some((h) => h.includes("fract_z"));
        if ((hasSymbol || hasLabel) && hasX && hasY && hasZ) {
            const iSym = headers.findIndex((h) => h.includes("type_symbol"));
            const iLab = headers.findIndex((h) => h.includes("label"));
            const iX = headers.findIndex((h) => h.includes("fract_x"));
            const iY = headers.findIndex((h) => h.includes("fract_y"));
            const iZ = headers.findIndex((h) => h.includes("fract_z"));
            while (j < lines.length && lines[j].trim() && !lines[j].trim().startsWith("_") && lines[j].trim().toLowerCase() !== "loop_") {
                const parts = lines[j].trim().split(/\s+/);
                let element = iSym >= 0 ? parts[iSym] : "";
                if (!/^[A-Z][a-z]?$/.test(element) && iLab >= 0) {
                    const label = parts[iLab];
                    element = label.match(/^[A-Za-z]+/)?.[0] || label;
                }
                const x = Number(parts[iX]);
                const y = Number(parts[iY]);
                const z = Number(parts[iZ]);
                if ([x, y, z].every(Number.isFinite) && /^[A-Z][a-z]?$/.test(element)) {
                    atoms.push({ element, x, y, z, direct: true });
                }
                j += 1;
            }
            break;
        }
        const next = lines.slice(j).findIndex((line) => line.trim().toLowerCase() === "loop_");
        if (next === -1) break;
        i = j + next;
    }
    return { title, atoms: normalizeAtoms(atoms) };
}

function normalizeAtoms(atoms) {
    if (!atoms.length) return atoms;
    const maxAbs = Math.max(...atoms.flatMap((atom) => [Math.abs(atom.x), Math.abs(atom.y), Math.abs(atom.z)]), 1);
    if (maxAbs <= 1.5) return atoms;
    const min = {
        x: Math.min(...atoms.map((atom) => atom.x)),
        y: Math.min(...atoms.map((atom) => atom.y)),
        z: Math.min(...atoms.map((atom) => atom.z)),
    };
    const max = {
        x: Math.max(...atoms.map((atom) => atom.x)),
        y: Math.max(...atoms.map((atom) => atom.y)),
        z: Math.max(...atoms.map((atom) => atom.z)),
    };
    return atoms.map((atom) => ({
        ...atom,
        x: (atom.x - min.x) / ((max.x - min.x) || 1),
        y: (atom.y - min.y) / ((max.y - min.y) || 1),
        z: (atom.z - min.z) / ((max.z - min.z) || 1),
    }));
}

function setupInteractiveStructureViewer(canvas, atoms) {
    const state = {
        atoms,
        rotX: -0.58,
        rotY: 0.72,
        zoom: 1.0,
        dragging: false,
        lastX: 0,
        lastY: 0,
        bonds: buildStructureBonds(atoms),
    };
    const redraw = () => {
        drawStructureCanvas(canvas, state);
        const axesCanvas = document.getElementById("structure-axes-canvas");
        if (axesCanvas) drawAxesTriad(axesCanvas, state);
    };
    if (window.ResizeObserver) {
        const resizeObserver = new ResizeObserver(redraw);
        resizeObserver.observe(canvas);
        const axesCanvas = document.getElementById("structure-axes-canvas");
        if (axesCanvas) resizeObserver.observe(axesCanvas);
        canvas._structureResizeObserver?.disconnect?.();
        canvas._structureResizeObserver = resizeObserver;
    }
    canvas.onpointerdown = (event) => {
        state.dragging = true;
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        canvas.setPointerCapture?.(event.pointerId);
    };
    canvas.onpointermove = (event) => {
        if (!state.dragging) return;
        const dx = event.clientX - state.lastX;
        const dy = event.clientY - state.lastY;
        // Grab-and-follow: the model tilts so the grabbed point chases the
        // cursor. Horizontal follows directly; vertical is inverted because
        // screen Y grows downward while the rotation's "up" is -Y.
        state.rotY += dx * 0.01;
        state.rotX -= dy * 0.01;
        state.rotX = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, state.rotX));
        state.lastX = event.clientX;
        state.lastY = event.clientY;
        redraw();
    };
    canvas.onpointerup = (event) => {
        state.dragging = false;
        canvas.releasePointerCapture?.(event.pointerId);
    };
    canvas.onpointerleave = () => {
        state.dragging = false;
    };
    canvas.onwheel = (event) => {
        event.preventDefault();
        state.zoom *= event.deltaY > 0 ? 0.92 : 1.08;
        state.zoom = Math.max(0.55, Math.min(2.6, state.zoom));
        redraw();
    };
    redraw();
}

function drawStructureCanvas(canvas, state) {
    const atoms = state.atoms;
    const ctx = canvas.getContext("2d");
    const width = canvas.clientWidth || 980;
    const height = canvas.clientHeight || 560;
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    /* light, plain background — no tech grid, no glow */
    ctx.fillStyle = "#f5f8fc";
    ctx.fillRect(0, 0, width, height);
    const projector = createStructureProjector(width, height, state);
    const projected = atoms.map((atom, index) => ({ ...atom, index, ...projector(atom) })).sort((a, b) => a.depth - b.depth);
    drawUnitCell(ctx, projector);
    drawStructureBonds(ctx, projected, state.bonds);
    projected.forEach((atom) => {
        const color = elementColor(atom.element);
        const radius = Math.max(4.5, 7.0 + atom.depth * 3.4) * Math.sqrt(state.zoom);
        ctx.beginPath();
        ctx.fillStyle = color;
        ctx.arc(atom.px, atom.py, radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = "rgba(11,31,58,0.55)";
        ctx.stroke();
    });
}

/* a / b / c basis-axis triad rendered in its own small canvas, sharing the
   main viewer's rotation so it stays in sync. a = red, b = green, c = blue. */
function drawAxesTriad(axesCanvas, state) {
    const ctx = axesCanvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const rect = axesCanvas.getBoundingClientRect();
    const width = Math.max(120, Math.round(rect.width || axesCanvas.clientWidth || 150));
    const height = Math.max(120, Math.round(rect.height || axesCanvas.clientHeight || 150));
    axesCanvas.width = width * dpr;
    axesCanvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, width, height);

    const cosX = Math.cos(state.rotX);
    const sinX = Math.sin(state.rotX);
    const cosY = Math.cos(state.rotY);
    const sinY = Math.sin(state.rotY);
    const size = Math.min(width, height);
    const armLen = size * 0.34;
    const cx = width / 2;
    const cy = height / 2 + size * 0.03;
    const project = (x, y, z) => {
        const x1 = x * cosY + z * sinY;
        const z1 = -x * sinY + z * cosY;
        const y1 = y * cosX - z1 * sinX;
        const z2 = y * sinX + z1 * cosX;
        const perspective = 1.18 / (1.55 - z2 * 0.42);
        return { px: cx + x1 * armLen * perspective, py: cy + y1 * armLen * perspective };
    };

    const axes = [
        { end: project(1, 0, 0), color: "#d9483f", label: "a" },
        { end: project(0, 1, 0), color: "#15a37f", label: "b" },
        { end: project(0, 0, 1), color: "#2563eb", label: "c" },
    ];
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    axes.forEach((axis) => {
        ctx.strokeStyle = axis.color;
        ctx.fillStyle = axis.color;
        ctx.lineCap = "round";
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(axis.end.px, axis.end.py);
        ctx.stroke();

        const angle = Math.atan2(axis.end.py - cy, axis.end.px - cx);
        const head = Math.max(9, size * 0.075);
        ctx.beginPath();
        ctx.moveTo(axis.end.px, axis.end.py);
        ctx.lineTo(axis.end.px - Math.cos(angle - 0.48) * head, axis.end.py - Math.sin(angle - 0.48) * head);
        ctx.lineTo(axis.end.px - Math.cos(angle + 0.48) * head, axis.end.py - Math.sin(angle + 0.48) * head);
        ctx.closePath();
        ctx.fill();

        const lx = cx + (axis.end.px - cx) * 1.22;
        const ly = cy + (axis.end.py - cy) * 1.22;
        ctx.font = "700 14px 'IBM Plex Mono', monospace";
        ctx.fillText(axis.label, lx, ly);
    });

    ctx.beginPath();
    ctx.fillStyle = "#0f233d";
    ctx.arc(cx, cy, 3.2, 0, Math.PI * 2);
    ctx.fill();
}

function createStructureProjector(width, height, state) {
    const cosX = Math.cos(state.rotX);
    const sinX = Math.sin(state.rotX);
    const cosY = Math.cos(state.rotY);
    const sinY = Math.sin(state.rotY);
    const scale = Math.min(width, height) * 0.78 * state.zoom;
    return (point) => {
        const x0 = point.x - 0.5;
        const y0 = point.y - 0.5;
        const z0 = point.z - 0.5;
        const x1 = x0 * cosY + z0 * sinY;
        const z1 = -x0 * sinY + z0 * cosY;
        const y1 = y0 * cosX - z1 * sinX;
        const z2 = y0 * sinX + z1 * cosX;
        const perspective = 1.18 / (1.55 - z2 * 0.42);
        return {
            px: width / 2 + x1 * scale * perspective,
            py: height / 2 + y1 * scale * perspective,
            depth: z2,
        };
    };
}

function drawUnitCell(ctx, projector) {
    const corners = [
        { x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 }, { x: 1, y: 1, z: 0 }, { x: 0, y: 1, z: 0 },
        { x: 0, y: 0, z: 1 }, { x: 1, y: 0, z: 1 }, { x: 1, y: 1, z: 1 }, { x: 0, y: 1, z: 1 },
    ].map(projector);
    const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
    ctx.strokeStyle = "rgba(91,113,140,0.55)";
    ctx.lineWidth = 1.1;
    edges.forEach(([a, b]) => {
        ctx.beginPath();
        ctx.moveTo(corners[a].px, corners[a].py);
        ctx.lineTo(corners[b].px, corners[b].py);
        ctx.stroke();
    });
}

function buildStructureBonds(atoms) {
    const pairs = [];
    const maxAtoms = Math.min(atoms.length, 260);
    const distances = [];
    for (let i = 0; i < maxAtoms; i += 1) {
        for (let j = i + 1; j < maxAtoms; j += 1) {
            const dx = atoms[i].x - atoms[j].x;
            const dy = atoms[i].y - atoms[j].y;
            const dz = atoms[i].z - atoms[j].z;
            const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (distance > 0.001) distances.push({ i, j, distance });
        }
    }
    distances.sort((a, b) => a.distance - b.distance);
    const threshold = Math.min(0.34, (distances[0]?.distance || 0.12) * 1.32);
    for (const item of distances) {
        if (item.distance > threshold || pairs.length > 900) break;
        pairs.push([item.i, item.j]);
    }
    return pairs;
}

function drawStructureBonds(ctx, projected, bonds) {
    const byIndex = new Map(projected.map((atom) => [atom.index, atom]));
    ctx.strokeStyle = "rgba(91,113,140,0.30)";
    ctx.lineWidth = 1.2;
    bonds.forEach(([a, b]) => {
        const atomA = byIndex.get(a);
        const atomB = byIndex.get(b);
        if (!atomA || !atomB) return;
        ctx.beginPath();
        ctx.moveTo(atomA.px, atomA.py);
        ctx.lineTo(atomB.px, atomB.py);
        ctx.stroke();
    });
}

/* Deterministic element→color map (CPK-flavoured). Stable per element so the
   legend always matches the spheres. Unknown elements fall back by hash. */
const ELEMENT_COLORS = {
    H: "#e6e6e6", He: "#c8d6ff", Li: "#c85a7a", Be: "#5fae9a", B: "#f6b73c",
    C: "#2b2b2b", N: "#3b7dd8", O: "#d9483f", F: "#5fd35f", Ne: "#c8d6ff",
    Na: "#c85a7a", Mg: "#5fae9a", Al: "#9aa0a6", Si: "#d9a066", P: "#ff8c42",
    S: "#e6c229", Cl: "#5fd35f", Ar: "#c8d6ff", K: "#c85a7a", Ca: "#5fae9a",
    Sc: "#d9a066", Ti: "#8c8c9a", V: "#6f9ed6", Cr: "#4a8ec2", Mn: "#9c59b6",
    Fe: "#d9655b", Co: "#3b6fb5", Ni: "#44c1c1", Cu: "#c87f3a", Zn: "#7d8cff",
    Ga: "#9aa0a6", Ge: "#4f7cc4", As: "#c87f3a", Se: "#e6c229", Br: "#8a3b2e",
    Rb: "#c85a7a", Sr: "#5fae9a", Y: "#6f9ed6", Zr: "#5fae9a", Nb: "#4a8ec2",
    Mo: "#3b7dd8", Tc: "#3b6fb5", Ru: "#44c1c1", Rh: "#2b8c8c", Pd: "#5fae9a",
    Ag: "#b8b8b8", Cd: "#7d8cff", In: "#9aa0a6", Sn: "#8c8c9a", Sb: "#c87f3a",
    Hf: "#5fae9a", Ta: "#4a8ec2", W: "#3b7dd8", Re: "#3b6fb5", Os: "#2b8c8c",
    Ir: "#2b8c8c", Pt: "#d9a066", Au: "#e0b638", Hg: "#b8b8b8", Tl: "#9aa0a6",
    Pb: "#575757", Bi: "#8a3b2e",
};
const ELEMENT_FALLBACK = ["#6f9ed6", "#5fae9a", "#d9a066", "#c85a7a", "#9c59b6", "#44c1c1", "#e6c229", "#4a8ec2"];

function elementColor(element) {
    const key = element.charAt(0).toUpperCase() + element.slice(1).toLowerCase();
    if (ELEMENT_COLORS[key]) return ELEMENT_COLORS[key];
    let hash = 0;
    for (const char of element) hash += char.charCodeAt(0);
    return ELEMENT_FALLBACK[hash % ELEMENT_FALLBACK.length];
}

function drawCanvasMessage(ctx, width, height, message) {
    ctx.fillStyle = "#f5f8fc";
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = "#5b718c";
    ctx.font = "500 15px 'IBM Plex Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(message, width / 2, height / 2);
}

async function browseDirectory(input) {
    const response = await fetch("/api/select_directory", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ current: input.value }),
    });
    const payload = await response.json();
    if (!response.ok) {
        appendLog(`[ERROR] ${payload.error || "Directory Picker Failed"}`);
        return;
    }
    if (!payload.cancelled && payload.path) {
        input.value = payload.path;
    }
}

function escapeHtml(value) {
    return value
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}

function sentenceCase(value) {
    const text = String(value || "");
    return text ? text.charAt(0).toUpperCase() + text.slice(1).toLowerCase() : "";
}

function formatBytes(value) {
    if (value < 1024) return `${value} B`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
    return `${(value / 1024 / 1024).toFixed(1)} MB`;
}

/* =====================================================================
   Instrument bar — live readout + unit-cell motif (signature element)
   The motif redraws BCC / FCC / HCP to mirror the active generate-tab
   crystal-structure selection, in copper hairline on graphite.
   ===================================================================== */

function initInstrumentBar() {
    drawHeaderMotif("bcc");
    updateHeaderCoord();
    const structureSelect = document.querySelector('#module-generate [name="crystal_structure"]');
    const latticeInput = document.querySelector('#module-generate [name="lattice_constant"]');
    if (structureSelect) {
        structureSelect.addEventListener("change", () => {
            drawHeaderMotif(structureSelect.value);
            updateHeaderCoord();
        });
    }
    if (latticeInput) {
        latticeInput.addEventListener("input", updateHeaderCoord);
    }
}

function updateHeaderCoord() {
    const coord = document.getElementById("header-coord");
    if (!coord) return;
    const select = document.querySelector('#module-generate [name="crystal_structure"]');
    const lattice = document.querySelector('#module-generate [name="lattice_constant"]');
    const structure = (select?.value || "bcc").toUpperCase();
    const a = parseFloat(lattice?.value || "3.0").toFixed(3);
    coord.textContent = `${structure} · a = ${a} Å`;
}

function drawHeaderMotif(structure) {
    const canvas = document.getElementById("header-motif");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const size = canvas.getAttribute("width") ? parseInt(canvas.getAttribute("width"), 10) : 46;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, size, size);

    const copper = "#22d3ee";
    const copperDim = "rgba(125,211,252,0.45)";
    const cx = size / 2;
    const cy = size / 2;
    const a = size * 0.326;       // half cell edge, scaled to canvas
    const corners = [
        [-a, -a, -a], [a, -a, -a], [a, a, -a], [-a, a, -a],
        [-a, -a, a], [a, -a, a], [a, a, a], [-a, a, a],
    ];
    const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];

    // isometric project
    const project = (p) => {
        const [x, y, z] = p;
        const px = cx + (x - z) * 0.62;
        const py = cy + (x + z) * 0.35 - y * 0.78;
        return [px, py];
    };

    // outer cell wireframe
    ctx.strokeStyle = copperDim;
    ctx.lineWidth = 1;
    edges.forEach(([i, j]) => {
        const [x1, y1] = project(corners[i]);
        const [x2, y2] = project(corners[j]);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    });

    // lattice points per structure
    const sites = latticeSites(structure, a);
    ctx.fillStyle = copper;
    const siteRadius = Math.max(1.4, size * 0.046);
    sites.forEach((s) => {
        const [px, py] = project(s);
        ctx.beginPath();
        ctx.arc(px, py, siteRadius, 0, Math.PI * 2);
        ctx.fill();
    });
}

function latticeSites(structure, a) {
    const corners = [
        [-a, -a, -a], [a, -a, -a], [a, a, -a], [-a, a, -a],
        [-a, -a, a], [a, -a, a], [a, a, a], [-a, a, a],
    ];
    if (structure === "fcc") {
        // corners + face centers
        const faces = [
            [0, -a, -a], [0, a, -a], [0, -a, a], [0, a, a],
            [-a, 0, -a], [a, 0, -a], [-a, 0, a], [a, 0, a],
            [-a, -a, 0], [a, -a, 0], [-a, a, 0], [a, a, 0],
        ];
        return [...corners, ...faces];
    }
    if (structure === "hcp") {
        // approximate: two-layer AB stacking motif
        return [
            [-a, -a, -a], [a, -a, -a], [a, a, -a], [-a, a, -a],
            [0, 0, -a],
            [-a, -a, a], [a, -a, a], [a, a, a], [-a, a, a],
            [a * 0.5, a * 0.5, 0], [-a * 0.5, -a * 0.5, 0],
        ];
    }
    // bcc: corners + body center
    return [...corners, [0, 0, 0]];
}
