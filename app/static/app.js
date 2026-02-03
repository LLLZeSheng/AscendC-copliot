const repoSelect = document.getElementById("repoSelect");
const opSearch = document.getElementById("opSearch");
const opList = document.getElementById("opList");
const metaRepo = document.getElementById("metaRepo");
const metaOp = document.getElementById("metaOp");
const heroStatus = document.getElementById("heroStatus");
const repoPanel = document.getElementById("repoPanel");
const repoCollapseBtn = document.getElementById("repoCollapseBtn");
const repoPanelSummary = document.getElementById("repoPanelSummary");
const resetSelectionBtn = document.getElementById("resetSelection");

const shapeInput = document.getElementById("shapeInput");
const shapePreset = document.getElementById("shapePreset");
const dtypeSelect = document.getElementById("dtypeSelect");
const platformSelect = document.getElementById("platformSelect");

const detailReadme = document.getElementById("detailReadme");
const detailFiles = document.getElementById("detailFiles");

const utScript = document.getElementById("utScript");
const utNote = document.getElementById("utNote");

const detectList = document.getElementById("detectList");
const detectNote = document.getElementById("detectNote");
const detectLog = document.getElementById("detectLog");
const workspaceFiles = document.getElementById("workspaceFiles");

const evolveTimeline = document.getElementById("evolveTimeline");
const evolveSummary = document.getElementById("evolveSummary");
const evolveLog = document.getElementById("evolveLog");
const evolveLogNote = document.getElementById("evolveLogNote");
const evolveLogDrawer = document.getElementById("evolveLogDrawer");
const evolveStages = document.getElementById("evolveStages");
const configDrawer = document.getElementById("configDrawer");
const evolveIterLabel = document.getElementById("evolveIterLabel");
const evolveStageLabel = document.getElementById("evolveStageLabel");
const bestVariantLabel = document.getElementById("bestVariantLabel");
const bestCheckpointLabel = document.getElementById("bestCheckpointLabel");
const bestLatencyLabel = document.getElementById("bestLatencyLabel");
const bestProgramCode = document.getElementById("bestProgramCode");
const bestProgramMeta = document.getElementById("bestProgramMeta");
const bestStatusChip = document.getElementById("bestStatusChip");
const bestDrawer = document.getElementById("bestDrawer");
const diffSelectA = document.getElementById("diffSelectA");
const diffSelectB = document.getElementById("diffSelectB");
const runDiffBtn = document.getElementById("runDiff");
const diffOutput = document.getElementById("diffOutput");
const diffSection = document.getElementById("diffSection");
const diffPlaceholder = document.getElementById("diffPlaceholder");
const consoleStage = document.getElementById("consoleStage");
const consoleRepo = document.getElementById("consoleRepo");
const consoleOp = document.getElementById("consoleOp");
const consoleSoc = document.getElementById("consoleSoc");
const consoleStep = document.getElementById("consoleStep");

const loadDetailBtn = document.getElementById("loadDetail");
const downloadPackageBtn = document.getElementById("downloadPackage");
const runUtBtn = document.getElementById("runUt");
const runDetectBtn = document.getElementById("runDetect");
const runEvolveBtn = document.getElementById("runEvolve");
const stepIndicators = document.querySelectorAll(".step");

let operators = [];
let selectedRepo = "";
let selectedOp = "";
let detailLoaded = false;
let shapeReady = false;
let detectDone = false;
let detectRunning = false;
let evolveRunning = false;
let lastWorkspace = {};
let utGenerated = false;
let evolveDone = false;
let bestAvgUs = null;
let bestVariant = null;
let bestCheckpointPath = null;
let lastOutputDir = null;
let evalCounter = 0;
let iterationCounter = 0;
let inInitialEval = false;
let inEval = false;
let evolveLogBuffer = [];
let evolveLogFlushHandle = null;
// timeline entries are tracked in iterationStates
let iterationStates = new Map();
let currentIterKey = null;
let checkpointOptions = new Map();

const STAGE_KEYS = ["replace", "build", "install", "run", "profile"];
const STAGE_LABELS = {
  replace: "Replace",
  build: "Build",
  install: "Install",
  run: "Run",
  profile: "Profile",
};

const tabs = document.querySelectorAll(".tab");
const tabPanels = {
  detail: document.getElementById("tab-detail"),
  ut: document.getElementById("tab-ut"),
  detect: document.getElementById("tab-detect"),
  evolve: document.getElementById("tab-evolve"),
};

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((item) => item.classList.remove("active"));
    tab.classList.add("active");
    Object.values(tabPanels).forEach((panel) => panel.classList.remove("active"));
    const target = tab.dataset.tab;
    tabPanels[target].classList.add("active");
  });
});

function switchTab(target) {
  tabs.forEach((item) => item.classList.remove("active"));
  Object.values(tabPanels).forEach((panel) => panel.classList.remove("active"));
  const tab = Array.from(tabs).find((item) => item.dataset.tab === target);
  if (tab) {
    tab.classList.add("active");
  }
  if (tabPanels[target]) {
    tabPanels[target].classList.add("active");
  }
}

function updateRepoPanelSummary() {
  if (!repoPanelSummary) return;
  if (selectedRepo && selectedOp) {
    repoPanelSummary.textContent = `${selectedRepo} / ${selectedOp}`;
  } else if (selectedRepo) {
    repoPanelSummary.textContent = `${selectedRepo} / 待选择`;
  } else {
    repoPanelSummary.textContent = "尚未选择";
  }
}

function setRepoPanelCollapsed(collapsed) {
  if (!repoPanel) return;
  repoPanel.classList.toggle("collapsed", collapsed);
  if (repoCollapseBtn) {
    repoCollapseBtn.textContent = collapsed ? "展开" : "折叠";
  }
}

function resetBestSummary() {
  bestAvgUs = null;
  bestVariant = null;
  bestCheckpointPath = null;
  if (bestVariantLabel) bestVariantLabel.textContent = "-";
  if (bestCheckpointLabel) bestCheckpointLabel.textContent = "-";
  if (bestLatencyLabel) bestLatencyLabel.textContent = "-";
  if (bestStatusChip) bestStatusChip.textContent = "暂无最佳版本";
  if (bestProgramMeta) bestProgramMeta.textContent = "等待最佳版本输出。";
  if (bestProgramCode) {
    bestProgramCode.textContent = "等待最佳版本输出…";
    bestProgramCode.className = "language-cpp";
  }
  if (bestDrawer) bestDrawer.open = false;
}

function formatMetric(value, digits = 4) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(digits);
}

function extractCheckpointLabel(variant) {
  if (!variant) return "-";
  const match = String(variant).match(/checkpoint_(\d+)/);
  if (match) return `#${match[1]}`;
  return variant;
}

function updateBestSummary() {
  if (bestVariantLabel) bestVariantLabel.textContent = bestVariant || "-";
  if (bestCheckpointLabel) bestCheckpointLabel.textContent = extractCheckpointLabel(bestVariant);
  if (bestLatencyLabel) {
    bestLatencyLabel.textContent =
      typeof bestAvgUs === "number" ? `${formatMetric(bestAvgUs)} us` : "-";
  }
  if (bestStatusChip) {
    bestStatusChip.textContent = bestVariant ? "已找到最佳版本" : "暂无最佳版本";
  }
}

function parseCombinedScore(note) {
  if (!note) return null;
  const match = note.match(/combined_score=([0-9.]+)/);
  if (!match) return null;
  const value = Number(match[1]);
  return Number.isFinite(value) ? value : null;
}

function parseAvgUsFromLog(line) {
  if (!line) return null;
  const patterns = [
    /Avg Time\(us\)\s*=\s*([0-9.]+)/,
    /Avg Time\(us\)\s*:\s*([0-9.]+)/,
    /avg_us\s*=\s*([0-9.]+)/i,
    /avg_us\s*:\s*([0-9.]+)/i,
  ];
  for (const pattern of patterns) {
    const match = line.match(pattern);
    if (!match) continue;
    const value = Number(match[1]);
    if (Number.isFinite(value)) return value;
  }
  return null;
}

function buildStageState(isInitial = false) {
  const state = {};
  STAGE_KEYS.forEach((key) => {
    state[key] = isInitial && key === "replace" ? "skipped" : "pending";
  });
  return state;
}

function applyStageState(stageContainer, stageState) {
  if (!stageContainer || !stageState) return;
  stageContainer.querySelectorAll(".stage-segment").forEach((segment) => {
    const key = segment.dataset.stage;
    segment.classList.remove("active", "done", "skipped");
    const status = stageState[key];
    if (status === "active") segment.classList.add("active");
    if (status === "done") segment.classList.add("done");
    if (status === "skipped") segment.classList.add("skipped");
  });
}

function createStageBar(stageState, size = "mini") {
  const bar = document.createElement("div");
  bar.className = `stage-track ${size}`;
  STAGE_KEYS.forEach((key) => {
    const seg = document.createElement("div");
    seg.className = "stage-segment";
    seg.dataset.stage = key;
    const label = document.createElement("span");
    label.textContent = key;
    seg.appendChild(label);
    bar.appendChild(seg);
  });
  applyStageState(bar, stageState);
  return bar;
}

function createTimelineState(key, label, isInitial = false) {
  const state = {
    key,
    label,
    isInitial,
    variant: isInitial ? "baseline" : "",
    checkpointPath: null,
    avgUs: null,
    score: null,
    logs: [],
    stageState: buildStageState(isInitial),
    programLoaded: false,
    completed: false,
    detailsEl: null,
    summaryEl: null,
    metricsEl: null,
    logEl: null,
    codeEl: null,
    stageEl: null,
    programSection: null,
    logSection: null,
  };
  iterationStates.set(key, state);
  return state;
}

function ensureTimelineEntry(key, label, isInitial = false) {
  let state = iterationStates.get(key);
  if (!state) {
    state = createTimelineState(key, label, isInitial);
  }
  if (state.detailsEl) return state;

  const details = document.createElement("details");
  details.className = "timeline-item pending";
  details.open = false;

  const summary = document.createElement("summary");
  const title = document.createElement("strong");
  title.textContent = label;
  summary.appendChild(title);

  const status = document.createElement("span");
  status.className = "muted";
  status.textContent = "运行中...";
  summary.appendChild(status);

  details.appendChild(summary);

  const body = document.createElement("div");
  body.className = "timeline-body";

  const metrics = document.createElement("div");
  metrics.className = "timeline-metrics";
  metrics.innerHTML = `
    <div class="timeline-metric"><span>Avg Time(us)</span><strong>-</strong></div>
  `;
  body.appendChild(metrics);

  const stageWrap = document.createElement("div");
  stageWrap.className = "timeline-section";
  const stageTitle = document.createElement("h4");
  stageTitle.textContent = "阶段进度";
  stageWrap.appendChild(stageTitle);
  const stageBar = createStageBar(state.stageState, "mini");
  stageWrap.appendChild(stageBar);
  body.appendChild(stageWrap);

  const programSection = document.createElement("details");
  programSection.className = "timeline-section";
  programSection.open = false;
  const programSummary = document.createElement("summary");
  programSummary.textContent = "修改后的 Program";
  programSection.appendChild(programSummary);
  const programPre = document.createElement("pre");
  programPre.className = "code-block";
  const programCode = document.createElement("code");
  programCode.className = "language-cpp";
  programCode.textContent = "展开后加载 program...";
  programPre.appendChild(programCode);
  programSection.appendChild(programPre);
  body.appendChild(programSection);

  const logSection = document.createElement("details");
  logSection.className = "timeline-section";
  logSection.open = false;
  const logSummary = document.createElement("summary");
  logSummary.textContent = "运行日志";
  logSection.appendChild(logSummary);
  const logPre = document.createElement("pre");
  logPre.className = "code-block inline";
  logPre.textContent = "等待日志...";
  logSection.appendChild(logPre);
  body.appendChild(logSection);

  details.appendChild(body);

  details.addEventListener("toggle", () => {
    if (details.open && state) {
      programSection.open = false;
      logSection.open = false;
      loadCheckpointProgram(state);
    }
  });

  evolveTimeline.appendChild(details);

  state.detailsEl = details;
  state.summaryEl = summary;
  state.metricsEl = metrics;
  state.logEl = logPre;
  state.codeEl = programCode;
  state.stageEl = stageBar;
  state.programSection = programSection;
  state.logSection = logSection;

  return state;
}

function updateTimelineMetrics(state) {
  if (!state || !state.metricsEl) return;
  const metrics = state.metricsEl.querySelectorAll(".timeline-metric strong");
  if (metrics.length >= 1) {
    metrics[0].textContent =
      typeof state.avgUs === "number" ? `${formatMetric(state.avgUs)} us` : "-";
  }
}

function updateTimelineLog(state) {
  if (!state || !state.logEl) return;
  if (!state.logs.length) {
    state.logEl.textContent = "等待日志...";
    return;
  }
  state.logEl.textContent = state.logs.join("\n");
}

function updateTimelineSummary(state) {
  if (!state || !state.summaryEl) return;
  const status = state.summaryEl.querySelector("span");
  const title = state.summaryEl.querySelector("strong");
  if (title) {
    const avgLabel = typeof state.avgUs === "number" ? `${formatMetric(state.avgUs)} us` : "";
    let label = state.variant ? `${state.label} · ${state.variant}` : state.label;
    if (state.variant && state.label === state.variant) {
      label = state.variant;
    }
    if (avgLabel) {
      label = `${label} · ${avgLabel}`;
    }
    title.textContent = label;
  }
  if (status) {
    status.textContent = state.completed || state.checkpointPath ? "已完成" : "运行中...";
  }
}

async function fetchFileContent(path) {
  const res = await fetch(`/api/workspace/file?path=${encodeURIComponent(path)}`);
  if (!res.ok) {
    throw new Error("file not found");
  }
  const data = await res.json();
  return data.content || "";
}

async function loadProgramFromCheckpoint(checkpointPath, targetCodeEl) {
  const candidates = ["best_program.cpp", "best_program.cc", "best_program.c", "best_program.py"];
  for (const name of candidates) {
    try {
      const content = await fetchFileContent(`${checkpointPath}/${name}`);
      if (targetCodeEl) {
        targetCodeEl.textContent = content || "文件为空";
        targetCodeEl.className = name.endsWith(".py") ? "language-python" : "language-cpp";
        highlightBlock(targetCodeEl);
      }
      return true;
    } catch (err) {
      continue;
    }
  }
  if (targetCodeEl) {
    targetCodeEl.textContent = "未找到 best_program 文件。";
  }
  return false;
}

async function loadCheckpointProgram(state) {
  if (!state) return;
  if (!state.checkpointPath) {
    if (state.codeEl) state.codeEl.textContent = "暂无 checkpoint program。";
    return;
  }
  if (state.programLoaded) return;
  if (state.codeEl) state.codeEl.textContent = "加载中...";
  const loaded = await loadProgramFromCheckpoint(state.checkpointPath, state.codeEl);
  state.programLoaded = loaded;
  try {
    const infoText = await fetchFileContent(`${state.checkpointPath}/best_program_info.json`);
    const info = JSON.parse(infoText);
    const metrics = info.metrics || {};
    if (typeof metrics.combined_score === "number") {
      state.score = metrics.combined_score;
      if (state.avgUs === null) {
        state.avgUs = 10.0 / metrics.combined_score;
      }
    }
    if (typeof metrics.avg_us === "number") {
      state.avgUs = metrics.avg_us;
    }
    updateTimelineMetrics(state);
  } catch (err) {
    // ignore
  }
}

async function loadBestDetails() {
  if (!bestCheckpointPath) return;
  if (bestProgramCode) bestProgramCode.textContent = "加载中...";
  await loadProgramFromCheckpoint(bestCheckpointPath, bestProgramCode);
  try {
    const infoText = await fetchFileContent(`${bestCheckpointPath}/best_program_info.json`);
    const info = JSON.parse(infoText);
    const metrics = info.metrics || {};
    const combined = metrics.combined_score;
    const avgUs = metrics.avg_us;
    if (typeof avgUs === "number") {
      bestAvgUs = avgUs;
      updateBestSummary();
    } else if (typeof combined === "number") {
      bestAvgUs = 10.0 / combined;
      updateBestSummary();
    }
    if (bestProgramMeta) {
      const pieces = [];
      if (bestVariant) pieces.push(`checkpoint=${bestVariant}`);
      if (info.id) pieces.push(`id=${info.id}`);
      const avgDisplay = typeof avgUs === "number" ? avgUs : bestAvgUs;
      if (typeof avgDisplay === "number") pieces.push(`avg_us=${formatMetric(avgDisplay)}`);
      if (info.iteration !== undefined) pieces.push(`iteration=${info.iteration}`);
      bestProgramMeta.textContent = pieces.join(" · ") || "最佳 program 信息已更新。";
    }
  } catch (err) {
    if (bestProgramMeta) bestProgramMeta.textContent = "未读取到最佳 program 信息。";
  }
}

function appendIterLog(state, line) {
  if (!state) return;
  state.logs.push(line);
  if (!state.logEl) {
    updateTimelineLog(state);
    return;
  }
  if (!state.pendingLines) state.pendingLines = [];
  state.pendingLines.push(line);
  if (state.flushHandle) return;
  state.flushHandle = requestAnimationFrame(() => {
    if (!state.logEl) {
      state.flushHandle = null;
      state.pendingLines = [];
      return;
    }
    if (state.logEl.textContent === "等待日志..." || !state.logEl.textContent) {
      state.logEl.textContent = "";
    }
    state.logEl.textContent += `${state.pendingLines.join("\n")}\n`;
    state.pendingLines = [];
    state.flushHandle = null;
    state.logEl.scrollTop = state.logEl.scrollHeight;
  });
}

function extractAvgFromLogs(logs) {
  if (!logs || !logs.length) return null;
  for (let i = logs.length - 1; i >= 0; i -= 1) {
    const value = parseAvgUsFromLog(logs[i]);
    if (value !== null) return value;
  }
  return null;
}

function updateDiffOptions(variant, checkpointPath) {
  if (!variant || !checkpointPath) return;
  if (checkpointOptions.has(variant)) return;
  checkpointOptions.set(variant, checkpointPath);

  const option = document.createElement("option");
  option.value = variant;
  option.textContent = variant;
  option.dataset.path = checkpointPath;

  if (diffSelectA) {
    diffSelectA.appendChild(option.cloneNode(true));
  }
  if (diffSelectB) {
    diffSelectB.appendChild(option);
  }
  syncDiffVisibility();
}

async function runCheckpointDiff() {
  if (!diffSelectA || !diffSelectB || !diffOutput) return;
  const optA = diffSelectA.selectedOptions[0];
  const optB = diffSelectB.selectedOptions[0];
  if (!optA || !optB) {
    diffOutput.textContent = "请选择两个 checkpoint。";
    return;
  }
  const pathA = optA.dataset.path;
  const pathB = optB.dataset.path;
  if (!pathA || !pathB) {
    diffOutput.textContent = "无法读取 checkpoint 路径。";
    return;
  }
  diffOutput.textContent = "正在生成 diff...";
  const res = await fetch(
    `/api/checkpoint/diff?checkpoint_a=${encodeURIComponent(pathA)}&checkpoint_b=${encodeURIComponent(pathB)}`
  );
  if (!res.ok) {
    diffOutput.textContent = "diff 生成失败，请查看后端日志。";
    return;
  }
  const data = await res.json();
  diffOutput.textContent = data.diff || "No differences.";
  diffOutput.className = "language-diff";
  highlightBlock(diffOutput);
}

function resetDiffOptions() {
  checkpointOptions = new Map();
  if (diffSelectA) {
    diffSelectA.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "选择 checkpoint";
    diffSelectA.appendChild(opt);
  }
  if (diffSelectB) {
    diffSelectB.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "选择 checkpoint";
    diffSelectB.appendChild(opt);
  }
  if (diffOutput) {
    diffOutput.textContent = "等待选择 checkpoint。";
    diffOutput.className = "language-diff";
  }
  syncDiffVisibility();
}

function resetSelectionState() {
  selectedOp = "";
  metaOp.textContent = "-";
  detailLoaded = false;
  detectDone = false;
  detectRunning = false;
  evolveRunning = false;
  utGenerated = false;
  evolveDone = false;
  shapeReady = false;
  lastOutputDir = null;
  evalCounter = 0;
  iterationCounter = 0;
  inInitialEval = false;
  inEval = false;
  evolveLogBuffer = [];
  evolveLogFlushHandle = null;
  iterationStates = new Map();
  currentIterKey = null;
  lastWorkspace = {};
  shapeInput.value = "";
  shapePreset.value = "";
  setStatus("请重新选择算子");
  detailReadme.textContent = "请选择算子以查看 README 详情。";
  detailFiles.innerHTML = "";
  detectList.innerHTML = "";
  detectNote.textContent = "等待 detect-LLM 运行。";
  detectLog.textContent = "等待日志输出…";
  evolveTimeline.innerHTML = "";
  evolveSummary.textContent = "等待 OpenEvolve 输出。";
  if (evolveLog) evolveLog.textContent = "等待日志输出…";
  if (evolveLogNote) evolveLogNote.textContent = "等待演化任务。";
  resetEvolveStages();
  setIterLabel("等待评测...");
  setStageLabel("等待阶段");
  resetBestSummary();
  resetDiffOptions();
  if (bestStatusChip) bestStatusChip.textContent = "演化中...";
  renderWorkspace(lastWorkspace);
  renderOperators(filterOperators(opSearch.value));
  setRepoPanelCollapsed(false);
  updateActionState();
}

function updateStepIndicators() {
  const statusMap = {
    detail: detailLoaded,
    ut: utGenerated,
    detect: detectDone,
    evolve: evolveDone,
  };
  const order = ["detail", "ut", "detect", "evolve"];
  stepIndicators.forEach((step) => {
    const key = step.dataset.step;
    step.classList.remove("active", "done");
    if (statusMap[key]) step.classList.add("done");
  });
  const activeKey = order.find((key) => !statusMap[key]);
  if (activeKey) {
    const target = Array.from(stepIndicators).find((step) => step.dataset.step === activeKey);
    target?.classList.add("active");
  }
}

function updateConsoleMeta() {
  if (consoleRepo) consoleRepo.textContent = selectedRepo || "-";
  if (consoleOp) consoleOp.textContent = selectedOp || "-";
  if (consoleSoc) consoleSoc.textContent = platformSelect?.value || "-";
  if (consoleStep) {
    const stepsDone = [detailLoaded, utGenerated, detectDone, evolveDone].filter(Boolean).length;
    consoleStep.textContent = `${stepsDone} / 4`;
  }
  updateRepoPanelSummary();
}

function updateActionState() {
  const hasSelection = Boolean(selectedRepo && selectedOp);
  const configLocked = utGenerated || detectDone || evolveRunning || evolveDone;
  shapeInput.disabled = !detailLoaded || configLocked;
  shapePreset.disabled = !detailLoaded || configLocked;
  dtypeSelect.disabled = !detailLoaded || utGenerated;
  platformSelect.disabled = !detailLoaded || evolveRunning;

  loadDetailBtn.disabled = !hasSelection || detailLoaded;
  runUtBtn.disabled = !(detailLoaded && shapeReady && !utGenerated);
  runDetectBtn.disabled = !(utGenerated && !detectDone) || detectRunning;
  runEvolveBtn.disabled = !(detectDone && !evolveDone) || evolveRunning;
  downloadPackageBtn.disabled = !hasSelection || evolveRunning;
  if (resetSelectionBtn) {
    resetSelectionBtn.disabled = !hasSelection || detectRunning || evolveRunning;
  }
  updateConsoleMeta();
  updateStepIndicators();
}

function setStatus(text) {
  heroStatus.textContent = text;
  if (consoleStage) {
    consoleStage.textContent = text;
  }
}

function parseShape(value) {
  try {
    const parsed = JSON.parse(value);
    if (Array.isArray(parsed)) {
      return parsed.map((v) => Number(v));
    }
  } catch (err) {
    return null;
  }
  return null;
}

function syncShapeState() {
  shapeReady = !!parseShape(shapeInput.value);
  updateActionState();
}

function highlightBlock(codeEl) {
  if (!codeEl) return;
  if (window.hljs) {
    window.hljs.highlightElement(codeEl);
  }
}

function getCodeLanguage(path, fallback) {
  if (!path) return fallback || "plaintext";
  const lower = path.toLowerCase();
  if (lower.endsWith(".py")) return "python";
  if (lower.endsWith(".cpp") || lower.endsWith(".cc") || lower.endsWith(".c") || lower.endsWith(".hpp") || lower.endsWith(".h")) {
    return "cpp";
  }
  return fallback || "plaintext";
}

async function fetchRepos() {
  const res = await fetch("/api/repos");
  const data = await res.json();
  repoSelect.innerHTML = "";
  data.repos.forEach((repo) => {
    const option = document.createElement("option");
    option.value = repo;
    option.textContent = repo;
    repoSelect.appendChild(option);
  });
  if (data.repos.length) {
    selectedRepo = data.repos[0];
    repoSelect.value = selectedRepo;
    await fetchOperators(selectedRepo);
    updateConsoleMeta();
    updateRepoPanelSummary();
  }
}

async function fetchOperators(repo) {
  const res = await fetch(`/api/operators?repo=${encodeURIComponent(repo)}`);
  const data = await res.json();
  operators = data.operators || [];
  renderOperators(operators);
}

function renderOperators(list) {
  opList.innerHTML = "";
  list.forEach((op) => {
    const item = document.createElement("div");
    item.className = "op-item" + (op === selectedOp ? " active" : "");
    item.textContent = op;
    item.addEventListener("click", () => {
      selectedOp = op;
      metaRepo.textContent = selectedRepo || "-";
      metaOp.textContent = selectedOp;
      updateConsoleMeta();
      setStatus(`已选择 ${selectedOp}`);
      setRepoPanelCollapsed(true);
      detailLoaded = false;
      detectDone = false;
      shapeReady = false;
      lastWorkspace = {};
      utGenerated = false;
      evolveDone = false;
      resetEvolveStages();
      setIterLabel("等待评测...");
      setStageLabel("等待阶段");
      resetBestSummary();
      resetDiffOptions();
      renderWorkspace(lastWorkspace);
      updateActionState();
      renderOperators(filterOperators(opSearch.value));
    });
    opList.appendChild(item);
  });
}

function filterOperators(keyword) {
  if (!keyword) return operators;
  return operators.filter((op) => op.toLowerCase().includes(keyword.toLowerCase()));
}

async function loadDetails() {
  if (!selectedRepo || !selectedOp) {
    setStatus("请先选择算子");
    return;
  }
  setStatus("正在加载算子信息...");
  const res = await fetch(
    `/api/operator/detail?repo=${encodeURIComponent(selectedRepo)}&op=${encodeURIComponent(selectedOp)}`
  );
  if (!res.ok) {
    setStatus("加载失败");
    return;
  }
  const data = await res.json();
  if (window.marked) {
    detailReadme.innerHTML = window.marked.parse(data.readme || "未找到 README 说明。");
    detailReadme.querySelectorAll("pre code").forEach((block) => highlightBlock(block));
  } else {
    detailReadme.textContent = data.readme || "未找到 README 说明。";
  }
  detailFiles.innerHTML = "";
  Object.entries(data.files).forEach(([key, files]) => {
    if (!files.length) return;
    const group = document.createElement("div");
    group.className = "file-group";
    const title = document.createElement("h4");
    title.textContent = key;
    const list = document.createElement("ul");
    files.slice(0, 10).forEach((file) => {
      const li = document.createElement("li");
      li.textContent = file;
      list.appendChild(li);
    });
    group.appendChild(title);
    group.appendChild(list);
    detailFiles.appendChild(group);
  });
  detailLoaded = true;
  detectDone = false;
  utGenerated = false;
  evolveDone = false;
  syncShapeState();
  updateActionState();
  switchTab("detail");
  setStatus("算子信息已更新，请选择或输入 shape");
}

async function runUt() {
  if (configDrawer && !configDrawer.open) {
    configDrawer.open = true;
  }
  const shape = parseShape(shapeInput.value);
  if (!shape) {
    if (configDrawer) configDrawer.open = true;
    setStatus("请输入正确的 shape JSON 数组");
    return;
  }
  if (!selectedRepo || !selectedOp) {
    setStatus("请选择算子");
    return;
  }
  setStatus("正在生成测试用例...");
  const res = await fetch("/api/ut-llm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      repo: selectedRepo,
      op: selectedOp,
      shape,
      dtype: dtypeSelect.value,
    }),
  });
  const data = await res.json();
  utScript.textContent = data.script;
  utScript.className = "language-python";
  highlightBlock(utScript);
  utNote.textContent = data.note;
  utGenerated = true;
  if (configDrawer) configDrawer.open = false;
  updateActionState();
  if (data.path_hint) {
    lastWorkspace = {
      ...lastWorkspace,
      ut_script: {
        path: data.path_hint,
      },
    };
    renderWorkspace(lastWorkspace);
  }
  switchTab("ut");
  setStatus("测试用例已生成");
}

async function runDetect() {
  if (!selectedRepo || !selectedOp) {
    setStatus("请选择算子");
    return;
  }
  if (!parseShape(shapeInput.value)) {
    setStatus("请先输入或选择 shape");
    return;
  }
  if (!utGenerated) {
    setStatus("请先生成测试用例");
    return;
  }
  detectRunning = true;
  detectDone = false;
  updateActionState();
  switchTab("detect");
  setStatus("正在检测优化点...");

  detectList.innerHTML = "";
  detectNote.textContent = "detect-LLM 运行中...";
  detectLog.textContent = "等待日志输出…";

  const logLimit = 0;
  const appendLog = (line) => {
    if (!line) return;
    const infoIndex = line.indexOf("INFO:");
    if (infoIndex === -1) return;
    line = line.slice(infoIndex + 5).trim();
    if (detectLog.textContent === "等待日志输出…") {
      detectLog.textContent = "";
    }
    detectLog.textContent += `${line}\n`;
    if (logLimit && detectLog.textContent.length > logLimit) {
      detectLog.textContent = detectLog.textContent.slice(-logLimit);
    }
    detectLog.scrollTop = detectLog.scrollHeight;
  };

  const streamUrl = `/api/detect-llm/stream?repo=${encodeURIComponent(
    selectedRepo
  )}&op=${encodeURIComponent(selectedOp)}`;
  const eventSource = new EventSource(streamUrl);

  eventSource.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "log") {
      appendLog(`[${payload.stream}] ${payload.message}`);
    }
    if (payload.type === "complete") {
      const data = payload.data || {};
      detectList.innerHTML = "";
      if (data.workspace && data.workspace.initial_program) {
        const initCard = document.createElement("div");
        initCard.className = "finding";
        const title = document.createElement("h4");
        title.textContent = "初始程序 (init program)";
        const funcs = document.createElement("p");
        const initFuncs = data.workspace.initial_program.functions || [];
        funcs.textContent = initFuncs.length ? `选中的函数: ${initFuncs.join(", ")}` : "选中的函数: 无";
        initCard.appendChild(title);
        initCard.appendChild(funcs);
        detectList.appendChild(initCard);
      }
      if (data.findings && data.findings.length) {
        data.findings.forEach((finding) => {
          const card = document.createElement("div");
          card.className = "finding";
          const title = document.createElement("h4");
          title.textContent = finding.path || "未提供路径";
          const funcs = document.createElement("p");
          funcs.textContent =
            finding.functions && finding.functions.length
              ? `选中的函数: ${finding.functions.join(", ")}`
              : "选中的函数: 无";
          card.appendChild(title);
          card.appendChild(funcs);
          if (finding.mode) {
            const mode = document.createElement("p");
            mode.textContent = `提取模式: ${finding.mode}`;
            card.appendChild(mode);
          }
          detectList.appendChild(card);
        });
      } else {
        const empty = document.createElement("div");
        empty.className = "note";
        empty.textContent = "未发现可优化的片段，请检查日志确认检测过程。";
        detectList.appendChild(empty);
      }

      const noteParts = [];
      if (data.note) noteParts.push(data.note);
      if (typeof data.duration_sec === "number") noteParts.push(`耗时 ${data.duration_sec}s`);
      if (data.report_path) noteParts.push(`报告: ${data.report_path}`);
      detectNote.textContent = noteParts.join(" · ") || "detect-LLM 已结束";

      lastWorkspace = data.workspace || {};
      renderWorkspace(lastWorkspace);
      detectRunning = false;
      detectDone = true;
      evolveDone = false;
      updateActionState();
      setStatus("优化点检测完成");
      eventSource.close();
    }
  };

  eventSource.onerror = () => {
    detectRunning = false;
    updateActionState();
    detectNote.textContent = "检测连接异常，请查看后端日志。";
    setStatus("检测失败");
    eventSource.close();
  };
}

function renderWorkspace(workspace) {
  workspaceFiles.innerHTML = "";
  const workspaceEntries = [
    { key: "initial_program", label: "Initial Program" },
    { key: "marked_original", label: "Marked Source" },
    { key: "ut_script", label: "UT Script" },
  ];
  workspaceEntries.forEach((item) => {
    const info = workspace[item.key];
    if (!info) return;
    const details = document.createElement("details");
    details.className = "workspace-item";

    const summary = document.createElement("summary");
    const caret = document.createElement("span");
    caret.className = "disclosure";
    caret.textContent = "▸";
    const labelSpan = document.createElement("span");
    labelSpan.textContent = item.label;
    summary.appendChild(caret);
    summary.appendChild(labelSpan);
    details.appendChild(summary);

    const meta = document.createElement("div");
    meta.className = "workspace-meta";
    if (info.mode) {
      const modeLine = document.createElement("div");
      modeLine.textContent = `Mode: ${info.mode}`;
      meta.appendChild(modeLine);
    }
    if (info.functions && info.functions.length) {
      const funcLine = document.createElement("div");
      funcLine.textContent = `Functions: ${info.functions.join(", ")}`;
      meta.appendChild(funcLine);
    }
    details.appendChild(meta);

    const pre = document.createElement("pre");
    pre.className = "code-block";
    const code = document.createElement("code");
    let fallbackLang = "cpp";
    if (item.key === "ut_script") fallbackLang = "python";
    if (item.key === "marked_original" || item.key === "initial_program") fallbackLang = "cpp";
    code.className = `language-${getCodeLanguage(info.path, fallbackLang)}`;
    code.textContent = "点击展开以加载文件内容。";
    pre.appendChild(code);
    details.appendChild(pre);

    details.addEventListener("toggle", async () => {
      if (!details.open || details.dataset.loaded) return;
      if (!info.path) {
        code.textContent = "无法读取该文件。";
        return;
      }
      code.textContent = "加载中...";
      try {
        const res = await fetch(
          `/api/workspace/file?path=${encodeURIComponent(info.path)}&repo=${encodeURIComponent(selectedRepo)}`
        );
        const data = await res.json();
        code.textContent = data.content || "内容为空";
        code.className = `language-${getCodeLanguage(info.path, fallbackLang)}`;
        highlightBlock(code);
        details.dataset.loaded = "1";
      } catch (err) {
        code.textContent = "加载失败，请查看后端日志。";
      }
    });

    workspaceFiles.appendChild(details);
  });
  if (!workspaceEntries.some((item) => workspace[item.key])) {
    const empty = document.createElement("div");
    empty.className = "note";
    empty.textContent = "暂未生成工作区文件，请先完成优化点检测或生成 UT。";
    workspaceFiles.appendChild(empty);
  }
}

function markEvolveStage(stageKey) {
  if (!evolveStages) return;
  const target = evolveStages.querySelector(`[data-stage="${stageKey}"]`);
  if (target) {
    target.classList.add("done");
    target.classList.remove("active");
  }
}

function resetEvolveStages() {
  if (!evolveStages) return;
  evolveStages.querySelectorAll(".stage-segment").forEach((pill) => pill.classList.remove("done"));
  evolveStages.querySelectorAll(".stage-segment").forEach((pill) => pill.classList.remove("active"));
  evolveStages.querySelectorAll(".stage-segment").forEach((pill) => pill.classList.remove("skipped"));
}

function setEvolveStageActive(stageKey) {
  if (!evolveStages) return;
  evolveStages.querySelectorAll(".stage-segment").forEach((pill) => pill.classList.remove("active"));
  const target = evolveStages.querySelector(`[data-stage="${stageKey}"]`);
  if (target && !target.classList.contains("done")) {
    target.classList.add("active");
  }
}

function setStageSkipped(stageKey) {
  if (!evolveStages) return;
  const target = evolveStages.querySelector(`[data-stage="${stageKey}"]`);
  if (target) {
    target.classList.add("skipped");
  }
}

function setIterLabel(text) {
  if (evolveIterLabel) {
    evolveIterLabel.textContent = text;
  }
}

function setStageLabel(text) {
  if (evolveStageLabel) {
    evolveStageLabel.textContent = text;
  }
}

function syncDiffVisibility() {
  const showDiff = !evolveRunning && checkpointOptions.size > 0;
  if (diffSection) diffSection.hidden = !showDiff;
  if (diffPlaceholder) {
    diffPlaceholder.hidden = showDiff;
    if (!showDiff) {
      if (evolveRunning) {
        diffPlaceholder.textContent = "迭代结束后可查看 checkpoint diff。";
      } else if (evolveDone) {
        diffPlaceholder.textContent = "暂无 checkpoint diff 可显示。";
      } else {
        diffPlaceholder.textContent = "启动演化后可查看 checkpoint diff。";
      }
    }
  }
}

async function runEvolve() {
  const shape = parseShape(shapeInput.value);
  if (!shape) {
    setStatus("请输入正确的 shape JSON 数组");
    return;
  }
  if (!selectedRepo || !selectedOp) {
    setStatus("请选择算子");
    return;
  }
  if (!detectDone) {
    setStatus("请先完成优化点检测");
    return;
  }
  evolveTimeline.innerHTML = "";
  const placeholder = document.createElement("div");
  placeholder.className = "note";
  placeholder.textContent = "等待迭代输出...";
  evolveTimeline.appendChild(placeholder);
  evolveSummary.textContent = "正在启动 OpenEvolve...";
  evolveRunning = true;
  evolveDone = false;
  resetBestSummary();
  lastOutputDir = null;
  evalCounter = 0;
  iterationCounter = 0;
  inInitialEval = false;
  inEval = false;
  iterationStates = new Map();
  currentIterKey = null;
  resetEvolveStages();
  setIterLabel("等待评测...");
  setStageLabel("等待阶段");
  resetDiffOptions();
  updateActionState();
  switchTab("evolve");
  setStatus("OpenEvolve 迭代中");
  if (evolveLogDrawer) evolveLogDrawer.open = true;
  if (evolveLog) evolveLog.textContent = "等待日志输出…";
  if (evolveLogNote) evolveLogNote.textContent = "演化任务启动中...";
  evolveLogBuffer = [];
  evolveLogFlushHandle = null;

  const res = await fetch("/api/openevolve/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      repo: selectedRepo,
      op: selectedOp,
      shape,
      dtype: dtypeSelect.value,
      platform: platformSelect.value,
    }),
  });
  if (!res.ok) {
    evolveRunning = false;
    updateActionState();
    evolveSummary.textContent = "启动失败，请查看后端日志。";
    if (evolveLogNote) evolveLogNote.textContent = "启动失败，请查看后端日志。";
    setStatus("启动失败");
    setStageLabel("启动失败");
    syncDiffVisibility();
    return;
  }
  const data = await res.json();
  const sessionId = data.session_id;
  lastOutputDir = data.output_dir || null;
  if (data.run_id) {
    evolveSummary.textContent = `已启动运行 ${data.run_id}，等待输出...`;
  }
  if (evolveLogNote && data.log) {
    evolveLogNote.textContent = `日志文件: ${data.log}`;
  }

  const beginEvalSession = () => {
    if (inEval) return;
    if (currentIterKey !== null) {
      const prev = iterationStates.get(currentIterKey);
      if (prev) {
        prev.completed = true;
        updateTimelineSummary(prev);
      }
    }
    evalCounter += 1;
    inEval = true;
    if (evalCounter === 1) {
      inInitialEval = true;
      currentIterKey = "initial";
      setIterLabel("初始评测（不替换）");
      setStageLabel("等待阶段");
      resetEvolveStages();
      setStageSkipped("replace");
      const state = ensureTimelineEntry(currentIterKey, "初始评测", true);
      state.logs = [];
      if (state.logEl) state.logEl.textContent = "等待日志...";
      state.detailsEl.classList.add("pending");
      updateTimelineSummary(state);
      applyStageState(evolveStages, state.stageState);
    } else {
      inInitialEval = false;
      iterationCounter += 1;
      currentIterKey = iterationCounter;
      setIterLabel(`第 ${iterationCounter} 轮`);
      setStageLabel("等待阶段");
      resetEvolveStages();
      const label = `checkpoint_${iterationCounter}`;
      const state = ensureTimelineEntry(currentIterKey, label);
      state.logs = [];
      if (state.logEl) state.logEl.textContent = "等待日志...";
      state.detailsEl.classList.add("pending");
      updateTimelineSummary(state);
      applyStageState(evolveStages, state.stageState);
    }
  };

  const eventSource = new EventSource(`/api/openevolve/stream?session_id=${sessionId}`);
  eventSource.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "initial_eval") {
      if (placeholder.parentElement) {
        placeholder.remove();
      }
      const state = ensureTimelineEntry("initial", "初始评测", true);
      state.completed = true;
      state.detailsEl.classList.remove("pending");
      const avgFromLogs = state.avgUs ?? extractAvgFromLogs(state.logs);
      if (avgFromLogs !== null && state.avgUs === null) {
        state.avgUs = avgFromLogs;
        updateTimelineMetrics(state);
      }
      updateTimelineSummary(state);
      if (evolveSummary && !bestVariant) {
        evolveSummary.textContent = "初始评测完成";
      }
    }

    if (payload.type === "log") {
      const line = payload.data?.message || "";
      const isBeginEval = line.includes("BEGIN eval");
      const isEndEval = line.includes("END eval");
      const isStepStart = line.includes("STEP_START:");
      const isStepOk = line.includes("STEP_OK:");
      if (isBeginEval || isStepStart || isStepOk) {
        if (placeholder.parentElement) {
          placeholder.remove();
        }
        beginEvalSession();
      }

      const state = currentIterKey !== null ? iterationStates.get(currentIterKey) : null;
      if (state && inEval) {
        appendIterLog(state, line);
        const avgFromLog = parseAvgUsFromLog(line);
        if (avgFromLog !== null) {
          state.avgUs = avgFromLog;
          updateTimelineMetrics(state);
          updateTimelineSummary(state);
        }
      }

      const updateStage = (stageKey, status) => {
        if (!state || !state.stageState) return;
        state.stageState[stageKey] = status;
        applyStageState(evolveStages, state.stageState);
        if (state.stageEl) applyStageState(state.stageEl, state.stageState);
        if (status === "active") {
          const label = STAGE_LABELS[stageKey] || stageKey;
          setStageLabel(`${label} 中`);
        }
        if (stageKey === "profile" && status === "done") {
          state.completed = true;
          updateTimelineSummary(state);
          setStageLabel("本轮完成");
        }
      };

      if (state && inEval) {
        if (!inInitialEval && line.includes("STEP_START: REPLACE")) updateStage("replace", "active");
        if (line.includes("STEP_START: BUILD")) updateStage("build", "active");
        if (line.includes("STEP_START: INSTALL")) updateStage("install", "active");
        if (line.includes("STEP_START: RUN")) updateStage("run", "active");
        if (line.includes("STEP_START: PROFILE")) updateStage("profile", "active");
        if (!inInitialEval && line.includes("STEP_OK: REPLACE")) updateStage("replace", "done");
        if (line.includes("STEP_OK: BUILD")) updateStage("build", "done");
        if (line.includes("STEP_OK: INSTALL")) updateStage("install", "done");
        if (line.includes("STEP_OK: RUN")) updateStage("run", "done");
        if (line.includes("STEP_OK: PROFILE")) updateStage("profile", "done");
      }

      if (evolveLog) {
        evolveLogBuffer.push(line);
        if (!evolveLogFlushHandle) {
          evolveLogFlushHandle = requestAnimationFrame(() => {
            if (!evolveLog) {
              evolveLogBuffer = [];
              evolveLogFlushHandle = null;
              return;
            }
            if (evolveLog.textContent === "等待日志输出…") {
              evolveLog.textContent = "";
            }
            evolveLog.textContent += `${evolveLogBuffer.join("\n")}\n`;
            evolveLogBuffer = [];
            evolveLogFlushHandle = null;
            evolveLog.scrollTop = evolveLog.scrollHeight;
          });
        }
      }

      if (isEndEval && state && inEval) {
        state.completed = true;
        updateTimelineSummary(state);
        inEval = false;
        currentIterKey = null;
        setStageLabel("本轮完成");
      }
    }

    if (payload.type === "iteration") {
      if (placeholder.parentElement) {
        placeholder.remove();
      }
      const idx = payload.data.index;
      const variant = payload.data.variant || `checkpoint_${idx}`;
      const avgValue = payload.data.avg_us;
      const notes = payload.data.notes || "";
      const scoreValue = parseCombinedScore(notes);

      const state = ensureTimelineEntry(idx, `checkpoint_${idx}`);
      const prevPath = state.checkpointPath;
      state.variant = variant;
      state.checkpointPath =
        payload.data.path || (lastOutputDir ? `${lastOutputDir}/checkpoints/${variant}` : null);
      if (state.checkpointPath && state.checkpointPath !== prevPath) {
        state.programLoaded = false;
        if (state.detailsEl && state.detailsEl.open) {
          loadCheckpointProgram(state);
        }
      }
      updateDiffOptions(variant, state.checkpointPath);
      state.completed = true;
      if (typeof avgValue === "number") {
        state.avgUs = avgValue;
      } else if (typeof scoreValue === "number") {
        state.avgUs = 10.0 / scoreValue;
      }
      state.score = typeof scoreValue === "number" ? scoreValue : state.score;
      state.detailsEl.classList.remove("pending");
      updateTimelineMetrics(state);
      updateTimelineSummary(state);

      let improved = false;
      if (typeof state.avgUs === "number") {
        improved = bestAvgUs === null || state.avgUs < bestAvgUs;
      }
      if (improved) {
        bestAvgUs = state.avgUs;
        bestVariant = variant;
        bestCheckpointPath = state.checkpointPath;
        state.detailsEl.classList.add("best");
        const avgText = typeof bestAvgUs === "number" ? formatMetric(bestAvgUs) : "--";
        evolveSummary.textContent = `当前最佳 ${bestVariant} · Avg ${avgText} us`;
        updateBestSummary();
        loadBestDetails();
      } else if (bestVariant) {
        const avgText = typeof bestAvgUs === "number" ? formatMetric(bestAvgUs) : "--";
        evolveSummary.textContent = `当前最佳 ${bestVariant} · Avg ${avgText} us`;
        updateBestSummary();
      }
    }

    if (payload.type === "complete") {
      const summary = payload.data || {};
      if (placeholder.parentElement) {
        placeholder.remove();
      }
      lastOutputDir = summary.output_dir || lastOutputDir;
      const success = summary.status === "success" && summary.best_variant;
      if (summary.status === "failed") {
        evolveSummary.textContent = `演化失败（exit ${summary.exit_code ?? "未知"}）。请查看日志 ${summary.log_file || ""}`;
        setStatus("演化失败");
        if (bestStatusChip) bestStatusChip.textContent = "演化失败";
        setStageLabel("演化失败");
      } else if (!summary.best_variant) {
        evolveSummary.textContent = `未检测到 checkpoint，请查看日志 ${summary.log_file || ""}`;
        setStatus("演化结束");
        if (bestStatusChip) bestStatusChip.textContent = "未找到最佳版本";
        setStageLabel("演化结束");
      } else {
        bestVariant = summary.best_variant;
        if (typeof summary.best_avg_us === "number") {
          bestAvgUs = summary.best_avg_us;
        }
        bestCheckpointPath =
          lastOutputDir && summary.best_variant ? `${lastOutputDir}/checkpoints/${summary.best_variant}` : bestCheckpointPath;
        updateBestSummary();
        loadBestDetails();
        const avgValue = summary.best_avg_us ?? bestAvgUs;
        const avgText = typeof avgValue === "number" ? formatMetric(avgValue) : "--";
        evolveSummary.textContent = `最佳版本 ${summary.best_variant} · Avg ${avgText} us`;
        setStatus("演化完成");
        setStageLabel("演化完成");
      }
      if (evolveLogNote) {
        evolveLogNote.textContent = `演化结束 · 日志: ${summary.log_file || "-"}`;
      }
      if (summary.best_variant) {
        const match = summary.best_variant.match(/checkpoint_(\d+)/);
        if (match) {
          const idx = Number(match[1]);
          const state = iterationStates.get(idx);
          if (state?.detailsEl) {
            state.detailsEl.classList.add("best");
          }
        }
      }
      evolveRunning = false;
      evolveDone = success;
      inEval = false;
      currentIterKey = null;
      updateActionState();
      syncDiffVisibility();
      eventSource.close();
    }
  };

  eventSource.onerror = () => {
    evolveRunning = false;
    inEval = false;
    currentIterKey = null;
    updateActionState();
    evolveSummary.textContent = "演化连接异常，请查看后端日志。";
    if (evolveLogNote) evolveLogNote.textContent = "日志连接异常，请查看后端日志。";
    setStatus("演化中断");
    setStageLabel("演化中断");
    syncDiffVisibility();
    eventSource.close();
  };
}

async function downloadPackage() {
  if (!selectedRepo || !selectedOp) {
    setStatus("请选择算子");
    return;
  }
  const hasBest = Boolean(bestVariant || bestCheckpointPath);
  setStatus(hasBest ? "正在打包最优版本... " : "未检测到最优版本，将下载原始算子包...");
  const requestBody = {
    repo: selectedRepo,
    op: selectedOp,
    include_generated: true,
  };
  if (bestCheckpointPath) {
    requestBody.best_checkpoint = bestCheckpointPath;
  } else if (bestVariant && lastOutputDir) {
    requestBody.output_dir = lastOutputDir;
    requestBody.best_variant = bestVariant;
  }
  const res = await fetch("/api/package", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });
  if (!res.ok) {
    setStatus("打包失败，请查看后端日志");
    return;
  }

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const bestTag =
    bestVariant || (bestCheckpointPath ? bestCheckpointPath.split("/").pop() : null);
  const suffix = bestTag ? `_best_${bestTag}` : "_package";
  a.download = `${selectedRepo}_${selectedOp.replaceAll("/", "_")}${suffix}.zip`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(url);
  setStatus("下载完成");
}

repoSelect.addEventListener("change", async (event) => {
  selectedRepo = event.target.value;
  selectedOp = "";
  metaRepo.textContent = selectedRepo;
  metaOp.textContent = "-";
  updateConsoleMeta();
  detailLoaded = false;
  detectDone = false;
  shapeReady = false;
  lastWorkspace = {};
  utGenerated = false;
  evolveDone = false;
  resetEvolveStages();
  setIterLabel("等待评测...");
  setStageLabel("等待阶段");
  resetBestSummary();
  resetDiffOptions();
  renderWorkspace(lastWorkspace);
  shapeInput.value = "";
  shapePreset.value = "";
  setRepoPanelCollapsed(false);
  updateActionState();
  await fetchOperators(selectedRepo);
});

opSearch.addEventListener("input", (event) => {
  renderOperators(filterOperators(event.target.value));
});

shapePreset.addEventListener("change", (event) => {
  if (event.target.value) {
    shapeInput.value = event.target.value;
  }
  syncShapeState();
});

shapeInput.addEventListener("input", () => {
  syncShapeState();
});

platformSelect.addEventListener("change", () => {
  updateConsoleMeta();
});

if (repoCollapseBtn) {
  repoCollapseBtn.addEventListener("click", () => {
    const collapsed = repoPanel?.classList.contains("collapsed");
    setRepoPanelCollapsed(!collapsed);
  });
}

if (resetSelectionBtn) {
  resetSelectionBtn.addEventListener("click", () => {
    resetSelectionState();
  });
}

loadDetailBtn.addEventListener("click", loadDetails);
runUtBtn.addEventListener("click", runUt);
runDetectBtn.addEventListener("click", runDetect);
runEvolveBtn.addEventListener("click", runEvolve);
downloadPackageBtn.addEventListener("click", downloadPackage);
if (runDiffBtn) runDiffBtn.addEventListener("click", runCheckpointDiff);

if (window.hljs) {
  window.hljs.highlightAll();
}

fetchRepos();
updateActionState();
renderWorkspace(lastWorkspace);
resetDiffOptions();
