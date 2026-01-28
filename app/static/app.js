const repoSelect = document.getElementById("repoSelect");
const opSearch = document.getElementById("opSearch");
const opList = document.getElementById("opList");
const metaRepo = document.getElementById("metaRepo");
const metaOp = document.getElementById("metaOp");
const heroStatus = document.getElementById("heroStatus");

const shapeInput = document.getElementById("shapeInput");
const dtypeSelect = document.getElementById("dtypeSelect");
const platformSelect = document.getElementById("platformSelect");

const detailReadme = document.getElementById("detailReadme");
const detailFiles = document.getElementById("detailFiles");

const utScript = document.getElementById("utScript");
const utNote = document.getElementById("utNote");

const detectList = document.getElementById("detectList");
const detectNote = document.getElementById("detectNote");

const evolveTimeline = document.getElementById("evolveTimeline");
const evolveSummary = document.getElementById("evolveSummary");

const loadDetailBtn = document.getElementById("loadDetail");
const downloadPackageBtn = document.getElementById("downloadPackage");
const runUtBtn = document.getElementById("runUt");
const runDetectBtn = document.getElementById("runDetect");
const runEvolveBtn = document.getElementById("runEvolve");

let operators = [];
let selectedRepo = "";
let selectedOp = "";

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

function setStatus(text) {
  heroStatus.textContent = text;
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
      setStatus(`已选择 ${selectedOp}`);
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
  detailReadme.textContent = data.readme || "未找到 README 说明。";
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
  setStatus("算子信息已更新");
}

async function runUt() {
  const shape = parseShape(shapeInput.value);
  if (!shape) {
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
  utNote.textContent = data.note;
  setStatus("测试用例已生成");
}

async function runDetect() {
  if (!selectedRepo || !selectedOp) {
    setStatus("请选择算子");
    return;
  }
  setStatus("正在检测优化点...");
  const res = await fetch("/api/detect-llm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      repo: selectedRepo,
      op: selectedOp,
    }),
  });
  const data = await res.json();
  detectList.innerHTML = "";
  data.findings.forEach((finding) => {
    const card = document.createElement("div");
    card.className = "finding";
    const title = document.createElement("h4");
    title.textContent = finding.path;
    const copy = document.createElement("p");
    copy.textContent = finding.summary;
    const pre = document.createElement("pre");
    pre.className = "code-block";
    pre.textContent = finding.excerpt || "";
    card.appendChild(title);
    card.appendChild(copy);
    card.appendChild(pre);
    detectList.appendChild(card);
  });
  detectNote.textContent = data.note;
  setStatus("优化点检测完成");
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
  evolveTimeline.innerHTML = "";
  evolveSummary.textContent = "正在启动 OpenEvolve...";
  setStatus("OpenEvolve 迭代中");

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
  const data = await res.json();
  const sessionId = data.session_id;

  const eventSource = new EventSource(`/api/openevolve/stream?session_id=${sessionId}`);
  eventSource.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "iteration") {
      const item = document.createElement("div");
      item.className = "timeline-item";
      item.innerHTML = `
        <strong>第 ${payload.data.index} 轮 · ${payload.data.variant}</strong>
        <div>Latency: ${payload.data.latency_ms} ms · Throughput: ${payload.data.throughput}</div>
        <div>${payload.data.notes}</div>
      `;
      evolveTimeline.appendChild(item);
    }
    if (payload.type === "complete") {
      evolveSummary.textContent = `最佳版本 ${payload.data.best_variant} · 延迟 ${payload.data.best_latency_ms} ms`;
      setStatus("演化完成");
      eventSource.close();
    }
  };
}

async function downloadPackage() {
  if (!selectedRepo || !selectedOp) {
    setStatus("请选择算子");
    return;
  }
  setStatus("正在打包... ");
  const res = await fetch("/api/package", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      repo: selectedRepo,
      op: selectedOp,
      include_generated: true,
    }),
  });

  const blob = await res.blob();
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${selectedRepo}_${selectedOp.replaceAll("/", "_")}_package.zip`;
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
  await fetchOperators(selectedRepo);
});

opSearch.addEventListener("input", (event) => {
  renderOperators(filterOperators(event.target.value));
});

loadDetailBtn.addEventListener("click", loadDetails);
runUtBtn.addEventListener("click", runUt);
runDetectBtn.addEventListener("click", runDetect);
runEvolveBtn.addEventListener("click", runEvolve);
downloadPackageBtn.addEventListener("click", downloadPackage);

fetchRepos();
