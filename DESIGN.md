# AscendC Copilot 设计与使用说明（优化版）

## 1. 核心能力
- **算子浏览**：自动扫描本地 `ops-x` 仓库，展示算子目录结构与 README 摘要。
- **UT 生成（ut-LLM）**：调用 `tools/utGenerator` 模板/LLM 生成 PyTorch+NPU 单测；异常时回退 NumPy stub。
- **优化点检测（detect-LLM）**：调用 `tools/detector/main.py` 识别真实 tiling 代码，裁剪函数并插入替换标记。
- **演化优化（OpenEvolve）**：用 `optimization/start_optimization.py` 驱动 OpenEvolve，对 tiling 函数进行迭代演化并流式返回 checkpoint。
- **一键打包**：将算子代码及生成物打包为 zip。

主要目录：
- `app/main.py`：FastAPI API 入口与业务编排。
- `tools/detector/`：检测脚本、标记插入器。
- `tools/utGenerator/`：UT 生成器（模板 + DeepSeek LLM）。
- `optimization/`：OpenEvolve 启动、评测与输出（日志、checkpoint）。

---

## 2. FastAPI 接口与实现要点

### 2.1 基础
| 路径 | 方法 | 功能 | 关键逻辑 |
| --- | --- | --- | --- |
| `/` | GET | 前端页面 | 渲染 `app/templates/index.html` |
| `/api/repos` | GET | 列仓库 | 枚举 `ops-x/*` 目录 |
| `/api/operators` | GET | 列算子 | 递归寻找含 `op_host/op_kernel/op_graph` 的目录 |

### 2.2 算子详情
`GET /api/operator/detail?repo=...&op=...`  
- 收集并分桶 tiling/op_host/op_kernel/tests/docs/other；截取 tiling_cpp 片段；读取 README（前 4K）。  
- 返回：文件分组、片段、README。

### 2.3 UT 生成（ut-LLM）
`POST /api/ut-llm`  
入参：`repo, op, shape[], dtype`  
流程：
1. 解析 category/op_name。
2. `_write_ut_file` → `tools/utGenerator.generate_test`：
   - 先模板 `_render_template`（已支持 `activation/swi_glu`、`activation/gelu_quant`）。
   - 否则构造 one-shot prompt 调用 DeepSeek（_llm_request）。
   - 失败回退 NumPy stub。
3. 产物落盘：`optimization/ut/<category>/test_<op>.py`。
返回：脚本内容、生成模式（template/llm/stub）、落盘路径。

### 2.4 优化点检测（detect-LLM）
`POST /api/detect-llm`  
入参：`repo, op`  
流程：
1. 调用 `tools/detector/main.py --base <op_path> --repo-root <repo_root> --result-root tools/detector/runs --write-report`。
2. 读取 `tools/detector/runs/<category>/<op>/report.json`，筛出 label=real 的文件，附证据与代码摘录。
返回：发现列表、stdout/stderr 末尾、结果目录。

### 2.5 OpenEvolve 演化
`POST /api/openevolve/start`  
入参：`repo, op, shape[], dtype, platform`  
流程：
1. 生成/复用 UT：`optimization/ut/<category>/test_<op>.py`。
2. 选择初始程序：优先最近 detect 的 `selected_initial_program`；否则 `operators/<op>/intitial.cpp`；再否则复制 tiling 文件并加 REPLACE 标记。
3. 解析 tiling 目标（优先 tiling_cpp）。
4. 启动：`python3 optimization/start_optimization.py <initial> --operator-name <op> --category <cat> --file-name <tiling_file> --test-file <ut> --iterations 30`。
5. 日志写入 `optimization/logs/openevolve_<ts>.log`。
返回：`session_id`、日志路径、输出目录 `optimization/<cat>/<op>/<test>`。

`GET /api/openevolve/stream?session_id=...`  
- 轮询输出目录 `checkpoints/checkpoint_*`，新文件即 SSE 事件 `iteration`。  
- 进程结束后推送 `complete`（最佳 checkpoint 名称、总迭代数）。

### 2.6 打包下载
`POST /api/package`  
- 打包算子目录；如 `include_generated=true`，附占位 `generated/gen_data.py` 与 `detect_report.json`。  
- 返回 zip 流。

---

## 3. 关键内部逻辑（app/main.py）
- `_parse_category_op`：拆出 `<category>, <op_name>`。
- `_collect_files/_collect_snippets`：目录分桶与代码节选。
- `_find_tiling_file`：优先 tiling_cpp，否则 op_host 首个 cpp。
- `_latest_detection_output`：读取 `tools/detector/runs/<cat>/<op>/` 最新结果。
- `_ensure_marked_copy`：复制文件并运行 `add_markers.py` 插入 `// [[[ REPLACE_START/END ]]]`。
- `_choose_initial_program`：检测产物 → `operators/intitial.cpp` → tiling 文件（加标记）。
- UT 链路：`_generate_ut_content`（模板/LLM/stub）→ `_write_ut_file`（落盘）。
- 演化状态：`_scan_checkpoints`、`_latest_checkpoint` 解析 checkpoint 事件。

路径约定：
- Detect 输出：`tools/detector/runs/<category>/<op>/`
- UT 输出：`optimization/ut/<category>/test_<op>.py`
- 演化输出：`optimization/<category>/<op>/<test>/checkpoints/`
- 演化日志：`optimization/logs/openevolve_<ts>.log`

---

## 4. 使用指南

### 4.1 环境
```bash
pip install -r requirements.txt
# 需要 Ascend 工具链 (build.sh, msprof)；如用 LLM，请配置 DeepSeek API Key。
```

### 4.2 启动
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
浏览器访问 http://localhost:8000 使用前端工作台。

### 4.3 API 调用示例
- 列仓库：`GET /api/repos`
- 列算子：`GET /api/operators?repo=ops-nn`
- 生成 UT：
```bash
curl -X POST http://localhost:8000/api/ut-llm \
  -H "Content-Type: application/json" \
  -d '{"repo":"ops-nn","op":"activation/swi_glu","shape":[48,1,9216],"dtype":"float16"}'
```
- 运行检测：
```bash
curl -X POST http://localhost:8000/api/detect-llm \
  -H "Content-Type: application/json" \
  -d '{"repo":"ops-nn","op":"activation/swi_glu"}'
```
- 启动演化并订阅进度：
```bash
curl -X POST http://localhost:8000/api/openevolve/start \
  -H "Content-Type: application/json" \
  -d '{"repo":"ops-nn","op":"activation/swi_glu","shape":[48,1,9216],"dtype":"float16","platform":"ascend910b"}'

# 假设返回 session_id=abc
curl -N "http://localhost:8000/api/openevolve/stream?session_id=abc"
```
- 打包下载：
```bash
curl -X POST http://localhost:8000/api/package \
  -H "Content-Type: application/json" \
  -d '{"repo":"ops-nn","op":"activation/swi_glu","include_generated":true}' --output pkg.zip
```

---

## 5. 产物定位
- Detect 报告/裁剪：`tools/detector/runs/<cat>/<op>/`
- UT：`optimization/ut/<cat>/test_<op>.py`
- 演化日志：`optimization/logs/openevolve_<ts>.log`
- 演化 checkpoint：`optimization/<cat>/<op>/<test>/checkpoints/checkpoint_*`
- 初始程序默认优先检测产物；否则 `operators/<op>/intitial.cpp`；再否则 tiling 文件（加标记）。

---

## 6. 常见问题
- **LLM 不可用**：UT 回退 stub；detect 结果可能为空。检查网络与 API Key。
- **Ascend 工具链缺失**：OpenEvolve 构建/评测失败，查看 `optimization/logs`。
- **进度无事件**：确认 checkpoint 目录是否生成；查看演化日志。
- **标记缺失**：`add_markers.py` 失败不会中断流程，可手动在目标函数外包裹 `// [[[ REPLACE_START: Func ]]] ... // [[[ REPLACE_END ]]]`。

---

## 7. 后续优化方向
- 在 `/api/package` 自动嵌入最新 detect 报告与 UT。
- 扩充更多算子模板到 `tools/utGenerator`，减少 LLM 调用成本。
- SSE 事件中附 msprof 解析，直接给出耗时/吞吐改进幅度。
