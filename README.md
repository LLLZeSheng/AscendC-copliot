# AscendC Copilot 设计与使用说明

## 1. 核心能力
- **算子浏览与详情**：扫描 `ops-x` 下的算子目录，返回 README 摘要、文件分组与 tiling 片段。
- **UT 生成（ut-LLM）**：优先模板生成，失败时调用 LLM（DeepSeek），再失败回退 NumPy stub。
- **优化点检测（detect-LLM）**：支持同步与 SSE 流式输出；抽取初始程序与标记后的源文件，回写工作区卡片。
- **OpenEvolve 演化**：启动异步演化，SSE 实时推送日志、checkpoint、最佳结果与结束摘要。
- **Checkpoint Diff**：对比两个 checkpoint 的 `best_program.*`，输出 unified diff。
- **一键打包**：下载算子源码与生成物（UT/检测报告/最佳程序/最佳 checkpoint 记录）。

关键目录：
- `app/main.py`：FastAPI API 入口与业务编排。
- `app/templates/index.html` / `app/static/`：前端页面与脚本。
- `tools/detector/`：优化点检测脚本与标记插入器。
- `tools/utGenerator/`：UT 生成器（模板 + LLM + stub）。
- `optimization/`：OpenEvolve 启动脚本、评测输出、日志与 checkpoint。
- `operators/`：内置初始程序兜底（如 `operators/<op>/intitial.cpp`）。

---

## 2. FastAPI 接口与实现要点

### 2.1 基础
| 路径 | 方法 | 功能 | 说明 |
| --- | --- | --- | --- |
| `/` | GET | 前端页面 | 渲染 `app/templates/index.html` |
| `/api/repos` | GET | 列仓库 | 枚举 `ops-x/*` 目录 |
| `/api/operators` | GET | 列算子 | 递归查找含 `op_host/op_kernel/op_graph` 的目录 |
| `/api/operator/detail` | GET | 算子详情 | README + 文件分组 + tiling 片段 |

### 2.2 UT 生成
`POST /api/ut-llm`  
入参：`repo, op, shape[], dtype`  
流程：
1. `_generate_ut_content`：模板 → LLM → stub。
2. 输出：`optimization/ut/<category>/test_<op_name>.py`。

返回：脚本内容、生成方式说明、输出路径提示。

### 2.3 优化点检测（支持 SSE）
- 同步：`POST /api/detect-llm`  
- 流式：`GET /api/detect-llm/stream?repo=...&op=...`  

SSE 事件类型：
- `log`：检测日志行
- `complete`：最终 payload（findings、workspace、report_path、duration 等）

### 2.4 工作区文件读取
`GET /api/workspace/file?path=...&repo=...`  
- 读取白名单目录下的文件内容（workspace 卡片与 checkpoint 详情渲染使用）。

### 2.5 Checkpoint Diff
`GET /api/checkpoint/diff?checkpoint_a=...&checkpoint_b=...`  
- 读取两个 checkpoint 下的 `best_program.*`，生成 unified diff。

### 2.6 OpenEvolve 演化
`POST /api/openevolve/start`  
入参：`repo, op, shape[], dtype, platform`  
流程：
1. 生成/复用 UT。
2. 选择初始程序：detect 产物 → `operators/<op>/intitial.cpp` → tiling 文件（加 REPLACE 标记）。
3. 调用 `optimization/start_optimization.py`，固定 `--iterations 30`。
4. 输出目录：`optimization/<category>/<op>/<test_name>_<run_id>/`。

返回：`session_id`、日志路径、输出目录与 `run_id`。

`GET /api/openevolve/stream?session_id=...`  
SSE 事件类型：
- `log`：演化日志行
- `initial_eval`：初始评测提示（无 replace）
- `iteration`：checkpoint 元信息（index/variant/avg_us/path）
- `complete`：结束摘要（best_variant/best_avg_us/status/output_dir 等）

### 2.7 打包下载
`POST /api/package`  
入参：`repo, op, include_generated, output_dir?, best_variant?, best_checkpoint?`  
- `include_generated=true` 时打包 `generated/` 目录，并附最佳程序与最佳 checkpoint 元数据（如可用）。

---

## 3. 运行指南

### 3.1 环境
```bash
pip install -r requirements.txt
# OpenEvolve 依赖 Ascend 工具链与本地编译/评测环境。
```

### 3.2 启动
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
浏览器访问 `http://localhost:8000`。

### 3.3 API 示例
```bash
# 生成 UT
curl -X POST http://localhost:8000/api/ut-llm \
  -H "Content-Type: application/json" \
  -d '{"repo":"ops-nn","op":"activation/swi_glu","shape":[48,1,9216],"dtype":"float16"}'

# detect-LLM 流式
curl -N "http://localhost:8000/api/detect-llm/stream?repo=ops-nn&op=activation/swi_glu"

# OpenEvolve 启动 + 订阅
curl -X POST http://localhost:8000/api/openevolve/start \
  -H "Content-Type: application/json" \
  -d '{"repo":"ops-nn","op":"activation/swi_glu","shape":[48,1,9216],"dtype":"float16","platform":"ascend910b"}'

# 假设返回 session_id=abc
curl -N "http://localhost:8000/api/openevolve/stream?session_id=abc"
```

---

## 4. 产物定位
- Detect 输出：`tools/detector/runs/<category>/<op>/<timestamp>/`
- UT 输出：`optimization/ut/<category>/test_<op_name>.py`
- 演化输出：`optimization/<category>/<op>/<test_name>_<run_id>/checkpoints/`
- 演化日志：`optimization/logs/openevolve_<timestamp>.log`
- 初始程序兜底：`operators/<op>/intitial.cpp`

---

## 5. 常见问题
- **LLM 不可用**：UT 生成会自动回退 stub；检查网络或 LLM 相关配置。
- **Ascend 工具链缺失**：OpenEvolve 构建/评测失败，查看 `optimization/logs`。
- **SSE 无事件**：确认检测/演化进程已启动，并检查日志文件与 checkpoint 目录是否产生。
- **标记缺失**：`add_markers.py` 失败不会中断流程，可手动在目标函数外包裹 `// [[[ REPLACE_START/END ]]]`。
