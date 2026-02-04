# run_example 调用逻辑与自定义示例文件

本文描述 `build.sh --run_example` 的解析与执行流程，并说明如何指定运行“非 `test_aclnn_*` / `test_geir_*` 前缀”的示例文件。

## 1. 命令示例（fast_gelu 自定义包）

```bash
bash build.sh --run_example fast_gelu eager cust --vendor_name=custom
```

解析结果：
- `OP_NAME=fast_gelu`
- `EXAMPLE_MODE=eager`
- `PKG_MODE=cust`
- `VENDOR_NAME=custom`

## 2. 调用流程（核心函数）

入口：`build.sh` 的 `main()`

1. `checkopts "$@"`
   - `getopts` 解析长参数。
   - 遇到 `--run_example` 会调用 `set_example_opt $2 $3 $4`，依次取出紧随其后的 3 个非 `-` 开头参数作为：
     - `OP_NAME`、`EXAMPLE_MODE`、`PKG_MODE`。
   - 同时解析 `--vendor_name=` 并设置 `VENDOR_NAME`，触发 `ENABLE_CUSTOM=TRUE`。

2. `assemble_cmake_args`
   - 仅拼装/打印 `CMAKE_ARGS`，`--run_example` 路径不会进行主工程 cmake 构建。

3. `build_example`
   - 根据 `EXAMPLE_MODE` 设置 `pattern`：
     - `eager` → `pattern=test_aclnn_`
     - `graph` → `pattern=test_geir_`
   - 查找示例源码：
     - `find ../ -path "*/${OP_NAME}/examples/${pattern}*.cpp"`
     - 若 `--soc=ascend910_95` 还会查 `examples/arch35/`。

4. `build_single_example`
   - `PKG_MODE=cust` 走自定义包路径：
     - `CUST_LIBRARY_PATH=${ASCEND_HOME_PATH}/opp/vendors/${VENDOR_NAME}_nn/op_api/lib`
     - `CUST_INCLUDE_PATH=${ASCEND_HOME_PATH}/opp/vendors/${VENDOR_NAME}_nn/op_api/include`
   - `g++` 编译并输出二进制到 `build/`：
     - `eager` → `build/test_aclnn_<example>`
     - `graph` → `build/test_geir_<example>`
   - 编译成功后立即执行该二进制。

## 3. fast_gelu 示例定位

`fast_gelu` 的 eager 示例源码：

```
activation/fast_gelu/examples/test_aclnn_fast_gelu.cpp
```

## 4. 指定运行“非 test_aclnn_* 前缀”的示例文件怎么办？

当前逻辑**只会匹配**：
- eager：`test_aclnn_*.cpp`
- graph：`test_geir_*.cpp`

因此如果要运行 **其他命名** 的示例文件，有两种方式：

### 方式 A：改文件名（无需改脚本）
将目标示例文件重命名为：
- eager：`test_aclnn_<name>.cpp`
- graph：`test_geir_<name>.cpp`

然后用 `--example_name=<name>` 精确指定：

```bash
bash build.sh --run_example fast_gelu eager --example_name=<name>
```

### 方式 B：修改脚本匹配规则（推荐改点）
要运行任意文件名，需要调整 `build.sh` 的 `build_example()` 中的匹配与生成逻辑。

建议修改位置：
- `build.sh` → `build_example()`
  - 目前 `pattern` 固定为 `test_aclnn_` / `test_geir_`。
  - 查找规则固定为：`*/${OP_NAME}/examples/${pattern}*.cpp`

可选改法：
1) **新增参数**（例如 `--example_file=<path>`）
   - 在 `checkopts()` 里解析 `example_file`，保存为 `EXAMPLE_FILE`。
   - 在 `build_example()` 中：若 `EXAMPLE_FILE` 非空，直接使用该文件，绕过 `pattern` 查找。

2) **放宽匹配**（适配其他前缀）
   - 把 `pattern` 改为可配置变量或直接匹配 `*.cpp`。
   - 同时调整 `example` 名称解析逻辑（目前通过剥离 `pattern` 获取名称）。

如果你希望我按上述方式直接改脚本，也可以告诉我偏好的方案。
