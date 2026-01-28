import argparse
import ast
import json
import os
from pathlib import Path
from textwrap import dedent

import requests


DTYPE_MAP = {
    "bfloat16": "torch.bfloat16",
    "bf16": "torch.bfloat16",
    "float16": "torch.float16",
    "fp16": "torch.float16",
    "float32": "torch.float32",
    "fp32": "torch.float32",
}


SWIGLU_TEMPLATE = """import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torch.nn.functional as F

torch.npu.config.allow_internal_format = False


class {class_name}(TestCase):

    def get_golden(self, input, dim):
        def high_precision(x):
            # 0.1版本，FP32格式运算，最后输出转成BF16
            x = torch.chunk(x, 2, dim=dim)
            x0 = x[0].type(torch.float32)
            x1 = x[1].type(torch.float32)
            output = F.silu(x0.cpu()) * x1.cpu()
            return output.type(input.type())

        def low_precision(x):
            # 0.2版本，Silu小算子拼接的版本，最后乘法使用BF16计算
            x = torch.chunk(x, 2, dim=dim)
            return F.silu(x[0].npu()) * x[1].npu()

        return high_precision(input).cpu()

    def run_swiglu(self, shape, dtype, dim):
        # 创建随机输入
        input_tensor = torch.randn(shape, dtype=dtype)

        # Golden 结果（CPU 上计算）
        golden = self.get_golden(input_tensor, dim)
        print("golden shape:", golden.shape)

        # NPU 算子结果
        for i in range(100):
            output_npu = torch_npu.npu_swiglu(input_tensor.npu(), dim).cpu()
        print("output_npu shape:", output_npu.shape)

        # 比较（注意：golden 和 output 都是相同 dtype）
        self.assertRtolEqual(output_npu, golden)

    def run_test(self, shape, dtype, dim=-1):
        print("======test case:", shape, dtype, dim)
        self.run_swiglu(shape, dtype, dim)

    def test_case(self):
        self.run_test({shape}, {dtype}, dim={dim})


if __name__ == "__main__":
    run_tests()
"""


GELU_QUANT_TEMPLATE = """import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torch.nn.functional as F

torch.npu.config.allow_internal_format = False


class {class_name}(TestCase):

    def get_golden(self, input_tensor, input_scale, input_offset, approximate, round_mode):
        x = input_tensor.float().cpu()
        scale = input_scale.float().cpu()
        offset = input_offset.float().cpu()

        gelu_out = F.gelu(x, approximate=approximate)
        y = gelu_out * scale + offset

        if round_mode in ("rint", "round"):
            y = torch.round(y)

        y = torch.clamp(y, -128, 127).to(torch.int8)
        return y

    def run_gelu_quant(self, shape, dtype, approximate, quant_mode, round_mode, dst_type):
        input_tensor = torch.randn(shape, dtype=dtype)
        input_scale = torch.ones([shape[-1]], dtype=dtype)
        input_offset = torch.zeros([shape[-1]], dtype=dtype)

        golden = self.get_golden(input_tensor, input_scale, input_offset, approximate, round_mode)

        result = torch_npu.npu_gelu_quant(
            input_tensor.npu(),
            input_scale.npu(),
            input_offset.npu(),
            approximate,
            quant_mode,
            round_mode,
            dst_type,
        )
        output_npu = result[0] if isinstance(result, (tuple, list)) else result
        output_npu = output_npu.cpu()

        self.assertRtolEqual(output_npu, golden)

    def run_test(self, shape, dtype):
        print("======test case:", shape, dtype)
        self.run_gelu_quant(
            shape,
            dtype,
            approximate="tanh",
            quant_mode="static",
            round_mode="rint",
            dst_type=2,
        )

    def test_case(self):
        self.run_test({shape}, {dtype})


if __name__ == "__main__":
    run_tests()
"""


def _parse_shape(shape_value):
    if isinstance(shape_value, (list, tuple)):
        return list(shape_value)
    if isinstance(shape_value, str):
        text = shape_value.strip()
        if text.startswith("["):
            return list(ast.literal_eval(text))
        parts = [p.strip() for p in text.split(",") if p.strip()]
        return [int(p) for p in parts]
    raise ValueError("shape must be list or string")


def _normalize_dtype(dtype_value):
    if isinstance(dtype_value, str):
        key = dtype_value.strip().lower()
        if key in DTYPE_MAP:
            return DTYPE_MAP[key]
    raise ValueError(f"unsupported dtype: {dtype_value}")


def _class_name_from_op(op_name):
    base = op_name.split("/")[-1]
    parts = [p for p in base.replace("-", "_").split("_") if p]
    return "Test" + "".join(p.capitalize() for p in parts)


def _file_name_from_op(op_name):
    base = op_name.split("/")[-1]
    base = base.replace("-", "_")
    return f"test_{base}.py"


def _render_template(op_name, shape, dtype, dim):
    base = op_name.split("/")[-1]
    if base != "swi_glu":
        if base != "gelu_quant":
            raise NotImplementedError(
                "only activation/swi_glu and activation/gelu_quant are supported right now"
            )
        return GELU_QUANT_TEMPLATE.format(
            class_name=_class_name_from_op(op_name),
            shape=shape,
            dtype=dtype,
        )
    return SWIGLU_TEMPLATE.format(
        class_name=_class_name_from_op(op_name),
        shape=shape,
        dtype=dtype,
        dim=dim,
    )


def _llm_request(messages, temperature=0.2, max_tokens=4000):
    base_url =  "https://api.deepseek.com/v1"
    api_key = "sk-d1e612e48d654731bffff6cbf35fdfd8"
    model = "deepseek-chat"
    if not api_key:
        raise RuntimeError("Missing LLM_API_KEY or OPENAI_API_KEY")

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _build_llm_prompt(op_name, shape, dtype, dim, oneshot_path):
    oneshot_text = Path(oneshot_path).read_text(encoding="utf-8")
    user_prompt = dedent(
        f"""
        你是PyTorch NPU算子测试用例生成器。请基于one-shot示例，生成一个新的测试文件。
        要求：
        - 输出必须是完整的Python测试文件源码，不要包含解释文字。
        - 保持与示例一致的风格与结构（类名、run_tests入口、golden对比、打印等）。
        - 输入算子名: {op_name}
        - 输入shape: {shape}
        - 输入dtype: {dtype}
        - 其他必要参数 diketahui时自行合理设置，缺省可参考示例。

        one-shot示例：
        ```python
        {oneshot_text}
        ```
        """
    ).strip()
    return [
        {
            "role": "system",
            "content": "你是严格的代码生成器，只输出代码，不要输出markdown围栏。",
        },
        {"role": "user", "content": user_prompt},
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate NPU op test file")
    parser.add_argument("--op", help="operator name, e.g. activation/swi_glu")
    parser.add_argument("--shape", help="input shape, e.g. [48,1,9216] or 48,1,9216")
    parser.add_argument("--dtype", help="dtype, e.g. bfloat16/float16/float32")
    parser.add_argument("--dim", type=int, default=-1, help="dim for swiglu")
    parser.add_argument(
        "--mode",
        choices=["template", "llm"],
        default="llm",
        help="generation mode",
    )
    parser.add_argument(
        "--oneshot-path",
        default=str(Path(__file__).with_name("test_swi_glu.py")),
        help="one-shot example file path for LLM prompt",
    )
    parser.add_argument(
        "--oneshot",
        help="JSON string or @file.json with keys: op, shape, dtype, dim(optional)",
    )
    parser.add_argument("--out", help="output path, defaults to test_<op>.py")
    args = parser.parse_args()

    if args.oneshot:
        raw = args.oneshot
        if raw.startswith("@"):
            raw = Path(raw[1:]).read_text(encoding="utf-8")
        data = json.loads(raw)
        op_name = data.get("op")
        shape = _parse_shape(data.get("shape"))
        dtype = _normalize_dtype(data.get("dtype"))
        dim = int(data.get("dim", -1))
    else:
        if not (args.op and args.shape and args.dtype):
            parser.error("--op/--shape/--dtype are required unless --oneshot is used")
        op_name = args.op
        shape = _parse_shape(args.shape)
        dtype = _normalize_dtype(args.dtype)
        dim = args.dim

    if args.mode == "llm":
        messages = _build_llm_prompt(op_name, shape, dtype, dim, args.oneshot_path)
        content = _llm_request(messages)
    else:
        content = _render_template(op_name, shape, dtype, dim)
    out_path = Path(args.out) if args.out else Path(_file_name_from_op(op_name))
    out_path.write_text(content, encoding="utf-8")
    print(f"Generated {out_path}")


if __name__ == "__main__":
    main()
