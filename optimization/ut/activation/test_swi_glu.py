import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torch.nn.functional as F

torch.npu.config.allow_internal_format = False


class TestSwiGlu(TestCase):

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
        self.run_test([24, 9216, 2560], torch.float16, dim=-1)


if __name__ == "__main__":
    run_tests()
