import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests
import torch.nn.functional as F

torch.npu.config.allow_internal_format = False


class TestGeluQuant(TestCase):

    def get_golden(self, input):
        def high_precision(x):
            # 使用FP32进行高精度GELU计算，然后转换回BF16
            x_fp32 = x.type(torch.float32)
            output_fp32 = F.gelu(x_fp32.cpu())
            return output_fp32.type(input.type())

        def low_precision(x):
            # 直接使用BF16在NPU上计算GELU
            return F.gelu(x.npu())

        return high_precision(input).cpu()

    def run_gelu_quant(self, shape, dtype):
        # 创建随机输入
        input_tensor = torch.randn(shape, dtype=dtype)

        # Golden 结果（CPU 上计算）
        golden = self.get_golden(input_tensor)
        print("golden shape:", golden.shape)

        # NPU 算子结果
        for i in range(100):
            output_npu = torch_npu.npu_gelu_quant(input_tensor.npu()).cpu()
        print("output_npu shape:", output_npu.shape)

        # 比较（注意：golden 和 output 都是相同 dtype）
        self.assertRtolEqual(output_npu, golden)

    def run_test(self, shape, dtype):
        print("======test case:", shape, dtype)
        self.run_gelu_quant(shape, dtype)

    def test_case(self):
        self.run_test([48, 1, 9216], torch.bfloat16)


if __name__ == "__main__":
    run_tests()