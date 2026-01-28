import torch
import torch_npu
from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.config.allow_internal_format = False


class TestLeakyRelu(TestCase):

    def get_golden(self, input, negative_slope):
        # 高精度计算，在CPU上使用FP32计算，最后转换回FP16
        input_fp32 = input.cpu().type(torch.float32)
        output_fp32 = torch.nn.functional.leaky_relu(input_fp32, negative_slope=negative_slope)
        return output_fp32.type(input.type())

    def run_leaky_relu(self, shape, dtype, negative_slope):
        # 创建随机输入
        input_tensor = torch.randn(shape, dtype=dtype)

        # Golden 结果（CPU 上计算）
        golden = self.get_golden(input_tensor, negative_slope)
        print("golden shape:", golden.shape)

        # NPU 算子结果
        for i in range(100):
            output_npu = torch_npu.npu_leaky_relu(input_tensor.npu(), negative_slope=negative_slope).cpu()
        print("output_npu shape:", output_npu.shape)

        # 比较（注意：golden 和 output 都是相同 dtype）
        self.assertRtolEqual(output_npu, golden)

    def run_test(self, shape, dtype, negative_slope=0.01):
        print("======test case:", shape, dtype, negative_slope)
        self.run_leaky_relu(shape, dtype, negative_slope)

    def test_case(self):
        self.run_test([256, 256, 256], torch.float16)


if __name__ == "__main__":
    run_tests()