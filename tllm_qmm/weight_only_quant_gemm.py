# coding=utf-8
# Copyright 2024 Zhihu Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .py_binding import WeightOnlyGroupwiseQuantMatmul
from .awq_utils import unpack_awq, reverse_awq_order, pack_columnwise
from .gptq_utils import unpack_4bit_to_32bit_signed
import torch

BITS = 4
W4A8_FP8 = 10
W4A16 = 2


class WeightOnlyGroupwiseQuantGEMM:

    def __init__(
        self,
        quant_algo,
        minM,
        maxM,
        maxK,
        maxN,
        group_size: int = 128,
        is_bf16: bool = False,
    ):
        self._plugin = WeightOnlyGroupwiseQuantMatmul(
            quant_algo, minM, maxM, maxK, maxN, group_size, is_bf16
        )

    def forward(self, inputs, pre_quant_scale, qweight, scales, zeros, bias, fp8_alpha):
        return self._plugin.forward(
            inputs, pre_quant_scale, qweight, scales, zeros, bias, fp8_alpha
        )

    def preprocess_weights_gptq(self, qweight, qzeros, scales):
        # Unpack the qweight and qzeros tensors
        iweight, izeros = unpack_4bit_to_32bit_signed(qweight, qzeros)

        # trt-llm uses signed int4, minus 8.
        iweight = iweight.to(torch.int8) - 8
        iweight = torch.bitwise_and(iweight, 0x0F)  # eventually correct overflow
        # swap weight/zeros, as zeros is column packed.
        qweight = pack_columnwise(iweight, BITS)

        qweight = self._plugin.preprocess_weights(
            qweight.view(torch.int8).cpu(),
        ).to(scales.device)

        # trt-llm uses additive float point zeros post-scaling, so scale first.
        qzeros = (-izeros).to(scales.dtype) * scales
        return qweight, qzeros

    def preprocess_weights(self, qweight, qzeros, scales):
        # Unpack the qweight and qzeros tensors
        iweight, izeros = unpack_awq(qweight, qzeros, BITS)
        # Reverse the order of the iweight and izeros tensors
        iweight, izeros = reverse_awq_order(iweight, izeros, BITS)

        # overflow checks
        iweight = torch.bitwise_and(iweight, (2**BITS) - 1)
        izeros = torch.bitwise_and(izeros, (2**BITS) - 1)

        # trt-llm uses signed int4, minus 8.
        iweight = iweight.to(torch.int8) - 8
        iweight = torch.bitwise_and(iweight, 0x0F)  # eventually correct overflow
        # swap weight/zeros, as zeros is column packed.
        qweight = pack_columnwise(iweight, BITS)

        qweight = self._plugin.preprocess_weights(
            qweight.view(torch.int8).cpu(),
        ).to(scales.device)

        # trt-llm uses additive float point zeros post-scaling, so scale first.
        qzeros = (-izeros).to(scales.dtype) * scales
        return qweight, qzeros


__all__ = ["W4A16", "W4A8_FP8", "WeightOnlyGroupwiseQuantGEMM"]
