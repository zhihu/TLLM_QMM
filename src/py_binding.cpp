/*
 * Copyright 2024 Zhihu Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include "tensorrt_llm/thop/thUtils.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/custom_class.h>
#include <torch/nn/functional.h>

#include <cutlass/numeric_types.h>

#include <ATen/cuda/CUDAGeneratorImpl.h>

#include "plugins/weightOnlyGroupwiseQuantMatmulPlugin/weightOnlyGroupwiseQuantMatmulPlugin.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"

#if defined(TORCH_VERSION_MAJOR) && ((TORCH_VERSION_MAJOR > 1) || ((TORCH_VERSION_MAJOR == 1) && (TORCH_VERSION_MINOR >= 9)))
#define TORCH_IS_AT_LEAST_v190
#endif

using namespace tensorrt_llm::plugins;
using namespace tensorrt_llm::common;
using namespace torch_ext;
using namespace tensorrt_llm::kernels::cutlass_kernels;

static constexpr int BIAS = int(1) << 0;
static constexpr int ZERO = int(1) << 1;
static constexpr int PRE_QUANT_SCALE = int(1) << 2;
static constexpr int FP8_ALPHA = int(1) << 3;

void check_quant_type_allowed(torch::ScalarType quant_type) {
#ifdef TORCH_IS_AT_LEAST_v190
    TORCH_CHECK(
        quant_type == torch::kInt8 || quant_type == at::ScalarType::QUInt4x2, "Must be int4 or int8 quantization");
#else
    TORCH_CHECK(quant_type == torch::kInt8, "Must be int8 quantization");
#endif
}

QuantType get_ft_quant_type(torch::ScalarType quant_type, torch::ScalarType activation_type = torch::kFloat16) {
    // Actually we need FP8 here, but current torch version does not support FP8. That's why INT8 is employed here
    if (activation_type == torch::kFloat8_e4m3fn) {
        return QuantType::W4_AFP8;
    } else if (quant_type == torch::kInt8) {
        return QuantType::W8_A16;
    }
#ifdef TORCH_IS_AT_LEAST_v190
    else if (quant_type == at::ScalarType::QUInt4x2) {
        return QuantType::W4_A16;
    }
#endif
    else {
        TORCH_CHECK(false, "Invalid quantization type");
    }
}

class WeightOnlyGroupwiseQuantMatmul {

    std::shared_ptr<WeightOnlyGroupwiseQuantMatmulPlugin> plugin;
    const int quant_algo;
    const int group_size;
    const bool is_bf16;

    WeightOnlyGroupwiseQuantMatmul(const int quant_algo, const int minM, const int maxM, const int maxK, const int maxN,
                                   const int group_size, const bool is_bf16) : quant_algo(quant_algo), group_size(group_size),
                                                                               is_bf16(is_bf16) {
        static GemmPluginProfilerManager<WeightOnlyGroupwiseQuantGemmPluginProfiler> gemmPluginProfileManager;
        std::shared_ptr<WeightOnlyGroupwiseQuantGemmPluginProfiler> profiler;
        profiler = gemmPluginProfileManager.createGemmPluginProfiler(false);
        plugin.reset(new WeightOnlyGroupwiseQuantMatmulPlugin(
            is_bf16 ? tensorrt_llm::DataType::TYPE_BF16 : tensorrt_llm::DataType::TYPE_FP16,
            quant_algo,
            group_size,
            profiler));
        plugin->configurePlugin(minM, maxM, maxK, maxN);
        plugin->initialize();
    }

public:
    at::Tensor
    forward(const at::Tensor &input, const at::Tensor &pre_quant_scale, const at::Tensor &weight, const at::Tensor &scale, const at::Tensor &zero, const at::Tensor &bias, const at::Tensor &alpha) {
        auto m = input.size(0);  // row major
        auto n = weight.size(1); // column major
        auto k = weight.size(0);
        at::Tensor output = at::empty(
            at::IntArrayRef(std::vector<int64_t>{m, n * FP16_INT4_RATIO}),
            at::TensorOptions()
                .device(input.device())
                .dtype(is_bf16 ? at::ScalarType::BFloat16 : at::ScalarType::Half));
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        int64_t ws_bytes = plugin->getWorkspaceSize();
        at::Tensor workspace = ws_bytes > 0 ? at::empty(
                                                  at::IntArrayRef(std::vector<int64_t>{ws_bytes}),
                                                  at::TensorOptions()
                                                      .device(input.device())
                                                      .dtype(is_bf16 ? at::ScalarType::BFloat16 : at::ScalarType::Half))
                                            : at::Tensor();
        plugin->run(input.data_ptr(), pre_quant_scale.data_ptr(), weight.data_ptr(), scale.data_ptr(), zero.data_ptr(), bias.data_ptr(), alpha.data_ptr(), output.data_ptr(), m, n, k, workspace.data_ptr(), ws_bytes, is_bf16, stream);
        return output;
    }

    at::Tensor preprocess_weights(at::Tensor row_major_quantized_weight) {
        auto _st = row_major_quantized_weight.scalar_type();
        auto quant_type = at::ScalarType::QUInt4x2;
        auto activation_type = quant_algo & FP8_ALPHA ? torch::kFloat8_e4m3fn : at::ScalarType::Half; // bf16 compactible, only bits matters.
        CHECK_CPU(row_major_quantized_weight);
        CHECK_CONTIGUOUS(row_major_quantized_weight);
        TORCH_CHECK(_st == torch::kInt8, "Quantized tensor must be int8 dtype");
        check_quant_type_allowed(quant_type);
        TORCH_CHECK(row_major_quantized_weight.dim() == 2 || row_major_quantized_weight.dim() == 3,
                    "Invalid dim. The dim of weight should be 2 or 3");

        QuantType ft_quant_type = get_ft_quant_type(quant_type, activation_type);
        const size_t bits_in_quant_type = get_weight_quant_bits(ft_quant_type);

        const size_t num_experts = row_major_quantized_weight.dim() == 2 ? 1 : row_major_quantized_weight.size(0);
        const size_t num_rows = row_major_quantized_weight.size(-2);
        const size_t num_cols = (8 / bits_in_quant_type) * row_major_quantized_weight.size(-1);

        at::Tensor processed_tensor = torch::zeros_like(row_major_quantized_weight);
        int8_t *input_byte_ptr = get_ptr<int8_t>(row_major_quantized_weight);
        int8_t *output_byte_ptr = get_ptr<int8_t>(processed_tensor);

        preprocess_weights_for_mixed_gemm(
            output_byte_ptr, input_byte_ptr, {num_experts, num_rows, num_cols}, ft_quant_type);

        return processed_tensor;
    }
    static WeightOnlyGroupwiseQuantMatmul create(const int quant_algo, const int minM, const int maxM, const int maxK, const int maxN, const int group_size, const bool is_bf16) {
        return WeightOnlyGroupwiseQuantMatmul(quant_algo, minM, maxM, maxK, maxN, group_size, is_bf16);
    }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "WeightOnly Quantized Gemm from TensorRT-LLM";
    py::class_<WeightOnlyGroupwiseQuantMatmul>(m, "WeightOnlyGroupwiseQuantMatmul")
        .def(py::init(&WeightOnlyGroupwiseQuantMatmul::create))
        .def("forward", &WeightOnlyGroupwiseQuantMatmul::forward)
        .def("preprocess_weights", &WeightOnlyGroupwiseQuantMatmul::preprocess_weights);
}
