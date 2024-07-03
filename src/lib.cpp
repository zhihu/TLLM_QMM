/*
 * Copyright 2023 The OpenBMB team.
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
#include "lib.h"
#include "plugins/w8a8Plugin/w8a8Plugin.h"
#include "plugins/weightOnlyGroupwiseQuantMatmulPlugin/weightOnlyGroupwiseQuantMatmulPlugin.h"
#include "plugins/weightOnlyQuantMatmulPlugin/weightOnlyQuantMatmulPlugin.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"

namespace tensorrt_llm {
namespace lib {

using namespace tensorrt_llm::plugins;
using namespace tensorrt_llm::common;

WeightOnlyQuantMatmulPlugin *get_w8a16_plugin(int weight_type_id) {
    static WeightOnlyQuantMatmulPlugin *plugin = nullptr;
    if (plugin == nullptr) {
        GemmPluginProfilerManager<WeightOnlyQuantGemmPluginProfiler> gemmPluginProfileManager;
        std::shared_ptr<WeightOnlyQuantGemmPluginProfiler> profiler;
        profiler = gemmPluginProfileManager.createGemmPluginProfiler(false);
        // int8
        plugin = new WeightOnlyQuantMatmulPlugin(
            tensorrt_llm::DataType::TYPE_FP16,
            weight_type_id,
            profiler);
    }
    return plugin;
}

WeightOnlyGroupwiseQuantMatmulPlugin *get_w4a16_plugin(const int group_size, const bool is_bf16) {
    static WeightOnlyGroupwiseQuantMatmulPlugin *plugin = nullptr;
    if (plugin == nullptr) {
        GemmPluginProfilerManager<WeightOnlyGroupwiseQuantGemmPluginProfiler> gemmPluginProfileManager;
        std::shared_ptr<WeightOnlyGroupwiseQuantGemmPluginProfiler> profiler;
        profiler = gemmPluginProfileManager.createGemmPluginProfiler(false);
        plugin = new WeightOnlyGroupwiseQuantMatmulPlugin(
            is_bf16 ? tensorrt_llm::DataType::TYPE_BF16 : tensorrt_llm::DataType::TYPE_FP16,
            2,
            group_size,
            profiler);
    }
    return plugin;
}

W8A8MatmulPlugin *get_w8a8_plugin(const bool is_bf16) {
    static W8A8MatmulPlugin *plugin = nullptr;
    if (plugin == nullptr) {
        GemmPluginProfilerManager<W8A8GemmPluginProfiler> gemmPluginProfileManager;
        std::shared_ptr<W8A8GemmPluginProfiler> profiler;
        profiler = gemmPluginProfileManager.createGemmPluginProfiler(false);
        // int8
        QuantMode quantMode = QuantMode::fromDescription(true, true, true, true);
        plugin = new W8A8MatmulPlugin(
            quantMode,
            is_bf16 ? tensorrt_llm::DataType::TYPE_BF16 : tensorrt_llm::DataType::TYPE_FP16,
            profiler);
    }
    return plugin;
}

void w8a16_initialize(const int weight_type_id, const int minM, const int maxM, const int maxK, const int maxN) {
    auto plugin = get_w8a16_plugin(weight_type_id);
    plugin->configurePlugin(minM, maxM, maxK, maxN);
    plugin->initialize();
}

void w4a16_initialize(const int group_size, const int minM, const int maxM, const int maxK, const int maxN, const bool is_bf16) {
    auto plugin = get_w4a16_plugin(group_size, is_bf16);
    plugin->configurePlugin(minM, maxM, maxK, maxN);
    plugin->initialize();
}

void w8a8_initialize(const int minM, const int maxM, const int maxK, const int maxN, const bool is_bf16) {
    auto plugin = get_w8a8_plugin(is_bf16);
    plugin->configurePlugin(minM, maxM, maxK, maxN);
    plugin->initialize();
}

void w8a16_matmul(const void *input, const void *weight, const void *scale, void *output,
                  const int m, const int n, const int k, const int weight_type_id, void *workspace, cudaStream_t stream) {
    auto plugin = get_w8a16_plugin(weight_type_id);
    plugin->run(input, weight, scale, output, m, n, k, workspace, stream);
}

void w4a16_matmul(const void *input, const void *pre_quant_scale, const void *weight, const void *scale, const void *zero, const void *alpha, void *output,
                  const int m, const int n, const int k, void *workspace, const size_t ws_bytes, const int group_size, const bool is_bf16, cudaStream_t stream) {
    auto plugin = get_w4a16_plugin(group_size, is_bf16);
    plugin->run(input, pre_quant_scale, weight, scale, zero, nullptr, alpha, output, m, n, k, workspace, ws_bytes, is_bf16, stream);
}

void w8a8_matmul(const void *input, const void *weight, const void *scale_x, const void *scale_weight, void *output,
                 const int m, const int n, const int k, void *workspace, const size_t ws_bytes, const bool is_bf16, cudaStream_t stream) {
    auto plugin = get_w8a8_plugin(is_bf16);
    plugin->run(input, weight, scale_x, scale_weight, output, m, n, k, workspace, ws_bytes, stream);
}

size_t w4a16_get_workspace_size(const int group_size, const bool is_bf16) {
    auto plugin = get_w4a16_plugin(group_size, is_bf16);
    return plugin->getWorkspaceSize();
}

size_t w8a8_get_workspace_size(const bool is_bf16) {
    auto plugin = get_w8a8_plugin(is_bf16);
    return plugin->getWorkspaceSize();
}

void preprocess_int4_weight(int8_t *out, const int8_t *in, const std::vector<size_t> &shape) {
    tensorrt_llm::kernels::cutlass_kernels::preprocess_weights_for_mixed_gemm(
        out, in, shape, tensorrt_llm::kernels::cutlass_kernels::QuantType::W4_A16);
}

/*
void gpu_preprocess_int4_weight(int8_t* out, int8_t* buffer, const int8_t* in, const std::vector<size_t>& shape, cudaStream_t stream){
    tensorrt_llm::kernels::cutlass_kernels::gpu_preprocess_weights_for_mixed_gemm(
        out, buffer, in, shape,
        tensorrt_llm::kernels::cutlass_kernels::QuantType::W4_A16,
        stream);
}
*/

} // namespace lib
} // namespace tensorrt_llm
