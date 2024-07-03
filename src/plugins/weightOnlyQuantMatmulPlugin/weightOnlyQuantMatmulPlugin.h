/*
 * Copyright 2023 The OpenBMB team.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv/kernelLauncher.h"
#include "tensorrt_llm/plugins/common/gemmPluginProfiler.h"
// #include "tensorrt_llm/plugins/common/plugin.h"

#include <cassert>
#include <cutlass/numeric_types.h>
#include <memory>
#include <set>
#include <string>
#include <vector>

// The blank line here is to avoid clang-format -sort-includes option reordering these two cutlass header files and
// breaking dependencies
#include "cutlass/integer_subbyte.h"

namespace tensorrt_llm::plugins
{
enum class WeightTypeId {
    INT8 = 1,
    INT4 = 2,
};

constexpr int32_t FP16_BITS = 16;
constexpr int32_t INT8_BITS = 8;
constexpr int32_t INT4_BITS = 4;
constexpr int32_t INT8_INT4_RATIO = INT8_BITS / INT4_BITS;
constexpr int32_t FP16_INT4_RATIO = FP16_BITS / INT4_BITS;

inline int32_t getWeightTypeMultiplier(WeightTypeId weightTypeId) {
    return weightTypeId == WeightTypeId::INT8 ? 1 : INT8_INT4_RATIO;
}

using WeightOnlyGemmRunner = tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;

class WeightOnlyQuantGemmPluginProfiler : public GemmPluginProfiler<tensorrt_llm::cutlass_extensions::CutlassGemmConfig,
                                              WeightOnlyGemmRunnerPtr, GemmIdCore, GemmIdCoreHash>
{
public:
    using Config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;

    void setWeightTypeId(WeightTypeId weightId) {
        mWeightTypeId = weightId;
    }

protected:
    void runTactic(int m, int n, int k, Config const &tactic, char *workspace, cudaStream_t const &stream) override;

    void computeTmpSize(int maxM, int n, int k) override;

    std::vector<Config> getTactics(int m, int n, int k) const override;

private:
    WeightTypeId mWeightTypeId;
};

class WeightOnlyQuantMatmulPlugin {
public:
    using PluginProfilerPtr = std::shared_ptr<WeightOnlyQuantGemmPluginProfiler>;
    WeightOnlyQuantMatmulPlugin() = delete;

    WeightOnlyQuantMatmulPlugin(DataType type, int weightTypeId, PluginProfilerPtr const &profiler);

    ~WeightOnlyQuantMatmulPlugin() = default;

    void configurePlugin(const int minM, const int maxM, const int maxK, const int maxN) noexcept;
    size_t getWorkspaceSize() const noexcept;
    int initialize() noexcept;
    void run(const void *inputs_ptr, const void *weights_ptr, const void *scales_ptr, void *output_ptr,
             const int m, const int n, const int k, void *workspace, cudaStream_t stream);

private:
    void init(DataType type, int weightTypeId);

    void configGemm();

private:
    const std::string mLayerName;

    WeightOnlyGemmRunnerPtr m_weightOnlyGemmRunner;
    size_t m_workspaceMaxSize;
    DataType mType;
    WeightTypeId mWeightTypeId;
    bool mCudaKernelEnabled;
    tensorrt_llm::kernels::weight_only::KernelType mCudaKernelType;
    int mArch;

    // When M is smaller than this value, we trigger a fast path
    // I.e. a tailored kernel instead of cutlass.
    static constexpr int SMALL_M_FAST_PATH = 5;

    GemmDims mDims{};
    GemmIdCore mGemmId{};

    PluginProfilerPtr mPluginProfiler;
};

} // namespace tensorrt_llm::plugins
