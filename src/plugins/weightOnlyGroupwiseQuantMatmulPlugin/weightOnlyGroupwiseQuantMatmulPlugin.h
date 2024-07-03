/*
 * Copyright 2024 Zhihu Inc.
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

#include "plugins/weightOnlyQuantMatmulPlugin/weightOnlyQuantMatmulPlugin.h"
#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/kernels/preQuantScaleKernel.h"
#include "tensorrt_llm/kernels/weightOnlyBatchedGemv//kernelLauncher.h"
#include "tensorrt_llm/plugins/common/gemmPluginProfiler.h"

#include <cutlass/numeric_types.h>

#include <cassert>
#include <cuda_runtime.h>
#include <memory>
#include <set>
#include <string>
#include <vector>

// The blank line here is to avoid clang-format -sort-includes option reordering these two cutlass header files and
// breaking dependencies
#include "cutlass/integer_subbyte.h"

namespace tensorrt_llm::plugins
{

using WeightOnlyGemmRunner = tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;

class WeightOnlyGroupwiseQuantGemmPluginProfiler
    : public GemmPluginProfiler<tensorrt_llm::cutlass_extensions::CutlassGemmConfig, WeightOnlyGemmRunnerPtr,
          GemmIdCore, GemmIdCoreHash>
{
public:
    using Config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;

    void setQuantAlgo(int quantAlgo)
    {
        mQuantAlgo = quantAlgo;
    }

    void setGroupSize(int groupSize)
    {
        mGroupSize = groupSize;
    }

protected:
    void runTactic(int m, int n, int k, Config const &tactic, char *workspace, cudaStream_t const &stream) override;

    void computeTmpSize(int maxM, int n, int k) override;

    std::vector<Config> getTactics(int m, int n, int k) const override;

private:
    int mQuantAlgo;
    int mGroupSize;
};

class WeightOnlyGroupwiseQuantMatmulPlugin {
public:
    using PluginProfilerPtr = std::shared_ptr<WeightOnlyGroupwiseQuantGemmPluginProfiler>;

    WeightOnlyGroupwiseQuantMatmulPlugin() = delete;

    WeightOnlyGroupwiseQuantMatmulPlugin(
        DataType type, int quant_algo, int group_size, PluginProfilerPtr const &profiler);

    ~WeightOnlyGroupwiseQuantMatmulPlugin() = default;

    void configurePlugin(const int minM, const int maxM, const int maxK, const int maxN) noexcept;
    size_t getWorkspaceSize() const noexcept;
    int initialize() noexcept;
    void run(const void *inputs_ptr, const void *pre_quant_scales_ptr, const void *weights_ptr, const void *scales_ptr, const void *zeros_ptr, const void *biases_ptr, const void *alpha_ptr, void *output_ptr,
             const int m, const int n, const int k, void *workspace, const size_t ws_bytes, const bool is_bf16, cudaStream_t stream);

private:
    // group_size: 64, 128
    void init(DataType type, int quant_algo, int group_size);

    void configGemm();

private:
    const std::string mLayerName;

    WeightOnlyGemmRunnerPtr m_weightOnlyGroupwiseGemmRunner;
    size_t m_workspaceMaxSize;
    DataType mType;
    bool mCudaKernelEnabled;
    tensorrt_llm::kernels::weight_only::KernelType mCudaKernelType;
    int mArch;

    // When M is smaller than this value, we trigger a fast path
    // I.e. a tailored kernel instead of cutlass.
    static constexpr int SMALL_M_FAST_PATH = 5;

    int mQuantAlgo;

    int mGroupSize;

    int mPreQuantScaleInputIdx;
    int mWeightInputIdx;
    int mScalesInputIdx;
    int mZerosInputIdx;
    int mBiasesInputIdx;
    int mAlphaInputIdx;

    GemmDims mDims{};
    GemmIdCore mGemmId{};

    PluginProfilerPtr mPluginProfiler;
};

} // namespace tensorrt_llm::plugins
