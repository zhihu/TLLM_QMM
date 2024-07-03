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
#include "tensorrt_llm/kernels/cutlass_kernels/int8_gemm/int8_gemm.h"
#include "tensorrt_llm/plugins/common/gemmPluginProfiler.h"

#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <vector>

// The blank line here is to avoid clang-format -sort-includes option reordering these two cutlass header files and
// breaking dependencies
//#include "cutlass/integer_subbyte.h"

namespace tensorrt_llm::plugins
{

using perfMapType = std::unordered_map<int, tensorrt_llm::cutlass_extensions::CutlassGemmConfig>;
using SqGemmRunnerPtr = std::shared_ptr<tensorrt_llm::kernels::cutlass_kernels::CutlassInt8GemmRunnerInterface>;

class W8A8GemmPluginProfiler
    : public GemmPluginProfiler<tensorrt_llm::cutlass_extensions::CutlassGemmConfig, SqGemmRunnerPtr,
          GemmIdCore, GemmIdCoreHash>
{
public:
    using Config = tensorrt_llm::cutlass_extensions::CutlassGemmConfig;

  	void setQuantMode(const tensorrt_llm::common::QuantMode& quantMode)
    {
        mQuantMode = quantMode;
    }

protected:
    void runTactic(int m, int n, int k, const Config& tactic, char* workspace, const cudaStream_t& stream) override;

    void computeTmpSize(int maxM, int n, int k) override;

    std::vector<Config> getTactics(int m, int n, int k) const override;

private:
    tensorrt_llm::common::QuantMode mQuantMode;
};

class W8A8MatmulPlugin 
{
public:
    using PluginProfilerPtr = std::shared_ptr<W8A8GemmPluginProfiler>;

    W8A8MatmulPlugin() = delete;

    W8A8MatmulPlugin(
        DataType type, int quant_algo, int group_size, const PluginProfilerPtr& profiler);

	W8A8MatmulPlugin(
    	tensorrt_llm::common::QuantMode quantMode, DataType type, const W8A8MatmulPlugin::PluginProfilerPtr& pluginProfiler);

    ~W8A8MatmulPlugin() ;

    void configurePlugin(const int minM, const int maxM, const int maxK, const int maxN) noexcept;
    size_t getWorkspaceSize() const noexcept;

    int initialize() noexcept;
    void run(const void* input, const void* weight, const void* scale_x, const void* scale_weight, void* output,
        const int m, const int n, const int k, void* workspace, const size_t ws_bytes, cudaStream_t stream);

private:
    // group_size: 64, 128
    void init(DataType type);

    void configGemm();

private:
    const std::string mLayerName;

    SqGemmRunnerPtr m_sqGemmRunner;
    tensorrt_llm::common::QuantMode mQuantMode;
    size_t m_workspaceMaxSize;
    DataType mType;

    GemmDims mDims{};
    GemmIdCore mGemmId{};

    PluginProfilerPtr mPluginProfiler;
};

} // namespace tensorrt_llm::plugins
