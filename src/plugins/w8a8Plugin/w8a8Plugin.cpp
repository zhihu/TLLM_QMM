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
#include "w8a8Plugin.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels::cutlass_kernels;
using tensorrt_llm::plugins::W8A8MatmulPlugin;
using tensorrt_llm::plugins::W8A8GemmPluginProfiler;

void W8A8GemmPluginProfiler::runTactic(int m, int n, int k, const W8A8GemmPluginProfiler::Config& tactic,
    char* workspace, const cudaStream_t& stream)
{
    int8_t* aTmp = reinterpret_cast<int8_t*>(workspace);
    int8_t* bTmp = nextWorkspacePtr(aTmp, m * k * sizeof(int8_t));
    void* cTmp = reinterpret_cast<void*>(nextWorkspacePtr(bTmp, n * k * sizeof(int8_t)));
    float* alphaRowTmp = reinterpret_cast<float*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(cTmp), m * n * (mType == DataType::TYPE_FP32 ? 4 : 2)));
    float* alphaColTmp
        = reinterpret_cast<float*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(alphaRowTmp), m * sizeof(float)));
    char* workspaceTmp
        = reinterpret_cast<char*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(alphaColTmp), n * sizeof(float)));

    const int wsSize = mRunner->getWorkspaceSize(m, n, k);

    mRunner->gemm(
        aTmp, bTmp, mQuantMode, alphaColTmp, alphaRowTmp, cTmp, m, n, k, tactic, workspaceTmp, wsSize, stream);
}

void W8A8GemmPluginProfiler::computeTmpSize(int maxM, int n, int k)
{
    std::vector<size_t> workspaces = {
        maxM * k * sizeof(int8_t),                                  // A
        n * k * sizeof(int8_t),                                     // B
        maxM * n * (mType == DataType::TYPE_FP32 ? 4u : 2u), // C
        maxM * sizeof(float),                                       // alphaRow
        n * sizeof(float),                                          // alphaCol
        mRunner->getWorkspaceSize(maxM, n, k)                       // workspace
    };
    std::uintptr_t constexpr kCudaMemAlign = 128;
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size(), kCudaMemAlign);
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<W8A8GemmPluginProfiler::Config> W8A8GemmPluginProfiler::getTactics(int m, int n, int k) const
{
    return mRunner->getConfigs();
}

W8A8MatmulPlugin::W8A8MatmulPlugin(
    QuantMode quantMode, DataType type, const W8A8MatmulPlugin::PluginProfilerPtr& pluginProfiler)
    : mQuantMode(quantMode)
    , mPluginProfiler(pluginProfiler)
{
    init(type);
}

void W8A8MatmulPlugin::init(DataType type)
{
    mType = type;
    if (mType == DataType::TYPE_FP16)
    {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<half>>();
    }
    else if (mType == DataType::TYPE_FP32)
    {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<float>>();
    }
    else if (mType == DataType::TYPE_INT32)
    {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<int32_t>>();
    }
    else if (mType == DataType::TYPE_BF16)
    {
        m_sqGemmRunner = std::make_shared<CutlassInt8GemmRunner<__nv_bfloat16>>();
    }
    else
    {
        // TODO: add bf16 support
        TLLM_THROW("no support dtype");
    }

    mPluginProfiler->setQuantMode(mQuantMode);

    mGemmId = GemmIdCore(mDims.n, mDims.k, mType);
}


size_t W8A8MatmulPlugin::getWorkspaceSize() const noexcept
{
    return m_workspaceMaxSize;
}

void W8A8MatmulPlugin::run(
	const void* input, const void *weight, 
	const void* scale_x,
	const void* scale_weight, 
	void* output, 
	const int m, const int n, const int k, 
    void* workspace,
	const size_t ws_bytes, 
    cudaStream_t stream) 
{
    const auto& bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
    TLLM_CHECK_WITH_INFO(bestTactic, "No valid SQ GEMM tactic");
    m_sqGemmRunner->gemm(
        reinterpret_cast<const int8_t*>(input), 
        reinterpret_cast<const int8_t*>(weight),
        mQuantMode, 
		reinterpret_cast<const float*>(scale_weight),
		reinterpret_cast<const float*>(scale_x),
        output, m, n, k, *bestTactic, reinterpret_cast<char*>(workspace), ws_bytes, stream);
}

int W8A8MatmulPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void W8A8MatmulPlugin::configGemm()
{
    mPluginProfiler->profileTactics(m_sqGemmRunner, mType, mDims, mGemmId);
}

void W8A8MatmulPlugin::configurePlugin(const int minM, const int maxM,
    const int maxK, const int maxN) noexcept
{
    if (!mDims.isInitialized())
    {
        mDims = {minM, maxM, maxN, maxK};
    }
    mGemmId = {maxN, maxK, mType};

    m_workspaceMaxSize = m_sqGemmRunner->getWorkspaceSize(maxM, maxN, maxK);
}
