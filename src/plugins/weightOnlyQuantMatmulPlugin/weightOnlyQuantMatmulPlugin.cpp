/*
 * Copyright 2023 The OpenBMB team.
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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
#include "weightOnlyQuantMatmulPlugin.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels::cutlass_kernels;
using tensorrt_llm::plugins::WeightOnlyQuantGemmPluginProfiler;
using tensorrt_llm::plugins::WeightOnlyQuantMatmulPlugin;

static char const *WOQ_MATMUL_PLUGIN_VERSION{"1"};
static char const *WOQ_MATMUL_PLUGIN_NAME{"WeightOnlyQuantMatmul"};

void WeightOnlyQuantGemmPluginProfiler::runTactic(int m, int n, int k,
                                                  WeightOnlyQuantGemmPluginProfiler::Config const &tactic, char *workspace, cudaStream_t const &stream) {
    int const originalN = n * getWeightTypeMultiplier(mWeightTypeId);
    half* actPtr = reinterpret_cast<half*>(workspace);
    int8_t* weightPtr
        = reinterpret_cast<int8_t*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(actPtr), m * k * sizeof(half)));
    half* scalesPtr = reinterpret_cast<half*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(weightPtr), originalN * k * sizeof(int8_t)));
    half* outputPtr
        = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(scalesPtr), originalN * sizeof(half)));
    char* workspacePtr
        = reinterpret_cast<char*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(outputPtr), m * originalN * sizeof(half)));

    int const wsSize = mRunner->getWorkspaceSize(m, n, k);

    if (mWeightTypeId == WeightTypeId::INT8) {
        mRunner->gemm(actPtr, weightPtr, scalesPtr, outputPtr, m, originalN, k, tactic, workspacePtr, wsSize, stream);
    } else {
        mRunner->gemm(actPtr, reinterpret_cast<cutlass::uint4b_t*>(weightPtr), scalesPtr, outputPtr, m, originalN, k,
            tactic, workspacePtr, wsSize, stream);
    }
}

void WeightOnlyQuantGemmPluginProfiler::computeTmpSize(int maxM, int n, int k)
{
    int const originalN = n * getWeightTypeMultiplier(mWeightTypeId);
    std::vector<size_t> workspaces = {
        maxM * k * sizeof(half),              // A
        originalN * k * sizeof(int8_t),       // B
        originalN * sizeof(half),             // scales
        maxM * originalN * sizeof(half),      // C
        mRunner->getWorkspaceSize(maxM, n, k) // workspace
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<WeightOnlyQuantGemmPluginProfiler::Config> WeightOnlyQuantGemmPluginProfiler::getTactics(
    int m, int n, int k) const
{
    return mRunner->getConfigs();
}

WeightOnlyQuantMatmulPlugin::WeightOnlyQuantMatmulPlugin(DataType type, int weightTypeId,
                                                         WeightOnlyQuantMatmulPlugin::PluginProfilerPtr const &pluginProfiler)
    : mPluginProfiler(pluginProfiler) {
    init(type, weightTypeId);
}

void WeightOnlyQuantMatmulPlugin::init(DataType type, int weightTypeId) {
    mArch = tensorrt_llm::common::getSMVersion();
    mType = type;
    mWeightTypeId = WeightTypeId(weightTypeId);

    if (mWeightTypeId == WeightTypeId::INT8) {
        if (mType == DataType::TYPE_FP16) {
            m_weightOnlyGemmRunner = std::make_shared<
                CutlassFpAIntBGemmRunner<half, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
            mCudaKernelEnabled = tensorrt_llm::kernels::weight_only::is_supported(
                mArch, tensorrt_llm::kernels::weight_only::KernelType::FP16Int8PerChannel);
            mCudaKernelType = tensorrt_llm::kernels::weight_only::KernelType::FP16Int8PerChannel;
        }
#if defined(ENABLE_BF16)
        else if (mType == DataType::TYPE_BF16) {
            m_weightOnlyGemmRunner = std::make_shared<
                CutlassFpAIntBGemmRunner<__nv_bfloat16, uint8_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
            mCudaKernelEnabled = tensorrt_llm::kernels::weight_only::is_supported(
                mArch, tensorrt_llm::kernels::weight_only::KernelType::BF16Int8PerChannel);
            mCudaKernelType = tensorrt_llm::kernels::weight_only::KernelType::BF16Int8PerChannel;
        }
#endif
        else {
            TLLM_CHECK(false);
        }
    } else if (mWeightTypeId == WeightTypeId::INT4) {
        if (mType == DataType::TYPE_FP16) {
            m_weightOnlyGemmRunner = std::make_shared<
                CutlassFpAIntBGemmRunner<half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
            mCudaKernelEnabled = tensorrt_llm::kernels::weight_only::is_supported(
                mArch, tensorrt_llm::kernels::weight_only::KernelType::FP16Int4PerChannel);
            mCudaKernelType = tensorrt_llm::kernels::weight_only::KernelType::FP16Int4PerChannel;
        }
#if defined(ENABLE_BF16)
        else if (mType == DataType::TYPE_BF16) {
            m_weightOnlyGemmRunner = std::make_shared<CutlassFpAIntBGemmRunner<__nv_bfloat16, cutlass::uint4b_t,
                                                                               cutlass::WeightOnlyQuantOp::PER_COLUMN_SCALE_ONLY>>();
            mCudaKernelEnabled = tensorrt_llm::kernels::weight_only::is_supported(
                mArch, tensorrt_llm::kernels::weight_only::KernelType::BF16Int4PerChannel);
            mCudaKernelType = tensorrt_llm::kernels::weight_only::KernelType::BF16Int4PerChannel;
        }
#endif
        else {
            TLLM_CHECK(false);
        }
    } else {
        TLLM_CHECK(false);
    }

    mPluginProfiler->setWeightTypeId(mWeightTypeId);

    mGemmId = GemmIdCore(mDims.n, mDims.k, mType);
}

void WeightOnlyQuantMatmulPlugin::run(const void *inputs_ptr, const void *weights_ptr, const void *scales_ptr, void *outputs_ptr,
                                      const int m, const int n, const int k, void *workspace, cudaStream_t stream) {
    // inputs
    //     mat1           [M1, M2,..., K]
    //     mat2           [K, N] for int8, [K, N/2] for int4
    //     scale_channels [N]
    // outputs
    //     mat [M, N]

    bool const use_cuda_kernel = m < SMALL_M_FAST_PATH && mCudaKernelEnabled;
#if defined(ENABLE_BF16)
    TLLM_CHECK_WITH_INFO(mType == DataType::TYPE_FP16 || mType == DataType::TYPE_BF16,
                         "No valid weightOnlyQuantMatmul configuration");
#else
    TLLM_CHECK_WITH_INFO(mType == DataType::TYPE_FP16, "No valid weightOnlyQuantMatmul configuration");
#endif
    int real_n = mWeightTypeId == WeightTypeId::INT4 ? n * INT8_INT4_RATIO : n;
    if (use_cuda_kernel) {
        void const *cuda_kernel_act_ptr = inputs_ptr;
        void const *cuda_kernel_weight_ptr = weights_ptr;
        void const *cuda_kernel_scales_ptr = scales_ptr;
        void *cuda_kernel_out_ptr = outputs_ptr;
        tensorrt_llm::kernels::weight_only::Params params(cuda_kernel_act_ptr, nullptr, cuda_kernel_weight_ptr,
                                                          cuda_kernel_scales_ptr, nullptr, nullptr, cuda_kernel_out_ptr, 1.f, m, real_n, k, 0, mCudaKernelType);
        tensorrt_llm::kernels::weight_only::kernel_launcher(mArch, params, stream);
    } else {
        int const ws_size = m_weightOnlyGemmRunner->getWorkspaceSize(m, real_n, k);

        auto const &bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
        TLLM_CHECK_WITH_INFO(bestTactic,
                             "No valid weight only per-channel GEMM tactic(It is usually caused by the failure to execute all candidate "
                             "configurations of the CUTLASS kernel, please pay attention to the warning information when building the "
                             "engine.)");

        m_weightOnlyGemmRunner->gemm(inputs_ptr, weights_ptr, scales_ptr, outputs_ptr, m, real_n, k, *bestTactic,
                                     reinterpret_cast<char *>(workspace), ws_size, stream);
    }

    return;
}

int WeightOnlyQuantMatmulPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}