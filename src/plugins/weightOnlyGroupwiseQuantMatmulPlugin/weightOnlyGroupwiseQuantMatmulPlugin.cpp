/*
 * Copyright 2024 Zhihu Inc.
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
#include "weightOnlyGroupwiseQuantMatmulPlugin.h"
#include <cuda_fp8.h>

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::kernels::cutlass_kernels;
using tensorrt_llm::plugins::WeightOnlyGroupwiseQuantGemmPluginProfiler;
using tensorrt_llm::plugins::WeightOnlyGroupwiseQuantMatmulPlugin;

// Flags for indicating whether the corresponding inputs are applied in mQuantAlgo
// mQuantAlgo = pre_quant_scale * PRE_QUANT_SCALE + zero * ZERO + bias * BIAS
// Here pre_quant_scale, zero and bias are boolean type
static constexpr int BIAS = int(1) << 0;
static constexpr int ZERO = int(1) << 1;
static constexpr int PRE_QUANT_SCALE = int(1) << 2;
static constexpr int FP8_ALPHA = int(1) << 3;

void WeightOnlyGroupwiseQuantGemmPluginProfiler::runTactic(int m, int n, int k,
                                                           WeightOnlyGroupwiseQuantGemmPluginProfiler::Config const &tactic, char *workspace, cudaStream_t const &stream) {
    // Quantized weights are packed in FP16 format (INT4*4 -> FP16)
    int const originalN = n * FP16_INT4_RATIO;
    half* actPtr = reinterpret_cast<half*>(workspace);
    cutlass::uint4b_t* weightPtr = reinterpret_cast<cutlass::uint4b_t*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(actPtr), m * k * sizeof(half)));
    half* inputScalesPtr
        = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(weightPtr), n * k * sizeof(float)));
    half* zerosPtr = reinterpret_cast<half*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(inputScalesPtr), k * originalN * sizeof(half) / mGroupSize));
    half* biasesPtr = reinterpret_cast<half*>(
        nextWorkspacePtr(reinterpret_cast<int8_t*>(zerosPtr), k * originalN * sizeof(half) / mGroupSize));
    half* outputPtr = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(biasesPtr), m * sizeof(half)));
    char* workspacePtr
        = reinterpret_cast<char*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(outputPtr), m * originalN * sizeof(half)));

    if ((mQuantAlgo & ZERO) == 0)
    {
        zerosPtr = nullptr;
    }

    if ((mQuantAlgo & BIAS) == 0)
    {
        biasesPtr = nullptr;
    }

    int const wsSize = mRunner->getWorkspaceSize(m, n, k);

    mRunner->gemm(actPtr, weightPtr, inputScalesPtr, zerosPtr, biasesPtr, outputPtr, m, originalN, k, mGroupSize,
        tactic, workspacePtr, wsSize, stream);
}

void WeightOnlyGroupwiseQuantGemmPluginProfiler::computeTmpSize(int maxM, int n, int k)
{
    // Quantized weights are packed in FP16 format (INT4*4 -> FP16)
    int const originalN = n * FP16_INT4_RATIO;
    std::vector<size_t> workspaces = {
        maxM * k * sizeof(half),                   // A
        k * n * sizeof(float),                     // B
        k * originalN * sizeof(half) / mGroupSize, // scales
        k * originalN * sizeof(half) / mGroupSize, // zeros
        maxM * sizeof(half),                       // biases
        maxM * originalN * sizeof(half),           // C
        mRunner->getWorkspaceSize(maxM, n, k)      // workspace
    };
    size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
    setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<WeightOnlyGroupwiseQuantGemmPluginProfiler::Config> WeightOnlyGroupwiseQuantGemmPluginProfiler::getTactics(
    int m, int n, int k) const
{
    return mRunner->getConfigs();
}

WeightOnlyGroupwiseQuantMatmulPlugin::WeightOnlyGroupwiseQuantMatmulPlugin(DataType type, int quant_algo,
                                                                           int group_size, WeightOnlyGroupwiseQuantMatmulPlugin::PluginProfilerPtr const &pluginProfiler)
    : mPluginProfiler(pluginProfiler) {
    init(type, quant_algo, group_size);
}

void WeightOnlyGroupwiseQuantMatmulPlugin::init(DataType type, int quant_algo, int group_size) {
    mArch = tensorrt_llm::common::getSMVersion();
    mType = type;
    mQuantAlgo = quant_algo;
    mGroupSize = group_size;

    // quant_algo = fp8_alpha * 8 + pre_quant_scale * 4 + zero * 2 + bias
    mPreQuantScaleInputIdx = (quant_algo & PRE_QUANT_SCALE) ? 1 : 0;
    mWeightInputIdx = mPreQuantScaleInputIdx + 1;
    mScalesInputIdx = mWeightInputIdx + 1;
    mZerosInputIdx = (quant_algo & ZERO) ? mScalesInputIdx + 1 : mScalesInputIdx;
    mBiasesInputIdx = (quant_algo & BIAS) ? mZerosInputIdx + 1 : mZerosInputIdx;
    mAlphaInputIdx = (quant_algo & FP8_ALPHA) ? mBiasesInputIdx + 1 : mBiasesInputIdx;

    if (mType == DataType::TYPE_FP16) {
        if (quant_algo & FP8_ALPHA) {
            // Ada & Hopper style kernels
            if (mArch < 89) {
                TLLM_THROW("W4A(fp)8 kernel is unsupported on pre-Ada (sm<89) architectures!");
            }
            if (quant_algo & ZERO) {
                // has zeros
                m_weightOnlyGroupwiseGemmRunner = std::make_shared<
                    tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<__nv_fp8_e4m3, cutlass::uint4b_t,
                                                                                     cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS, half, half, half>>();
            } else {
                // no zeros
                m_weightOnlyGroupwiseGemmRunner = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<__nv_fp8_e4m3,
                                                                                                                                    cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY, half, half, half>>();
            }
        } else {
            if (quant_algo & ZERO) {
                // has zeros
                m_weightOnlyGroupwiseGemmRunner = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<half,
                                                                                                                                    cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
            } else {
                // no zeros
                m_weightOnlyGroupwiseGemmRunner = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<half,
                                                                                                                                    cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>>();
            }
        }
        mCudaKernelEnabled = tensorrt_llm::kernels::weight_only::is_supported(
            mArch, tensorrt_llm::kernels::weight_only::KernelType::FP16Int4Groupwise);
        mCudaKernelType = tensorrt_llm::kernels::weight_only::KernelType::FP16Int4Groupwise;
    }
#if defined(ENABLE_BF16)
    else if (mType == DataType::TYPE_BF16) {
        if (quant_algo & FP8_ALPHA) {
            // FP8 requires at least sm89 devices
            if (mArch < 89) {
                TLLM_THROW("W4A(fp)8 kernel is unsupported on pre-Ada (sm<89) architectures!");
            }
            TLLM_THROW("FP8 is unsupported on with BF16 scales and zero-points!");
        } else {
            if (quant_algo & ZERO) {
                // has zeros
                m_weightOnlyGroupwiseGemmRunner = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<__nv_bfloat16,
                                                                                                                                    cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS>>();
            } else {
                // no zeros
                m_weightOnlyGroupwiseGemmRunner = std::make_shared<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<__nv_bfloat16,
                                                                                                                                    cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>>();
            }
        }
        mCudaKernelEnabled = tensorrt_llm::kernels::weight_only::is_supported(
            mArch, tensorrt_llm::kernels::weight_only::KernelType::BF16Int4Groupwise);
        mCudaKernelType = tensorrt_llm::kernels::weight_only::KernelType::BF16Int4Groupwise;
    }
#endif
    else {
        TLLM_THROW("Unsupported data type");
    }
    mPluginProfiler->setQuantAlgo(mQuantAlgo);
    mPluginProfiler->setGroupSize(mGroupSize);

    mGemmId = GemmIdCore(mDims.n, mDims.k, mType);
}

void WeightOnlyGroupwiseQuantMatmulPlugin::configGemm()
{
    mPluginProfiler->profileTactics(m_weightOnlyGroupwiseGemmRunner, mType, mDims, mGemmId);
}

void WeightOnlyGroupwiseQuantMatmulPlugin::configurePlugin(const int minM, const int maxM, const int maxK, const int maxN) noexcept {

    auto const K = maxK;
    auto const N = maxN / FP16_INT4_RATIO;

    if (!mDims.isInitialized()) {
        mDims = {minM, maxM, N, K};
    }
    mGemmId = {N, K, mType};

    size_t smoothedActSize = static_cast<size_t>(maxM) * static_cast<size_t>(maxK) * 2;
    m_workspaceMaxSize = smoothedActSize + m_weightOnlyGroupwiseGemmRunner->getWorkspaceSize(maxM, maxN, maxK);
}

size_t WeightOnlyGroupwiseQuantMatmulPlugin::getWorkspaceSize() const noexcept {
    return m_workspaceMaxSize;
}

int WeightOnlyGroupwiseQuantMatmulPlugin::initialize() noexcept
{
    configGemm();
    return 0;
}

void WeightOnlyGroupwiseQuantMatmulPlugin::run(const void *inputs_ptr, const void *pre_quant_scales_ptr, const void *weights_ptr, const void *scales_ptr, const void *zeros_ptr, const void *bias_ptr, const void *alpha_ptr, void *outputs_ptr,
                                               const int m, const int n, const int k, void *workspace, const size_t ws_bytes, const bool is_bf16, cudaStream_t stream) {

    // inputs
    //   0 activations      [M, K]
    //   1 pre-quant scales [K]
    //   2 weights          [K, N/2]
    //   3 scales           [K // group_size, N]
    //   4 zeros            [K // group_size, N]
    //   5 biases           [M]
    //   6 alpha            [1]
    // outputs
    //   mat                [M, N]

    // bool use_cuda_kernel = m < SMALL_M_FAST_PATH && mCudaKernelEnabledm && !(QuantAlgo & PRE_QUANT_SCALE);
    // bool use_cuda_kernel = m < SMALL_M_FAST_PATH && mCudaKernelEnabledm;
    bool use_cuda_kernel = false; // fast path is slower and inaccurate.
    bool use_pre_quant_scale = mQuantAlgo & PRE_QUANT_SCALE;

    half const *act_ptr = reinterpret_cast<half const *>(inputs_ptr);
    half const *biases_ptr = nullptr;
    if (mQuantAlgo & BIAS)
        biases_ptr = reinterpret_cast<half const *>(bias_ptr);

    float alpha = 1.0;
    if (mQuantAlgo & FP8_ALPHA) {
        cudaMemcpy(&alpha, const_cast<void *>(alpha_ptr), sizeof(float), cudaMemcpyDeviceToHost);
    }

    if (use_pre_quant_scale && !use_cuda_kernel) {
        // Apply pre-quant per channel scale on activations
        act_ptr = reinterpret_cast<half const *>(workspace);
        if (mType == DataType::TYPE_FP16) {
            if (mQuantAlgo & FP8_ALPHA) {
                tensorrt_llm::kernels::apply_per_channel_scale_kernel_launcher<half, __nv_fp8_e4m3>(
                    reinterpret_cast<__nv_fp8_e4m3 *>(workspace), reinterpret_cast<half const *>(inputs_ptr),
                    reinterpret_cast<half const *>(pre_quant_scales_ptr), m, k, stream);
            } else {
                tensorrt_llm::kernels::apply_per_channel_scale_kernel_launcher<half, half>(
                    reinterpret_cast<half *>(workspace), reinterpret_cast<half const *>(inputs_ptr),
                    reinterpret_cast<half const *>(pre_quant_scales_ptr), m, k, stream);
            }
        }
#if defined(ENABLE_BF16)
        else if (mType == DataType::TYPE_BF16) {
            if (mQuantAlgo & FP8_ALPHA) {
                tensorrt_llm::kernels::apply_per_channel_scale_kernel_launcher<__nv_bfloat16, __nv_fp8_e4m3>(
                    reinterpret_cast<__nv_fp8_e4m3 *>(workspace), reinterpret_cast<__nv_bfloat16 const *>(inputs_ptr),
                    reinterpret_cast<__nv_bfloat16 const *>(pre_quant_scales_ptr), m, k, stream);
            } else {
                tensorrt_llm::kernels::apply_per_channel_scale_kernel_launcher<__nv_bfloat16, __nv_bfloat16>(
                    reinterpret_cast<__nv_bfloat16 *>(workspace), reinterpret_cast<__nv_bfloat16 const *>(inputs_ptr),
                    reinterpret_cast<__nv_bfloat16 const *>(pre_quant_scales_ptr), m, k, stream);
            }
        }
#endif
    }

#if defined(ENABLE_BF16)
    TLLM_CHECK_WITH_INFO(mType == DataType::TYPE_FP16 || mType == DataType::TYPE_BF16,
                         "No valid weightOnlyGropwiseQuantMatmul configuration");
#else
    TLLM_CHECK_WITH_INFO(mType == DataType::TYPE_FP16, "No valid weightOnlyGropwiseQuantMatmul configuration");
#endif

    // Quantized weights are packed in FP16 format (INT4*4 -> FP16)
    int real_n = n * FP16_INT4_RATIO;
    if (use_cuda_kernel) {
        void const *cuda_kernel_act_ptr = act_ptr;
        void const *cuda_kernel_act_scale_ptr = use_pre_quant_scale ? pre_quant_scales_ptr : nullptr;
        void const *cuda_kernel_weight_ptr = weights_ptr;
        void const *cuda_kernel_scales_ptr = scales_ptr;
        void const *cuda_kernel_zeros_ptr = zeros_ptr;
        void const *cuda_kernel_bias_ptr = biases_ptr;
        void *cuda_kernel_out_ptr = outputs_ptr;
        tensorrt_llm::kernels::weight_only::Params params{cuda_kernel_act_ptr, cuda_kernel_act_scale_ptr,
                                                          cuda_kernel_weight_ptr, cuda_kernel_scales_ptr, cuda_kernel_zeros_ptr, cuda_kernel_bias_ptr,
                                                          cuda_kernel_out_ptr, alpha, m, real_n, k, mGroupSize, mCudaKernelType,
                                                          static_cast<bool>(mQuantAlgo & FP8_ALPHA)};
        tensorrt_llm::kernels::weight_only::kernel_launcher(mArch, params, stream);
    } else {
        int const ws_bytes = m_weightOnlyGroupwiseGemmRunner->getWorkspaceSize(m, n, k);

        int32_t *weight_ptr = const_cast<int32_t *>(reinterpret_cast<int32_t const *>(weights_ptr));

        auto const &bestTactic = mPluginProfiler->getBestConfig(m, mGemmId);
        TLLM_CHECK_WITH_INFO(bestTactic,
                             "No valid weight only groupwise GEMM tactic(It is usually caused by the failure to execute all "
                             "candidate "
                             "configurations of the CUTLASS kernel, please pay attention to the warning information when building "
                             "the "
                             "engine.)");
        // std::cout << "bestConfig: tile_config=" << int(bestTactic->tile_config) << ", split_k_style=" << int(bestTactic->split_k_style) << ", split_k_factor=" << bestTactic->split_k_factor << ", stages=" << bestTactic->stages << std::endl;
        int stages = -1;
        m_weightOnlyGroupwiseGemmRunner->gemm(act_ptr, weight_ptr, scales_ptr, zeros_ptr, biases_ptr,
                                              alpha, outputs_ptr, m, real_n, k, mGroupSize, *bestTactic,
                                              reinterpret_cast<char *>(workspace) + m * k * sizeof(half), ws_bytes, stream);
    }
    return;
}