/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/plugins/common/gemmPluginProfiler.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/kernels/cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "tensorrt_llm/kernels/cutlass_kernels/int8_gemm/int8_gemm.h"

namespace tensorrt_llm::plugins {

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::GemmPluginProfiler() {
    mMNKProfileMap = std::make_shared<MNKProfileMap>();

    // set SKIP_GEMM_PLUGIN_PROFILINGS=1 to avoid tactics profilings
    const auto skipEnv = std::getenv("SKIP_GEMM_PLUGIN_PROFILINGS");
    mSkip = (skipEnv != NULL && std::stoi(skipEnv));
    if (mSkip) {
        TLLM_LOG_DEBUG(
            "SKIP_GEMM_PLUGIN_PROFILINGS is set. Skipping GEMM plugin profilings. It could result in runtime error "
            "if default tactic is not defined.");
    }
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTactics(
    const RunnerPtr &runner, const DataType &type, const GemmDims &dims, const GemmIdType &gemmId) {
    writer_lock lock(mMNKProfileMap->mutex);

    if (!dims.isInitialized()) {
        return;
    }

    mRunner = runner;
    mType = type;

    const int maxM = std::min(nextPowerOfTwo(dims.maxM), MAX_PROFILE_M);
    computeTmpSize(maxM, dims.n, dims.k);

    if (!mMNKProfileMap->existsMProfileMap(gemmId)) {
        // Create map for GEMM ID
        mMNKProfileMap->createMProfileMap(gemmId);
    }

    if (mSkip) {
        return;
    }

    auto mProfileMap = mMNKProfileMap->getMProfileMap(gemmId);

    auto profileTactics = [&mProfileMap, this](int m, int n, int k) {
        if (mProfileMap->count(m) == 0) {
            const auto tactics = this->getTactics(m, n, k);
            // Profile different tactics for particular m and insert best config to the map
            mProfileMap->insert({m, this->profileTacticsForProblem(m, n, k, tactics)});
        }
    };

    // Allocate tmp data to run GEMMs
    allocateTmpData();
    const int startMinMRounded = nextPowerOfTwo(dims.minM);
    for (int m = startMinMRounded; m < maxM; m *= 2) {
        profileTactics(m, dims.n, dims.k);
    }

    profileTactics(maxM, dims.n, dims.k);
    // Free tmp data
    freeTmpData();
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
std::optional<Config> GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::getBestConfig(
    int m, const GemmIdType &gemmId) const {
    reader_lock lock(mMNKProfileMap->mutex);

    if (mSkip) {
        // return std::nullopt;
        return Config(cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64,
                      cutlass_extensions::SplitKStyle::NO_SPLIT_K,
                      1,
                      3);
    }

    const int mRounded = std::min(nextPowerOfTwo(m), MAX_PROFILE_M);
    auto mp = mMNKProfileMap->getMProfileMap(gemmId);

    return mMNKProfileMap->getMProfileMap(gemmId)->at(mRounded);
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::allocateTmpData() {
    TLLM_CHECK_WITH_INFO(mTmpWorkspaceSizeInBytes > 0, "tmpWorkspaceSizeInBytes must be larger than 0");
    const auto status = cudaMalloc(&mWorkspaceTmp, mTmpWorkspaceSizeInBytes);
    TLLM_CHECK_WITH_INFO(status == cudaSuccess, cudaGetErrorString(status));
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::freeTmpData() {
    const auto status = cudaFree(mWorkspaceTmp);
    TLLM_CHECK_WITH_INFO(status == cudaSuccess, "Can't free tmp workspace for GEMM tactics profiling.");
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
std::optional<Config> GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTacticsForProblem(
    int m, int n, int k, const std::vector<Config> &tactics) {
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

    float bestTime = std::numeric_limits<float>::max();
    Config bestConfig;
    bool foundOne = false;

    // Iterate over all tactics for given M, N and K
    for (int ii = 0; ii < tactics.size(); ++ii) {
        const Config &candidateConfig = tactics[ii];
        float time = std::numeric_limits<float>::max();
        try {
            if (!checkTactic(m, n, k, candidateConfig)) {
                continue;
            }
            // Profile particualar tactic for given M, N and K
            time = profileTacticForProblem(m, n, k, candidateConfig);
            foundOne = true;
        } catch (const std::exception &e) {
            std::ostringstream msg;
            msg << "Cannot profile configuration " << ii << " (for"
                << " m=" << m << ", n=" << n << ", k=" << k << "). Skipped";
            TLLM_LOG_TRACE(msg.str());
            TLLM_LOG_TRACE(e.what());
            cudaGetLastError(); // Reset the last cudaError to cudaSuccess.
            continue;
        }

        // Choose the fastest tactic
        if (time < bestTime) {
            bestConfig = candidateConfig;
            bestTime = time;
        }
    }

    if (!foundOne) {
        std::ostringstream msg;
        msg << "Have not found any valid GEMM config for shape ("
            << "m=" << m << ", n=" << n << ", k=" << k << "). Will try to use default or fail at runtime";
        TLLM_LOG_WARNING(msg.str());
        return std::nullopt;
    }
    return {bestConfig};
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
float GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTacticForProblem(
    int m, int n, int k, const Config &tactic) {
    constexpr int warmup = 5;
    constexpr int runs = 10;

    cudaStream_t stream = cudaStreamDefault;
    // Warmup the execution
    for (int i = 0; i < warmup; ++i) {
        runTactic(m, n, k, tactic, mWorkspaceTmp, stream);
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);

    // Profile GEMM
    for (int i = 0; i < runs; ++i) {
        runTactic(m, n, k, tactic, mWorkspaceTmp, stream);
    }

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed / runs;
}

template class GemmPluginProfiler<tensorrt_llm::cutlass_extensions::CutlassGemmConfig,
                                  std::shared_ptr<tensorrt_llm::kernels::cutlass_kernels::CutlassInt8GemmRunnerInterface>, GemmIdCore,
                                  GemmIdCoreHash>;

template class GemmPluginProfiler<tensorrt_llm::cutlass_extensions::CutlassGemmConfig,
                                  std::shared_ptr<tensorrt_llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface>, GemmIdCore,
                                  GemmIdCoreHash>;

} // namespace tensorrt_llm::plugins
