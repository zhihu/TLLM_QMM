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
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace tensorrt_llm {
namespace lib {

void w8a16_initialize(const int weight_type_id, const int minM, const int maxM, const int maxK, const int maxN, const bool is_bf16);

void w4a16_initialize(const int group_size, const int minM, const int maxM, const int maxK, const int maxN, const bool is_bf16);

void w8a8_initialize(const int minM, const int maxM, const int maxK, const int maxN, const bool is_bf16);

void w8a16_matmul(const void *input, const void *weight, const void *scale, void *output,
                  const int m, const int n, const int k, const int weight_type_id, void *workspace, const bool is_bf16, cudaStream_t stream);

/*
 * brief: call a groupwise int4 matmul
 * input: the input tensor with shape (m, k), and dtype is float16
 * weight: the weight tensor with shape (k, n/2), and dtype is int4
 * scale: the scale of weight tensor with shape (m, k/group_size), and dtype is float16
 * zero: the zero of weight tensor with shape (m, k/group_size), and dtype is float16
 * m: the rows of input tensor
 * n: the cols of output tensor
 * k: the cols of input tensor
 * workspace: the workspace tensor with size=ws_bytes
 * ws_bytes: the size of workspace tensor
 * group_size: currently, only support 128
 * is_bf16: whether to use bfloat16
 * stream: a cuda stream
 */
void w4a16_matmul(const void *input, const void *pre_quant_scale, const void *weight, const void *scale, const void *zero, const void *alpha, void *output,
                  const int m, const int n, const int k, void *workspace, const size_t ws_bytes, const int group_size, const bool is_bf16, cudaStream_t stream);

/*
 * brief: call (int8 * int8) matmul
 * input: the input tensor with shape (m, k), and dtype is int8_t
 * weight: the weight tensor with shape (k, n), and dtype is int8_t
 * scale_x: the scale of input tensor with shape (m, ), and dtype is float
 * scale_weight: the scale of weight tensor with shape (, n), and dtype is float
 * m: the rows of input tensor
 * n: the cols of output tensor
 * k: the cols of input tensor
 * workspace: the workspace tensor with size=ws_bytes
 * ws_bytes: the size of workspace tensor
 * is_bf16: whether to use bfloat16
 * stream: a cuda stream
 */
void w8a8_matmul(const void *input, const void *weight, const void *scale_x, const void *scale_weight, void *output,
                 const int m, const int n, const int k, void *workspace, const size_t ws_bytes, const bool is_bf16, cudaStream_t stream);

size_t w4a16_get_workspace_size(const int group_size, const bool is_bf16);

size_t w8a8_get_workspace_size(const bool is_bf16);

/*
 * brief: preprocess int4 weight on cpu
 * out: the out tensor with shape (dim_in, dim_out/2)
 * buffer : the buffer tensor with shape (dim_in, dim_out/2)
 * in : the input tensor with shape (dim_in, dim_out/2)
 * shape: the int4 weight shape (dim_in, dim_out)
 * stream: a cuda stream
 */
void preprocess_int4_weight(int8_t *out, const int8_t *in, const std::vector<size_t> &shape);

/*
 * brief: preprocess int4 weight on gpu
 * out: the out tensor with shape (dim_in, dim_out/2)
 * buffer : the buffer tensor with shape (dim_in, dim_out/2)
 * in : the input tensor with shape (dim_in, dim_out/2)
 * shape: the int4 weight shape (dim_in, dim_out)
 * stream: a cuda stream
 */
void gpu_preprocess_int4_weight(int8_t *out, int8_t *buffer, const int8_t *in, const std::vector<size_t> &shape, cudaStream_t stream);

} // namespace lib
} // namespace tensorrt_llm


