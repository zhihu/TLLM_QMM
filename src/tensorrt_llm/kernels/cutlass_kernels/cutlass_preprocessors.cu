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
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"
#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/stringUtils.h"

#include "cutlass_extensions/gemm/kernel/mixed_gemm_B_layout.h"

#include <cuda_runtime.h>
using namespace tensorrt_llm::common;
namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

__global__ void kernel_permute_B_rows_for_mixed_gemm(
    const int num_experts,
    const int num_rows,
    const int num_vec_cols,
    const int B_ROWS_PER_MMA,
    const int ELTS_PER_REG,
    const uint32_t* input_byte_ptr,
    uint32_t* output_byte_ptr
){
    const int expert = blockIdx.z;
    const int base_row = (blockIdx.y * blockDim.y + threadIdx.y) * B_ROWS_PER_MMA;
    const int write_col = blockIdx.x * blockDim.x + threadIdx.x;
    if(expert < num_experts){
        const int64_t matrix_offset = expert * int64_t(num_rows) * int64_t(num_vec_cols);
        if(base_row < num_rows && write_col < num_vec_cols){
            for(int tile_row = 0; tile_row < B_ROWS_PER_MMA; ++tile_row){
                const int write_row = base_row + tile_row;
                const int tile_read_row
                    = 8 * (((tile_row % ELTS_PER_REG) / 2)) + tile_row % 2 + 2 * (tile_row / ELTS_PER_REG);
                const int read_row = base_row + tile_read_row;
                const int read_col = write_col;

                const int64_t read_offset = matrix_offset + int64_t(read_row) * num_vec_cols + read_col;
                const int64_t write_offset = matrix_offset + int64_t(write_row) * num_vec_cols + write_col;

                output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
            }
        }
    }
}
 
void gpu_permute_B_rows_for_mixed_gemm(int8_t* permuted_quantized_tensor, const int8_t* quantized_tensor,
    const std::vector<size_t>& shape, QuantType quant_type, const int64_t arch_version, cudaStream_t stream)
{
    // We only want to run this step for weight only quant.
    //TLLM_CHECK(quant_type == QuantType::PACKED_INT4_WEIGHT_ONLY || quant_type == QuantType::INT8_WEIGHT_ONLY);

    TLLM_CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");
    const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
    const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

    const int BITS_PER_ELT = get_weight_quant_bits(quant_type);
    const int K = 16 / BITS_PER_ELT;
    //const int ELTS_PER_BYTE = 8 / BITS_PER_ELT;
    const int ELTS_PER_REG = 32 / BITS_PER_ELT;

    const uint32_t* input_byte_ptr = reinterpret_cast<const uint32_t*>(quantized_tensor);
    uint32_t* output_byte_ptr = reinterpret_cast<uint32_t*>(permuted_quantized_tensor);

    int MMA_SHAPE_N = 8;
    int B_ROWS_PER_MMA = 8 * K;
    const int elts_in_int32 = 32 / BITS_PER_ELT;

    const int num_vec_cols = num_cols / elts_in_int32;

    TLLM_CHECK_WITH_INFO(
        arch_version >= 75, "Unsupported Arch. Pre-volta not supported. Column interleave not needed on Volta.");

    TLLM_CHECK_WITH_INFO(num_rows % B_ROWS_PER_MMA == 0,
        fmtstr("Invalid shape for quantized tensor. Number of rows of quantized matrix must be a multiple of %d",
            B_ROWS_PER_MMA));
    TLLM_CHECK_WITH_INFO(num_cols % MMA_SHAPE_N == 0,
        fmtstr("Invalid shape for quantized tensor. On turing/Ampere, the number of cols must be a multiple of %d.",
            MMA_SHAPE_N));

    dim3 blockDim(32, 32, 1);
    dim3 gridDim((num_vec_cols + 31)/32, (num_rows + 31)/32, num_experts);

    kernel_permute_B_rows_for_mixed_gemm<<<gridDim, blockDim, 0, stream>>>(
        num_experts, 
        num_rows, 
        num_vec_cols,
        B_ROWS_PER_MMA,
        ELTS_PER_REG,
        input_byte_ptr,
        output_byte_ptr
    );
}


template <int M_TILE_L1, int N_TILE_L1, int ELTS_PER_BYTE>
__global__ void kernel_subbyte_transpose_impl(
    const size_t num_experts,
    const size_t num_rows,
    const size_t num_cols,
    const size_t col_bytes,
    const size_t col_bytes_trans,
    const int VECTOR_WIDTH,
    const QuantType quant_type,
    const uint8_t *input_byte_ptr,
    uint8_t* output_byte_ptr
){
    const size_t expert = blockIdx.z;
    const size_t row_tile_start = (blockIdx.y * blockDim.y + threadIdx.y) * M_TILE_L1;
    const size_t col_tile_start_byte = (blockIdx.x * blockDim.x + threadIdx.x) * N_TILE_L1;
    uint8_t cache_buf[M_TILE_L1][N_TILE_L1];
    if(expert < num_experts){
        const size_t matrix_offset = expert * num_rows * col_bytes;
        if(row_tile_start < num_rows && col_tile_start_byte < col_bytes){
            const int row_limit = std::min(row_tile_start + M_TILE_L1, num_rows);
            const int col_limit = std::min(col_tile_start_byte + N_TILE_L1, col_bytes);
            for (int ii = 0; ii < M_TILE_L1; ++ii)
            {
                const int row = row_tile_start + ii;

                for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH)
                {
                    const int col = col_tile_start_byte + jj;

                    const size_t logical_src_offset = matrix_offset + row * col_bytes + col;

                    if (row < row_limit && col < col_limit)
                    {
                        for (int v = 0; v < VECTOR_WIDTH; ++v)
                        {
                            cache_buf[ii][jj + v] = input_byte_ptr[logical_src_offset + v];
                        }
                    }
                }
            }

            if (quant_type == QuantType::W8_A16) {
                for (int ii = 0; ii < M_TILE_L1; ++ii)
                {
                    for (int jj = ii + 1; jj < N_TILE_L1; ++jj)
                    {
                        //std::swap(cache_buf[ii][jj], cache_buf[jj][ii]);
                        auto tmp = cache_buf[ii][jj];
                        cache_buf[ii][jj] = cache_buf[jj][ii];
                        cache_buf[jj][ii] = tmp;
                    }
                }
            } else if (quant_type == QuantType::W4_A16) {
                for (int ii = 0; ii < M_TILE_L1; ++ii)
                {
                    // Using M_TILE_L1 here is deliberate since we assume that the cache tile
                    // is square in the number of elements (not necessarily the number of bytes).
                    for (int jj = ii + 1; jj < M_TILE_L1; ++jj)
                    {
                        const int ii_byte = ii / ELTS_PER_BYTE;
                        const int ii_bit_offset = ii % ELTS_PER_BYTE;

                        const int jj_byte = jj / ELTS_PER_BYTE;
                        const int jj_bit_offset = jj % ELTS_PER_BYTE;

                        uint8_t src_elt = 0xF & (cache_buf[ii][jj_byte] >> (4 * jj_bit_offset));
                        uint8_t tgt_elt = 0xF & (cache_buf[jj][ii_byte] >> (4 * ii_bit_offset));

                        cache_buf[ii][jj_byte] &= (0xF0 >> (4 * jj_bit_offset));
                        cache_buf[jj][ii_byte] &= (0xF0 >> (4 * ii_bit_offset));

                        cache_buf[ii][jj_byte] |= (tgt_elt << (4 * jj_bit_offset));
                        cache_buf[jj][ii_byte] |= (src_elt << (4 * ii_bit_offset));
                    }
                }
            } else {
                //TLLM_CHECK_WITH_INFO(false, "Unsupported quantization type.");
            }

            const size_t row_tile_start_trans = col_tile_start_byte * ELTS_PER_BYTE;
            const size_t col_tile_start_byte_trans = row_tile_start / ELTS_PER_BYTE;

            const int row_limit_trans = std::min(row_tile_start_trans + M_TILE_L1, num_cols);
            const int col_limit_trans = std::min(col_tile_start_byte_trans + N_TILE_L1, col_bytes_trans);

            for (int ii = 0; ii < M_TILE_L1; ++ii)
            {
                const int row = row_tile_start_trans + ii;
                for (int jj = 0; jj < N_TILE_L1; jj += VECTOR_WIDTH)
                {
                    const int col = col_tile_start_byte_trans + jj;

                    const size_t logical_tgt_offset = matrix_offset + row * col_bytes_trans + col;

                    if (row < row_limit_trans && col < col_limit_trans)
                    {
                        for (int v = 0; v < VECTOR_WIDTH; ++v)
                        {
                            output_byte_ptr[logical_tgt_offset + v] = cache_buf[ii][jj + v];
                        }
                    }
                }
            }
        }
    }
}

// We need to use this transpose to correctly handle packed int4 and int8 data
// The reason this code is relatively complex is that the "trivial" loops took a substantial
// amount of time to transpose leading to long preprocessing times. This seemed to be a big
// issue for relatively large models.
template <QuantType quant_type>
void gpu_subbyte_transpose_impl(
    int8_t* transposed_quantized_tensor, const int8_t* quantized_tensor, const std::vector<size_t>& shape, cudaStream_t stream)
{
    const int bits_per_elt = get_weight_quant_bits(quant_type);

    TLLM_CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");
    const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
    const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

    const size_t col_bytes = num_cols * bits_per_elt / 8;
    const size_t col_bytes_trans = num_rows * bits_per_elt / 8;

    const uint8_t* input_byte_ptr = reinterpret_cast<const uint8_t*>(quantized_tensor);
    uint8_t* output_byte_ptr = reinterpret_cast<uint8_t*>(transposed_quantized_tensor);

    static_assert(quant_type == QuantType::W8_A16 || quant_type == QuantType::W4_A16, "");
    static constexpr int ELTS_PER_BYTE = quant_type == QuantType::W8_A16 ? 1 : 2;

    static constexpr int M_TILE_L1 = 64;
    static constexpr int N_TILE_L1 = M_TILE_L1 / ELTS_PER_BYTE;

    static constexpr int VECTOR_WIDTH = std::min(32, N_TILE_L1);

    // We assume the dims are a multiple of vector width. Our kernels only handle dims which are multiples
    // of 64 for weight-only quantization. As a result, this seemed like a reasonable tradeoff because it
    // allows GCC to emit vector instructions.
    TLLM_CHECK_WITH_INFO(!(col_bytes_trans % VECTOR_WIDTH) && !(col_bytes % VECTOR_WIDTH),
        fmtstr("Number of bytes for rows and cols must be a multiple of %d. However, num_rows_bytes = %ld and "
               "num_col_bytes = %ld.",
            VECTOR_WIDTH, col_bytes_trans, col_bytes));

    dim3 blockDim(32, 32, 1);
    dim3 gridDim((col_bytes + 31)/32, (num_rows + 31)/32, num_experts);
    kernel_subbyte_transpose_impl<M_TILE_L1, N_TILE_L1, ELTS_PER_BYTE><<<gridDim, blockDim, 0, stream>>>(
        num_experts, 
        num_rows, 
        num_cols,
        col_bytes,
        col_bytes_trans,
        VECTOR_WIDTH,
        quant_type,
        input_byte_ptr,
        output_byte_ptr
    );
}

void gpu_subbyte_transpose(int8_t* transposed_quantized_tensor, const int8_t* quantized_tensor,
    const std::vector<size_t>& shape, QuantType quant_type, cudaStream_t stream)
{

    if (quant_type == QuantType::W8_A16) {
        gpu_subbyte_transpose_impl<QuantType::W8_A16>(transposed_quantized_tensor, quantized_tensor, shape, stream);
    } else if (quant_type == QuantType::W4_A16) {
        gpu_subbyte_transpose_impl<QuantType::W4_A16>(
            transposed_quantized_tensor, quantized_tensor, shape, stream);
    } else {
        TLLM_CHECK_WITH_INFO(false, "Invalid quant_tye");
    }
}

__global__ void kernel_interleave_column_major_tensor(
    const int num_experts,
    const int num_vec_rows,
    const int num_cols,
    const int interleave,
    const int vec_rows_per_tile,
    const uint32_t* input_byte_ptr,
    uint32_t* output_byte_ptr
){
    const int expert = blockIdx.z;
    const int read_col = blockIdx.y * blockDim.y + threadIdx.y;
    const int base_vec_row = (blockIdx.x * blockDim.x + threadIdx.x) * vec_rows_per_tile;
    if(expert < num_experts){
        const int64_t matrix_offset = expert * int64_t(num_vec_rows) * int64_t(num_cols);
        if(read_col < num_cols && base_vec_row < num_vec_rows){
            const int64_t write_col = read_col / interleave;
            for (int vec_read_row = base_vec_row;
                 vec_read_row < std::min(num_vec_rows, base_vec_row + vec_rows_per_tile); ++vec_read_row)
            {
                const int64_t vec_write_row = interleave * base_vec_row
                    + vec_rows_per_tile * (read_col % interleave) + vec_read_row % vec_rows_per_tile;

                const int64_t read_offset = matrix_offset + int64_t(read_col) * num_vec_rows + vec_read_row;
                const int64_t write_offset
                    = matrix_offset + int64_t(write_col) * num_vec_rows * interleave + vec_write_row;
                output_byte_ptr[write_offset] = input_byte_ptr[read_offset];
            }
        }
    }
}
    
void gpu_interleave_column_major_tensor(int8_t* interleaved_quantized_tensor, const int8_t* quantized_tensor,
    const std::vector<size_t>& shape, QuantType quant_type, LayoutDetails details, cudaStream_t stream)
{
    // We only want to run this step for weight only quant.
    TLLM_CHECK(quant_type == QuantType::W4_A16 || quant_type == QuantType::W8_A16);

    TLLM_CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");
    const size_t num_experts = shape.size() == 2 ? 1 : shape[0];
    const size_t num_rows = shape.size() == 2 ? shape[0] : shape[1];
    const size_t num_cols = shape.size() == 2 ? shape[1] : shape[2];

    const int BITS_PER_ELT = get_weight_quant_bits(quant_type);
    const int elts_in_int32 = 32 / BITS_PER_ELT;

    const int rows_per_tile = details.rows_per_column_tile;

    TLLM_CHECK_WITH_INFO(!(num_rows % elts_in_int32),
        fmtstr("The number of rows must be a multiple of %d but the number of rows is %ld.", elts_in_int32, num_rows));

    const uint32_t* input_byte_ptr = reinterpret_cast<const uint32_t*>(quantized_tensor);
    uint32_t* output_byte_ptr = reinterpret_cast<uint32_t*>(interleaved_quantized_tensor);

    TLLM_CHECK_WITH_INFO(!(num_rows % rows_per_tile),
        fmtstr("The number of rows must be a multiple of %d but the number of rows is %ld.", rows_per_tile, num_rows));

    const int num_vec_rows = num_rows / elts_in_int32;
    const int vec_rows_per_tile = rows_per_tile / elts_in_int32;
    const int interleave = details.columns_interleaved;

    dim3 blockDim(32, 32, 1);
    dim3 gridDim((num_vec_rows + 31)/32, (num_cols + 31)/32, num_experts);
    kernel_interleave_column_major_tensor<<<gridDim, blockDim, 0, stream>>>(
        num_experts,
        num_vec_rows,
        num_cols,
        interleave,
        vec_rows_per_tile,
        input_byte_ptr,
        output_byte_ptr
    );
}


__global__ void kernel_add_bias_and_interleave_int4s_inplace_step1(
    const int num_bytes,
    int8_t *packed_int4_tensor
){
    const size_t ii = threadIdx.x + blockIdx.x * blockDim.x;
    if(ii < num_bytes){
        int8_t transformed_packed_int4s = 0;
        int8_t transformed_first_elt
            = (int8_t(packed_int4_tensor[ii] << 4) >> 4) + 8; // The double shift here is to ensure sign extension
        int8_t transformed_second_elt = (packed_int4_tensor[ii] >> 4) + 8;

        //TLLM_CHECK_WITH_INFO(
        //    transformed_first_elt >= 0 && transformed_first_elt <= 15, "Illegal result for int4 transform (first elt)");
        //TLLM_CHECK_WITH_INFO(transformed_second_elt >= 0 && transformed_second_elt <= 15,
        //    "Illegal result for int4 transform (second elt)");

        // We don't need to mask in these ops since everything should be in the range 0-15
        transformed_packed_int4s |= transformed_first_elt;
        transformed_packed_int4s |= (transformed_second_elt << 4);
        packed_int4_tensor[ii] = transformed_packed_int4s;
    }
}

__global__ void kernel_add_bias_and_interleave_int4s_inplace_step2(
    const int num_registers,
    uint32_t*register_ptr
){
    const size_t ii = threadIdx.x + blockIdx.x * blockDim.x;
    if(ii < num_registers)
    {
        const uint32_t current_register = register_ptr[ii];
        uint32_t transformed_register = 0;

        for (int dest_idx = 0; dest_idx < 8; ++dest_idx)
        {
            const int src_idx = dest_idx < 4 ? 2 * dest_idx : 2 * (dest_idx - 4) + 1;
            const int src_shift = 4 * src_idx;
            const int dest_shift = 4 * dest_idx;

            const uint32_t src_bits = (current_register >> src_shift) & 0xF;
            transformed_register |= (src_bits << dest_shift);
        }
        register_ptr[ii] = transformed_register;
    }
}

void gpu_add_bias_and_interleave_int4s_inplace(int8_t* packed_int4_tensor, const size_t num_elts, cudaStream_t stream)
{
    const int num_bytes = num_elts / 2;

    // Step 1 will be to transform all the int4s to unsigned in order to make the dequantize take as little
    // instructions as possible in the CUDA code.
    dim3 blockDim(512, 1, 1);
    dim3 gridDim((num_bytes + 511) / 512, 1, 1);
    kernel_add_bias_and_interleave_int4s_inplace_step1<<<gridDim, blockDim, 0, stream>>>(
        num_bytes, packed_int4_tensor);

    // Step 2 will transform the layout of a 32-bit register in CUDA in order to minimize the number of shift & logical
    // instructions That are needed to extract the int4s in the GEMM main loop. Pictorially, the loop below will do the
    // following: Take as input a 32 bit register with layout: bit 32 0
    //      [elt_7  elt_6  elt_5  elt_4  elt_3  elt_2  elt_1  elt_0] (each elt occupies 4 bits)
    //
    // And it will rearrange the output 32 bit register to be the following:
    // bit 32                                                      0
    //      [elt_7  elt_5  elt_3  elt_1  elt_6  elt_4  elt_2  elt_0] (each elt occupies 4 bits)

    TLLM_CHECK_WITH_INFO(num_bytes % 4 == 0, "Dimensions of int4 tensor must be a multiple of 8 for register relayout");
    const size_t num_registers = num_bytes / 4;
    uint32_t* register_ptr = reinterpret_cast<uint32_t*>(packed_int4_tensor);
    {
        dim3 blockDim(512, 1, 1);
        dim3 gridDim((num_registers + 511) / 512, 1, 1);
        kernel_add_bias_and_interleave_int4s_inplace_step2<<<gridDim, blockDim, 0, stream>>>(
            num_registers, register_ptr);
    }
}

void gpu_add_bias_and_interleave_quantized_tensor_inplace(int8_t* tensor, const size_t num_elts, QuantType quant_type, cudaStream_t stream)
{
    if (quant_type == QuantType::W8_A16) {
        TLLM_CHECK_WITH_INFO(false, "doesn't support gpu preprocess yet.");
    } else if (quant_type == QuantType::W4_A16) {
        gpu_add_bias_and_interleave_int4s_inplace(tensor, num_elts, stream);
    } else {
        TLLM_CHECK_WITH_INFO(false, "Invalid quantization type for interleaving.");
    }
}


void gpu_preprocess_weights_for_mixed_gemm(int8_t* preprocessed_quantized_weight, int8_t* buffer, const int8_t* row_major_quantized_weight,
    const std::vector<size_t>& shape, QuantType quant_type, cudaStream_t stream)
{
    int arch = getSMVersion();
    LayoutDetails details = getLayoutDetailsForTransform(quant_type, arch);

    TLLM_CHECK_WITH_INFO(shape.size() == 2 || shape.size() == 3, "Shape must be 2-D or 3-D");

    size_t num_elts = 1;
    for (const auto& dim : shape)
    {
        num_elts *= dim;
    }

    const size_t num_bytes = num_elts * get_weight_quant_bits(quant_type) / 8;

    int8_t *src_buf = preprocessed_quantized_weight;
    int8_t *dst_buf = buffer;
    cudaMemcpyAsync(src_buf, row_major_quantized_weight, num_bytes, cudaMemcpyDeviceToDevice, stream);

    // Works on row major data, so issue this permutation first.
    if (details.uses_imma_ldsm)
    {
        const int arch = getSMVersion();
        gpu_permute_B_rows_for_mixed_gemm(dst_buf, src_buf, shape, quant_type, arch, stream);
        std::swap(src_buf, dst_buf);
    }

    if (details.layoutB == LayoutDetails::Layout::COLUMN_MAJOR)
    {
        gpu_subbyte_transpose(dst_buf, src_buf, shape, quant_type, stream);
        std::swap(src_buf, dst_buf);
    }

    if (details.columns_interleaved > 1)
    {
        gpu_interleave_column_major_tensor(dst_buf, src_buf, shape, quant_type, details, stream);
        std::swap(src_buf, dst_buf);
    }

    gpu_add_bias_and_interleave_quantized_tensor_inplace(src_buf, num_elts, quant_type, stream);
    if(src_buf != preprocessed_quantized_weight){
        cudaMemcpyAsync(preprocessed_quantized_weight, src_buf, num_bytes, cudaMemcpyDeviceToDevice, stream);
    }
}

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm
