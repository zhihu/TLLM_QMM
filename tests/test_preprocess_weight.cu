#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{
namespace cutlass_kernels
{

void test_permute_B_rows_for_mixed_gemm(){
    std::vector<size_t> shape = {4096, 8192};

    int N = 1;
    for(int i = 0; i < shape.size(); i++){
        N *= shape[i];
    }

    std::vector<int8_t> weight(N);
    for(int i = 0; i < N; i++){
        weight[i] = i % 128;
    }

    shape[1] *= 2;

    std::vector<int8_t> preprocess_weight(N);
    const int arch = getSMVersion();
    auto quant_type = QuantType::W4_A16;
    permute_B_rows_for_mixed_gemm(preprocess_weight.data(), weight.data(), shape, quant_type, arch);

    {
        //call gpu
        int8_t* d_weight, *d_preprocess_weight;
        cudaMalloc((void**)&d_weight, N);
        cudaMalloc((void**)&d_preprocess_weight, N);
        cudaMemcpy(d_weight, weight.data(), N, cudaMemcpyHostToDevice);
        gpu_permute_B_rows_for_mixed_gemm(d_preprocess_weight, d_weight, shape, quant_type, arch, 0);
        std::vector<int8_t> out(N);
        cudaMemcpy(out.data(), d_preprocess_weight, N, cudaMemcpyDeviceToHost);
        for(int i = 0; i < N; i++){
            TLLM_CHECK_WITH_INFO(out[i] == preprocess_weight[i], "check failed");
        }
        std::cout << "test_permute_B_rows_for_mixed_gemm success\n";
    }
}

void test_subbyte_transpose(){
    std::vector<size_t> shape = {4096, 8192};

    int N = 1;
    for(int i = 0; i < shape.size(); i++){
        N *= shape[i];
    }

    std::vector<int8_t> weight(N);
    for(int i = 0; i < N; i++){
        weight[i] = i % 128;
    }
    shape[1] *= 2;

    std::vector<int8_t> preprocess_weight(N);
    auto quant_type = QuantType::W4_A16;
    subbyte_transpose(preprocess_weight.data(), weight.data(), shape, quant_type);

    {
        //call gpu
        int8_t* d_weight, *d_preprocess_weight;
        cudaMalloc((void**)&d_weight, N);
        cudaMalloc((void**)&d_preprocess_weight, N);
        cudaMemcpy(d_weight, weight.data(), N, cudaMemcpyHostToDevice);
        gpu_subbyte_transpose(d_preprocess_weight, d_weight, shape, quant_type, 0);
        std::vector<int8_t> out(N);
        cudaMemcpy(out.data(), d_preprocess_weight, N, cudaMemcpyDeviceToHost);
        for(int i = 0; i < N; i++){
            TLLM_CHECK_WITH_INFO(out[i] == preprocess_weight[i], "check failed");
        }
        std::cout << "test_subbyte_transpose success\n";
    }
}

/*
void test_interleave_column_major_tensor(){
    std::vector<size_t> shape = {4096, 8192};

    int N = 1;
    for(int i = 0; i < shape.size(); i++){
        N *= shape[i];
    }

    std::vector<int8_t> weight(N);
    for(int i = 0; i < N; i++){
        weight[i] = i % 128;
    }
    shape[1] *= 2;

    std::vector<int8_t> preprocess_weight(N);
    auto quant_type = QuantType::PACKED_INT4_WEIGHT_ONLY;
    LayoutDetails details = getLayoutDetailsForTransform(quant_type);
    interleave_column_major_tensor(preprocess_weight.data(), weight.data(), shape, quant_type, details);

    {
        //call gpu
        int8_t* d_weight, *d_preprocess_weight;
        cudaMalloc((void**)&d_weight, N);
        cudaMalloc((void**)&d_preprocess_weight, N);
        cudaMemcpy(d_weight, weight.data(), N, cudaMemcpyHostToDevice);
        gpu_interleave_column_major_tensor(d_preprocess_weight, d_weight, shape, quant_type, details, 0);
        std::vector<int8_t> out(N);
        cudaMemcpy(out.data(), d_preprocess_weight, N, cudaMemcpyDeviceToHost);
        for(int i = 0; i < N; i++){
            TLLM_CHECK_WITH_INFO(out[i] == preprocess_weight[i], "check failed");
        }
        std::cout << "test_interleave_column_major_tensor success\n";
    }
}
*/

void test_add_bias_and_interleave_quantized_tensor_inplace(){
    std::vector<size_t> shape = {4096, 8192};

    int N = 1;
    for(int i = 0; i < shape.size(); i++){
        N *= shape[i];
    }

    std::vector<int8_t> weight(N);
    for(int i = 0; i < N; i++){
        weight[i] = i % 128;
    }
    shape[1] *= 2;
    std::vector<int8_t> weight_bak(weight);

    int arch = getSMVersion();
    auto quant_type = QuantType::W4_A16;
    LayoutDetails details = getLayoutDetailsForTransform(quant_type, arch);
    add_bias_and_interleave_quantized_tensor_inplace(weight.data(), N/2, quant_type);

    {
        //call gpu
        int8_t* d_weight;
        cudaMalloc((void**)&d_weight, N);
        cudaMemcpy(d_weight, weight_bak.data(), N, cudaMemcpyHostToDevice);
        gpu_add_bias_and_interleave_quantized_tensor_inplace(d_weight, N/2, quant_type, 0);
        std::vector<int8_t> out(N);
        cudaMemcpy(out.data(), d_weight, N, cudaMemcpyDeviceToHost);
        for(int i = 0; i < N; i++){
            TLLM_CHECK_WITH_INFO(out[i] == weight[i], "check failed");
        }
        std::cout << "test_add_bias_and_interleave_quantized_tensor_inplace success\n";
    }
}

void test_preprocess_weights_for_mixed_gemm(){
    std::vector<size_t> shape = {4096, 4096};

    int N = 1;
    for(int i = 0; i < shape.size(); i++){
        N *= shape[i];
    }

    std::vector<int8_t> weight(N);
    for(int i = 0; i < N; i++){
        weight[i] = i % 128;
    }

    shape[1] *= 2; //int4

    std::vector<int8_t> preprocess_weight(N);
    auto quant_type = QuantType::W4_A16;
    preprocess_weights_for_mixed_gemm(preprocess_weight.data(), weight.data(), shape, quant_type);

    {
        //call gpu
        int8_t* d_weight, *d_preprocess_weight, *d_buffer;
        cudaMalloc((void**)&d_weight, N);
        cudaMalloc((void**)&d_preprocess_weight, N);
        cudaMalloc((void**)&d_buffer, N);
        cudaMemcpy(d_weight, weight.data(), N, cudaMemcpyHostToDevice);
        gpu_preprocess_weights_for_mixed_gemm(d_preprocess_weight, d_buffer, d_weight, shape, quant_type, 0);
        std::vector<int8_t> out(N);
        cudaMemcpy(out.data(), d_preprocess_weight, N, cudaMemcpyDeviceToHost);
        for(int i = 0; i < N; i++){
            TLLM_CHECK_WITH_INFO(out[i] == preprocess_weight[i], "check failed");
        }
        std::cout << "test_preprocess_weights_for_mixed_gemm success\n";
    }
}

} // namespace cutlass_kernels
} // namespace kernels
} // namespace tensorrt_llm

int main(){
    using namespace tensorrt_llm::kernels::cutlass_kernels;
    test_permute_B_rows_for_mixed_gemm();
    test_subbyte_transpose();
    // test_interleave_column_major_tensor();
    test_add_bias_and_interleave_quantized_tensor_inplace();
    test_preprocess_weights_for_mixed_gemm();
    return 0;
}

