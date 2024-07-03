
<p style="text-align: center;">TLLM_QMM: TensorRT-LLM Quantized MatMul Operators for Pytorch</p>

# Introduction

TLLM_QMM strips the implementation of quantized kernels of Nvidia's TensorRT-LLM, removing NVInfer dependency and exposes ease of use Pytorch module. We modified the dequantation and weight preprocessing to align with popular quantization alogirthms such as AWQ and GPTQ, and combine them with new FP8 quantization.

TensorRT-LLM and former FasterTransformer leverages reusable code base and mupltiple optimzation techniques under various generation of GPUs. We belive this can benifit the development of new quantization methods compared to diversed implementations of multiple open-sourced kernels.

# Installation

## Prerequisities

* Ada Loverance(SM89), Hopper(currently disabled) for FP8.
* CUDA>=12.4(for fp8 on Ada Loverance)
* Python>=3.8
* cmake>=3.18

## Build

1. python setup.py build

## Install

1. pip install .

# Integration

## vLLM

TODO

# Supported Quantization Alogrithms

   * AWQ
   * AWQ+FP8(dynamic act scales)
   * GPTQ
   * GPTQ+FP8(dynamic act scales)

The major differences betweeen W4A16 method of TensorRT-LLM and AWQ/GPTQ are:
   1. TensorRT-LLM uses `int4_weight * scale + zero` with float type zero points for nvcc's fmad optimization. While AWQ/GPTQ uses `(int4_weight - zero) * scale` with integer zero points. So we do a prescaling with qzeros in the preprocess stage;
   2. TensorRT-LLM dequantizes int4 weight into -8.0 to 7.0 float points, while AWQ uses 0 to 15.0. We have modified the kernel to aligin with AWQ and GPTQ;
   3. TensorRT-LLM uses a column-major interleaved memory layout of weight tensor, so we unpack and reorder AWQ/GPTQ's weight before preprocessing.

# Performance

TODO

# Versioning

TLLM_QMM's major and minor versions are aligned with TensorRT-LLM, while the patch version would be used for bug fixes.
