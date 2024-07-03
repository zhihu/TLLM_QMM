cmake_minimum_required(VERSION 3.18)
project(weight_only_kernels)

enable_language(C)
enable_language(CXX)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89")
set(TORCH_CUDA_ARCH_LIST "8.0;8.6;8.9")
enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_BUILD_RPATH $ORIGIN)
set(CMAKE_INSTALL_RPATH $ORIGIN)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

if (NOT CMAKE_BUILD_TYPE MATCHES "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif()

add_definitions(-DENABLE_BF16)
add_definitions(-DENABLE_FP8)

#if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL "11.8")
#  add_definitions("-DENABLE_FP8")
#  message(
#    STATUS
#      "CUDAToolkit_VERSION ${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR} is greater or equal than 11.8, enable -DENABLE_FP8 flag"
#  )
#endif()

list(APPEND CMAKE_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
#execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))" OUTPUT_VARIABLE PYTORCH_CXX_ABI OUTPUT_STRIP_TRAILING_WHITESPACE)
#list(APPEND CMAKE_CXX_FLAGS " -D_GLIBCXX_USE_CXX11_ABI=${PYTORCH_CXX_ABI}")
#list(APPEND CMAKE_CUDA_FLAGS " -D_GLIBCXX_USE_CXX11_ABI=${PYTORCH_CXX_ABI}")
#message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")

find_package(Python COMPONENTS Interpreter Development)
# find_program(PYTHON_EXECUTABLE python NO_PACKAGE_ROOT_PATH NO_CMAKE_PATH)
message("PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE}")
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())" OUTPUT_VARIABLE PYBIND11_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_PREFIX_PATH}")
list(APPEND CMAKE_PREFIX_PATH "${PYTORCH_CMAKE_PREFIX_PATH}/Torch")
message("PYBIND11_CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_PREFIX_PATH}")
message("PYTORCH_CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH}")
message("CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")
find_package(pybind11 CONFIG)
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

message (STATUS "CMAKE_INSTALL_RPATH: ${CMAKE_INSTALL_RPATH}")


include(cutlass)
include(tensorrt_llm)

file(GLOB_RECURSE SOURCES
    "src/lib.cpp"
    "src/tensorrt_llm/*.cpp"
    "src/plugins/*.cpp"
    "src/plugins/*.cu"
    "3rd/TensorRT-LLM/cpp/tensorrt_llm/common/*.cpp"
    "3rd/TensorRT-LLM/cpp/tensorrt_llm/common/*.cu"
    "3rd/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/*.cpp"
    "3rd/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/*.cu"
    "3rd/TensorRT-LLM/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/*.cpp"
    "3rd/TensorRT-LLM/cpp/tensorrt_llm/kernels/weightOnlyBatchedGemv/*.cu"
    "3rd/TensorRT-LLM/cpp/tensorrt_llm/kernels/preQuantScaleKernel.cu"
    "3rd/TensorRT-LLM/cpp/tensorrt_llm/cutlass_extensions/*.cpp"
    "3rd/TensorRT-LLM/cpp/tensorrt_llm/cutlass_extensions/*.cu"
)

list(FILTER SOURCES EXCLUDE REGEX "thop/.*")
list(FILTER SOURCES EXCLUDE REGEX "common/cudaAllocator.cpp")
list(FILTER SOURCES EXCLUDE REGEX "common/mpiUtils.cpp")
list(FILTER SOURCES EXCLUDE REGEX "kernels/customAllReduceKernels.cu")
list(FILTER SOURCES EXCLUDE REGEX "kernels/cutlass_kernels/moe_gemm/*.cpp")
list(FILTER SOURCES EXCLUDE REGEX "kernels/cutlass_kernels/moe_gemm/*.cu")
#list(FILTER SOURCES EXCLUDE REGEX "3rd/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp")
#list(FILTER SOURCES EXCLUDE REGEX "src/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp")

add_library(${PROJECT_NAME} STATIC ${SOURCES})

set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(${PROJECT_NAME}
    #PUBLIC "./"
    PUBLIC "src" # override tensorrt_llm modified.
    PUBLIC "src/tensorrt_llm/cutlass_extensions/include/"
    PUBLIC "3rd/cutlass/include"
    PUBLIC "3rd/TensorRT-LLM/cpp"
    PUBLIC "3rd/TensorRT-LLM/cpp/include"
    PUBLIC "3rd/TensorRT-LLM/cpp/tensorrt_llm/cutlass_extensions/include/"
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

file(GLOB_RECURSE TEST_SRC
    "tests/*.cu"
)

add_executable(test ${TEST_SRC})
target_link_libraries(test
    ${PROJECT_NAME}
)

file(GLOB_RECURSE PYBINDING_SRC_NN
    "src/py_binding.cpp"
    "tensorrt_llm/thop/*.cpp"
)

pybind11_add_module(py_binding ${PYBINDING_SRC_NN})
target_compile_definitions(py_binding
    PRIVATE TORCH_EXTENSION_NAME=py_binding
    PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})

target_include_directories(py_binding PRIVATE "tensorrt_llm")

target_link_libraries(py_binding PRIVATE
    "weight_only_kernels"
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions"
    "pthread"
    "${TORCH_LIBRARIES}"
    "${TORCH_PYTHON_LIBRARY}"
)


