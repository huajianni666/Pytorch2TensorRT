#ifndef _COMMON_H_
#define _COMMON_H_
#pragma once
#include <cassert>
#include <cstring>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <stdexcept>
#include "NvInfer.h"
#include "NvCaffeParser.h"

#define CHECK(status) { if (status != 0) throw std::runtime_error(__FILE__ +  __LINE__ + std::string{"CUDA Error: "} + std::to_string(status)); }

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

#define CUDA_CHECK(condition)                                  \
  do {                                                         \
    cudaError_t error = condition;                             \
    CHECK_EQ(error, cudaSuccess) << cudaGetErrorString(error); \
  } while (0)

// CUDA: use 16*16<512 threads per block
const int NumThreads = 512;

// CUDA: number of blocks for threads.
inline int GetBlocks(const int N)
{
    return (N + NumThreads - 1) / NumThreads;
}

// Helpers to move data to/from the GPU.
nvinfer1::Weights copyToDevice(const void* hostData, int count);

int copyFromDevice(char* hostBuffer, nvinfer1::Weights deviceWeights);

#endif
