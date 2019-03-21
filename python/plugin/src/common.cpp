#include "common.h"

// Helpers to move data to/from the GPU.
nvinfer1::Weights copyToDevice(const void* hostData, int count)
{
        void* deviceData;
        CHECK(cudaMalloc(&deviceData, count * sizeof(float)));
        CHECK(cudaMemcpy(deviceData, hostData, count * sizeof(float), cudaMemcpyHostToDevice));
        return nvinfer1::Weights{nvinfer1::DataType::kFLOAT, deviceData, count};
}

int copyFromDevice(char* hostBuffer, nvinfer1::Weights deviceWeights)
{
        *reinterpret_cast<int*>(hostBuffer) = deviceWeights.count;
        CHECK(cudaMemcpy(hostBuffer + sizeof(int), deviceWeights.values, deviceWeights.count * sizeof(float), cudaMemcpyDeviceToHost));
        return sizeof(int) + deviceWeights.count * sizeof(float);
}
