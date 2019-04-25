#ifndef SHADOW_FINETUNELOC_HPP
#define SHADOW_FINETUNELOC_HPP

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "NvInfer.h"
using namespace std;

void FinetuneLoc(int batchSize, int numPriorboxes, const float* arm_loc,
                 const float* priorbox_loc, float* loc);

#if NV_TENSORRT_MAJOR >= 5

class FinetuneLocPlugin : public nvinfer1::IPluginV2 {
 public:
  FinetuneLocPlugin(const std::string name) : mLayerName(name) {}

  FinetuneLocPlugin(const std::string name, const void* data, size_t length)
      : mLayerName(name) {
    assert(length == sizeof(int));
    const int* d = reinterpret_cast<const int*>(data);
    numPriorboxes = d[0];
  }

  const char* getPluginType() const override { return "finetuneloc"; }

  const char* getPluginVersion() const override { return "v1"; }

  int getNbOutputs() const override { return 1; }

  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                     int nbInputDims) override {
    return nvinfer1::DimsCHW{static_cast<int>(inputs[0].d[0]),
                             static_cast<int>(inputs[0].d[1] * 2),
                             static_cast<int>(inputs[0].d[2])};
  }

  bool supportsFormat(nvinfer1::DataType type,
                      nvinfer1::PluginFormat format) const override {
    return type == nvinfer1::DataType::kFLOAT &&
           format == nvinfer1::PluginFormat::kNCHW;
  }

  void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs,
                           const nvinfer1::Dims* outputDims, int nbOutputs,
                           nvinfer1::DataType type,
                           nvinfer1::PluginFormat format,
                           int maxBatchSize) override {
    assert(nbInputs == 2);
    assert(nbOutputs == 1);
    numPriorboxes = inputDims[0].d[0];
  }

  int initialize() override { return 0; }

  void terminate() override {}

  size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

  int enqueue(int batchSize, const void* const* inputs, void** outputs,
              void* workspace, cudaStream_t stream) override {
    const float* arm_loc = reinterpret_cast<const float*>(inputs[0]);
    const float* priorbox_loc = reinterpret_cast<const float*>(inputs[1]);
    float* decodePriorbox = reinterpret_cast<float*>(outputs[0]);
    FinetuneLoc(batchSize, numPriorboxes, arm_loc, priorbox_loc,
                decodePriorbox);
    return 0;
  }

  size_t getSerializationSize() const override { return sizeof(int); }

  void serialize(void* buffer) const override {
    int* d = reinterpret_cast<int*>(buffer);
    d[0] = numPriorboxes;
  }

  void destroy() override { delete this; }

  IPluginV2* clone() const override {
    return new FinetuneLocPlugin(mLayerName);
  }

  void setPluginNamespace(const char* pluginNamespace) override {
    mNamespace = pluginNamespace;
  }

  const char* getPluginNamespace() const override { return mNamespace.c_str(); }

 private:
  int numPriorboxes;
  const std::string mLayerName;
  std::string mNamespace;
};

class FinetuneLocPluginCreator : public nvinfer1::IPluginCreator {
 public:
  FinetuneLocPluginCreator() {
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
  }

  const char* getPluginName() const override { return "finetuneloc"; }

  const char* getPluginVersion() const override { return "v1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() override {
    return &mFC;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override {
    return new FinetuneLocPlugin(name);
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serialData,
                                         size_t serialLength) override {
    return new FinetuneLocPlugin(name, serialData, serialLength);
  }

  void setPluginNamespace(const char* pluginNamespace) override {
    mNamespace = pluginNamespace;
  }

  const char* getPluginNamespace() const override { return mNamespace.c_str(); }

 private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};

#endif

#endif  // SHADOW_FINETUNELOC_HPP
