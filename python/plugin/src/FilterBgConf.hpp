#ifndef SHADOW_FILTERBGCONF_HPP
#define SHADOW_FILTERBGCONF_HPP

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

void FilterBgConf(int batchSize, int numPriorboxes, int numClasses,
                  float objectness_score, const float* arm_conf,
                  const float* odm_conf, float* conf);

#if NV_TENSORRT_MAJOR >= 5

class FilterBgConfPlugin : public nvinfer1::IPluginV2 {
 public:
  FilterBgConfPlugin(const std::string name, float _objectness_score)
      : mLayerName(name) {
    objectness_score = _objectness_score;
  }

  FilterBgConfPlugin(const std::string name, const void* data, size_t length)
      : mLayerName(name) {
    assert(length == 3 * sizeof(float));
    const float* d = reinterpret_cast<const float*>(data);
    objectness_score = d[0];
    numClasses = d[1];
    numPriorboxes = d[2];
  }

  const char* getPluginType() const override { return "filterbgconf"; }

  const char* getPluginVersion() const override { return "v1"; }

  int getNbOutputs() const override { return 1; }

  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                     int nbInputDims) override {
    assert(index == 0 && nbInputDims == 2);
    return inputs[1];
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
    numPriorboxes = static_cast<float>(inputDims[0].d[0]);
    numClasses = static_cast<float>(inputDims[1].d[1]);
  }

  int initialize() override { return 0; }

  void terminate() override {}

  size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

  int enqueue(int batchSize, const void* const* inputs, void** outputs,
              void* workspace, cudaStream_t stream) override {
    const float* arm_conf = reinterpret_cast<const float*>(inputs[0]);
    const float* odm_conf = reinterpret_cast<const float*>(inputs[1]);
    float* conf = reinterpret_cast<float*>(outputs[0]);
    FilterBgConf(batchSize, numPriorboxes, numClasses, objectness_score,
                 arm_conf, odm_conf, conf);
    return 0;
  }

  size_t getSerializationSize() const override { return 3 * sizeof(float); }

  void serialize(void* buffer) const override {
    float* d = reinterpret_cast<float*>(buffer);
    d[0] = objectness_score;
    d[1] = numClasses;
    d[2] = numPriorboxes;
  }

  void destroy() override { delete this; }

  IPluginV2* clone() const override {
    return new FilterBgConfPlugin(mLayerName, objectness_score);
  }

  void setPluginNamespace(const char* pluginNamespace) override {
    mNamespace = pluginNamespace;
  }

  const char* getPluginNamespace() const override { return mNamespace.c_str(); }

 private:
  float objectness_score;
  float numClasses;
  float numPriorboxes;
  const std::string mLayerName;
  std::string mNamespace;
};

class FilterBgConfPluginCreator : public nvinfer1::IPluginCreator {
 public:
  FilterBgConfPluginCreator() {
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "objectness_score", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
  }

  const char* getPluginName() const override { return "filterbgconf"; }

  const char* getPluginVersion() const override { return "v1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() override {
    return &mFC;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override {
    float objectness_score;

    const nvinfer1::PluginField* fields = fc->fields;

    assert(fc->nbFields == 1);
    for (int i = 0; i < fc->nbFields; i++) {
      if (strcmp(fields[i].name, "objectness_score") == 0) {
        assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
        objectness_score = *(static_cast<const float*>(fields[i].data));
      }
    }

    return new FilterBgConfPlugin(name, objectness_score);
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serialData,
                                         size_t serialLength) override {
    return new FilterBgConfPlugin(name, serialData, serialLength);
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

#endif  // SHADOW_FILTERBGCONF_HPP
