#ifndef SHADOW_UPSAMPLE_HPP
#define SHADOW_UPSAMPLE_HPP

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "NvInfer.h"
using namespace std;

template <typename T>
void Resize(const T* in_data, const vector<int>& in_shape, int type,
            const vector<int>& out_shape, T* out_data);

#if NV_TENSORRT_MAJOR >= 5

class UpSamplePlugin : public nvinfer1::IPluginV2 {
 public:
  UpSamplePlugin(const std::string name, float _scale, int _type)
      : mLayerName(name) {
    scale = _scale;
    up_type = _type;
  }

  UpSamplePlugin(const std::string name, const void* data, size_t length)
      : mLayerName(name) {
    assert(length == 5 * sizeof(float));
    const float* d = reinterpret_cast<const float*>(data);
    scale = d[0];
    input_c = d[1];
    input_h = d[2];
    input_w = d[3];
    up_type = d[4];
  }

  const char* getPluginType() const override { return "upsample"; }

  const char* getPluginVersion() const override { return "v1"; }

  int getNbOutputs() const override { return 1; }

  nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs,
                                     int nbInputDims) override {
    assert(index == 0 && nbInputDims == 1);
    return nvinfer1::DimsCHW{static_cast<int>(inputs[0].d[0]),
                             static_cast<int>(inputs[0].d[1] * scale),
                             static_cast<int>(inputs[0].d[2] * scale)};
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
    assert(nbInputs == 1);
    assert(nbOutputs == 1);
    input_c = inputDims[0].d[0];
    input_h = inputDims[0].d[1];
    input_w = inputDims[0].d[2];
  }

  int initialize() override { return 0; }

  void terminate() override {}

  size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

  int enqueue(int batchSize, const void* const* inputs, void** outputs,
              void* workspace, cudaStream_t stream) override {
    vector<int> input_shape = {batchSize, input_c, input_h, input_w};
    vector<int> output_shape = {batchSize, input_c, int(input_h * scale),
                                int(input_w * scale)};
    Resize((const float*)inputs[0], input_shape, up_type, output_shape,
           (float*)outputs[0]);
    return 0;
  }

  size_t getSerializationSize() const override { return 5 * sizeof(float); }

  void serialize(void* buffer) const override {
    float* d = reinterpret_cast<float*>(buffer);
    d[0] = scale;
    d[1] = input_c;
    d[2] = input_h;
    d[3] = input_w;
    d[4] = up_type;
  }

  void destroy() override { delete this; }

  IPluginV2* clone() const override {
    return new UpSamplePlugin(mLayerName, scale, up_type);
  }

  void setPluginNamespace(const char* pluginNamespace) override {
    mNamespace = pluginNamespace;
  }

  const char* getPluginNamespace() const override { return mNamespace.c_str(); }

 private:
  float scale;
  int up_type, input_c, input_h, input_w;
  const std::string mLayerName;
  std::string mNamespace;
};

class UpSamplePluginCreator : public nvinfer1::IPluginCreator {
 public:
  UpSamplePluginCreator() {
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "scale", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField(
        "type", nullptr, nvinfer1::PluginFieldType::kINT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
  }

  const char* getPluginName() const override { return "upsample"; }

  const char* getPluginVersion() const override { return "v1"; }

  const nvinfer1::PluginFieldCollection* getFieldNames() override {
    return &mFC;
  }

  nvinfer1::IPluginV2* createPlugin(
      const char* name, const nvinfer1::PluginFieldCollection* fc) override {
    float scale;
    int type;

    const nvinfer1::PluginField* fields = fc->fields;

    assert(fc->nbFields == 2);
    for (int i = 0; i < fc->nbFields; i++) {
      if (strcmp(fields[i].name, "scale") == 0) {
        assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
        scale = *(static_cast<const float*>(fields[i].data));
      } else if (strcmp(fields[i].name, "type") == 0) {
        assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
        type = *(static_cast<const int*>(fields[i].data));
      }
    }

    return new UpSamplePlugin(name, scale, type);
  }

  nvinfer1::IPluginV2* deserializePlugin(const char* name,
                                         const void* serialData,
                                         size_t serialLength) override {
    return new UpSamplePlugin(name, serialData, serialLength);
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

#endif  // SHADOW_UPSAMPLE_HPP
