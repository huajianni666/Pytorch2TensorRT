#include "UpSample.hpp"

#if NV_TENSORRT_MAJOR >= 5

nvinfer1::PluginFieldCollection UpSamplePluginCreator::mFC{};
std::vector<nvinfer1::PluginField> UpSamplePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(UpSamplePluginCreator);

#endif
