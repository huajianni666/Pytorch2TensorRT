#include "UpSample.hpp"

nvinfer1::PluginFieldCollection UpSamplePluginCreator::mFC{};
std::vector<nvinfer1::PluginField> UpSamplePluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(UpSamplePluginCreator);
