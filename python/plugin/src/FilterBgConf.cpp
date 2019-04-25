#include "FilterBgConf.hpp"

#if NV_TENSORRT_MAJOR >= 5

nvinfer1::PluginFieldCollection FilterBgConfPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> FilterBgConfPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FilterBgConfPluginCreator);

#endif
