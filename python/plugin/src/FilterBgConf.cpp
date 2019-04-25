#include "FilterBgConf.hpp"

nvinfer1::PluginFieldCollection FilterBgConfPluginCreator::mFC{};
std::vector<nvinfer1::PluginField>
    FilterBgConfPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FilterBgConfPluginCreator);
