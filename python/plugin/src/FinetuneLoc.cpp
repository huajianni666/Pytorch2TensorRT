#include "FinetuneLoc.hpp"

nvinfer1::PluginFieldCollection FinetuneLocPluginCreator::mFC{};
std::vector<nvinfer1::PluginField>
    FinetuneLocPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FinetuneLocPluginCreator);
