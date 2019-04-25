#include "FinetuneLoc.hpp"

#if NV_TENSORRT_MAJOR >= 5

nvinfer1::PluginFieldCollection FinetuneLocPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> FinetuneLocPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FinetuneLocPluginCreator);

#endif
