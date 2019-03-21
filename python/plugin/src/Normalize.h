#ifndef _NORMALIZE_H_
#define _NORMALIZE_H_
#include "NvInfer.h"
#include "NvInferPlugin.h"

nvinfer1::IPlugin *NormalizePlugin(const nvinfer1::Weights *scales, bool acrossSpatial, bool channelShared, float eps){
    return (nvinfer1::IPlugin *)nvinfer1::plugin::createSSDNormalizePlugin(scales, acrossSpatial, channelShared, eps);
}
#endif
