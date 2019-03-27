#ifndef _PRIORBOX_H_
#define _PRIORBOX_H_
#include "NvInfer.h"
#include "NvInferPlugin.h"

nvinfer1::IPlugin *PriorBoxPlugin(int numMinSize,
                                          int numMaxSize,
                                          int numAspectRatios,
                                          std::vector<float> minSizeVector,
                                          std::vector<float> maxSizeVector,
                                          std::vector<float> aspectRatiosVector,
                                          bool flip,
                                          bool clip,
                                          std::vector<float> varianceVector,
                                          int imgH,
                                          int imgW,
                                          float StepH,
                                          float StepW,
                                          float offset){
    float *minSize = minSizeVector.data();
    float *maxSize = maxSizeVector.data();
    float *aspectRatios = aspectRatiosVector.data();
    float *variance = varianceVector.data();
    assert(numMinSize == minSizeVector.size());
    assert(numMaxSize == maxSizeVector.size());
    assert(numAspectRatios == aspectRatiosVector.size());
    assert(varianceVector.size() == 4);
    return (nvinfer1::IPlugin *)nvinfer1::plugin::createSSDPriorBoxPlugin({minSize, maxSize, aspectRatios, numMinSize, numMaxSize, numAspectRatios, flip, clip, {variance[0],variance[1],variance[2],variance[3]}, imgH, imgW, StepH, StepW, offset});    
}

#endif
