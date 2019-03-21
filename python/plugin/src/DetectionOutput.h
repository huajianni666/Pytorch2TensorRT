#ifndef _DETECTIONOUTPUT_H_
#define _DETECTIONOUTPUT_H_
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <iostream>
nvinfer1::IPlugin *DetectionOutputPlugin(bool 	shareLocation,
                                                bool 	varianceEncodedInTarget,
                                                int 	backgroundLabelId,
                                                int 	numClasses,
                                                int 	topK,
                                                int 	keepTopK,
                                                float 	confidenceThreshold,
                                                float 	nmsThreshold){
    return (nvinfer1::IPlugin *)nvinfer1::plugin::createSSDDetectionOutputPlugin({shareLocation, varianceEncodedInTarget, backgroundLabelId, numClasses, topK, keepTopK, confidenceThreshold, nmsThreshold, nvinfer1::plugin::CodeTypeSSD::CENTER_SIZE, {0 ,1 , 2}, false, true});
}
#endif
