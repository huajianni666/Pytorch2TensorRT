#ifndef _FILTER_BG_CONF_H_
#define _FILTER_BG_CONF_H_
#include <iostream>
#include <cassert>
#include <cstring>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <stdexcept>
#include "Common.h"
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <vector>
using namespace std;

void FilterBgConf(int batchSize, int numPriorboxes, int numClasses, float objectness_score, const float *arm_conf, const float *odm_conf, float *conf);

class FilterBgConfPlugin: public nvinfer1::IPluginExt
{
public:
	// In this simple case we're going to infer the number of output channels from the bias weights.
	// The knowledge that the kernel weights are weights[0] and the bias weights are weights[1] was
	// divined from the caffe innards
	
        FilterBgConfPlugin(float objectness_score_)
        {
            objectness_score = objectness_score_;
        }

	// Create the plugin at runtime from a byte stream.
	FilterBgConfPlugin(const void* data, size_t length)
        {
            assert(length == 3 * sizeof(float));
            const float *d = reinterpret_cast<const float *>(data);
            objectness_score = d[0];
            numClasses = d[1];
            numPriorboxes = d[2];

        }

	virtual int getNbOutputs() const override { return 1; }

	virtual nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override
	{
            assert(index == 0 && nbInputDims == 2);
	    return inputs[1];
	}

	virtual int initialize() override
	{
	    return 0;
	}

	virtual void terminate() override
	{
	}

        // This plugin requires no workspace memory during build time.
	virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

	virtual int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override
	{
	    const float *arm_conf = reinterpret_cast<const float *>(inputs[0]);
            const float *odm_conf = reinterpret_cast<const float *>(inputs[1]);
            float *conf = reinterpret_cast<float *>(outputs[0]);
            FilterBgConf(batchSize, numPriorboxes, numClasses, objectness_score, arm_conf, odm_conf, conf);
            return 0;
        }

	// For this sample, we'll only support float32 with NCHW.
	virtual bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override
	{
	    return (type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kNCHW);
	}

	void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize)
	{
	    assert(nbInputs == 2);
            numPriorboxes = static_cast<float>(inputDims[0].d[0]);
            numClasses = static_cast<float>(inputDims[1].d[1]);
        }

	virtual size_t getSerializationSize() override
	{
	    return 3 * sizeof(float);
	}

	virtual void serialize(void* buffer) override
	{
	    float *d = reinterpret_cast<float *>(buffer);
            d[0] = objectness_score;
            d[1] = numClasses;
            d[2] = numPriorboxes;

        }

	// Free buffers.
	virtual ~FilterBgConfPlugin(){}

private:
        float objectness_score;
        float numClasses;
        float numPriorboxes;

};

#endif
