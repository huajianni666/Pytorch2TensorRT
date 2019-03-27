#ifndef _UP_SAMPLE_H_
#define _UP_SAMPLE_H_

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

template <typename T>
void Resize(const T* in_data, const vector<int>& in_shape, int type,
            const vector<int>& out_shape, T* out_data);

class UpSamplePlugin: public nvinfer1::IPluginExt
{
public:
	// In this simple case we're going to infer the number of output channels from the bias weights.
	// The knowledge that the kernel weights are weights[0] and the bias weights are weights[1] was
	// divined from the caffe innards
	
        UpSamplePlugin(float scale_, int type_)
        {
            scale = scale_;
            type = type_;
        }

	// Create the plugin at runtime from a byte stream.
	UpSamplePlugin(const void* data, size_t length)
        {
            assert(length == 5 * sizeof(float));
            const float *d = reinterpret_cast<const float *>(data);
            scale = d[0];
            input_c = d[1];
            input_h = d[2];
            input_w = d[3];
            type = d[4];
        }

	virtual int getNbOutputs() const override { return 1; }

	virtual nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override
	{
            assert(index == 0 && nbInputDims == 1);
	    return nvinfer1::DimsCHW{static_cast<int>(inputs[0].d[0]), static_cast<int>(inputs[0].d[1]*scale), static_cast<int>(inputs[0].d[2]*scale)};
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
            //const float *input_data = reinterpret_cast<const float *>(inputs[0]);
            //float *output_data = reinterpret_cast<float *>(outputs[0]);
            vector<int> input_shape = { batchSize, input_c, input_h, input_w };
            vector<int> output_shape = { batchSize, input_c, int(input_h * scale), int(input_w * scale) };
            Resize((const float*)inputs[0], input_shape, type, output_shape, (float*)outputs[0]);     
            return 0;
	}

	// For this sample, we'll only support float32 with NCHW.
	virtual bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override
	{
	    return (type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kNCHW);
	}

	void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize)
	{
	    assert(nbInputs == 1);
            assert(nbOutputs == 1);
            input_c = inputDims[0].d[0];
            input_h = inputDims[0].d[1];
            input_w = inputDims[0].d[2];
	}

	virtual size_t getSerializationSize() override
	{
	    return 5 * sizeof(float);
	}

	virtual void serialize(void* buffer) override
	{
	    float* d = reinterpret_cast<float*>(buffer);
	    d[0] = scale;
            d[1] = input_c;
            d[2] = input_h;
            d[3] = input_w;
            d[4] = type;
	}

	// Free buffers.
	virtual ~UpSamplePlugin(){}

private:
        float scale;
        int type;
        int input_c, input_h, input_w;
};

#endif
