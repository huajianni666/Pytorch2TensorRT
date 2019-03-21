#ifndef _FINETUNE_LOC_H_
#define _FINETUNE_LOC_H_
#include <iostream>
#include <cassert>
#include <cstring>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <stdexcept>
#include "common.h"
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <vector>
using namespace std;

void FinetuneLoc(int batchSize, int numPriorboxes, const float *arm_loc, const float *priorbox_loc, float *loc);

class FinetuneLocPlugin: public nvinfer1::IPluginExt
{
public:
	// In this simple case we're going to infer the number of output channels from the bias weights.
	// The knowledge that the kernel weights are weights[0] and the bias weights are weights[1] was
	// divined from the caffe innards
	
        FinetuneLocPlugin()
        {
        }

	// Create the plugin at runtime from a byte stream.
	FinetuneLocPlugin(const void* data, size_t length)
        {
            assert(length == sizeof(int));
            const int *d = reinterpret_cast<const int *>(data);
            numPriorboxes = d[0];
        }

	virtual int getNbOutputs() const override { return 1; }

	virtual nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override
	{
	    return nvinfer1::DimsCHW{static_cast<int>(inputs[0].d[0]), static_cast<int>(inputs[0].d[1]*2), static_cast<int>(inputs[0].d[2])};;
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
            const float *arm_loc = reinterpret_cast<const float *>(inputs[0]);
            const float *priorbox_loc = reinterpret_cast<const float *>(inputs[1]);
            float *decodePriorbox = reinterpret_cast<float *>(outputs[0]);
            FinetuneLoc(batchSize, numPriorboxes, arm_loc, priorbox_loc, decodePriorbox);
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
            assert(nbOutputs == 1);
            numPriorboxes = inputDims[0].d[0];
	}

	virtual size_t getSerializationSize() override
	{
	    return sizeof(int);
	}

	virtual void serialize(void* buffer) override
	{
	    int* d = reinterpret_cast<int*>(buffer);
	    d[0] = numPriorboxes;
	}

	// Free buffers.
	virtual ~FinetuneLocPlugin(){}

private:
        int numPriorboxes;
};

class FinetuneLocPluginFactory : public nvcaffeparser1::IPluginFactoryExt, public nvinfer1::IPluginFactory
{
public:

	bool isPlugin(const char* name) override { return isPluginExt(name); }

	bool isPluginExt(const char* name) override { return !strcmp(name, "FinetuneLocOp");; }

        // Create a plugin using provided weights.
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
	{       
            assert(isPluginExt(layerName));
	    assert(mPlugin == nullptr);
            // This plugin will need to be manually destroyed after parsing the network, by calling destroyPlugin.
	    mPlugin = new FinetuneLocPlugin();
            return mPlugin;
	}

        // Create a plugin from serialized data.
	virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
	{
            assert(isPlugin(layerName));
            // This will be automatically destroyed when the engine is destroyed.
	    return new FinetuneLocPlugin(serialData, serialLength);
	}

        // User application destroys plugin when it is safe to do so.
        // Should be done after consumers of plugin (like ICudaEngine) are destroyed.
	void destroyPlugin() 
        {
            delete mPlugin;   
        }


        FinetuneLocPlugin* mPlugin{ nullptr };     

};

#endif
