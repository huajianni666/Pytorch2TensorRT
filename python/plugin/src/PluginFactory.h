#ifndef _GLOBAL_PLUGIN_FACTORY_H_
#define _GLOBAL_PLUGIN_FACTORY_H_
#include <cassert>
#include <cstring>
#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <stdexcept>
#include "UpSample.h"
#include "DetectionOutput.h"
#include "Normalize.h"
#include "PriorBox.h"
#include "FilterBgConf.h"
#include "FinetuneLoc.h"
#include "NvInferPlugin.h"
#include "NvInfer.h"
#include "NvCaffeParser.h"
#include <string>

class GlobalPluginFactory : public nvcaffeparser1::IPluginFactoryExt, public nvinfer1::IPluginFactory
{
public:
        bool isPlugin(const char* name) override { return isPluginExt(name); }

        bool isPluginExt(const char* name) override 
        {
            bool UpSamplePlugin = (strstr(name,"upsample") == NULL); 
            return !strcmp(name, "fc") || !strcmp(name, "detectionoutput") || !strcmp(name, "filterbgconf") || !strcmp(name, "finetuneloc") || !UpSamplePlugin; 
        }

        // Create a plugin using provided weights.
        virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const nvinfer1::Weights* weights, int nbWeights) override
        {
            return nullptr;
        }

        // Create a plugin from serialized data.
        virtual nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override
        {
            assert(isPlugin(layerName));
            if(!strcmp(layerName, "detectionoutput"))
            {
                detectionOutputPlugin = (nvinfer1::IPlugin *)nvinfer1::plugin::createSSDDetectionOutputPlugin(serialData,serialLength);
                return detectionOutputPlugin;
            }
            else if(!strcmp(layerName, "filterbgconf"))
            {    
                filterBgConfPlugin = new FilterBgConfPlugin{serialData, serialLength};
                return filterBgConfPlugin;
            }
            else if(!strcmp(layerName, "finetuneloc"))
            {
                finetuneLocPlugin = new FinetuneLocPlugin{serialData, serialLength};
                return finetuneLocPlugin;
            }
            else if(strstr(layerName, "upsample") != NULL)
            {
                UpSamplePlugin* ptr = new UpSamplePlugin{serialData, serialLength};
                upSamplePlugin.push_back(ptr);
                return ptr;
            }
        }

        // User application destroys plugin when it is safe to do so.
        // Should be done after consumers of plugin (like ICudaEngine) are destroyed.
        void destroyPlugin() 
        { 
            if(!filterBgConfPlugin) delete filterBgConfPlugin;
            if(!finetuneLocPlugin) delete finetuneLocPlugin;
            for(int iter = 0; iter < upSamplePlugin.size(); iter ++) { delete upSamplePlugin.at(iter);}
            if(!detectionOutputPlugin) delete detectionOutputPlugin;
        }

private:

    FilterBgConfPlugin* filterBgConfPlugin{ nullptr };
    FinetuneLocPlugin* finetuneLocPlugin{ nullptr };
    std::vector<UpSamplePlugin*> upSamplePlugin{ nullptr };
    nvinfer1::IPlugin* detectionOutputPlugin{ nullptr };
};

#endif
