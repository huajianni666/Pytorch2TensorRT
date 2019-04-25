#include "FilterBgConf.hpp"
#include "common.hpp"

__global__ void KernelFilterBgConf(int batchSize, int numPriorboxes, int numClasses, float objectness_score, const float *arm_conf, const float *odm_conf, float *conf)
{
    int priorboxesId = threadIdx.x + blockIdx.x * blockDim.x;
    if (priorboxesId < batchSize * numPriorboxes)
    {
        if (arm_conf[2 * priorboxesId + 1] < objectness_score)
        {
            for (int c = 0; c < numClasses; ++c)
            {
                if (c != 0)
                    conf[priorboxesId * numClasses + c] = 0.0;
                else
                    conf[priorboxesId * numClasses + c] = 1.0;
            }
        }
        else
        {
            for (int c = 0; c < numClasses; c++)
                conf[priorboxesId * numClasses + c] = odm_conf[priorboxesId * numClasses + c];
        }
    }
}

void FilterBgConf(int batchSize, int numPriorboxes, int numClasses, float objectness_score, const float *arm_conf, const float *odm_conf, float *conf)
{
    int block = GetBlocks(batchSize * numPriorboxes);
    int grid = (batchSize * numPriorboxes + block - 1) / block;
    KernelFilterBgConf<<<grid, block>>>(batchSize, numPriorboxes, numClasses, objectness_score, arm_conf, odm_conf, conf);
}

