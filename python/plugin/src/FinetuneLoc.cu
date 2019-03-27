#include "FinetuneLoc.h"
#include "Common.h"

__global__ void KernelFinetuneLoc(int batchSize, int numPriorboxes, const float *arm_loc, const float *priorbox_loc, float *loc)
{
    int box_id = threadIdx.x + blockIdx.x * blockDim.x;
    int beginAddress = box_id / numPriorboxes * numPriorboxes * 8;
    int box_id_image = box_id % numPriorboxes;
    float var1 = 0.1;
    float var2 = 0.1;
    float var3 = 0.2;
    float var4 = 0.2;
    if (box_id < batchSize * numPriorboxes)
    {   
        /*
        float xmin = priorbox_loc[box_id_image * 4],
              ymin = priorbox_loc[box_id_image * 4 + 1],
              xmax = priorbox_loc[box_id_image * 4 + 2],
              ymax = priorbox_loc[box_id_image * 4 + 3];
        */
        float bbox1 = arm_loc[box_id * 4],
              bbox2 = arm_loc[box_id * 4 + 1],
              bbox3 = arm_loc[box_id * 4 + 2],
              bbox4 = arm_loc[box_id * 4 + 3];
        
        float prior_width = priorbox_loc[box_id_image * 4 + 2],
              prior_height = priorbox_loc[box_id_image * 4 + 3],
              prior_center_x = priorbox_loc[box_id_image * 4],
              prior_center_y = priorbox_loc[box_id_image * 4 + 1];
        float decode_bbox_center_x = var1 * bbox1 * prior_width + prior_center_x,
              decode_bbox_center_y = var2 * bbox2 * prior_height + prior_center_y,
              decode_bbox_width = exp(var3 * bbox3) * prior_width,
              decode_bbox_height = exp(var4 * bbox4) * prior_height;

        loc[beginAddress + box_id_image * 4] = decode_bbox_center_x - decode_bbox_width / 2;
        loc[beginAddress + box_id_image * 4 + 1] = decode_bbox_center_y - decode_bbox_height / 2;
        loc[beginAddress + box_id_image * 4 + 2] = decode_bbox_center_x + decode_bbox_width / 2;
        loc[beginAddress + box_id_image * 4 + 3] = decode_bbox_center_y + decode_bbox_height / 2;
        loc[beginAddress + (box_id_image + numPriorboxes) * 4] = var1;
        loc[beginAddress + (box_id_image + numPriorboxes) * 4 + 1] = var2;
        loc[beginAddress + (box_id_image + numPriorboxes) * 4 + 2] = var3;
        loc[beginAddress + (box_id_image + numPriorboxes) * 4 + 3] = var4;
    }
}

void FinetuneLoc(int batchSize, int numPriorboxes, const float *arm_loc, const float *priorbox_loc, float *loc)
{
    int block = GetBlocks(batchSize * numPriorboxes);
    int grid = (batchSize * numPriorboxes + block - 1) / block;
    KernelFinetuneLoc<<<grid, block>>>(batchSize, numPriorboxes, arm_loc, priorbox_loc, loc);
}

