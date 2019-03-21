import torch
from torch import nn
import pdb
import torch
from math import sqrt as sqrt

def getPriorLayer(step, feature_maps, img_wh, steps, min_sizes, max_sizes, aspect_ratios, use_max_sizes, clip):
    mean = []
    for k in range(step):
        f = feature_maps[k]
        grid_h, grid_w = f[1], f[0]
        for i in range(grid_h):
            for j in range(grid_w):
                f_k_h = img_wh[1] / steps[k][1]
                f_k_w = img_wh[0] / steps[k][0]
                # unit center x,y
                cx = (j + 0.5) / f_k_w
                cy = (i + 0.5) / f_k_h

                # aspect_ratio: 1
                # rel size: min_size
                s_k_h = min_sizes[k] / img_wh[1]
                s_k_w = min_sizes[k] / img_wh[0]
                mean.append([cx, cy, s_k_w, s_k_h])

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                if(use_max_sizes):
                    s_k_prime_w = sqrt(s_k_w * (max_sizes[k] / img_wh[0]))
                    s_k_prime_h = sqrt(s_k_h * (max_sizes[k] / img_wh[1]))
                    mean.append([cx, cy, s_k_prime_w, s_k_prime_h])

                for ar in aspect_ratios[k]:
                    mean.append([cx, cy, s_k_w * sqrt(ar), s_k_h / sqrt(ar)])

    # back to torch land
    output = torch.Tensor(mean).view(-1, 4)
    if clip:
        output.clamp_(max=1, min=0)
    return output
