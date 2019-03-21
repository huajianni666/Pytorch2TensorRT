import torch
from torch import nn
import pdb
import torch
from math import sqrt as sqrt
from torch.jit import Tensor
from typing import List, Tuple

'''
class PriorBox(torch.jit.ScriptModule):
    def __init__(self):
        super(PriorBox, self).__init__()

    def forward(self, step, feature_maps, img_wh, steps, min_sizes, max_sizes, aspect_ratios, use_max_sizes, clip):
        mean = torch.jit.annotate(List[Tuple[Tensor, Tensor]], [])
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
'''

class Refinedet(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, cfg, backbone, pretrained = False):
        super(Refinedet, self).__init__()
        self.num_classes = cfg['NUM_CLASSES']
        # SSD network
        self.extractor = backbone
        self.arm_channels = cfg['ARM_CHANNELS']
        self.num_anchors = cfg['NUM_ANCHORS']
        self.odm_channels = cfg['ODM_CHANNELS']
        self.arm_num_classes = 2
        self.arm_loc, self.arm_conf, self.odm_loc, self.odm_conf = self._getConfAndLocLayer(cfg) 
        
        self.img_wh = cfg['SIZE']
        self.num_priors = len(cfg['ASPECT_RATIOS'])
        self.feature_maps = cfg['FEATURE_MAPS']
        self.variance = cfg['VARIANCE']
        self.min_sizes = cfg['MIN_SIZES']
        self.use_max_sizes = cfg['USE_MAX_SIZE']
        self.max_sizes = cfg['MAX_SIZES']
        self.steps = cfg['STEPS']
        self.aspect_ratios = cfg['ASPECT_RATIOS']
        self.clip = cfg['CLIP']
        
        self.softmax = nn.Softmax(dim=-1)
        #self.priors_layer = PriorBox();
        #self.priors = self.priors_layer.forward(len(self.feature_maps), self.feature_maps, self.img_wh, self.steps, self.min_sizes, self.max_sizes, self.aspect_ratios, self.use_max_sizes, self.clip)


    def _getConfAndLocLayer(self, cfg):
        arm_loc_layers = []
        arm_conf_layers = []
        odm_loc_layers = []
        odm_conf_layers = []
        for i in range(len(self.arm_channels)):
            arm_loc_layers.append(nn.Conv2d(self.arm_channels[i], self.num_anchors[i] * 4, kernel_size = 3, padding = 1))
            arm_conf_layers.append(nn.Conv2d(self.arm_channels[i], self.num_anchors[i] * self.arm_num_classes, kernel_size = 3, padding = 1))            
            odm_loc_layers.append(nn.Conv2d(self.odm_channels[i], self.num_anchors[i] * 4, kernel_size = 3, padding = 1))
            odm_conf_layers.append(nn.Conv2d(self.odm_channels[i], self.num_anchors[i] * self.num_classes, kernel_size = 3, padding = 1))
        return nn.ModuleList(arm_loc_layers), nn.ModuleList(arm_conf_layers), nn.ModuleList(odm_loc_layers), nn.ModuleList(odm_conf_layers)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        arm_loc = list()
        arm_conf = list()
        odm_loc = list()
        odm_conf = list()
        arm_xs, odm_xs = self.extractor(x)
        for i, y in enumerate(odm_xs):
            ol = self.odm_loc[i](y).permute(0, 2, 3, 1).contiguous()
            odm_loc.append(ol.view(1, -1))
            oc = self.odm_conf[i](y).permute(0, 2, 3, 1).contiguous()
            odm_conf.append(oc.view(1, -1))
        for i, y in enumerate(arm_xs):
            al = self.arm_loc[i](y).permute(0, 2, 3, 1).contiguous()
            arm_loc.append(al.view(1, -1))
            ac = self.arm_conf[i](y).permute(0, 2, 3, 1).contiguous()
            arm_conf.append(ac.view(1, -1))
        
        arm_loc = torch.cat([o  for o in arm_loc], 1)
        arm_conf = torch.cat([o for o in arm_conf], 1)
        arm_conf_softmax = self.softmax(arm_conf.view(1, -1, self.arm_num_classes))
        odm_loc = torch.cat([o for o in odm_loc], 1)
        odm_conf = torch.cat([o for o in odm_conf], 1)
        odm_conf_softmax = self.softmax(odm_conf.view(1, -1, self.num_classes))
        
        output = (arm_loc.view(1, -1, 4),
                  arm_conf_softmax.view(1, -1, self.arm_num_classes),
                  odm_loc.view(1, -1, 4),
                  odm_conf_softmax.view(1, -1, self.num_classes))
        return output

