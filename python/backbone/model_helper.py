# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
import pdb


class FpnAdapter(nn.Module):
    def __init__(self, block, fpn_num):
        super(FpnAdapter, self).__init__()
        trans_layers = list();
        latent_layers = list();
        up_layers = list();
        for i in range(fpn_num):
            trans_layers.append(self._getTransLayer(block[i]))
            latent_layers.append(self._getLatentLayer())
        self.up_layers = list();
        for i in range(fpn_num-1):
            up_layers.append(self._getUpLayer())
        
        self.trans_layers , self.up_layers, self.latent_layers = nn.ModuleList(trans_layers), nn.ModuleList(up_layers), nn.ModuleList(latent_layers)

    def _getLatentLayer(self):
        return nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _getUpLayer(self):
        return nn.Upsample(scale_factor=2, mode='nearest')

    def _getTransLayer(self,block):
        return nn.Sequential(
                nn.Conv2d(block, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        trans_layers_list = list()
        fpn_out = list()
        for i in range(len(self.trans_layers)):
            trans_layers_list.append(self.trans_layers[i](x[i]))
        last = F.relu(self.latent_layers[-1](trans_layers_list[-1]))
        _up = self.up_layers[-1](last)

        tmp = F.relu(trans_layers_list[-2] + _up)        
        last2 = F.relu(self.latent_layers[-2](tmp))
        _up = self.up_layers[-2](last2) 
       
        tmp = F.relu(trans_layers_list[-3] + _up)
        last1 = F.relu(self.latent_layers[-3](tmp)) 
        _up = self.up_layers[-3](last1)

        tmp = F.relu(trans_layers_list[-4] + _up)
        last0 = F.relu(self.latent_layers[-4](tmp))
        fpn_out = [last0,last1,last2,last]
        return fpn_out
