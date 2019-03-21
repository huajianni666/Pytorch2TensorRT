# -*- coding: utf-8 -*-
# Written by yq_yao

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from backbone.model_helper import FpnAdapter

import pdb

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out

class BasicBlockDownsample(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample):
        super(BasicBlockDownsample, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class RefineResnet(nn.Module):
    def __init__(self, block,blockDown, num_blocks, size):
        super(RefineResnet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer_downsample(block, blockDown, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer_downsample(block, blockDown, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer_downsample(block, blockDown, 512, num_blocks[3], stride=2)
        self.inchannel = block.expansion * 512
        self.extras = self._add_extras(self.inchannel)
        self.smooth1 = nn.Conv2d(self.inchannel, 512, kernel_size=3, stride=1, padding=1)
        self.fpn = FpnAdapter([128, 256, 512, 256], 4)
    
    def _add_extras(self, in_channel):
        layers = []
        layers.append(nn.Conv2d(in_channel, 256, kernel_size=1, stride=1));
        layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1));
        return nn.ModuleList(layers)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        layers = []
        layers.append(block(self.inplanes, planes, stride))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _make_layer_downsample(self, block, blockDown, planes, blocks, stride=1):
        downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(blockDown(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        x = c5
        c5_ = self.smooth1(c5)
        x = F.relu(self.extras[0](x))
        c6 = F.relu(self.extras[1](x))
        
        arm_sources = [c3, c4, c5_, c6]
        odm_sources = self.fpn(arm_sources)
        '''
        a3 = torch.reshape(arm_sources[0], (-1,))
        a4 = torch.reshape(arm_sources[1], (-1,))
        a5 = torch.reshape(arm_sources[2], (-1,))
        a6 = torch.reshape(arm_sources[3], (-1,))

        o3 = torch.reshape(odm_sources[0], (-1,))
        o4 = torch.reshape(odm_sources[1], (-1,))
        o5 = torch.reshape(odm_sources[2], (-1,))
        o6 = torch.reshape(odm_sources[3], (-1,))
        result = torch.cat((a3,a4,a5,a6,o3,o4,o5,o6),0)
        '''
        return arm_sources , odm_sources

def RefineResnet18(size, channel_size='48'):
    return RefineResnet(BasicBlock,BasicBlockDownsample, [2, 2, 2, 2], size)
