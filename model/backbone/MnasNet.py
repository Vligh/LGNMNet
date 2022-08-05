# -*- coding: utf-8 -*-

__author__ = 'kohou.wang'
__time__ = '19-9-24'
__email__ = 'oukohou@outlook.com'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.
# Well, I'm kidding... Always, Welcome to contact me.

"""Description for the script:
train SSR-Net.
"""
import os
import torch
import torch.nn as nn

from torchvision.models.mnasnet import mnasnet0_5
# from torchvision.models.resnet import resnet50
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sys
# sys.path.append('../')
# from head.head_def import HeadFactory
class MnasNet(torch.nn.Module):
    def __init__(self, num_classes = 2, embedding_size=128, pretrained=True):
        super(MnasNet, self).__init__()
        base = mnasnet0_5(pretrained=pretrained)
        self.backbone = base
        self.CovBnRelu = nn.Sequential(
            nn.Conv2d(1280, 32, (1, 1), 1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        self.Linear = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(1000, embedding_size), # same as line 98 x.view
            # nn.ReLU6(inplace=True)
        )
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.backbone(x)
        # aa = x.size()
        # print(x.size())
        x = self.Linear(x)
        # print(x.size())
        x = self.flatten(x)
        # bb = x.size()
        # print(x.size())
        # x = x.view(-1, self.dim) # CC.size[3]*CC.size[2]
        # x = self.Linear(x)
        # cc = x.size()
        # print('cc', cc)
        # x = self.classifier(x)
        # x = x.squeeze(1)
        return x