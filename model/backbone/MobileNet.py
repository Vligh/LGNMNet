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

from torchvision.models.mobilenet import mobilenet_v2
# from torchvision.models.resnet import resnet50
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sys
# sys.path.append('../')
# from head.head_def import HeadFactory
class MobileNet(torch.nn.Module):
    def __init__(self, num_classes = 2, embedding_size=128, pretrained=True):
        super(MobileNet, self).__init__()
        base = mobilenet_v2(pretrained=pretrained)
        self.backbone = base.features
        self.CovBnRelu = nn.Sequential(
            nn.Conv2d(1280, 32, (1, 1), 1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        # self.Linear = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(20480, embedding_size), # same as line 98 x.view
        #     # nn.ReLU6(inplace=True)
        # )
        # self.classifier = nn.Sequential(
        #     nn.Linear(fea_dim, num_classes),
        # )
        self.flatten = nn.Flatten()


    def forward(self, x):
        x = self.backbone(x)
        # aa = x.size()
        # print(x.size())
        x = self.CovBnRelu(x)
        # print(x.size())
        x = self.flatten(x)
        # bb = x.size()
        # print(x.size())
        # x = x.view(-1, self.dim) # CC.size[3]*CC.size[2]
        # x = self.Linear(x)
        # cc = x.size()
        # print(x.size())
        # x = self.classifier(x)
        return x