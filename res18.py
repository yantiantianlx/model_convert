# -*- coding: utf-8 -*-
"""
Created on 2020-08-27 14:31

@author: tiantian
"""

# -*- coding: utf-8 -*-
"""
Created on 2020-08-27 11:47

@author: tiantian
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import random


class ResNet18SiameseNetwork(nn.Module):
    def __init__(self):
        super(ResNet18SiameseNetwork, self).__init__()
        self.net = models.resnet18(pretrained=True)

    def forward(self, x):
        output = self.net.conv1(x)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)

        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)

        output = self.net.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.net.fc(output)
        return output

