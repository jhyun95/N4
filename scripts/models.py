# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 03:13:28 2018

@author: jhyun_000
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

def transform_default():
    return [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]

def transform_list():
    return [transforms.ToTensor(), 
            lambda x: x > 0.5,
            lambda x: x.float()]
    
class LinNet(nn.Module):
    def __init__(self):
        super(LinNet, self).__init__()
        self.lin1 = nn.Linear(784,112)
        self.lin2 = nn.Linear(112,16)
        self.lin3 = nn.Linear(16,10)
#        self.lin1 = l0module.L0Linear(784,112)
#        self.lin2 = l0module.L0Linear(112,16)
#        self.lin3 = l0module.L0Linear(16,10)
        
    def forward(self, x):
        x = x.view(-1,784)
        x = self.lin1(x); x = F.relu(x)
        x = self.lin2(x); x = F.relu(x)
        x = self.lin3(x); x = F.log_softmax(x)
        return x

class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class StnNet(nn.Module):
    # From http://pytorch.org/tutorials/intermediate/spatial_transformer_tutorial.html?highlight=mnist
    def __init__(self):
        super(StnNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.fill_(0)
        self.fc_loc[2].bias.data = torch.FloatTensor([1, 0, 0, 0, 1, 0])

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 3 * 3)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
