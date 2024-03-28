"""
A customized CNN model
"""

import torch
import torch.nn as nn

# based on 
# https://github.com/nhungnoon/Final-Project-Group3/blob/master/Nhung-Nguyen-individual-project/Code/pytorch_ver.py
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3),)
        self.convnorm1 = nn.BatchNorm2d(32)
        self.drop = nn.Dropout(0.25)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.convnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d((2, 2))

        self.linear1 = nn.Linear(64 * 5 * 5, 400)
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(0.5)
        self.linear2 = nn.Linear(400, 10)
        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        return self.linear2(x)
