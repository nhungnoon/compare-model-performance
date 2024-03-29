"""
A customized CNN model
"""

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.metrics import accuracy_score


# based on
# https://github.com/nhungnoon/Final-Project-Group3/blob/master/Nhung-Nguyen-individual-project/Code/pytorch_ver.py
class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, (3, 3))
        self.convnorm1 = nn.BatchNorm2d(16)
        self.drop = nn.Dropout(0.25)
        self.pool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, (3, 3))
        self.convnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((2, 2))

        self.linear1 = nn.Linear(32 * 5 * 5, 400)
        self.linear1_bn = nn.BatchNorm1d(400)
        self.drop = nn.Dropout(0.25)
        self.linear2 = nn.Linear(400, 10)
        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.drop(self.linear1_bn(self.act(self.linear1(x.view(len(x), -1)))))
        return self.linear2(x)


def calculate_accuracy(transform_inputs, y_test):
    with torch.no_grad():
        _, pred_labels = torch.max(transform_inputs.data, 1)

    return 100 * accuracy_score(y_test, pred_labels)


def train_model(
    num_epoch: int,
    train_loader: DataLoader,
    x_test: Variable,
    y_test: Variable,
):
    cnn_model = CNN_model()
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        losses = []
        cnn_model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            loss = criterion(cnn_model(inputs), labels.squeeze_())
            loss.backward()
            optimizer.step()
            losses.append(loss.data.mean())
        print("[%d/%d] Train Loss: %.3f" % (epoch + 1, num_epoch, np.mean(losses)))
        cnn_model.eval()
        with torch.no_grad():
            x_test_split = Variable(x_test)
            loss = criterion(cnn_model(x_test_split), y_test.squeeze_())
            loss_test_split = loss.item()

        transform_inputs_test = cnn_model(x_test)
        print(
            "Epoch {} Valid Loss: {:.5f}, Valid Acc: {:.2f}".format(
                epoch,
                loss_test_split,
                calculate_accuracy(transform_inputs_test, y_test),
            )
        )

    return cnn_model
