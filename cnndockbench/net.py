import torch
import torch.nn as nn
import numpy as np


class ConvProt(nn.Module):

    def __init__(self):
        super(ConvProt, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv3d(7, 16, kernel_size=3, stride=1),
                                    nn.BatchNorm3d(16),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=2, stride=1))

        self.layer2 = nn.Sequential(nn.Conv3d(16, 32, kernel_size=3, stride=1),
                                    nn.BatchNorm3d(32),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=2, stride=1))

        self.layer3 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, stride=1),
                                    nn.BatchNorm3d(32),
                                    nn.ReLU(),
                                    nn.MaxPool3d(kernel_size=2, stride=1))
        self.fc = nn.Linear(32 * 3 * 3 * 3, 1024)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        for i in range(5):
            out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


class FullNN(nn.Module):
    def __init__(self):
        super(FullNN, self).__init__()

        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024 + 1024, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)

        self.fcOut1 = nn.Linear(128, 17)
        self.fcOut2 = nn.Linear(128, 17)
        self.fcOut3 = nn.Linear(128, 17)

    def forward(self, x, p):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.fc3(torch.cat((out, p), 1))
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)

        out1 = self.fcOut1(out)
        out2 = self.fcOut2(out)
        out3 = self.fcOut3(out)

        return out1, out2, out3