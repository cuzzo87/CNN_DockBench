import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLegs(nn.Module):
    def __init__(self, channels=8, fp_size=1024, desc_size=199):
        super(TwoLegs, self).__init__()
        self.channels = channels
        self.fp_size = fp_size

        self.linear_fp1 = nn.Linear(fp_size + desc_size, 1024)
        self.linear_fp2 = nn.Linear(1024, 512)
        self.linear_fp3 = nn.Linear(512, 512)

        self.conv_voxel1 = nn.Conv3d(self.channels, 64, kernel_size=3)
        self.conv_voxel2 = nn.Conv3d(64, 64, kernel_size=3)
        self.max_voxel1 = nn.MaxPool3d(kernel_size=3)
        self.conv_voxel3 = nn.Conv3d(64, 64, kernel_size=1)
        self.conv_voxel4 = nn.Conv3d(64, 64, kernel_size=3)
        self.conv_voxel5 = nn.Conv3d(64, 64, kernel_size=3)

        self.linear_cat1 = nn.Linear(1024, 128)
        self.linear_out1 = nn.Linear(128, 17)
        self.linear_out2 = nn.Linear(128, 17)
        self.linear_out3 = nn.Linear(128, 17)

    def _fingerprint_forward(self, fp):
        x = F.relu(self.linear_fp1(fp))
        x = F.relu(self.linear_fp2(x))
        return F.relu(self.linear_fp3(x))

    def _pocket_forward(self, voxel):
        x = F.relu(self.conv_voxel1(voxel))
        x = F.relu(self.conv_voxel2(x))
        x = self.max_voxel1(x)
        x = F.relu(self.conv_voxel3(x))
        x = F.relu(self.conv_voxel4(x))
        x = F.relu(self.conv_voxel5(x))
        return x.view(x.shape[0], -1)

    def forward(self, voxel, fp):
        voxel_out = self._pocket_forward(voxel)
        fp_out = self._fingerprint_forward(fp)
        x = torch.cat([voxel_out, fp_out], dim=1)
        x = self.linear_cat1(x)
        out1 = F.relu(self.linear_out1(x))
        out2 = F.relu(self.linear_out2(x))
        out3 = F.relu(self.linear_out3(x))
        return out1, out2, out3
