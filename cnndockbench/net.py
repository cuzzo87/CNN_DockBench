import torch
import torch.nn as nn
import torch.nn.functional as F

from cnndockbench.preprocess import N_PROTOCOLS


class TwoLegs(nn.Module):
    def __init__(self, desc_size=1207, channels=8):
        super(TwoLegs, self).__init__()
        self.desc_size = desc_size
        self.channels = channels

        self.linear_fp1 = nn.Linear(desc_size, 1024)
        self.linear_fp2 = nn.Linear(1024, 512)
        self.linear_fp3 = nn.Linear(512, 512)

        self.conv_voxel1 = nn.Conv3d(self.channels, 64, kernel_size=3)
        self.conv_voxel2 = nn.Conv3d(64, 64, kernel_size=3)
        self.max_voxel1 = nn.MaxPool3d(kernel_size=3)
        self.conv_voxel3 = nn.Conv3d(64, 64, kernel_size=1)
        self.conv_voxel4 = nn.Conv3d(64, 64, kernel_size=3)
        self.conv_voxel5 = nn.Conv3d(64, 64, kernel_size=3)

        self.linear_cat1 = nn.Linear(1024, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.linear_out1 = nn.Linear(128, N_PROTOCOLS)
        self.linear_out2 = nn.Linear(128, N_PROTOCOLS)
        self.linear_out3 = nn.Linear(128, N_PROTOCOLS)

    def _ligand_forward(self, fp):
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
        protein_out = self._pocket_forward(voxel)
        ligand_out = self._ligand_forward(fp)
        x = torch.cat([protein_out, ligand_out], dim=1)
        x = self.linear_cat1(x)
        x = self.bn1(x)
        out1 = F.relu(self.linear_out1(x))
        out2 = F.relu(self.linear_out2(x))
        out3 = F.relu(self.linear_out3(x))
        return out1, out2, out3
