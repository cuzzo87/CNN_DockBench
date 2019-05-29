import numpy as np
import torch
from torch import nn
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, SDMolSupplier
from torch.utils.data import Dataset

from moleculekit.molecule import Molecule
from moleculekit.tools.voxeldescriptors import getVoxelDescriptors

from preprocess import FAIL_FLAG


class Featurizer(Dataset):
    def __init__(self, coords, grid_centers, channels, centers,
                 coords_ligand, grid_centers_ligand, channels_ligand, centers_ligand,
                 rmsd_min, rmsd_ave, n_rmsd):
        """
        Produces voxels for protein pocket and fingerprint for ligand.
        """
        self.coords = coords
        self.grid_centers = grid_centers
        self.channels = channels
        self.centers = centers
        self.coords_ligand = coords_ligand
        self.grid_centers_ligand = grid_centers_ligand
        self.channels_ligand = channels_ligand
        self.centers_ligand = centers_ligand
        self.rmsd_min = rmsd_min
        self.rmsd_ave = rmsd_ave
        self.n_rmsd = n_rmsd

    def __getitem__(self, index):
        rmsd_min = np.load(self.rmsd_min[index]).astype(np.float32)
        rmsd_ave = np.load(self.rmsd_ave[index]).astype(np.float32)
        n_rmsd = np.load(self.n_rmsd[index]).astype(np.float32)
        prot_feat, _ = get_voxel_features(usercoords=np.load(self.coords[index]),
                                            usercenters=np.load(self.grid_centers[index]),
                                            userchannels=np.load(self.channels[index]),
                                            rotate_over=np.load(self.centers[index]))
        prot_feat = np.transpose(prot_feat.reshape((24, 24, 24, 8)),
                                 axes=(3, 0, 1, 2)).astype(np.float32)
        lig_feat = get_voxel_features(usercoords=np.load(self.coords_ligand[index]),
                                            usercenters=np.load(self.grid_centers_ligand[index]),
                                            userchannels=np.load(self.channels_ligand[index]),
                                            rotate_over=np.load(self.centers_ligand[index]))
        lig_feat = np.transpose(lig_feat.reshape((24, 24, 24, 8)),
                                 axes=(3, 0, 1, 2)).astype(np.float32)
        mask = (rmsd_min != FAIL_FLAG).astype(np.uint8)
        return torch.from_numpy(prot_feat), torch.from_numpy(lig_feat),\
               torch.from_numpy(rmsd_min), torch.from_numpy(rmsd_ave),\
               torch.from_numpy(n_rmsd), torch.from_numpy(mask)

    def __len__(self):
        return len(self.coords)


class CombinedLoss(nn.Module):
    """
    MSELoss for rmsd_min / rmsd_ave and PoissonNLLLoss for n_rmsd
    """
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.loss_rmsd_min = nn.MSELoss()
        self.loss_rmsd_ave = nn.MSELoss()
        self.loss_n_rmsd = nn.PoissonNLLLoss(log_input=True, full=True)

    def forward(self, out1, out2, out3, rmsd_min, rmsd_ave, n_rmsd):
        loss_rmsd_min = self.loss_rmsd_min(out1, rmsd_min)
        loss_rmsd_ave = self.loss_rmsd_ave(out2, rmsd_ave)
        loss_n_rmsd = self.loss_n_rmsd(out3, n_rmsd)
        return loss_rmsd_min, loss_rmsd_ave, loss_n_rmsd


def get_voxel_features(usercoords, usercenters, userchannels, rotate_over=None, boxsize=[24]*3):
    """
    Featurizes protein pocket using 3D voxelization
    """
    if rotate_over is not None:
        alpha = np.random.uniform(0, 2 * np.pi)
        beta = np.random.uniform(0, 2 * np.pi)
        gamma = np.random.uniform(0, 2 * np.pi)        
        
        usercoords = (usercoords.squeeze() - rotate_over)
        usercoords = usercoords.T.copy()
        Rz = np.array([[np.cos(alpha), -np.sin(alpha), 0], [np.sin(alpha), np.cos(alpha), 0], [0, 0, 1]])
        Ry = np.array([[np.cos(beta), 0, np.sin(beta)], [0, 1, 0], [-np.sin(beta), 0, np.cos(beta)]])
        Rx = np.array([[1, 0, 0], [0, np.cos(gamma), -np.sin(gamma)], [0, np.sin(gamma), np.cos(gamma)]])
        R = Rz @ Ry @ Rx
        usercoords = ((R @ usercoords).T.copy() + rotate_over).astype(np.float32)

    features = getVoxelDescriptors(mol=None,
                                   usercoords=usercoords,
                                   usercenters=usercenters,
                                   userchannels=userchannels)
    return features
