import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, SDMolSupplier

import torch
from htmdmol.molecule import Molecule
from htmdmol.tools.voxeldescriptors import getVoxelDescriptors
from torch.utils.data import Dataset


def get_protein_features(usercoords, usercenters, userchannels, center=None, boxsize=[24]*3):
    if center is not None:
        # TODO rotations
        pass
    features = getVoxelDescriptors(mol=None,
                                   usercoords=usercoords,
                                   usercenters=usercenters,
                                   userchannels=userchannels)
    return features


def get_ligand_features(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, 2, nBits=1024, useChirality=True)
    arr = np.zeros((1,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class Featurizer(Dataset):
    def __init__(self, coords, grid_centers, channels, centers, ligands, rmsd_min, rmsd_ave, n_rmsd):
        self.coords = coords
        self.grid_centers = grid_centers
        self.channels = channels
        self.centers = centers
        self.ligands = ligands
        self.rmsd_min = rmsd_min
        self.rmsd_ave = rmsd_ave
        self.n_rmsd = n_rmsd

    def __getitem__(self, index):
        center = np.load(self.centers[index])
        ligand = next(SDMolSupplier(self.ligands[index]))
        rmsd_min = np.load(self.rmsd_min[index])
        rmsd_ave = np.load(self.rmsd_ave[index])
        n_rmsd = np.load(self.n_rmsd[index])
        prot_feat, _ = get_protein_features(usercoords=np.load(self.coords[index]),
                                            usercenters=np.load(
                                                self.grid_centers[index]),
                                            userchannels=np.load(self.channels[index]))
        lig_feat = get_ligand_features(ligand)
        return prot_feat, lig_feat, rmsd_min, rmsd_ave, n_rmsd

    def __len__(self):
        return len(self.coords)


if __name__ == '__main__':
    import os
    from cnndockbench import home
    from cnndockbench.utils import get_data
    path = os.path.join(home(), 'data')
    coords, grid_centers, channels, ligands, centers, rmsd_min, rmsd_ave, n_rmsd = get_data(
        path)

    feat = Featurizer(coords, grid_centers, channels, centers,
                      ligands, rmsd_min, rmsd_ave, n_rmsd)
