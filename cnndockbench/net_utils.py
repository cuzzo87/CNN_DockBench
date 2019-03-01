import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, SDMolSupplier

import torch
from htmdmol.molecule import Molecule
from htmdmol.tools.voxeldescriptors import getVoxelDescriptors
from torch.utils.data import Dataset


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
        rmsd_min = np.load(self.rmsd_min[index]).astype(np.float32)
        rmsd_ave = np.load(self.rmsd_ave[index]).astype(np.float32)
        n_rmsd = np.load(self.n_rmsd[index]).astype(np.uint8)
        prot_feat, _ = get_protein_features(usercoords=np.load(self.coords[index]),
                                            usercenters=np.load(self.grid_centers[index]),
                                            userchannels=np.load(self.channels[index]),
                                            rotate_over=center)
        prot_feat = np.transpose(prot_feat.reshape((24, 24, 24, 8)),
                                 axes=(3, 0, 1, 2)).astype(np.float32)
        lig_feat = get_ligand_features(ligand)
        return torch.from_numpy(prot_feat), torch.from_numpy(lig_feat),\
               torch.from_numpy(rmsd_min), torch.from_numpy(rmsd_ave),\
               torch.from_numpy(n_rmsd)

    def __len__(self):
        return len(self.coords)


def get_protein_features(usercoords, usercenters, userchannels, rotate_over=None, boxsize=[24]*3):
    """
    Featurizes protein pocket using 3D voxelization
    """
    if rotate_over is not None:
        # TODO rotations
        pass
    features = getVoxelDescriptors(mol=None,
                                   usercoords=usercoords,
                                   usercenters=usercenters,
                                   userchannels=userchannels)
    return features


def get_ligand_features(mol):
    """
    Featurizes ligand using a Morgan fingerprint
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, 2, nBits=1024, useChirality=True)
    arr = np.zeros((1,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

