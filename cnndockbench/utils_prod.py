import os

import numpy as np
import torch
from rdkit.Chem import MolFromSmiles
from torch.utils.data import Dataset
from tqdm import tqdm

from cnndockbench.net_utils import get_ligand_features, get_protein_features
from cnndockbench.train import DEVICE


class FeaturizerProd(Dataset):
    """
    Similar to the featurizer in `utils.py`, but without returning
    target values. It also only considers a single protein pocket
    and reuses its featurization.
    """
    def __init__(self, coords, grid_centers, channels, mols):
        self.coords = coords
        self.grid_centers = grid_centers
        self.channels = channels
        self.mols = mols

        
        self.desc_cols = np.load(os.path.join(os.path.dirname(__file__), 'data', 'desc_cols.npy'))
        self.avg_feat = np.load(os.path.join(os.path.dirname(__file__), 'data', 'avg.npy'))
        self.std_feat = np.load(os.path.join(os.path.dirname(__file__), 'data', 'std.npy'))

        self.prot_feat, _ = get_protein_features(usercoords=coords,
                                                 usercenters=grid_centers,
                                                 userchannels=channels)
        self.prot_feat = np.transpose(self.prot_feat.reshape((24, 24, 24, 8)),
                                      axes=(3, 0, 1, 2)).astype(np.float32)

    def __getitem__(self, index):
        # TODO: augmentation
        mol = MolFromSmiles(self.mols[index])
        fp, desc = get_ligand_features(mol)
        std_desc = (desc[self.desc_cols] - self.avg_feat) / self.std_feat
        lig_feat = np.concatenate((fp, std_desc))
        return self.prot_feat, lig_feat

    def __len__(self):
        return len(self.mols)


def prod_loop(loader, model):
    """
    Evaluation loop for production.
    """
    model = model.eval()
    progress = tqdm(loader)

    rmsd_min_pred = []
    rmsd_ave_pred = []
    n_rmsd_pred = []

    for voxel, fp in progress:
        with torch.no_grad():
            voxel = voxel.to(DEVICE)
            fp = fp.to(DEVICE)

            out1, out2, out3 = model(voxel, fp)
            out3 = torch.round(torch.exp(out3)).clamp(max=20).type(torch.int)

            rmsd_min_pred.append(out1.cpu())
            rmsd_ave_pred.append(out2.cpu())
            n_rmsd_pred.append(out3.cpu())
    return torch.cat(rmsd_min_pred), torch.cat(rmsd_ave_pred), torch.cat(n_rmsd_pred)
