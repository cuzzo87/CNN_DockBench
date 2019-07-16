import os

from tqdm impor tqdm
import numpy as np
from rdkit.Chem import MolFromSmiles
from torch.utils.data import Dataset
from net_utils import get_protein_features, get_ligand_features


class FeaturizerProd(Dataset):
    def __init__(self, coords, grid_centers, channels, mols):
        self.coords = coords
        self.grid_centers = grid_centers
        self.channels = channels
        self.mols = mols
        if isinstance(self.mols, list):
            self.mols = iter(self.mols)
        
        self.desc_cols = np.load(os.path.join(os.path.dirname(__file__), 'data', 'desc_cols.npy'))
        self.avg_feat = np.load(os.path.join(os.path.dirname(__file__), 'data', 'avg.npy'))
        self.std_feat = np.load(os.path.join(os.path.dirname(__file__), 'data', 'std.npy'))

        self.prot_feat, _ = get_protein_features(usercoords=coords,
                                                 usercenters=grid_centers,
                                                 userchannels=channels)
        self.prot_feat = np.transpose(self.prot_feat.reshape((24, 24, 24, 8)),
                                      axes=(3, 0, 1, 2)).astype(np.float32)

    def __getitem__(self, _):
        mol = next(self.mols)
        if isinstance(mol, str):
            mol = MolFromSmiles(mol)
        fp, desc = get_ligand_features(mol)
        std_desc = (desc[self.desc_cols] - self.avg_feat) / self.std_feat
        lig_feat = np.concatenate((fp, std_desc))
        return self.prot_feat, lig_feat


def prod_loop(loader, model):
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