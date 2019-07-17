import os

import numpy as np
import torch
from rdkit.Chem import SDMolSupplier
from torch.utils.data import DataLoader

from moleculekit.molecule import Molecule
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.tools.voxeldescriptors import getCenters, getChannels
from preprocess import BOXSIZE
from prod_utils import FeaturizerProd, prod_loop
from train import BATCH_SIZE, DEVICE, NUM_WORKERS
from utils import home


class DockNet:
    def __init__(self, pdb, ligands, x=None, y=None, z=None):
        self.pdb = pdb
        self.ligands = ligands
        self.center = (x, y, z)
        self.compute_center = True

        self.sanity_checks()

        self.process_ligands()
        if self.compute_center:
            self.geom_center()

        self.process_protein()

    def process_protein(self):
        protein = Molecule(self.pdb)
        protein.filter('protein')
        protein = prepareProteinForAtomtyping(protein, verbose=False)
        self.coords = protein.coords
        self.grid_centers, _ = getCenters(
            protein, boxsize=[BOXSIZE]*3, center=self.center)
        self.channels, _ = getChannels(protein)

    def process_ligands(self):
        if self.ftype == 'sdf':
            self.mols = SDMolSupplier(self.ligands)
        else:
            with open(self.ligands, 'r+') as handle:
                self.ligands = handle.readlines()
            self.ligands.split('\n')
            self.mols = [sm.strip('\n') for sm in self.ligands]

    def run_net(self):
        featurizer = FeaturizerProd(
            self.coords, self.grid_centers, self.channels, self.mols)
        loader = DataLoader(featurizer,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            shuffle=False)
        model = torch.load(os.path.join(
            home(), 'models', 'production.pt')).to(DEVICE)
        return prod_loop(loader, model)

    def geom_center(self):
        centers = []
        for mol in SDMolSupplier(self.ligands):
            centers.append(mol.GetConformer().GetPositions().mean(axis=0))
        self.center = np.mean(centers, axis=0).astype(np.float32)

    def sanity_checks(self):
        if self.ligands.endswith('.sdf'):
            self.ftype = 'sdf'
        elif self.ligands.endswith('.smi'):
            self.ftype = 'smi'
        else:
            raise ValueError('Ligand filetype needs to be either .sdf or .smi')

        if self.ftype == 'smi' and self.center == (None, None, None):
            raise ValueError('If an .smi file is supplied,',
                             'coordinates for the center of the binding pocket',
                             '(x, y, z) need to be provided.')

        if self.ftype == 'sdf' and self.center is all([isinstance(x, float) for x in self.center]):
            self.compute_center = False
            self.center = np.array(self.center, dtype=np.float32)


if __name__ == '__main__':
    pdb = '/shared/jose/janssen/pde2_glide/protein.pdb'
    ligands = '/shared/jose/janssen/pde2_glide/pde2_clean.sdf'
    dn = DockNet(pdb, ligands)
