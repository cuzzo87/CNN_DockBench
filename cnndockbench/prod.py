import os
import logging, warnings

import numpy as np
import pandas as pd
import torch
from rdkit.Chem import MolFromSmiles
from torch.utils.data import DataLoader

from moleculekit.molecule import Molecule
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.tools.voxeldescriptors import getCenters, getChannels
from cnndockbench.preprocess import BOXSIZE, PROTOCOLS
from cnndockbench.train import BATCH_SIZE, DEVICE, NUM_WORKERS
from cnndockbench.utils import home
from cnndockbench.utils_prod import FeaturizerProd, prod_loop


logging.basicConfig(level=logging.INFO, format='%(asctime)s || %(name)s | %(levelname)s: %(message)s',  datefmt='%Y/%m/%d %I:%M:%S %p')
LOGGER = logging.getLogger('DockBenchPred')
warnings.filterwarnings('ignore')


class DockNet:
    def __init__(self, pdb, ligands, x, y, z):
        """
        Base production class.
        """
        self.pdb = pdb
        self.ligands = ligands
        self.center = (x, y, z)

        LOGGER.info('Performing sanity checks...')
        self.sanity_checks()

        LOGGER.info('Processing ligands...')
        self.process_ligands()
        LOGGER.info('Preparing the protein...')
        self.process_protein()

    def process_protein(self):
        """
        Prepares input protein for atomtyping and creates necessary
        files for voxelization.
        """
        protein = Molecule(self.pdb)
        protein.filter('protein')
        protein = prepareProteinForAtomtyping(protein, verbose=False)
        self.coords = protein.coords
        self.grid_centers, _ = getCenters(
            protein, boxsize=[BOXSIZE]*3, center=np.array(self.center, dtype=np.float32))
        self.channels, _ = getChannels(protein)

    def process_ligands(self):
        """
        Processes input ligands and removes those that cannot be
        read by rdkit.
        """
        with open(self.ligands, 'r+') as handle:
            self.ligands = handle.readlines()
        self.mols = [sm.strip('\n') for sm in self.ligands]
        if self.mols[-1] == '':
            self.mols.pop(-1)
        
        # Remove ligands that cannot be read by rdkit
        self.mols = [m for m in self.mols if MolFromSmiles(m) is not None]

    def run_net(self):
        """
        Creates featurizer objects and outputs network predictions.
        """
        LOGGER.info('Now predicting...')
        featurizer = FeaturizerProd(self.coords, self.grid_centers, self.channels, self.mols)
        loader = DataLoader(featurizer,
                            batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS,
                            shuffle=False)
        model = torch.load(os.path.join(home(), 'models', 'production.pt'), map_location=DEVICE)
        rmsd_min, rmsd_ave, n_rmsd = prod_loop(loader, model)
        return self.prettify_res(rmsd_min, rmsd_ave, n_rmsd)

    def prettify_res(self, rmsd_min, rmsd_ave, n_rmsd):
        """
        Converts results to pandas dataframes for easier readability.
        """
        rmsd_min_df = pd.DataFrame(rmsd_min.numpy(), columns=PROTOCOLS, index=self.mols)
        rmsd_ave_df = pd.DataFrame(rmsd_ave.numpy(), columns=PROTOCOLS, index=self.mols)
        n_rmsd_df = pd.DataFrame(n_rmsd.numpy(), columns=PROTOCOLS, index=self.mols)
        return rmsd_min_df, rmsd_ave_df, n_rmsd_df

    def sanity_checks(self):
        """
        Simple sanity checks for input and types.
        """
        if not self.ligands.endswith('.smi'):
            raise ValueError('Ligand filetype needs to be .smi')
        if not all([isinstance(c, float) for c in self.center]):
            raise ValueError('x, y and z need to be floats')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Input for DockBenchPred')
    parser.add_argument('-pdb', dest='pdb', type=str, required=True, help='Path to the protein .pdb file.')
    parser.add_argument('-smi', dest='smi', type=str, required=True, help='Path to a ligands .smi file.')
    parser.add_argument('-x', dest='x', type=float, required=True, help='x pocket location')
    parser.add_argument('-y', dest='y', type=float, required=True, help='y pocket location')
    parser.add_argument('-z', dest='z', type=float, required=True, help='z pocket location')
    parser.add_argument('-output', dest='output', type=str, required=False, default='.', help='Location to store results. Defaults to local directory.')

    args = parser.parse_args()

    dn = DockNet(pdb=args.pdb, ligands=args.smi, x=args.x, y=args.y, z=args.z)
    rmsd_min, rmsd_ave, n_rmsd = dn.run_net()

    os.makedirs(args.output, exist_ok=True)
    rmsd_min.to_csv(os.path.join(args.output, 'rmsd_min.csv'))
    rmsd_ave.to_csv(os.path.join(args.output, 'rmsd_ave.csv'))
    n_rmsd.to_csv(os.path.join(args.output, 'n_rmsd.csv'))
