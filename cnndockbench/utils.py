import os
import numpy as np
from glob import glob

import numpy as np

NEEDED_FILES = ['coords.npy', 'grid_centers.npy', 'channels.npy', 'ligand.sdf',
                'center.npy', 'rmsd_min.npy', 'rmsd_ave.npy', 'n_rmsd.npy']


def geom_center(mol):
    """
    Returns molecule geometric center
    """
    return np.mean(np.squeeze(mol.coords), axis=0)


def get_data(path):
    """
    Returns paths for all available data.
    """
    coords = []
    grid_centers = []
    channels = []
    centers = []
    ligands = []
    rmsd_min = []
    rmsd_ave = []
    n_rmsd = []

    for subfolder in glob(os.path.join(path, '*/')):
        all_available = True
        available_files = set([os.path.basename(f)
                               for f in glob(os.path.join(subfolder, '*'))])
        for need_file in NEEDED_FILES:
            if need_file not in available_files:
                all_available = False

        if all_available:
            coords.append(os.path.join(subfolder, 'coords.npy'))
            grid_centers.append(os.path.join(subfolder, 'grid_centers.npy'))
            channels.append(os.path.join(subfolder, 'channels.npy'))
            ligands.append(os.path.join(subfolder, 'ligand.sdf'))
            centers.append(os.path.join(subfolder, 'center.npy'))
            rmsd_min.append(os.path.join(subfolder, 'rmsd_min.npy'))
            rmsd_ave.append(os.path.join(subfolder, 'rmsd_ave.npy'))
            n_rmsd.append(os.path.join(subfolder, 'n_rmsd.npy'))
    return coords, grid_centers, channels, centers, ligands, rmsd_min, rmsd_ave, n_rmsd


class Splitter:
    def __init__(self, coords, grid_centers, channels, centers, ligands, rmsd_min, rmsd_ave, n_rmsd):
        self.coords = np.array(coords)
        self.grid_centers = np.array(grid_centers)
        self.channels = np.array(channels)
        self.centers = np.array(centers)
        self.ligands = np.array(ligands)
        self.rmsd_min = np.array(rmsd_min)
        self.rmsd_ave = np.array(rmsd_ave)
        self.n_rmsd = np.array(n_rmsd)
        self.n = len(self.grid_centers)

        assert len(self.coords) == len(self.channels) == len(self.ligands) == len(self.centers) == \
            len(self.rmsd_min) == len(self.rmsd_ave)

    def _random_split(self, p):
        all_indexes = np.arange(0, self.n)
        test_indexes = np.random.choice(all_indexes, size=int(p * self.n))
        train_indexes = np.setdiff1d(all_indexes, test_indexes)
        return (self.coords[train_indexes].tolist(), self.grid_centers[train_indexes].tolist(), self.channels[train_indexes].tolist(),
                self.centers[train_indexes].tolist(), self.ligands[train_indexes].tolist(), self.rmsd_min[train_indexes].tolist(),
                self.rmsd_ave[train_indexes].tolist(), self.n_rmsd[train_indexes].tolist()), (self.coords[test_indexes].tolist(),
                self.grid_centers[test_indexes].tolist(), self.channels[test_indexes].tolist(),
                self.centers[test_indexes].tolist(), self.ligands[test_indexes].tolist(), self.rmsd_min[test_indexes].tolist(),
                self.rmsd_ave[test_indexes].tolist(), self.n_rmsd[test_indexes].tolist())

    def _ligand_scaffold_split(self, p):
        # TODO
        pass

    def split(self, p, mode='random'):
        if mode == 'random':
            return self._random_split(p)

        elif mode == 'ligand_scaffold':
            return self._ligand_scaffold_split(p)

        else:
            raise ValueError('Splitting procedure not recognised. Available ones: {}'.format(self._available_methods()))

    def _available_methods(self):
        return ['random', 'ligand_scaffold']
