import os
from glob import glob

import numpy as np

from rdkit import DataStructs
from rdkit.Chem import AllChem, SDMolSupplier

from sklearn.model_selection import KFold
from sklearn.cluster import MiniBatchKMeans


FP_SIZE = 1024
REQUIRED_FILES = ['coords.npy', 'grid_centers.npy', 'channels.npy', 'ligand.sdf',
                  'center.npy', 'rmsd_min.npy', 'rmsd_ave.npy', 'n_rmsd.npy', 'resolution.npy']


def home():
    return os.path.dirname(__file__)


def geom_center(mol):
    """
    Returns molecule geometric center
    """
    return np.mean(np.squeeze(mol.coords), axis=0)


def check_required_files(path, required_files):
    """
    Checks the required files are available in the folder
    before retrieving them.
    """
    all_available = True
    available_files = set([os.path.basename(f) for f in glob(os.path.join(path, '*'))])
    for req_file in required_files:
        if req_file not in available_files:
            all_available = False
    return all_available


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
    resolution = []

    for subfolder in glob(os.path.join(path, '*/')):
        if check_required_files(subfolder, REQUIRED_FILES):
            coords.append(os.path.join(subfolder, 'coords.npy'))
            grid_centers.append(os.path.join(subfolder, 'grid_centers.npy'))
            channels.append(os.path.join(subfolder, 'channels.npy'))
            ligands.append(os.path.join(subfolder, 'ligand.sdf'))
            centers.append(os.path.join(subfolder, 'center.npy'))
            rmsd_min.append(os.path.join(subfolder, 'rmsd_min.npy'))
            rmsd_ave.append(os.path.join(subfolder, 'rmsd_ave.npy'))
            n_rmsd.append(os.path.join(subfolder, 'n_rmsd.npy'))
            resolution.append(os.path.join(subfolder, 'resolution.npy'))
    return coords, grid_centers, channels, centers, ligands, rmsd_min, rmsd_ave, n_rmsd, resolution


class Splitter:
    def __init__(self, coords, grid_centers, channels, centers, ligands, rmsd_min, rmsd_ave, n_rmsd, resolution,
                 n_splits=5, method='random', random_state=1337):
        """
        Base class for splitting data into train and test sets.
        """
        self.coords = np.array(coords)
        self.grid_centers = np.array(grid_centers)
        self.channels = np.array(channels)
        self.centers = np.array(centers)
        self.ligands = np.array(ligands)
        self.rmsd_min = np.array(rmsd_min)
        self.rmsd_ave = np.array(rmsd_ave)
        self.n_rmsd = np.array(n_rmsd)
        self.resolution = np.array(resolution)
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_samples = len(self.grid_centers)

        assert len(self.coords) == len(self.channels) == len(self.ligands) == len(self.centers) == \
            len(self.rmsd_min) == len(self.rmsd_ave) == len(self.resolution)

        if method == 'random':
            self._random_split()

        elif method == 'ligand_scaffold':
            self._ligand_scaffold_split()
        else:
            raise ValueError('Splitting procedure not recognised. Available ones: {}'.format(self._available_methods()))

    def _random_split(self):
        """
        Random split into train/test set
        """
        all_indices = np.arange(0, self.n_samples)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        self.splits = list(kf.split(all_indices))

    def _ligand_scaffold_split(self):
        """
        Ligand-based scaffold split using Morgan fingerprints
        and k-means clustering.
        """
        km = MiniBatchKMeans(n_clusters=self.n_splits, random_state=self.random_state)
        feat = np.zeros((self.n_samples, 1024), dtype=np.uint8)

        for idx, sdf in enumerate(self.ligands):
            mol = next(SDMolSupplier(sdf))
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=FP_SIZE)
            arr = np.zeros((1,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            feat[idx] = arr.copy()

        labels = km.fit_predict(feat)
        self.splits = []
        for split_no in range(self.n_splits):
            indices_train = np.where(labels != split_no)[0]
            indices_test = np.where(labels == split_no)[0]
            self.splits.append((indices_train, indices_test))

    def get_split(self, split_no, mode='random'):
        train_idx, test_idx = self.splits[split_no] 
        return (self.coords[train_idx], self.grid_centers[train_idx], self.channels[train_idx],
                self.centers[train_idx], self.ligands[train_idx], self.rmsd_min[train_idx],
                self.rmsd_ave[train_idx], self.n_rmsd[train_idx]), (self.coords[test_idx],
                self.grid_centers[test_idx], self.channels[test_idx], self.centers[test_idx],
                self.ligands[test_idx], self.rmsd_min[test_idx], self.rmsd_ave[test_idx],
                self.n_rmsd[test_idx])

    def _available_methods(self):
        return ['random', 'ligand_scaffold']
