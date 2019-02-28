import os
from glob import glob

import numpy as np

NEEDED_FILES = ['protein.mol2', 'ligand.sdf', 'center.npy',
                'rmsd_min.npy', 'rmsd_ave.npy', 'n_rmsd.npy']


def geom_center(mol):
    """
    Returns molecule geometric center
    """
    return np.mean(np.squeeze(mol.coords), axis=0)


def get_data(path):
    """
    Returns paths for all available data.
    """
    proteins = []
    ligands = []
    centers = []
    rmsd_min = []
    rmsd_ave = []
    n_rmsd = []

    for subfolder in glob(os.path.join(path, '*/')):
        all_available = True
        available_files = set([os.path.basename(f) for f in glob(os.path.join(subfolder, '*'))])
        for need_file in NEEDED_FILES:
            if need_file not in available_files:
                all_available = False

        if all_available:
            proteins.append(os.path.join(subfolder, 'protein.mol2'))
            ligands.append(os.path.join(subfolder, 'ligand.sdf'))
            centers.append(os.path.join(subfolder, 'center.npy'))
            rmsd_min.append(os.path.join(subfolder, 'rmsd_min.npy'))
            rmsd_ave.append(os.path.join(subfolder, 'rmsd_ave.npy'))
            n_rmsd.append(os.path.join(subfolder, 'n_rmsd.npy'))
    return proteins, ligands, centers, rmsd_min, rmsd_ave, n_rmsd
