import numpy as np

import torch
from htmdmol.molecule import Molecule


def geom_center(mol):
    return np.mean(np.squeeze(mol.coords), axis=0)
