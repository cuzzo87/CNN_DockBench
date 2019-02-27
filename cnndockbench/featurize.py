import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from htmdmol.molecule import Molecule
from htmdmol.tools.voxeldescriptors import getPointDescriptors


def getProteinFeatures(molfile, center, box=[24]*3, channels=None):
    if channels is None:
        channels = list(range(8))
    m = Molecule(molfile)
    features = getPointDescriptors(m, center, box)
    return features[:, :, :, channels]


def getLigandFeatures(molfile):
    if molfile.endswith('.sdf'):
        ligmol = Chem.SDMolSupplier(molfile, removeHs=False)[0]
    elif molfile.endswith('.mol2'):
        ligmol = Chem.MolFromMol2File(molfile, removeHs=False)

    if ligmol is None:
        return ligmol
    fp = AllChem.GetMorganFingerprintAsBitVect(
        ligmol, 2, nBits=1024, useChirality=True)
    arr = np.zeros((1,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
