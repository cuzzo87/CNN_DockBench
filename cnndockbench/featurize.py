import numpy as np
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from htmdmol.tools.voxeldescriptors import getPointDescriptors


def getProteinFeatures(mol, center, box=[24]*3):
    features = getPointDescriptors(mol, center, box)
    return features


def getLigandFeatures(molfile):
    if molfile.endswith('.sdf'):
        ligmol = Chem.SDMolSupplier(molfile, removeHs=False)[0]
    elif molfile.endswith('.mol2'):
        ligmol = Chem.MolFromMol2File(molfile, removeHs=False)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        ligmol, 2, nBits=1024, useChirality=True)
    arr = np.zeros((1,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr
