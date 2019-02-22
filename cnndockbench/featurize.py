from htmdmol.tools.voxeldescriptors import  getPointDescriptors
from htmdmol.molecule import Molecule
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def getProteinFeatures(molfile, center, box=[24 ,24 ,24], channels=[0 ,1 ,2 ,3 ,4 ,5 ,7]):
    m = Molecule(molfile)
    features = getPointDescriptors(m, center, box)
    return features[: ,: ,: ,channels]

def getLigandFeatures(molfile):
    try:
        ligmol = Chem.SDMolSupplier(molfile ,removeHs=False)[0]
    except:
        molfile.replace('.sdf', '.mol2')
        ligmol = Chem.MolFromMol2File(molfile, removeHs=False)

    if ligmol is None:
        return ligmol
    fp = AllChem.GetMorganFingerprintAsBitVect(ligmol, 2 ,nBits=1024 ,useChirality=True)
    bitstring = fp.ToBitString()
    intmap = map(int, bitstring)

    return np.array(list(intmap))
