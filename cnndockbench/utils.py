from htmdmol.molecule import Molecule
import numpy as np
import torch

def getCenters(prot, ligname):

    m = Molecule(prot)
    center = np.mean(m.coords[m.resname == ligname.upper()], axis=0)

    return center.reshape(3).tolist()

def getTensor(narray, atype, device):
    narray = narray.astype(atype)

    t = torch.from_numpy(narray)
    return  t.to(device)