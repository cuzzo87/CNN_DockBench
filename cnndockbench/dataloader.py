import os
from glob import glob

import numpy as np
import tqdm

from cnndockbench import home
from cnndockbench.utils import getCenters
from htmdmol.molecule import Molecule

PROTOCOLS = {'autodock-ga': 0,
             'autodock-lga': 1,
             'autodock-ls': 2,
             'glide-sp': 3,
             'gold-asp': 4,
             'gold-chemscore': 5,
             'gold-goldscore': 6,
             'gold-plp': 7,
             'moe-AffinitydG': 8,
             'moe-GBVIWSA': 9,
             'moe-LondondG': 10,
             'plants-chemplp': 11,
             'plants-plp': 12,
             'plants-plp95': 13,
             'rdock-solv': 14,
             'rdock-std': 15,
             'vina-std': 16}


def loadDatasets():
    dataFolder = home('Cases')
    datasets = glob(dataFolder + '/*')
    sdatasets = {}
    for d in datasets:
        basename = os.path.basename(d)
        _id, _system, _nproteins, _nprotocols = basename.split('_')
        l, p, r, d, = loadDataSet(d, int(_nprotocols))
        sdatasets[_id] = [_system, _nproteins, _nprotocols, l, p, r, d]
    return sdatasets


def loadDataSet(dataset, nprotocols):
    listFiles = glob(dataset + '/*')

    ligands = getLigands(listFiles[0])
    pdbs = getComplexes(listFiles[1])
    receptors = getProteins(listFiles[2])
    data = getData(listFiles[3], nprotocols)

    ligandsSorted, centers, receptorsSorted, dataSorted = sortDataset(
        ligands, pdbs, receptors, data)

    return ligandsSorted, centers, receptorsSorted, dataSorted


def sortDataset(ligs, pdbs, recs, data):
    sligs = []
    spbds = []
    srecs = []
    sdata = []

    for complex, value in data.items():
        l, p = complex.split('-')

        ligFile = [_ for _ in ligs if l in _][0]
        protFile = [_ for _ in recs if p in _][0]
        pdbFile = [_ for _ in pdbs if p in _][0]

        sligs.append(ligFile)
        spbds.append(pdbFile)
        srecs.append(protFile)
        sdata.append(value)
    lignames = [os.path.basename(l).split('-')[0].upper() for l in sligs]
    centers = [getCenters(p, l) for p, l in zip(spbds, lignames)]
    return sligs, centers, srecs, np.array(sdata)


def getData(dataFile, nprotocols):
    # if docking went wrong we assigned 99 as values. How do we have to deal with this?
    complexes = {}
    header = False
    curr_complex = None
    curr_protocol = None
    for n, line in enumerate(open(dataFile, 'r')):
        if len(line) == 2:
            header = True
            continue
        if header:
            header = False
            structures, protocol = line.split('_')
            ligname, protein = structures.split('-')
            protocol = protocol.strip().replace('min-', '')
            curr_protocol = protocol
            curr_complex = structures
            if curr_complex not in complexes:
                complexes[curr_complex] = np.zeros((3, nprotocols))

        if line.startswith('RMSDmin'):
            complexes[curr_complex][0][PROTOCOLS[curr_protocol]
                                       ] = float(line.strip().split()[-1])
        if line.startswith('RMSDave'):
            complexes[curr_complex][1][PROTOCOLS[curr_protocol]
                                       ] = float(line.strip().split()[-1])
        if line.startswith('N(RMSD<R)'):
            complexes[curr_complex][2][PROTOCOLS[curr_protocol]
                                       ] = float(line.strip().split()[-1])

    return complexes


def getLigands(folderpath):
    l_files = glob(folderpath + '/*.sdf')
    return l_files


def getProteins(folderpath):
    p_files = glob(folderpath + '/*.mol2')
    return p_files


def getComplexes(folderpath):
    p_files = glob(folderpath + '/*.pdb')
    return p_files


def getComplexesCenters(folderpath, ligands, proteins):
    pdbs = glob(folderpath + '/*.pdb')

    complexes = []
    for l in tqdm(ligands):
        tmp = []
        lname, p = os.path.splitext(l)[0].split('-')
        lname = os.path.basename(lname)
        p = p.split('_')[0]
        tmp.append([_ for _ in proteins if p in _][0])
        tmp.append(l)
        pdb = [_p for _p in pdbs if p in _p][0]
        m = Molecule(pdb)
        center = np.mean(m.coords[m.resname == lname.upper()], axis=0)
        tmp.append(center.reshape(3).tolist())
        complexes.append(tmp)
    return complexes
