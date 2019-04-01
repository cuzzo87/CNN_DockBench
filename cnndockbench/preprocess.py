import os
import re
import shutil
from glob import glob

import numpy as np
from rdkit.Chem import SDMolSupplier
from tqdm import tqdm

from cnndockbench import home
from cnndockbench.utils import geom_center
from htmdmol.molecule import Molecule
from htmdmol.tools.atomtyper import prepareProteinForAtomtyping
from htmdmol.tools.voxeldescriptors import getCenters, getChannels

DATA_PATH = os.path.join(home(), 'cases')
OUTDIR = os.path.join(home(), 'data')
REGEX_PATTERN = r'\w*\-\w*_min\-\w*-\w*'
PROTOCOLS = ['autodock-ga', 'autodock-lga', 'autodock-ls', 'glide-sp', 'gold-asp', 'gold-chemscore',
             'gold-goldscore', 'gold-plp', 'moe-AffinitydG', 'moe-GBVIWSA', 'moe-LondondG',
             'plants-chemplp', 'plants-plp95', 'plants-plp', 'rdock-solv', 'rdock-std', 'vina-std']
N_PROTOCOLS = len(PROTOCOLS)


def build_guide(path):
    """
    Builds a convenient guide for preprocessing data.
    """
    cases = glob(os.path.join(path, '*/'))
    pat = re.compile(REGEX_PATTERN)

    guide = {}

    for case in cases:
        ref_file = os.path.join(case, 'RMSD.txt')

        with open(ref_file, 'r+') as handle:
            lines = handle.readlines()
            for i, line in enumerate(lines):
                current_pdbid = None
                res = pat.match(line)
                if res is not None:
                    match = res.group().split('-')
                    resname = match[0]
                    pdbid = match[1].split('_')[0]
                    protocol = match[2] + '-' + match[3]

                    rmsd_min = float(
                        lines[i + 2].split(':')[1].strip('\n').strip('\t'))
                    rmsd_ave = float(
                        lines[i + 3].split(':')[1].strip('\n').strip('\t'))
                    if rmsd_min == 99 or rmsd_ave == 99:
                        continue
                    n_rmsd = int(lines[i + 4].split(':')
                                 [1].strip('\n').strip('\t'))

                    resolution = float(
                        lines[i + 5].split(':')[1].split(' ')[0].strip('\t'))

                    if pdbid not in guide:
                        guide[pdbid] = {}

                    guide[pdbid]['case'] = os.path.basename(
                        os.path.dirname(case))
                    guide[pdbid]['resname'] = resname
                    guide[pdbid]['resolution'] = resolution
                    guide[pdbid][protocol] = (rmsd_min, rmsd_ave, n_rmsd)

    # Iterate over guide to remove missing cases
    to_exclude = []

    for pdbid in guide:
        if len(guide[pdbid]) != N_PROTOCOLS + 3:
            to_exclude.append(pdbid)

    for exclude in to_exclude:
        guide.pop(exclude, None)
    return guide


def clean_data(guide, path, outpath):
    """
    Creates subfolder structure.
    """
    protein_exclude = []
    ligand_exclude = []

    for pdbid in tqdm(guide.keys()):
        case = os.path.join(path, guide[pdbid]['case'])
        cocrystal_dir = os.path.join(case, 'cocrystals')
        ligand_dir = os.path.join(case, 'ligands')
        receptor_dir = os.path.join(case, 'receptors')

        os.makedirs(os.path.join(outpath, pdbid), exist_ok=True)

        # Get pocket center
        cocrystal = Molecule(os.path.join(
            cocrystal_dir, '{}.pdb'.format(pdbid)), keepaltloc="all")
        ligand = cocrystal.copy()
        ligand.filter('resname {}'.format(guide[pdbid]['resname'].upper()))
        center = geom_center(ligand)

        # Featurize protein and copy ligand in their corresponding folder
        protein_path = os.path.join(receptor_dir, '{}.mol2'.format(pdbid))
        ligand_path = os.path.join(
            ligand_dir, '{}-{}_min.sdf'.format(guide[pdbid]['resname'], pdbid))
        protein = Molecule(protein_path)
        protein.filter('protein')
        protein = prepareProteinForAtomtyping(protein, verbose=False)
        if protein.atomselect('not protein').sum():
            protein_exclude.append(pdbid)
            continue

        ligand = next(SDMolSupplier(ligand_path))
        if ligand is None:
            ligand_exclude.append(pdbid)
            continue

        grid_centers, _ = getCenters(protein, boxsize=[24]*3, center=center)
        channels, _ = getChannels(protein)

        np.save(os.path.join(outpath, pdbid, 'center.npy'), arr=center)
        np.save(os.path.join(outpath, pdbid, 'coords.npy'), arr=protein.coords)
        np.save(os.path.join(outpath, pdbid, 'grid_centers.npy'), arr=grid_centers)
        np.save(os.path.join(outpath, pdbid, 'channels.npy'), arr=channels)

        shutil.copy(ligand_path, os.path.join(outpath, pdbid, 'ligand.sdf'))

        # Correctly format protocols
        rmsd_min = []
        rmsd_ave = []
        n_rmsd = []
        for protocol in PROTOCOLS:
            rmsd_min.append(guide[pdbid][protocol][0])
            rmsd_ave.append(guide[pdbid][protocol][1])
            n_rmsd.append(guide[pdbid][protocol][2])

        resolution = np.array(guide[pdbid]['resolution'])

        np.save(os.path.join(outpath, pdbid, 'rmsd_min.npy'), arr=rmsd_min)
        np.save(os.path.join(outpath, pdbid, 'rmsd_ave.npy'), arr=rmsd_ave)
        np.save(os.path.join(outpath, pdbid, 'n_rmsd.npy'), arr=n_rmsd)
        np.save(os.path.join(outpath, pdbid, 'resolution.npy'), arr=resolution)
    return protein_exclude, ligand_exclude

if __name__ == '__main__':
    import sys
    print('Cleaning input data...')
    outdir = sys.argv[1]
    os.makedirs(outdir, exist_ok=True)
    guide = build_guide(DATA_PATH)
    protein_exclude, ligand_exclude = clean_data(guide, DATA_PATH, outdir)
    if len(protein_exclude) > 0:
        print('Several proteins could not be processed: {}'.format(protein_exclude))

    if len(ligand_exclude) > 0:
        print('Several ligands could not be read by RDkit: {}'.format(ligand_exclude))
