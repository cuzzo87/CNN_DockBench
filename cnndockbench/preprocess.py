import os
import pickle
import re
import shutil
from glob import glob

import numpy as np
from rdkit.Chem import SDMolSupplier
from tqdm import tqdm

from moleculekit.molecule import Molecule
from moleculekit.tools.atomtyper import prepareProteinForAtomtyping
from moleculekit.tools.voxeldescriptors import getCenters, getChannels
from moleculekit.smallmol.smallmol import SmallMol
from utils import REQUIRED_FILES, check_required_files, geom_center, home


DATA_PATH = os.path.join(home(), 'cases')
OUTDIR = os.path.join(home(), 'data')
REGEX_PATTERN = r'\w*\-\w*_min\-\w*-\w*'
PROTOCOLS = ['autodock-ga', 'autodock-lga', 'autodock-ls', 'glide-sp', 'gold-asp', 'gold-chemscore',
             'gold-goldscore', 'gold-plp', 'moe-AffinitydG', 'moe-GBVIWSA', 'moe-LondondG',
             'plants-chemplp', 'plants-plp95', 'plants-plp', 'rdock-solv', 'rdock-std', 'vina-std']
SKIP_IDS = ['1r9l', '1px4', '2f2h', '3i3b', '3mv0', '3t09', '3t0d', '3t2q', '3vdb', '3f37', '3k1j', '3f34', '3dyo',
            '3muz', '3t08', '3t0b', '3vd4', '3vd9', '3f33']
N_PROTOCOLS = len(PROTOCOLS)
FAIL_FLAG = 99.0


def build_guide(path):
    """
    Builds a convenient guide for preprocessing data.
    """
    cases = glob(os.path.join(path, '*/'))
    pat = re.compile(REGEX_PATTERN)

    guide = {}

    not_provided_cases = []

    for case in cases:
        ref_file = os.path.join(case, 'RMSD.txt')
        if not os.path.exists(ref_file):
            continue

        with open(ref_file, 'r+') as handle:
            lines = handle.readlines()
            for i, line in enumerate(lines):
                res = pat.match(line)
                if res is not None:
                    match = res.group().split('-')
                    resname = match[0]
                    pdbid = match[1].split('_')[0]
                    protocol = match[2] + '-' + match[3]

                    if 'RESOLUTION' not in lines[i + 5]:
                        not_provided_cases.append(os.path.basename(
                            os.path.dirname(case)) + '-' + line)
                        continue

                    rmsd_min = float(
                        lines[i + 2].split(':')[1].strip('\n').strip('\t'))
                    rmsd_ave = float(
                        lines[i + 3].split(':')[1].strip('\n').strip('\t'))
                    if rmsd_min == FAIL_FLAG or rmsd_ave == FAIL_FLAG:
                        n_rmsd = 99
                    else:
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

    # Iterate over guide to remove incorrectly processed cases
    not_complete_cases = []
    for pdbid in guide:
        if len(guide[pdbid]) != N_PROTOCOLS + 3:
            not_complete_cases.append(pdbid)

    for pdbid in not_complete_cases:
        guide.pop(pdbid, None)

    # Check cases where everything is missing
    all_missing_cases = []
    for pdbid in guide:
        all_missing = True
        for protocol in PROTOCOLS:
            if guide[pdbid][protocol] != (FAIL_FLAG, FAIL_FLAG, int(FAIL_FLAG)):
                all_missing = False
        if all_missing:
            all_missing_cases.append(pdbid)

    for pdbid in all_missing_cases:
        guide.pop(pdbid, None)
    return guide, not_complete_cases, all_missing_cases, not_provided_cases


def clean_data(guide, path, outpath):
    """
    Creates subfolder structure.
    """
    protein_exclude = []
    ligand_exclude = []

    progress = tqdm(guide.keys())

    for pdbid in progress:
        progress.set_postfix({'pdbid': pdbid})
        pdboutdir = os.path.join(outpath, pdbid)
        if pdbid in SKIP_IDS:
            continue

        if check_required_files(pdboutdir, REQUIRED_FILES):
            continue

        case = os.path.join(path, guide[pdbid]['case'])
        cocrystal_dir = os.path.join(case, 'cocrystals')
        ligand_dir = os.path.join(case, 'ligands')
        receptor_dir = os.path.join(case, 'receptors')

        # Get pocket center
        try:
            cocrystal = Molecule(os.path.join(cocrystal_dir, '{}.pdb'.format(pdbid)), keepaltloc="all")
        except Exception as _:
            protein_exclude.append(pdbid)
            continue
        ligand = cocrystal.copy()
        ligand.filter('resname {}'.format(guide[pdbid]['resname'].upper()))
        if ligand.numAtoms == 0:
            ligand_exclude.append(pdbid)
            continue
        center_pocket = geom_center(ligand)

        # Featurize protein and copy ligand in their corresponding folder
        protein_path = os.path.join(receptor_dir, '{}.mol2'.format(pdbid))
        ligand_path = os.path.join(ligand_dir, '{}-{}_min.sdf'.format(guide[pdbid]['resname'], pdbid))
        protein = Molecule(protein_path)
        protein.filter('protein')
        try:
            protein = prepareProteinForAtomtyping(protein, verbose=False)
        except Exception as _:
            protein_exclude.append(pdbid)
            continue

        if protein.atomselect('not protein').sum():
            protein_exclude.append(pdbid)
            continue

        ligand = [mol for mol in SDMolSupplier(ligand_path)][0]
        try:
            ligand = SmallMol(ligand)
        except Exception as _:
            ligand_exclude.append(pdbid)
            continue

        center_ligand = geom_center(ligand)

        grid_centers, _ = getCenters(protein, boxsize=[24]*3, center=center_pocket)
        grid_centers_ligand, _ = getCenters(ligand, boxsize=[24]*3, center=center_ligand)

        try:
            channels, _ = getChannels(protein)
        except Exception as _:
            protein_exclude.append(pdbid)
            continue

        try:
            channels_ligand, _ = getChannels(ligand)
        except Exception as _:
            ligand_exclude.append(pdbid)

        os.makedirs(pdboutdir, exist_ok=True)

        np.save(os.path.join(pdboutdir, 'center.npy'), arr=center_pocket)
        np.save(os.path.join(pdboutdir, 'coords.npy'), arr=protein.coords)
        np.save(os.path.join(pdboutdir, 'grid_centers.npy'), arr=grid_centers)
        np.save(os.path.join(pdboutdir, 'channels.npy'), arr=channels)
        np.save(os.path.join(pdboutdir, 'center_ligand.npy'), arr=center_ligand)
        np.save(os.path.join(pdboutdir, 'coords_ligand.npy'), arr=ligand._coords)
        np.save(os.path.join(pdboutdir, 'grid_centers_ligand.npy'), arr=grid_centers_ligand)
        np.save(os.path.join(pdboutdir, 'channels_ligand.npy'), arr=channels_ligand)

        # Correctly format protocols
        rmsd_min = []
        rmsd_ave = []
        n_rmsd = []
        for protocol in PROTOCOLS:
            rmsd_min.append(guide[pdbid][protocol][0])
            rmsd_ave.append(guide[pdbid][protocol][1])
            n_rmsd.append(guide[pdbid][protocol][2])

        resolution = np.array(guide[pdbid]['resolution'])

        np.save(os.path.join(pdboutdir, 'rmsd_min.npy'), arr=rmsd_min)
        np.save(os.path.join(pdboutdir, 'rmsd_ave.npy'), arr=rmsd_ave)
        np.save(os.path.join(pdboutdir, 'n_rmsd.npy'), arr=n_rmsd)
        np.save(os.path.join(pdboutdir, 'resolution.npy'), arr=resolution)
    return protein_exclude, ligand_exclude


if __name__ == '__main__':
    print('Cleaning input data...')
    os.makedirs(OUTDIR, exist_ok=True)
    guide, not_complete_cases, all_missing_cases, not_provided_cases = build_guide(
        DATA_PATH)
    print('After cleaning input files, {} cases were correctly parsed.'.format(len(guide)),
          '{} contained incomplete docking cases, with ids: {}.'.format(
              len(not_complete_cases), not_complete_cases),
          '{} featured all missing cases, with ids: {}.'.format(
              len(all_missing_cases), all_missing_cases),
          '{} cases were in the files, but not provided, with matches: {}'.format(len(not_provided_cases), not_provided_cases), sep='\n')

    protein_exclude, ligand_exclude = clean_data(guide, DATA_PATH, OUTDIR)
    if protein_exclude:
        print('Several proteins were not correctly read, filtered or could not be featurized: {}'.format(
            protein_exclude))

    if ligand_exclude:
        print('Several ligands could not be read by RDkit: {}'.format(ligand_exclude))

    if SKIP_IDS:
        print('Some ids could not be processed for other reasons: {}'.format(SKIP_IDS))

    # Save failed cases for future reference
    with open(os.path.join(OUTDIR, 'not_complete.pkl'), 'wb') as handle:
        pickle.dump(not_complete_cases, handle)

    with open(os.path.join(OUTDIR, 'all_missing.pkl'), 'wb') as handle:
        pickle.dump(all_missing_cases, handle)

    with open(os.path.join(OUTDIR, 'protein_exclude.pkl'), 'wb') as handle:
        pickle.dump(protein_exclude, handle)

    with open(os.path.join(OUTDIR, 'ligand_exclude.pkl'), 'wb') as handle:
        pickle.dump(ligand_exclude, handle)

    with open(os.path.join(OUTDIR, 'other_exclude.pkl'), 'wb') as handle:
        pickle.dump(SKIP_IDS, handle)
