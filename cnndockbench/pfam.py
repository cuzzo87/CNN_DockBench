import os
import pickle
from glob import glob

import numpy as np
from tqdm import tqdm

from train import EVAL_MODES, N_SPLITS
from eval import ligand_eval
from utils import home

CLASSES_PATH = os.path.join(home(), 'proteinClasses')
RES_PATH = os.path.join(home(), 'results')
PFAM_FILES = glob(os.path.join(CLASSES_PATH, '*.list'))


def build_pfam_map(pfam_files):
    pdbid_pfam_map = {}

    for pfam_file in tqdm(pfam_files):
        pfam_family = os.path.basename(pfam_file).strip('.list')
        with open(pfam_file, 'r+') as handle:
            pdbids = handle.readlines()
        
        pdbids = [pdbid.strip('\n').lower() for pdbid in pdbids]

        for pdbid in pdbids:
            pdbid_pfam_map[pdbid] = pfam_family

    with open(os.path.join(CLASSES_PATH, 'pdbid_pfam_map.pt'), 'wb') as handle:
        pickle.dump(pdbid_pfam_map, handle)
    return pdbid_pfam_map


if __name__ == "__main__":
    pdbid_pfam_map = build_pfam_map(PFAM_FILES)

    with open(os.path.join(RES_PATH, 'split_pdbids.pt'), 'rb') as handle:
        split_pdbids = pickle.load(handle)

    results_family = {}

    for mode in EVAL_MODES:
        results_family.setdefault(mode, {})
        for split_no in range(N_SPLITS):
            pdbids = results_family[mode][split_no]
            families = [pdbid_pfam_map[pdbid] for pdbid in pdbids]
            mask = np.load(os.path.join(RES_PATH, 'mask_{}_{}.npy'.format(mode, split_no))).astype(np.bool)
            rmsd_ave_test = np.load(os.path.join(RES_PATH, 'rmsd_ave_test_{}_{}.npy'.format(mode, split_no)))
            rmsd_ave_pred = np.load(os.path.join(RES_PATH, 'rmsd_ave_pred_{}_{}.npy'.format(mode, split_no)))
            corrs, _ = ligand_eval(rmsd_ave_test, rmsd_ave_pred, mask)

            for family, corr in zip(families, corrs):
                results_family[mode].setdefault(family, [])
                results_family[mode][family].append(corr)
