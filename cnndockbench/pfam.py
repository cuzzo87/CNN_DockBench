import os
import pickle
from glob import glob

import numpy as np
from tqdm import tqdm

from cnndockbench.train import EVAL_MODES, N_SPLITS
from cnndockbench.eval import ligand_eval
from cnndockbench.utils import home

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


def pfam_population(pdbid_pfam_map, split_pdbids):
    pdbids = [pdbid for sublist in split_pdbids['random'] for pdbid in sublist]
    families = [pdbid_pfam_map[pdbid] if pdbid in pdbid_pfam_map else None for pdbid in pdbids]

    population_family = {}

    for family in families:
        population_family.setdefault(family, 0)
        population_family[family] += 1
    return population_family


if __name__ == "__main__":
    pdbid_pfam_map = build_pfam_map(PFAM_FILES)

    with open(os.path.join(RES_PATH, 'split_pdbids.pt'), 'rb') as handle:
        split_pdbids = pickle.load(handle)

    corr_family = {}
    rmse_family = {}

    for mode in EVAL_MODES:
        corr_family.setdefault(mode, {})
        rmse_family.setdefault(mode, {})

        for split_no in range(N_SPLITS):
            pdbids = split_pdbids[mode][split_no]
            families = [pdbid_pfam_map[pdbid] if pdbid in pdbid_pfam_map else None for pdbid in pdbids]
            mask = np.load(os.path.join(RES_PATH, 'mask_{}_{}.npy'.format(mode, split_no))).astype(np.bool)
            rmsd_ave_test = np.load(os.path.join(RES_PATH, 'rmsd_ave_test_{}_{}.npy'.format(mode, split_no)))
            rmsd_ave_pred = np.load(os.path.join(RES_PATH, 'rmsd_ave_pred_{}_{}.npy'.format(mode, split_no)))
            corrs, _, rmses = ligand_eval(rmsd_ave_test, rmsd_ave_pred, mask)

            for family, corr, rmse in zip(families, corrs, rmses):
                corr_family[mode].setdefault(family, [])
                rmse_family[mode].setdefault(family, [])
                corr_family[mode][family].append(corr)
                rmse_family[mode][family].append(rmse)

    # Average results
    corr_avg_family = {}
    corr_std_family = {}

    rmse_avg_family = {}
    rmse_std_family = {}

    for mode in EVAL_MODES:
        corr_avg_family.setdefault(mode, {})
        corr_std_family.setdefault(mode, {})

        rmse_avg_family.setdefault(mode, {})
        rmse_std_family.setdefault(mode, {})

        for (family, corr), (_, rmse) in zip(corr_family[mode].items(), rmse_family[mode].items()):
            corr_avg_family[mode][family] = np.mean(corr)
            corr_std_family[mode][family] = np.std(corr)

            rmse_avg_family[mode][family] = np.mean(rmse)
            rmse_std_family[mode][family] = np.std(rmse) 

    with open(os.path.join(RES_PATH, 'corr_avg_family.pt'), 'wb') as handle:
        pickle.dump(corr_avg_family, handle)

    with open(os.path.join(RES_PATH, 'corr_std_family.pt'), 'wb') as handle:
        pickle.dump(corr_std_family, handle)

    with open(os.path.join(RES_PATH, 'rmse_avg_family.pt'), 'wb') as handle:
        pickle.dump(rmse_avg_family, handle)

    with open(os.path.join(RES_PATH, 'rmse_std_family.pt'), 'wb') as handle:
        pickle.dump(rmse_std_family, handle)

    # Family population
    population_family = pfam_population(pdbid_pfam_map, split_pdbids)

    with open(os.path.join(RES_PATH, 'pfam_population.pt'), 'wb') as handle:
        pickle.dump(population_family, handle)
