import os

import numpy as np
import pandas as pd

from cnndockbench import home

RES_DIR = os.path.join(home(), 'results')
EVAL_MODES = ['random', 'ligand_scaffold']
N_SPLITS = 5


def compute_score(rmsd_ave, n_rmsd, resolution, n_complex):
    """
    Computes target score. @cuzzo check this out please!
    """
    add_one = (rmsd_ave < resolution).astype(int)
    add_two = (n_rmsd > 10).astype(int)
    score = add_one + add_two
    indices = np.where((rmsd_ave == np.min(rmsd_ave, axis=1).reshape(n_complex, 1)) & (n_rmsd == np.max(n_rmsd, axis=1).reshape(n_complex, 1)))
    score[indices] += 1
    return score


if __name__ == '__main__':
    # results = {}
    # for mode in EVAL_MODES:
    #     results[mode] = []

    for mode in EVAL_MODES:
        for split in range(N_SPLITS):
            resolution = np.load(os.path.join(RES_DIR, 'resolution_{}_{}.npy'.format(mode, split)))
            # test set
            # rmsd_min_test = np.load(os.path.join(RES_DIR, 'rmsd_min_test_{}_{}.npy'.format(mode, split)))
            rmsd_ave_test = np.load(os.path.join(RES_DIR, 'rmsd_ave_test_{}_{}.npy'.format(mode, split)))
            n_rmsd_test = np.load(os.path.join(RES_DIR, 'n_rmsd_test_{}_{}.npy'.format(mode, split)))
            # prediction
            # rmsd_min_pred = np.load(os.path.join(RES_DIR, 'rmsd_min_pred_{}_{}.npy'.format(mode, split)))
            rmsd_ave_pred = np.load(os.path.join(RES_DIR, 'rmsd_ave_pred_{}_{}.npy'.format(mode, split)))
            n_rmsd_pred = np.load(os.path.join(RES_DIR, 'n_rmsd_pred_{}_{}.npy'.format(mode, split)))

            n_complex = rmsd_ave_pred.shape[0]
            n_protocols = rmsd_ave_pred.shape[1]
            resolution = np.transpose(np.tile(resolution, (n_protocols, 1)))

            score_test = compute_score(rmsd_ave_test, n_rmsd_test, resolution, n_complex)
            score_pred = compute_score(rmsd_ave_pred, n_rmsd_pred, resolution, n_complex)

            # TODO: compute multiclass classification metrics with sklearn
