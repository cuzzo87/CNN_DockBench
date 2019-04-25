import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score)

from cnndockbench import home
from cnndockbench.preprocess import PROTOCOLS
from cnndockbench.train import EVAL_MODES, N_SPLITS

RES_DIR = os.path.join(home(), 'results')


def rmse(y, ypred):
    return np.sqrt(np.mean((y - ypred)**2))


def corr(y, ypred):
    return np.corrcoef((y, ypred))[0, 1]


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


def regression_metrics(rmsd_test, rmsd_pred):
    n_protocols = len(PROTOCOLS)
    rmses = [rmse(rmsd_test[:, i], rmsd_pred[:, i]) for i in range(n_protocols)]
    corrs = [corr(rmsd_test[:, i], rmsd_pred[:, i]) for i in range(n_protocols)]
    return rmses, corrs


def ordinal_metrics(score_test, score_pred):
    #TODO: check https://github.com/ayrna/orca#performance-metrics
    pass


if __name__ == '__main__':
    results = {}

    for mode in EVAL_MODES:
        for split_no in range(N_SPLITS):
            resolution = np.load(os.path.join(RES_DIR, 'resolution_{}_{}.npy'.format(mode, split_no)))

            rmsd_ave_test = np.load(os.path.join(RES_DIR, 'rmsd_ave_test_{}_{}.npy'.format(mode, split_no)))
            n_rmsd_test = np.load(os.path.join(RES_DIR, 'n_rmsd_test_{}_{}.npy'.format(mode, split_no)))

            rmsd_ave_pred = np.load(os.path.join(RES_DIR, 'rmsd_ave_pred_{}_{}.npy'.format(mode, split_no)))
            n_rmsd_pred = np.round(np.load(os.path.join(RES_DIR, 'n_rmsd_pred_{}_{}.npy'.format(mode, split_no)))).astype(np.int32)

            n_complex = rmsd_ave_pred.shape[0]
            n_protocols = rmsd_ave_pred.shape[1]
            resolution = np.transpose(np.tile(resolution, (n_protocols, 1)))

            score_test = compute_score(rmsd_ave_test, n_rmsd_test, resolution, n_complex)
            score_pred = compute_score(rmsd_ave_pred, n_rmsd_pred, resolution, n_complex)

            rmses_ave, corrs_ave = regression_metrics(rmsd_ave_test, rmsd_ave_pred)

            results.setdefault(mode, {})

            for i, protocol in enumerate(PROTOCOLS):
                results[mode].setdefault(protocol, {})
                results[mode][protocol].setdefault('rmse_ave', []).append(rmses_ave[i])
                results[mode][protocol].setdefault('corr_ave', []).append(corrs_ave[i])


    with open(os.path.join(RES_DIR, 'results.pkl'), 'wb') as handle:
        pickle.dump(results, handle)
