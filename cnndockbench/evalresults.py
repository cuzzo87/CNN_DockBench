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


def metrics(score_test, score_pred, strategy='weighted'):
    """
    Computes standard multiclass classification metrics per protocol.
    """
    n_protocols = len(PROTOCOLS)
    accs = [accuracy_score(score_test[:, i], score_pred[:, i]) for i in range(n_protocols)]
    precs = [precision_score(score_test[:, i], score_pred[:, i], average=strategy) for i in range(n_protocols)]
    recs = [recall_score(score_test[:, i], score_pred[:, i], average=strategy) for i in range(n_protocols)]
    f1s = [f1_score(score_test[:, i], score_pred[:, i], average=strategy) for i in range(n_protocols)]
    mccs = [matthews_corrcoef(score_test[:, i], score_pred[:, i]) for i in range(n_protocols)]
    return accs, precs, recs, mccs


if __name__ == '__main__':
    results = {}

    for mode in EVAL_MODES:
        for split in range(N_SPLITS):
            resolution = np.load(os.path.join(RES_DIR, 'resolution_{}_{}.npy'.format(mode, split)))

            rmsd_ave_test = np.load(os.path.join(RES_DIR, 'rmsd_ave_test_{}_{}.npy'.format(mode, split)))
            n_rmsd_test = np.load(os.path.join(RES_DIR, 'n_rmsd_test_{}_{}.npy'.format(mode, split)))

            rmsd_ave_pred = np.load(os.path.join(RES_DIR, 'rmsd_ave_pred_{}_{}.npy'.format(mode, split)))
            n_rmsd_pred = np.load(os.path.join(RES_DIR, 'n_rmsd_pred_{}_{}.npy'.format(mode, split)))

            n_complex = rmsd_ave_pred.shape[0]
            n_protocols = rmsd_ave_pred.shape[1]
            resolution = np.transpose(np.tile(resolution, (n_protocols, 1)))

            score_test = compute_score(rmsd_ave_test, n_rmsd_test, resolution, n_complex)
            score_pred = compute_score(rmsd_ave_pred, n_rmsd_pred, resolution, n_complex)

            accs, precs, recs, mccs = metrics(score_test, score_pred)

            results.setdefault(mode, {})

            for i, protocol in enumerate(PROTOCOLS):
                results[mode].setdefault(protocol, {})
                results[mode][protocol].setdefault('acc', []).append(accs[i])
                results[mode][protocol].setdefault('prec', []).append(precs[i])
                results[mode][protocol].setdefault('rec', []).append(recs[i])
                results[mode][protocol].setdefault('mcc', []).append(mccs[i])

    with open(os.path.join(RES_DIR, 'results.pkl'), 'wb') as handle:
        pickle.dump(results, handle)
