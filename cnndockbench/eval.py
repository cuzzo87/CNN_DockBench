import os
import pickle

import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score

from preprocess import PROTOCOLS
from train import EVAL_MODES, N_SPLITS
from utils import home


RES_DIR = os.path.join(home(), 'results')


class MeanError:
    """
    Convenient class for handling micro and macro
    averaging of errors.
    """
    def __init__(self, metric='mae', strategy='micro'):
        self.strategy = strategy
        self.base = self.mae if metric == 'mae' else self.rmse

    def mae(self, y, ypred):
        """
        Mean absolute error between `y` and `ypred`.
        """
        return np.mean(abs(y - ypred))

    def rmse(self, y, ypred):
        """
        Mean squared error between `y` and `ypred`.
        """
        return np.sqrt(np.mean((y - ypred)**2))

    def compute(self, y, ypred):
        if self.strategy == 'micro':
            return self.base(y, ypred)
        elif self.strategy == 'macro':
            n_classes = np.unique(y)
            measures = []
            for class_ in n_classes:
                y_ = y[y == class_]
                ypred_ = ypred[y == class_]
                measures.append(self.base(y_, ypred_))
            return np.mean(measures)


def corr(y, ypred):
    """
    Pearson corr. coef. wrapper
    """
    return np.corrcoef((y, ypred))[0, 1]


def compute_score(rmsd_ave, n_rmsd, resolution, n_complex):
    """
    Computes dockbench score.
    """
    add_one = (rmsd_ave < resolution).astype(int)
    add_two = (n_rmsd > 10).astype(int)
    score = add_one + add_two
    indices = np.where((rmsd_ave == np.min(rmsd_ave, axis=1).reshape(n_complex, 1)) &
                       (n_rmsd == np.max(n_rmsd, axis=1).reshape(n_complex, 1)))
    score[indices] += 1
    return score


def regression_metrics(rmsd_test, rmsd_pred, mask):
    """
    Computes root mean squared error and correlation for `rmsd_test`
    and `rmsd_pred` only using the samples marked by `mask`.
    """
    rmses = []
    corrs = []

    for idx_protocol in range(len(PROTOCOLS)):
        r_t, r_p = (rmsd_test[:, idx_protocol])[mask[:, idx_protocol].astype(np.bool)], \
                   (rmsd_pred[:, idx_protocol])[mask[:, idx_protocol].astype(np.bool)]
        rmses.append(MeanError(metric='rmse').compute(r_t, r_p))
        u_t, u_p = np.unique(r_t), np.unique(r_p)
        if len(np.intersect1d(u_t, u_p)) == 1:
            corrs.append(1.0)
            continue
        elif len(u_t) == 1 or len(u_p) == 1:
            corrs.append(0.0)
            continue
        else:
            corrs.append(corr(r_t, r_p))
    return rmses, corrs


def ordinal_metrics(score_test, score_pred, mask):
    """
    Computes ordinal metrics such as micro and macro mean absolute
    and squared errors for the available `score_test` and `score_pred`
    dockbench scores using `mask`-
    """
    # TODO: check https://github.com/ayrna/orca#performance-metrics
    mae_micros = []
    mae_macros = []
    rmse_micros = []
    rmse_macros = []
    rhos = []
    taus = []
    kappas_lin = []
    kappas_quad = []

    for idx_protocol in range(len(PROTOCOLS)):
        s_t, s_p = (score_test[:, idx_protocol])[mask[:, idx_protocol].astype(np.bool)], \
                   (score_pred[:, idx_protocol])[mask[:, idx_protocol].astype(np.bool)]
        # Check for more than unique value in s_p, otherwise rho and tau calculation fails
        mae_micros.append(MeanError(metric='mae', strategy='micro').compute(s_t, s_p))
        mae_macros.append(MeanError(metric='mae', strategy='macro').compute(s_t, s_p))
        rmse_micros.append(MeanError(metric='rmse', strategy='micro').compute(s_t, s_p))
        rmse_macros.append(MeanError(metric='rmse', strategy='macro').compute(s_t, s_p))
        u_t, u_p = np.unique(s_t), np.unique(s_p)
        if len(np.intersect1d(u_t, u_p)) == 1:
            rhos.append(1.0)
            taus.append(1.0)
            kappas_lin.append(1.0)
            kappas_quad.append(1.0)
            continue
        elif len(u_t) == 1 or len(u_p) == 1:
            rhos.append(0.0)
            taus.append(0.0)
            kappas_lin.append(0.0)
            kappas_quad.append(0.0)
            continue
        else:
            rhos.append(spearmanr(s_t, s_p, nan_policy='raise')[0])
            taus.append(kendalltau(s_t, s_p, nan_policy='raise')[0])
            kappas_lin.append(cohen_kappa_score(s_t, s_p, weights='linear'))
            kappas_quad.append(cohen_kappa_score(s_t, s_p, weights='quadratic'))
    return mae_micros, mae_macros, rmse_micros, rmse_macros, rhos, taus, kappas_lin, kappas_quad


if __name__ == '__main__':
    results = {}
    np.seterr(all='ignore')

    for mode in EVAL_MODES:
        for split_no in range(N_SPLITS):
            resolution = np.load(os.path.join(RES_DIR, 'resolution_{}_{}.npy'.format(mode, split_no)))
            mask = np.load(os.path.join(RES_DIR, 'mask_{}_{}.npy'.format(mode, split_no))).astype(np.bool)

            rmsd_ave_test = np.load(os.path.join(RES_DIR, 'rmsd_ave_test_{}_{}.npy'.format(mode, split_no)))
            n_rmsd_test = np.load(os.path.join(RES_DIR, 'n_rmsd_test_{}_{}.npy'.format(mode, split_no)))

            rmsd_ave_pred = np.load(os.path.join(RES_DIR, 'rmsd_ave_pred_{}_{}.npy'.format(mode, split_no)))
            n_rmsd_pred = np.load(os.path.join(RES_DIR, 'n_rmsd_pred_{}_{}.npy'.format(mode, split_no)))

            n_complex = rmsd_ave_pred.shape[0]
            n_protocols = rmsd_ave_pred.shape[1]
            resolution = np.transpose(np.tile(resolution, (n_protocols, 1)))

            score_test = compute_score(rmsd_ave_test, n_rmsd_test, resolution, n_complex)
            score_pred = compute_score(rmsd_ave_pred, n_rmsd_pred, resolution, n_complex)

            rmses_ave, corrs_ave = regression_metrics(rmsd_ave_test, rmsd_ave_pred, mask)
            mae_micros, mae_macros, rmse_micros, rmse_macros, rhos, taus, kappas_lin, kappas_quad = ordinal_metrics(score_test, score_pred, mask)

            results.setdefault(mode, {})

            for i, protocol in enumerate(PROTOCOLS):
                results[mode].setdefault(protocol, {})
                results[mode][protocol].setdefault('rmse_ave', []).append(rmses_ave[i])
                results[mode][protocol].setdefault('corr_ave', []).append(corrs_ave[i])
                results[mode][protocol].setdefault('mae_micro', []).append(mae_micros[i])
                results[mode][protocol].setdefault('mae_macro', []).append(mae_macros[i])
                results[mode][protocol].setdefault('rmse_micro', []).append(rmse_micros[i])
                results[mode][protocol].setdefault('rmse_macro', []).append(rmse_macros[i])
                results[mode][protocol].setdefault('rho_ave', []).append(rhos[i])
                results[mode][protocol].setdefault('tau_ave', []).append(taus[i])
                results[mode][protocol].setdefault('kappa_lin', []).append(kappas_lin[i])
                results[mode][protocol].setdefault('kappa_quad', []).append(kappas_quad[i])


    with open(os.path.join(RES_DIR, 'results.pkl'), 'wb') as handle:
        pickle.dump(results, handle)
