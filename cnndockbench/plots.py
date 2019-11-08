import os

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from train import EVAL_MODES, N_SPLITS
from utils import home

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


RES_DIR = os.path.join(home(), 'results')
PLOT_DIR = os.path.join(home(), 'plots')

NAMES_EVAL_MODES = {
    'random': 'random',
    'ligand_scaffold': 'ligand scaffold',
    'protein_classes': 'protein classes',
    'protein_classes_distribution': 'protein classes balanced'
}

def ligand_res(rmsd_test, rmsd_pred, mask):
    r_ts = []
    r_ps = []
    for sample in range(rmsd_test.shape[0]):
        r_t, r_p = (rmsd_test[sample, :])[mask[sample, :].astype(np.bool)].tolist(), \
                   (rmsd_pred[sample, :])[mask[sample, :].astype(np.bool)].tolist()
        r_ts.extend(r_t)
        r_ps.extend(r_p)
    return r_ts, r_ps


if __name__ == "__main__":
    ligand_true = {}
    ligand_pred = {}

    for mode in EVAL_MODES:
        ligand_true.setdefault(mode, [])
        ligand_pred.setdefault(mode, [])

        for split_no in range(N_SPLITS):
            mask = np.load(os.path.join(RES_DIR, 'mask_{}_{}.npy'.format(mode, split_no))).astype(np.bool)
            rmsd_ave_test = np.load(os.path.join(RES_DIR, 'rmsd_ave_test_{}_{}.npy'.format(mode, split_no)))
            rmsd_ave_pred = np.load(os.path.join(RES_DIR, 'rmsd_ave_pred_{}_{}.npy'.format(mode, split_no)))
            r_ts, r_ps = ligand_res(rmsd_ave_test, rmsd_ave_pred, mask)
            ligand_true[mode].extend(r_ts)
            ligand_pred[mode].extend(r_ps)

    f, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    idx = 0

    for row in range(2):
        for col in range(2):
            mode = EVAL_MODES[idx]
            lt = ligand_true[mode]
            lp = ligand_pred[mode]
            ax[row, col].set_axisbelow(True)
            ax[row, col].grid()
            ax[row, col].scatter(lt, lp, s=0.2, c='slategrey')
            ax[row, col].set_xlim(0, 25)
            ax[row, col].set_ylim(0, 25)
            ax[row, col].set_title(NAMES_EVAL_MODES[mode])

            idx += 1

    f.text(0.5, 0.04, r'Experimental $\mathrm{RMSD}_{\mathrm{ave}}$', ha='center')
    f.text(0.04, 0.45, r'Predicted $\mathrm{RMSD}_{\mathrm{ave}}$', ha='center', rotation='vertical')

    # plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'ligand_eval.png'), format='png')
    plt.close()

    # plt.show()