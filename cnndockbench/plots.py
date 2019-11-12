import os
import pickle

import numpy as np
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from train import EVAL_MODES, N_SPLITS
from utils import home

from matplotlib import rc
# rc('font',**{'family':'sans-serif', 'sans-serif':['Helvetica'], 'size': 6})
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


def ligand_plot():
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

    f, ax = plt.subplots(nrows=2, ncols=2, dpi=400, sharex=True,
                         sharey=True)
    idx = 0

    for row in range(2):
        for col in range(2):
            mode = EVAL_MODES[idx]
            lt = ligand_true[mode]
            lp = ligand_pred[mode]
            ax[row, col].set_axisbelow(True)
            ax[row, col].grid()
            ax[row, col].scatter(lt, lp, s=0.05, c='slategrey')
            ax[row, col].set_xlim(0, 25)
            ax[row, col].set_ylim(0, 25)
            ax[row, col].set_title(NAMES_EVAL_MODES[mode])
            x0, x1 = ax[row, col].get_xlim()
            y0, y1 = ax[row, col].get_ylim()
            ax[row, col].set_aspect(abs(x1 - x0) / abs(y1 - y0))
            idx += 1

    f.text(0.5, 0.04, r'Experimental $\mathrm{RMSD}_{\mathrm{ave}}$', ha='center')
    f.text(0.04, 0.45, r'Predicted $\mathrm{RMSD}_{\mathrm{ave}}$', ha='center', rotation='vertical')

    plt.savefig(os.path.join(PLOT_DIR, 'ligand_eval.png'), format='png')
    plt.close()


def family_plot(num_families=30):
    with open(os.path.join(RES_DIR, 'avg_family.pt'), 'rb') as handle:
        avg_family = pickle.load(handle)
    
    with open(os.path.join(RES_DIR, 'std_family.pt'), 'rb') as handle:
        std_family = pickle.load(handle)

    with open(os.path.join(RES_DIR, 'pfam_population.pt'), 'rb') as handle:
        population_family = pickle.load(handle)

    population_family.pop(None)
    sorted_population = sorted(population_family.items(), key=lambda kv: kv[1], reverse=True)
    selected_families = [t[0] for t in sorted_population[:num_families]]

    f, ax = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(12, 8))
    idx = 0

    for row in range(2):
        for col in range(2):
            mode = EVAL_MODES[idx]
            avg_top = [avg_family[mode][fam] for fam in selected_families]
            std_top = [std_family[mode][fam] for fam in selected_families]
            x = np.arange(len(avg_top))

            ax[row, col].bar(x, avg_top, yerr=std_top, color='slategrey')
            ax[row, col].set_xticks(np.arange(len(x)))
            ax[row, col].set_xticklabels(selected_families, rotation=65)
            ax[row, col].set_title(NAMES_EVAL_MODES[mode])
            ax[row, col].yaxis.grid(True)

            idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'family_eval.pdf'), format='pdf')
    plt.close()

if __name__ == "__main__":
    # ligand_plot()
    family_plot()
