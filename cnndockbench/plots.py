import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc

from cnndockbench.preprocess import PROTOCOLS, FAIL_FLAG
from cnndockbench.train import EVAL_MODES, N_SPLITS
from cnndockbench.utils import home

matplotlib.use("Agg")


rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 12})
rc("text", usetex=True)

RES_DIR = os.path.join(home(), "results")
PLOT_DIR = os.path.join(home(), "plots")

NAMES_EVAL_MODES = {
    "random": "random",
    "ligand_scaffold": "ligand scaffold",
    "protein_classes": "protein classes",
    "protein_classes_distribution": "protein classes balanced",
}


def ligand_res(rmsd_test, rmsd_pred, mask):
    r_ts = []
    r_ps = []
    for sample in range(rmsd_test.shape[0]):
        r_t, r_p = (
            (rmsd_test[sample, :])[mask[sample, :].astype(np.bool)].tolist(),
            (rmsd_pred[sample, :])[mask[sample, :].astype(np.bool)].tolist(),
        )
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
            mask = np.load(
                os.path.join(RES_DIR, "mask_{}_{}.npy".format(mode, split_no))
            ).astype(np.bool)
            rmsd_ave_test = np.load(
                os.path.join(RES_DIR, "rmsd_ave_test_{}_{}.npy".format(mode, split_no))
            )
            rmsd_ave_pred = np.load(
                os.path.join(RES_DIR, "rmsd_ave_pred_{}_{}.npy".format(mode, split_no))
            )
            r_ts, r_ps = ligand_res(rmsd_ave_test, rmsd_ave_pred, mask)
            ligand_true[mode].extend(r_ts)
            ligand_pred[mode].extend(r_ps)

    f, ax = plt.subplots(nrows=2, ncols=2, dpi=400, sharex=True, sharey=True)
    idx = 0

    for row in range(2):
        for col in range(2):
            mode = EVAL_MODES[idx]
            lt = ligand_true[mode]
            lp = ligand_pred[mode]
            ax[row, col].set_axisbelow(True)
            ax[row, col].grid()
            ax[row, col].scatter(lt, lp, s=0.05, c="slategrey")
            ax[row, col].set_xlim(0, 25)
            ax[row, col].set_ylim(0, 25)
            ax[row, col].set_title(NAMES_EVAL_MODES[mode])
            x0, x1 = ax[row, col].get_xlim()
            y0, y1 = ax[row, col].get_ylim()
            ax[row, col].set_aspect(abs(x1 - x0) / abs(y1 - y0))
            idx += 1

    f.text(0.5, 0.04, r"Experimental $\mathrm{RMSD}_{\mathrm{ave}}$", ha="center")
    f.text(
        0.04,
        0.45,
        r"Predicted $\mathrm{RMSD}_{\mathrm{ave}}$",
        ha="center",
        rotation="vertical",
    )

    plt.savefig(os.path.join(PLOT_DIR, "ligand_eval.png"), format="png")
    plt.close()


def family_plot(avg, std, outf, num_families=30, color="slategrey"):
    with open(os.path.join(RES_DIR, "pfam_population.pt"), "rb") as handle:
        population_family = pickle.load(handle)

    population_family.pop(None)
    sorted_population = sorted(
        population_family.items(), key=lambda kv: kv[1], reverse=True
    )
    selected_families = [t[0] for t in sorted_population[:num_families]]

    f, ax = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=(12, 8))
    idx = 0

    for row in range(2):
        for col in range(2):
            mode = EVAL_MODES[idx]
            avg_top = [avg[mode][fam] for fam in selected_families if fam in avg[mode]]
            std_top = [std[mode][fam] for fam in selected_families if fam in std[mode]]
            x = np.arange(len(avg_top))

            ax[row, col].bar(x, avg_top, yerr=std_top, color=color)
            ax[row, col].set_xticks(np.arange(len(x)))
            ax[row, col].set_xticklabels(selected_families, rotation=65)
            ax[row, col].set_title(NAMES_EVAL_MODES[mode])
            ax[row, col].yaxis.grid(True)

            idx += 1

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, outf), format="pdf")
    plt.close()
    # plt.show()


def desc_plot():
    rmsds_min = []
    rmsds_ave = []
    n_rmsds = []
    masks = []

    for split_no in range(N_SPLITS):
        rmsds_min.append(
            np.load(
                os.path.join(RES_DIR, "rmsd_min_test_random_{}.npy".format(split_no))
            )
        )
        rmsds_ave.append(
            np.load(
                os.path.join(RES_DIR, "rmsd_ave_test_random_{}.npy".format(split_no))
            )
        )
        n_rmsds.append(
            np.load(os.path.join(RES_DIR, "n_rmsd_test_random_{}.npy".format(split_no)))
        )
        masks.append(
            np.load(os.path.join(RES_DIR, "mask_random_{}.npy".format(split_no)))
        )

    rmsds_min = np.vstack(rmsds_min)
    rmsds_ave = np.vstack(rmsds_ave)
    n_rmsds = np.vstack(n_rmsds)
    masks = np.vstack(masks)

    protocols = []
    rmsd_min_long = []
    rmsd_ave_long = []
    n_rmsd_long = []

    for idx_protocol, protocol in enumerate(PROTOCOLS):
        rmsd_min_protocol = rmsds_min[
            masks[:, idx_protocol].astype(np.bool), idx_protocol
        ].tolist()
        rmsd_ave_protocol = rmsds_ave[
            masks[:, idx_protocol].astype(np.bool), idx_protocol
        ].tolist()
        n_rmsd_protocol = n_rmsds[
            masks[:, idx_protocol].astype(np.bool), idx_protocol
        ].tolist()
        rmsd_min_long.extend(rmsd_min_protocol)
        rmsd_ave_long.extend(rmsd_ave_protocol)
        n_rmsd_long.extend(n_rmsd_protocol)
        protocols.extend([protocol] * len(rmsd_min_protocol))

    df_desc = pd.DataFrame(
        {
            "protocol": protocols,
            "rmsd_min": rmsd_min_long,
            "rmsd_ave": rmsd_ave_long,
            "n_rmsd": n_rmsd_long,
        }
    )

    common_params = {
        "linewidth": 1,
        "fliersize": 1,
        "palette": "tab20",
        "showfliers": False,
    }

    f, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax0 = sns.boxplot(
        y="protocol", x="rmsd_min", data=df_desc, orient="h", ax=ax[0], **common_params
    )
    ax0.set(ylabel="", xlabel=r"$\mathrm{RMSD}_\mathrm{min}$")
    ax1 = sns.boxplot(
        y="protocol", x="rmsd_ave", data=df_desc, orient="h", ax=ax[1], **common_params
    )
    ax1.set(ylabel="", yticklabels=[], xlabel=r"$\mathrm{RMSD}_\mathrm{ave}$")
    ax2 = sns.boxplot(
        y="protocol", x="n_rmsd", data=df_desc, orient="h", ax=ax[2], **common_params
    )
    ax2.set(ylabel="", yticklabels=[], xlabel=r"$n\mathrm{RMSD}$")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rmsd_plot.pdf"))
    plt.close()


def rmsd_plot():
    # CNNDockbench evaluation (w. mode & split)
    dist_min_results = {}
    dist_ave_results = {}
    dist_n_rmsd_results = {}

    for mode in EVAL_MODES:
        dist_min_results.setdefault(mode, [])
        dist_ave_results.setdefault(mode, [])
        dist_n_rmsd_results.setdefault(mode, [])

        for split_no in range(N_SPLITS):
            rmsd_min_test = np.load(
                os.path.join(RES_DIR, "rmsd_min_test_{}_{}.npy".format(mode, split_no))
            )
            rmsd_ave_test = np.load(
                os.path.join(RES_DIR, "rmsd_ave_test_{}_{}.npy".format(mode, split_no))
            )
            n_rmsd_test = np.load(
                os.path.join(RES_DIR, "n_rmsd_test_{}_{}.npy".format(mode, split_no))
            )

            rmsd_min_pred = np.load(
                os.path.join(RES_DIR, "rmsd_min_pred_{}_{}.npy".format(mode, split_no))
            )
            rmsd_ave_pred = np.load(
                os.path.join(RES_DIR, "rmsd_ave_pred_{}_{}.npy".format(mode, split_no))
            )
            n_rmsd_pred = np.load(
                os.path.join(RES_DIR, "n_rmsd_pred_{}_{}.npy".format(mode, split_no))
            )

            argmin_min = np.argmin(rmsd_min_pred, axis=1)
            dist_min = np.array(
                [rmsd_min_test[idx, argmin_min[idx]] for idx in range(len(argmin_min))]
            )
            dist_min = dist_min[dist_min != FAIL_FLAG]

            dist_min_results[mode].extend(dist_min)

            argmin_ave = np.argmin(rmsd_ave_pred, axis=1)
            dist_ave = np.array(
                [rmsd_ave_test[idx, argmin_ave[idx]] for idx in range(len(argmin_ave))]
            )
            dist_ave = dist_ave[dist_ave != FAIL_FLAG]

            dist_ave_results[mode].extend(dist_ave)

            argmax_n_rsmd = np.argmax(n_rmsd_pred, axis=1)
            dist_n_rmsd = np.array(
                [
                    n_rmsd_test[idx, argmax_n_rsmd[idx]]
                    for idx in range(len(argmax_n_rsmd))
                ]
            )
            dist_n_rmsd = dist_n_rmsd[dist_n_rmsd != int(FAIL_FLAG)]

            dist_n_rmsd_results[mode].extend(dist_n_rmsd)

    # Protocol evaluation

    rmsds_min = []
    rmsds_ave = []
    n_rmsds = []
    masks = []

    for split_no in range(N_SPLITS):
        rmsds_min.append(
            np.load(
                os.path.join(RES_DIR, "rmsd_min_test_random_{}.npy".format(split_no))
            )
        )
        rmsds_ave.append(
            np.load(
                os.path.join(RES_DIR, "rmsd_ave_test_random_{}.npy".format(split_no))
            )
        )
        n_rmsds.append(
            np.load(os.path.join(RES_DIR, "n_rmsd_test_random_{}.npy".format(split_no)))
        )
        masks.append(
            np.load(os.path.join(RES_DIR, "mask_random_{}.npy".format(split_no)))
        )

    rmsds_min = np.vstack(rmsds_min)
    rmsds_ave = np.vstack(rmsds_ave)
    n_rmsds = np.vstack(n_rmsds)
    masks = np.vstack(masks)

    protocols = []
    rmsd_min_long = []
    rmsd_ave_long = []
    n_rmsd_long = []

    for idx_protocol, protocol in enumerate(PROTOCOLS):
        rmsd_min_protocol = rmsds_min[
            masks[:, idx_protocol].astype(np.bool), idx_protocol
        ].tolist()
        rmsd_ave_protocol = rmsds_ave[
            masks[:, idx_protocol].astype(np.bool), idx_protocol
        ].tolist()
        n_rmsd_protocol = n_rmsds[
            masks[:, idx_protocol].astype(np.bool), idx_protocol
        ].tolist()
        rmsd_min_long.extend(rmsd_min_protocol)
        rmsd_ave_long.extend(rmsd_ave_protocol)
        n_rmsd_long.extend(n_rmsd_protocol)
        protocols.extend([protocol] * len(rmsd_min_protocol))

    protocols_min = protocols[:]
    protocols_ave = protocols[:]
    protocols_n_rmsd = protocols[:]

    for mode in EVAL_MODES:
        rmsd_min_long.extend(dist_min_results[mode])
        rmsd_ave_long.extend(dist_ave_results[mode])
        n_rmsd_long.extend(dist_n_rmsd_results[mode])

        protocols_min.extend(
            ["This work ({})".format(NAMES_EVAL_MODES[mode])]
            * len(dist_min_results[mode])
        )
        protocols_ave.extend(
            ["This work ({})".format(NAMES_EVAL_MODES[mode])]
            * len(dist_ave_results[mode])
        )
        protocols_n_rmsd.extend(
            ["This work ({})".format(NAMES_EVAL_MODES[mode])]
            * len(dist_n_rmsd_results[mode])
        )

    df_min = pd.DataFrame({"protocol": protocols_min, "rmsd_min": rmsd_min_long,})

    df_ave = pd.DataFrame({"protocol": protocols_ave, "rmsd_ave": rmsd_ave_long,})

    df_n_rmsd = pd.DataFrame({"protocol": protocols_n_rmsd, "n_rmsd": n_rmsd_long,})

    common_params = {
        "linewidth": 1,
        "fliersize": 1,
        "palette": "tab20",
        "showfliers": False,
    }

    f, ax = plt.subplots(1, 3, figsize=(12, 6))
    ax0 = sns.boxplot(
        y="protocol", x="rmsd_min", data=df_min, orient="h", ax=ax[0], **common_params
    )
    ax0.set(ylabel="", xlabel=r"$\mathrm{RMSD}_\mathrm{min}$")
    ax1 = sns.boxplot(
        y="protocol", x="rmsd_ave", data=df_ave, orient="h", ax=ax[1], **common_params
    )
    ax1.set(ylabel="", yticklabels=[], xlabel=r"$\mathrm{RMSD}_\mathrm{ave}$")
    ax2 = sns.boxplot(
        y="protocol", x="n_rmsd", data=df_n_rmsd, orient="h", ax=ax[2], **common_params
    )
    ax2.set(ylabel="", yticklabels=[], xlabel=r"$n\mathrm{RMSD}$")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "rmsd_plot_w_model.pdf"))
    plt.close()


if __name__ == "__main__":
    ligand_plot()

    with open(os.path.join(RES_DIR, 'corr_avg_family.pt'), 'rb') as handle:
        corr_avg_family = pickle.load(handle)

    with open(os.path.join(RES_DIR, 'corr_std_family.pt'), 'rb') as handle:
        corr_std_family = pickle.load(handle)

    with open(os.path.join(RES_DIR, 'rmse_avg_family.pt'), 'rb') as handle:
        rmse_avg_family = pickle.load(handle)

    with open(os.path.join(RES_DIR, 'rmse_std_family.pt'), 'rb') as handle:
        rmse_std_family = pickle.load(handle)

    family_plot(corr_avg_family, corr_std_family, 'corr_family_eval.pdf')
    family_plot(rmse_avg_family,
                rmse_std_family,
                'rmse_family_eval.pdf',
                color='yellowgreen')

    desc_plot()

    rmsd_plot()
