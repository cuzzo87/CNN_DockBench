import os

import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind
import pandas as pd

from cnndockbench.preprocess import PROTOCOLS, FAIL_FLAG
from cnndockbench.train import EVAL_MODES, N_SPLITS
from cnndockbench.utils import home
from cnndockbench.plots import NAMES_EVAL_MODES, RES_DIR



def get_long_data():
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
    return df_min, df_ave, df_n_rmsd


if __name__ == "__main__":
    df_min, df_ave, df_n_rmsd = get_long_data()

    # Means
    mean_min = df_min.groupby(["protocol"]).mean()
    mean_ave = df_ave.groupby(["protocol"]).mean()
    mean_n_rmsd = df_n_rmsd.groupby(["protocol"]).mean()

    # Significance analysis
    min_work = [
        df_min.loc[df_min.protocol == "This work ({})".format(NAMES_EVAL_MODES[mode])][
            "rmsd_min"
        ].to_list()
        for mode in EVAL_MODES
    ]
    ave_work = [
        df_ave.loc[df_ave.protocol == "This work ({})".format(NAMES_EVAL_MODES[mode])][
            "rmsd_ave"
        ].to_list()
        for mode in EVAL_MODES
    ]
    n_rmsd_work = [
        df_n_rmsd.loc[
            df_n_rmsd.protocol == "This work ({})".format(NAMES_EVAL_MODES[mode])
        ]["n_rmsd"].to_list()
        for mode in EVAL_MODES
    ]

    min_protocols = []
    ave_protocols = []
    n_rmsd_protocols = []

    for protocol in PROTOCOLS:
        min_protocols.append(
            df_min.loc[df_min.protocol == protocol]["rmsd_min"].to_list()
        )
        ave_protocols.append(
            df_ave.loc[df_ave.protocol == protocol]["rmsd_ave"].to_list()
        )
        n_rmsd_protocols.append(
            df_n_rmsd.loc[df_n_rmsd.protocol == protocol]["n_rmsd"].to_list()
        )

    p_values_min = np.zeros((len(ave_work), len(ave_protocols)), dtype=np.float32)
    for i, values_min_work in enumerate(min_work):
        for j, values_min_protocol in enumerate(min_protocols):
            p_values_min[i, j] = mannwhitneyu(
                values_min_work, values_min_protocol, alternative="less"
            ).pvalue

    p_values_ave = np.zeros((len(ave_work), len(ave_protocols)), dtype=np.float32)
    for i, values_ave_work in enumerate(ave_work):
        for j, values_ave_protocol in enumerate(ave_protocols):
            p_values_ave[i, j] = mannwhitneyu(
                values_ave_work, values_ave_protocol, alternative="less"
            ).pvalue

    p_values_n_rmsd = np.zeros(
        (len(n_rmsd_work), len(n_rmsd_protocols)), dtype=np.float32
    )
    for i, values_n_rmsd_work in enumerate(n_rmsd_work):
        for j, values_n_rmsd_protocol in enumerate(n_rmsd_protocols):
            p_values_n_rmsd[i, j] = mannwhitneyu(
                values_n_rmsd_work, values_n_rmsd_protocol, alternative="greater"
            ).pvalue
