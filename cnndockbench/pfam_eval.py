import numpy as np
import pandas as pd
from utils import  get_data
from eval import compute_score, MeanError, corr
import os
from glob import glob
from tqdm import tqdm
import pickle

# FUNCTIONS


def loadFromResults(mode, no_split):
    aveTest = np.load('results/rmsd_ave_test_{}_{}.npy'.format(mode, no_split))
    minTest = np.load('results/rmsd_min_test_{}_{}.npy'.format(mode, no_split))
    nrmsdTest = np.load('results/n_rmsd_test_{}_{}.npy'.format(mode, no_split))

    avePred = np.load('results/rmsd_ave_pred_{}_{}.npy'.format(mode, no_split))
    minPred = np.load('results/rmsd_min_pred_{}_{}.npy'.format(mode, no_split))
    nrmsdPred = np.load('results/n_rmsd_pred_{}_{}.npy'.format(mode, no_split))

    masks = np.load('results/mask_{}_{}.npy'.format(mode, no_split))
    resolutions = np.load('results/resolution_{}_{}.npy'.format(mode, no_split))

    datas_aveTest = []

    for t in aveTest:
        data = np.round(t, 2)
        datas_aveTest.append(data)
    return datas_aveTest, minTest, nrmsdTest, avePred, minPred, nrmsdPred, masks, resolutions


def loadFromData():
    listFiles = glob('data/*/rmsd_ave.npy')
    datas = [np.round(np.load(d), 2).astype(np.float32) for d in listFiles]
    return datas, listFiles


def getDictPDdIdIdx(dataFromData, fnamesFromData, dataFromResult):
    dictPdbid_idx = {}
    for n, dR in enumerate(tqdm(dataFromResult)):
        for dD in dataFromData:
            if np.all(dR == dD):
                pdbid = fnamesFromData[n].split('/')[1]
                dictPdbid_idx[pdbid] = n
    return dictPdbid_idx


def regression_metricsPFAM(rmsd_test, rmsd_pred, mask):
    """
    Computes root mean squared error and correlation for `rmsd_test`
    and `rmsd_pred` only using the samples marked by `mask`.
    """
    rmses = []
    corrs = []

    r_t, r_p = rmsd_test[mask.astype(np.bool)], rmsd_pred[mask.astype(np.bool)]
    rmses.append(MeanError(metric='rmse').compute(r_t, r_p))

    u_t, u_p = np.unique(r_t), np.unique(r_p)
    if len(np.intersect1d(u_t, u_p)) == 1:
        corrs.append(1.0)

    elif len(u_t) == 1 or len(u_p) == 1:
        corrs.append(0.0)
    else:
        corrs.append(corr(r_t, r_p))
    return rmses, corrs

def aggregate_results(results_dict, fun):
    """
    Computes average results over splits.
    """
    avg_results = {}
    for mode in SPLITMODES:
        avg_results.setdefault(mode, {})
        for pfam in results_dict[mode].keys():
            avg_results[mode].setdefault(pfam, {})
            for metric, values in results_dict[mode][pfam].items():
                avg_results[mode][pfam][metric] = fun(values)
    return avg_results

def cleanFromNan(agg_rmse, agg_stf):
    excluded = []
    for mode in SPLITMODES:
        for pfam in agg_rmse[mode]:
            rmse = agg_rmse['random'][pfam]['corr_ave']
            if np.isnan(rmse) and pfam not in excluded:
                excluded.append(pfam)

    for mode in SPLITMODES:
        for pfam in excluded:
            del agg_rmse[mode][pfam]
            del agg_std[mode][pfam]
    return agg_rmse, agg_std

def pfamsNMembers(classesFiles):

    clMember = {}
    for clF in classesFiles:
        f = open(clF)
        nl = len(f.readlines())
        f.close()
        clMember[clF] = nl

    sortedClNames = {cl.split('/')[-1].split('.')[0]:clMember[cl] for cl in
                         sorted(clMember, key=lambda k: clMember[k], reverse=True)}

    return sortedClNames



# Loading
Results = np.load('results/results.pkl')

# Main Variables
SPLITMODES = list(Results.keys())
PROTOCOLS  = list(Results[SPLITMODES[0]].keys())
NKFOLD = len(Results[SPLITMODES[0]][PROTOCOLS[0]]['rmse_min'])




if __name__ == '__main__':
    print('Evaluating performance based on PFAM')
    print('Splits: ', SPLITMODES)
    print('Protocols: ', PROTOCOLS)
    print('N Kfold: ', NKFOLD)


    data = get_data('data/')
    classesFiles = glob('proteinClassesSelection/*.list')
    pfamMembers = pfamsNMembers(classesFiles)

    # loading rmsd_ave data for comparison. This will be used to associate each sample of the test set to the PDB code
    dataFromDataFolder, fnamesFromDataFolder = loadFromData()

    RES_DIR = 'data/'

    results = {}
    for mode in SPLITMODES:
        print('Evaluating ', mode)

        for i in range(NKFOLD):
            print('\tn KFold ', i+1)
            aveTest, minTest, nrmsdTest, avePred, minPred, nrmsdPred, masks, ress = loadFromResults(mode, i)
            dictPdbId_idx = getDictPDdIdIdx(dataFromDataFolder, fnamesFromDataFolder, aveTest)
            # print('protein_classes_distribution --> ', i)
            results.setdefault(mode, {})
            for cl in classesFiles:
                clName = cl.split('/')[-1].split('.')[0]
                results[mode].setdefault(clName, {})
                results[mode][clName].setdefault('rmse_ave', [])
                results[mode][clName].setdefault('corr_ave', [])
                f = open(cl, 'r')
                pdbids = [pdbid.strip().lower() for pdbid in f.readlines()]
                f.close()
                rmsd_min_test_all = []
                rmsd_ave_test_all = []
                n_rmsd_test_all = []

                rmsd_min_pred_all = []
                rmsd_ave_pred_all = []
                n_rmsd_pred_all = []

                masks_all = []
                resolution_all = []
                for pdbid in pdbids:
                    if pdbid not in dictPdbId_idx:
                        continue
                    idx = dictPdbId_idx[pdbid.lower()]

                    resolution = ress[idx]
                    mask = masks[idx]
                    resolution_all.append(resolution)
                    masks_all.append(mask)

                    rmsd_min_test = minTest[idx]
                    rmsd_ave_test = aveTest[idx]
                    n_rmsd_test = nrmsdTest[idx]
                    rmsd_min_test_all.append(rmsd_min_test)
                    rmsd_ave_test_all.append(rmsd_ave_test)
                    n_rmsd_test_all.append(n_rmsd_test)

                    rmsd_min_pred = minPred[idx]
                    rmsd_ave_pred = avePred[idx]
                    n_rmsd_pred = nrmsdPred[idx]
                    rmsd_min_pred_all.append(rmsd_min_pred)
                    rmsd_ave_pred_all.append(rmsd_ave_pred)
                    n_rmsd_pred_all.append(n_rmsd_pred)

                if len(resolution_all) == 0:
                    # print(cl, ' --> NO DATA')
                    continue
                rmsd_min_test_all = np.asarray(rmsd_min_test_all)
                rmsd_ave_test_all = np.asarray(rmsd_ave_test_all)
                n_rmsd_test_all = np.asarray(n_rmsd_test_all)

                rmsd_min_pred_all = np.asarray(rmsd_min_pred_all)
                rmsd_ave_pred_all = np.asarray(rmsd_ave_pred_all)
                n_rmsd_pred_all = np.asarray(n_rmsd_pred_all)

                masks_all = np.asarray(masks_all)
                resolution_all = np.asarray(resolution_all)
                n_complex = rmsd_ave_pred_all.shape[0]
                n_protocols = rmsd_ave_pred_all.shape[1]

                resolution_all = np.transpose(np.tile(resolution_all, (n_protocols, 1)))
                score_test = compute_score(rmsd_ave_test_all, n_rmsd_test_all, resolution_all, n_complex)
                score_pred = compute_score(rmsd_ave_pred_all, n_rmsd_pred_all, resolution_all, n_complex)

                rmses_min, corrs_min = regression_metricsPFAM(rmsd_min_test_all, rmsd_min_pred_all, masks_all)
                rmses_ave, corrs_ave = regression_metricsPFAM(rmsd_ave_test_all, rmsd_ave_pred_all, masks_all)
                results[mode][clName]['rmse_ave'].append(rmses_ave)
                results[mode][clName]['corr_ave'].append(corrs_ave)
                # print(rmsd_ave_test_all.shape, len(rmses_ave))
                # print(cl, ' ', resolution_all.shape[0], ' --> ',
                #                   'RMSE ', np.nanmean(rmses_ave), '+/-', np.nanstd(rmses_ave),
                #                   'CORR', np.nanmean(corrs_ave), '+/-', np.nanstd(corrs_ave) )

    print('Finished !!! Writing the results in tmp_results.pkl')
    agg_rmse = aggregate_results(results, np.nanmean)
    agg_std = aggregate_results(results, np.nanstd)

    agg_rmse, agg_std = cleanFromNan(agg_rmse, agg_std)

    pickle.dump(agg_rmse, open('Pfam_agg_rmse.pkl', 'wb'))
    pickle.dump(agg_std, open('Pfam_agg_std.pkl', 'wb'))
    pickle.dump(pfamMembers, open('Pfam_members.pkl', 'wb'))

