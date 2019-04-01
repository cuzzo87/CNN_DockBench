import pandas as pd
import os
import numpy as np

# computing the accuracy
modes = ['random', 'ligand_scaffold']

# preparing list for accuracy for each mode and split
results = {'random':[], 'ligand_scaffold':[]}
for m in modes:
    for i in range(5):
        ### loading all the files
        # resolution
        resolution = np.load('results/resolution_{}_{}.npy'.format(m, i))
        # test set
        rmsd_min_test = np.load('results/rmsd_min_test_{}_{}.npy'.format(m, i))
        rmsd_ave_test = np.load('results/rmsd_ave_test_{}_{}.npy'.format(m, i))
        n_rmsd_test = np.load('results/n_rmsd_test_{}_{}.npy'.format(m, i))
        # prediction
        rmsd_min_pred = np.load('results/rmsd_min_pred_{}_{}.npy'.format(m, i))
        rmsd_ave_pred = np.load('results/rmsd_ave_pred_{}_{}.npy'.format(m, i))
        n_rmsd_pred = np.load('results/n_rmsd_pred_{}_{}.npy'.format(m, i))

        # get ncomplexes and nprotocols
        n_complex = rmsd_min_pred.shape[0]
        n_protocols = rmsd_min_pred.shape[1]
        
        # repeat reolution values for all the complexes
        resolution_matrix = np.transpose(np.tile(resolution, (n_protocols,1)))[0]
        
        # points test
        points_test = np.zeros((n_complex, n_protocols), dtype=int)
        addpoints1_test = (rmsd_ave_test < resolution_matrix).astype(int)
        addpoints2_test = (n_rmsd_test > 10).astype(int)
        points_test = points_test + addpoints1_test + addpoints2_test
        idx_points3_test = np.where( (rmsd_ave_test == np.min(rmsd_ave_test, axis=1).reshape(n_complex,1)) & (n_rmsd_test == np.max(n_rmsd_test, axis=1).reshape(n_complex,1))) 
        points_test[idx_points3_test] += 1
        
        # points pred
        points_pred = np.zeros((n_complex, n_protocols), dtype=int)
        addpoints1_pred = (rmsd_ave_pred < resolution_matrix).astype(int)
        addpoints2_pred = (n_rmsd_pred > 10).astype(int)
        points_pred = points_pred + addpoints1_pred + addpoints2_pred
        idx_points3_pred = np.where( (rmsd_ave_pred == np.min(rmsd_ave_pred, axis=1).reshape(n_complex,1)) & (n_rmsd_pred == np.max(n_rmsd_pred, axis=1).reshape(n_complex,1))) 
        points_pred[idx_points3_pred] += 1
        
        match = np.equal(np.array(points_test), np.array(points_pred) ).astype(int)
        
        n_data = match.shape[0] * match.shape[1]
        total = np.sum(match)
        
        res = total * 100. / n_data
        
        results[m].append(res)

df = pd.DataFrame(columns=['Mode', 'min', 'max', 'avg', 'std'])
for k, v in results.items():
    i = len(df)
    a = np.array(v)
    df.loc[i] = [k, np.min(a), np.max(a), np.average(a), np.std(a)]
    df.to_csv(os.path.join('results', 'evalresults.csv')) 
    print(df)
