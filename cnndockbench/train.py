import multiprocessing
import os

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from cnndockbench import home
from cnndockbench.net import TwoLegs
from cnndockbench.net_utils import CombinedLoss, Featurizer
from cnndockbench.utils import Splitter, get_data

GPU_DEVICE = torch.device('cuda')
NUM_WORKERS = int(multiprocessing.cpu_count() / 2)

DATA_PATH = os.path.join(home(), 'data')
RES_PATH = os.path.join(home(), 'results')

N_EPOCHS = 50
N_SPLITS = 5
BATCH_SIZE = 32
EVAL_MODES = ['random', 'ligand_scaffold']


def training_loop(loader, model, loss_cl, opt):
    """
    Training loop of `model` using data from `loader` and
    loss functions from `loss_cl` using optimizer `opt`.
    """
    model = model.train()
    progress = tqdm(loader)

    for voxel, fp, rmsd_min, rmsd_ave, n_rmsd, mask in progress:
        voxel = voxel.to(GPU_DEVICE)
        fp = fp.to(GPU_DEVICE)
        rmsd_min = rmsd_min.to(GPU_DEVICE)[mask]
        rmsd_ave = rmsd_ave.to(GPU_DEVICE)[mask]
        n_rmsd = n_rmsd.to(GPU_DEVICE)[mask]

        opt.zero_grad()

        out1, out2, out3 = model(voxel, fp)
        loss_rmsd_min, loss_rmsd_ave, loss_n_rmsd = loss_cl(
            out1[mask], out2[mask], out3[mask], rmsd_min, rmsd_ave, n_rmsd)
        loss = loss_rmsd_min + loss_rmsd_ave + loss_n_rmsd
        loss.backward()
        opt.step()

        progress.set_postfix({'loss_rmsd_min': loss_rmsd_min.item(),
                              'loss_rmsd_ave': loss_rmsd_ave.item(),
                              'loss_n_rmsd': loss_n_rmsd.item(),
                              'loss': loss.item()})


def eval_loop(loader, model):
    """
    Evaluation loop using `model` and data from `loader`.
    """
    model = model.eval()
    progress = tqdm(loader)

    rmsd_min_all = []
    rmsd_ave_all = []
    n_rmsd_all = []
    masks = []

    rmsd_min_pred = []
    rmsd_ave_pred = []
    n_rmsd_pred = []

    for voxel, fp, rmsd_min, rmsd_ave, n_rmsd, mask in progress:
        with torch.no_grad():
            voxel = voxel.to(GPU_DEVICE)
            fp = fp.to(GPU_DEVICE)

            out1, out2, out3 = model(voxel, fp)
            out3 = torch.round(torch.exp(out3)).clamp(max=20).type(torch.int)

            rmsd_min_all.append(rmsd_min)
            rmsd_ave_all.append(rmsd_ave)
            n_rmsd_all.append(n_rmsd.type(torch.int))
            masks.append(mask)

            rmsd_min_pred.append(out1.cpu())
            rmsd_ave_pred.append(out2.cpu())
            n_rmsd_pred.append(out3.cpu())
    return torch.cat(rmsd_min_all), torch.cat(rmsd_ave_all), torch.cat(n_rmsd_all), \
           torch.cat(rmsd_min_pred), torch.cat(rmsd_ave_pred), torch.cat(n_rmsd_pred), \
           torch.cat(masks)


if __name__ == '__main__':
    data = get_data(DATA_PATH)
    for mode in EVAL_MODES:
        sp = Splitter(*data, n_splits=N_SPLITS, method=mode)
        for split_no in range(N_SPLITS):
            print('Now evaluating split {}/{} with strategy {}'.format(split_no + 1, N_SPLITS, mode))
            train_data, test_data = sp.get_split(split_no=split_no, mode=mode)

            feat_train = Featurizer(*train_data)
            feat_test = Featurizer(*test_data)

            loader_train = DataLoader(feat_train,
                                      batch_size=BATCH_SIZE,
                                      num_workers=NUM_WORKERS,
                                      shuffle=True)

            loader_test = DataLoader(feat_test,
                                     batch_size=BATCH_SIZE,
                                     num_workers=NUM_WORKERS,
                                     shuffle=False)

            model = TwoLegs().cuda()
            loss_cl = CombinedLoss()
            opt = Adam(model.parameters())

            print('Training model...')
            for i in range(N_EPOCHS):
                print('Epoch {}/{}...'.format(i + 1, N_EPOCHS))
                training_loop(loader_train, model, loss_cl, opt)

            print('Evaluating model...')
            rmsd_min_test, rmsd_ave_test, n_rmsd_test, rmsd_min_pred, rmsd_ave_pred, n_rmsd_pred, mask = eval_loop(loader_test, model)
            _, test_idx = sp.splits[split_no]
            test_resolution = np.array([np.load(f) for f in sp.resolution[test_idx]])

            os.makedirs(RES_PATH, exist_ok=True)

            # Save results for later evaluation
            np.save(os.path.join(RES_PATH, 'rmsd_min_test_{}_{}.npy'.format(mode, split_no)), arr=rmsd_min_test.numpy())
            np.save(os.path.join(RES_PATH, 'rmsd_ave_test_{}_{}.npy'.format(mode, split_no)), arr=rmsd_ave_test.numpy())
            np.save(os.path.join(RES_PATH, 'n_rmsd_test_{}_{}.npy'.format(mode, split_no)), arr=n_rmsd_test.numpy())
            np.save(os.path.join(RES_PATH, 'rmsd_min_pred_{}_{}.npy'.format(mode, split_no)), arr=rmsd_min_pred.numpy())
            np.save(os.path.join(RES_PATH, 'rmsd_ave_pred_{}_{}.npy'.format(mode, split_no)), arr=rmsd_ave_pred.numpy())
            np.save(os.path.join(RES_PATH, 'n_rmsd_pred_{}_{}.npy'.format(mode, split_no)), arr=n_rmsd_pred.numpy())
            np.save(os.path.join(RES_PATH, 'mask_{}_{}.npy'.format(mode, split_no)), arr=mask.numpy())
            np.save(os.path.join(RES_PATH, 'resolution_{}_{}.npy'.format(mode, split_no)), arr=test_resolution)
