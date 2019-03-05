import os
import multiprocessing

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from cnndockbench import home
from cnndockbench.net import TwoLegs
from cnndockbench.net_utils import CombinedLoss, Featurizer
from cnndockbench.utils import get_data, Splitter


GPU_DEVICE = torch.device('cuda')
NUM_WORKERS = int(multiprocessing.cpu_count() / 2)
N_EPOCHS = 20
BATCH_SIZE = 32


def training_loop(loader, model, loss_cl, opt):
    model = model.train()
    progress = tqdm(loader)

    for voxel, fp, rmsd_min, rmsd_ave, n_rmsd in progress:
        voxel = voxel.to(GPU_DEVICE)
        fp = fp.to(GPU_DEVICE)
        rmsd_min = rmsd_min.to(GPU_DEVICE)
        rmsd_ave = rmsd_ave.to(GPU_DEVICE)
        n_rmsd = n_rmsd.to(GPU_DEVICE)

        opt.zero_grad()

        out1, out2, out3 = model(voxel, fp)
        loss_rmsd_min, loss_rmsd_ave, loss_n_rmsd = loss_cl(
            out1, out2, out3, rmsd_min, rmsd_ave, n_rmsd)
        loss = loss_rmsd_min + loss_rmsd_ave + loss_n_rmsd
        loss.backward()
        opt.step()

        progress.set_postfix({'loss_rmsd_min': loss_rmsd_min.item(),
                              'loss_rmsd_ave': loss_rmsd_ave.item(),
                              'loss_n_rmsd': loss_n_rmsd.item(),
                              'loss': loss.item()})


def eval_loop(loader, model):
    model = model.eval()
    progress = tqdm(loader)

    rmsd_min_all = []
    rmsd_ave_all = []
    n_rmsd_all = []

    rmsd_min_pred = []
    rmsd_ave_pred = []
    n_rmsd_pred = []

    for voxel, fp, rmsd_min, rmsd_ave, n_rmsd in progress:
        with torch.no_grad():
            voxel = voxel.to(GPU_DEVICE)
            fp = fp.to(GPU_DEVICE)

            out1, out2, out3 = model(voxel, fp)
            out3 = torch.round(torch.exp(out3)).clamp(max=20).type(torch.int)

            rmsd_min_all.append(rmsd_min)
            rmsd_ave_all.append(rmsd_ave)
            n_rmsd_all.append(n_rmsd.type(torch.int))

            rmsd_min_pred.append(out1.cpu())
            rmsd_ave_pred.append(out2.cpu())
            n_rmsd_pred.append(out3.cpu())
    return torch.cat(rmsd_min_all), torch.cat(rmsd_ave_all), torch.cat(n_rmsd_all), \
           torch.cat(rmsd_min_pred), torch.cat(rmsd_ave_pred), torch.cat(n_rmsd_pred)


if __name__ == '__main__':
    path = os.path.join(home(), 'data')
    data = get_data(path)

    sp = Splitter(*data)
    train_data, test_data = sp.split(p=.25, mode='random')

    feat_train = Featurizer(*train_data)
    feat_test = Featurizer(*test_data)

    loader_train = DataLoader(feat_train,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True)

    loader_test = DataLoader(feat_test,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             shuffle=True)

    model = TwoLegs().cuda()
    loss_cl = CombinedLoss()
    opt = Adam(model.parameters())

    print('Training model...')
    for i in range(N_EPOCHS):
        print('Epoch {}/{}...'.format(i + 1, N_EPOCHS))
        training_loop(loader_train, model, loss_cl, opt)

    print('Evaluating model...')
    rmsd_min_test, rmsd_ave_test, n_rmsd_test, rmsd_min_pred, rmsd_ave_pred, n_rmsd_pred = eval_loop(loader_test, model)
