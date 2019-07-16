import os

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from net import TwoLegs
from net_utils import Featurizer, CombinedLoss
from train import EVAL_MODES, DEVICE, NUM_WORKERS, DATA_PATH, N_EPOCHS, BATCH_SIZE, \
                  training_loop
from utils import home, get_data

MODEL_PATH = os.path.join(home(), 'models')

if __name__ == '__main__':
    data = get_data(DATA_PATH)
    feat = Featurizer(*data)
    loader = DataLoader(feat,
                        batch_size=BATCH_SIZE,
                        num_workers=NUM_WORKERS,
                        shuffle=True)
    model = TwoLegs.to(DEVICE)
    loss_cl = CombinedLoss()
    opt = Adam(model.parameters())
    scheduler = ExponentialLR(opt, gamma=0.95)

    print('Training production model...')
    for i in range(N_EPOCHS):
        print('Epoch {}/{}...'.format(i + 1, N_EPOCHS))
        training_loop(loader, model, loss_cl, opt)
        scheduler.step()

    os.makedirs(MODEL_PATH, exist_ok=True)

    torch.save(model, os.path.join(MODEL_PATH, 'production.pt'))
