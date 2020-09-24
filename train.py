from os.path import splitext
import os
import glob
import logging
import numpy as np
import random
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.dataset import CustomDataset

# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.cuda.manual_seed(seed)
# torch.manual_seed(seed)
# torch.backends.cudnn.deterministic = True


train_data_dir = "data/train"
test_data_dir = "data/test"

train_mask_dir = "mask/train"
test_mask_dir = "mask/test"

# data_dir = "./data"
# mask_dir = "./mask"
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Preapare train and valid csv
# images = glob.glob("./data/test/*.*")
# mask = glob.glob("./mask/test/*.*")

# train_list = pd.DataFrame(np.concatenate(
#     [np.asarray(images).reshape(-1, 1), np.asarray(mask).reshape(-1, 1)], axis=1))
# train_list.to_csv('test.csv', index=False)


def train_net(net, device, epochs=5, batch_size=1, lr=0.001, save_cp=True):

    train = CustomDataset(train_df, train_data_dir, train_mask_dir, 'train')
    test = CustomDataset(test_df, test_data_dir, test_mask_dir, 'test')

    n_train = len(train)
    n_test = len(test)

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr,
                              weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=2)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'{epoch + 1} / {epoch}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.long

                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)

                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
