from os.path import splitext
import os
import glob
import logging
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import pandas as pd
import argparse
from tqdm import tqdm
# import wandb
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from utils.dataset import CustomDataset

from model.unet import UNet
from eval import eval_net

# wandb.init(project="unet")
# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.cuda.manual_seed(seed)
# torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

train_data_dir = "data/train"
test_data_dir = "data/test"

train_mask_dir = "mask/train"
test_mask_dir = "mask/test"

dir_checkpoint = 'checkpoints/'

# data_dir = "./data"
# mask_dir = "./mask"
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Preapare train and valid csv
images = glob.glob("data/train/*.*")
# mask = glob.glob("mask/test/*.*")
mask = []
for i in images:
    z = i.replace('data','mask')
    x = z.replace('jpg','png')
    mask.append(x)

train_list = pd.DataFrame(np.concatenate([np.asarray(images).reshape(-1, 1), np.asarray(mask).reshape(-1, 1)], axis=1))
train_list.to_csv('train.csv', index=False)



def train_net(net, device, epochs=200, batch_size=1, lr=0.001, save_cp=True):

    train = CustomDataset(train_df, train_data_dir, train_mask_dir, 'train')
    test = CustomDataset(test_df, test_data_dir, test_mask_dir, 'test')

    n_train = len(train)
    n_test = len(test)

    # print(n_train, n_test)

    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True, drop_last=True)

    global_step = 0

    optimizer = optim.RMSprop(net.parameters(), lr=lr,
                              weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    criterion = nn.BCEWithLogitsLoss()


    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']

                # print(imgs.shape, net.n_channels)
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                try:
                    masks_pred = net(imgs)
                except RuntimeError:
                    # print(batch['image_path'])
                    continue
                
                loss = criterion(masks_pred, true_masks)

                # wandb.log({"loss":loss,"epoch":epoch})

                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])

                global_step += 1

                if global_step % (n_train // (10 * batch_size)) == 0:

                    val_score = eval_net(net, test_loader, device)
                    scheduler.step(val_score)

                    # wandb.log({'val_score':val_score})

                    if net.n_classes > 1:
                        logging.info(
                            'Validation cross entropy: {}'.format(val_score))
                        # writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info(
                            'Validation Dice Coeff: {}'.format(val_score))
                        # writer.add_scalar('Dice/test', val_score, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            if ((epoch + 1)  % 20 == 0):
                torch.save(net.state_dict(), dir_checkpoint +f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')


if __name__ == '__main__':

    net = UNet(n_channels=3, n_classes=1)
    device = "cuda:0"
    net = net.to(device)
    # net.load_state_dict(torch.load(r"checkpoints/CP_epoch10.pth", map_location=device))
    # print(f'Model loaded')
    # wandb.watch(net)
    train_net(net, device=device)