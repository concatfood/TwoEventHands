import argparse
import logging
import os
from shutil import copyfile
import sys

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

from TEHNet import TEHNet

# relative directories
dir_events = 'data/events/'
dir_mano = 'data/mano/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'

# early stopping
patience = 10

# LNES window size
l_lnes = 200

# weights
weight_mask = 1.0
weight_mano = 1.0


# training function
def train_net(net,
              device,
              epochs=100,
              batch_size=16,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1.0):

    # setup data loader
    dataset = BasicDataset(dir_events, dir_mano, dir_mask, l_lnes, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # TensorBoard
    writer = SummaryWriter()

    # optimization and scheduling
    optimizer = optim.RMSprop(net.parameters(), lr=lr)

    # losses
    criterion_mask = nn.CrossEntropyLoss()
    criterion_mano = nn.MSELoss()

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Scaling:         {img_scale}
    ''')

    # early stopping values
    epoch_best = -1
    loss_valid_best = float('inf')

    # redefine to number of batches
    n_train = len(train_loader)

    # dataset loop
    for epoch in range(epochs):
        net.train()
        loss_train = 0

        # epoch loop
        with tqdm(total=n_train, desc='training phase') as pbar:
            for batch in train_loader:
                # load data
                lnes = batch['lnes']
                true_masks = batch['mask']
                true_mano = batch['mano']

                # send to device
                lnes = lnes.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                true_mano = true_mano.to(device=device, dtype=torch.float32)

                # forward and loss computation
                masks_pred, mano_pred = net(lnes)
                loss_mask = criterion_mask(masks_pred, true_masks)
                loss_mano = criterion_mano(mano_pred, true_mano)
                loss_total = weight_mask * loss_mask + weight_mano * loss_mano
                loss_train += loss_total.item()

                # backward propagation
                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                pbar.update()

        writer.add_scalar('train loss', loss_train / n_train)

        # validation phase
        net.eval()
        loss_valid = eval_net(net, val_loader, device)
        writer.add_scalar('valid loss', loss_valid)

        # check if best epoch
        if loss_valid < loss_valid_best:
            epoch_best = epoch
            loss_valid_best = loss_valid

        # early stopping
        if epoch - epoch_best > patience:
            break

        # save checkpoint
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch}.pth')
            logging.info(f'Checkpoint {epoch} saved !')

            if epoch == epoch_best:
                copyfile(dir_checkpoint + f'CP_epoch{epoch}.pth', dir_checkpoint + f'CP_epoch_best.pth')

    writer.close()


# parse arguments
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on LNES and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1.0,
                        help='Downscaling factor')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


# main function
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = TEHNet()

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    # save model just in case
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')

        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
