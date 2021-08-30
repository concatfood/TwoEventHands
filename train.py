import argparse
from eval import eval_net
import logging
import math
import numpy as np
import os
from shutil import copyfile
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from TEHNet import TEHNet

# relative directories
dir_events = os.path.join('data', 'train', 'events')
dir_mano = os.path.join('data', 'train', 'mano')
dir_mask = os.path.join('data', 'train', 'masks')
dir_checkpoint = 'checkpoints'

# early stopping
patience = 100

# resolution
res = (240, 180)

# LNES window size
l_lnes = 200

# weights
weight_mano = 1.0
weight_mask = 0.1

# use UNet for masks
use_unet = False


# training function
def train_net(net, device, epochs=1000, batch_size=16, lr=0.0001, val_percent=0.1, save_cp=True, checkpoint=None):
    # setup data loader
    dataset = BasicDataset(dir_events, dir_mano, dir_mask, res, l_lnes, use_unet=use_unet)

    # split such that validation set consists of the midmost parts of all sequences
    list_train_np = [np.array(list(range(0,
                                         int(round(num_frames / 2 * (1 - val_percent))))) +
                              list(range(int(round(num_frames / 2 * (1 + val_percent))),
                                         num_frames)))
                     for num_frames in dataset.num_frames]

    list_val_np = [np.array(list(range(int(round(num_frames / 2 * (1 - val_percent))),
                                       int(round(num_frames / 2 * (1 + val_percent))))))
                   for num_frames in dataset.num_frames]

    for s in range(1, len(list_train_np)):
        list_train_np[s] += dataset.len_until[s]

    list_train = np.concatenate(list_train_np).tolist()

    for s in range(1, len(list_val_np)):
        list_val_np[s] += dataset.len_until[s]

    list_val = np.concatenate(list_val_np).tolist()

    train = torch.utils.data.Subset(dataset, list_train)
    val = torch.utils.data.Subset(dataset, list_val)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # optimization and scheduling
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    # losses
    criterion_mask = nn.CrossEntropyLoss()
    criterion_mano = nn.MSELoss()

    # early stopping values
    epoch_best = -1 if checkpoint is None else checkpoint['epoch_best']
    loss_valid_best = float('inf') if checkpoint is None else checkpoint['loss_valid_best']

    # start where we left of last time
    epoch_start = 0 if checkpoint is None else checkpoint['epoch'] + 1
    lr = lr if checkpoint is None else optimizer.param_groups[0]['lr']

    global_step = 0 if checkpoint is None else checkpoint['global_step']

    logging.info(f'''Starting training:
      Start epoch:     {epoch_start}
      Maximum epochs:  {epochs}
      Batch size:      {batch_size}
      Learning rate:   {lr}
      Device:          {device.type}
    ''')

    n_train = len(train_loader)
    n_val = len(val_loader)
    percentage_data = 0.001
    iters_train = int(round(percentage_data * n_train))
    iters_val = int(round(percentage_data * n_val))

    # TensorBoard
    writer = SummaryWriter('runs/TEHNet')

    # dataset loop
    for epoch in range(epoch_start, epochs):
        loss_train_mano = 0
        loss_train_mask = 0
        loss_train = 0

        net.train()

        # epoch loop
        # with tqdm(total=iters_train, desc='training phase') as pbar:
        for it, batch in enumerate(train_loader):
            if it == iters_train:
                break

            # log phase, epoch and iteration
            with open('phase_epoch_iteration.txt', "w") as f:
                f.write('training, ' + str(epoch) + ', ' + str(it))

            # load data
            if use_unet:
                lnes, true_masks, true_mano = batch['lnes'], batch['mask'], batch['mano']
            else:
                lnes, true_mano = batch['lnes'], batch['mano']

            # send to device
            lnes = lnes.to(device=device, dtype=torch.float32)

            if use_unet:
                true_masks = true_masks.to(device=device, dtype=torch.long)

            true_mano = true_mano.to(device=device, dtype=torch.float32)

            # forward and loss computation
            if use_unet:
                masks_pred, mano_pred = net(lnes)
            else:
                mano_pred = net(lnes)

            loss_mano = criterion_mano(mano_pred, true_mano)
            loss_train_mano += loss_mano.item()

            if use_unet:
                loss_mask = criterion_mask(masks_pred, true_masks)
                loss_train_mask += loss_mask.item()
                loss_total = weight_mask * loss_mask + weight_mano * loss_mano
            else:
                loss_total = weight_mano * loss_mano

            loss_train += loss_total.item()

            # backward propagation
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            global_step += 1

            # pbar.update()

        loss_train /= iters_train
        loss_train_mano /= iters_train
        loss_train_mask /= iters_train

        # validation phase
        net.eval()
        loss_valid, loss_valid_mano, loss_valid_mask = eval_net(net, val_loader, device, epoch, iters_val, use_unet)
        loss_valid /= iters_val
        loss_valid_mano /= iters_val
        loss_valid_mask /= iters_val
        scheduler.step(loss_valid)

        # log to TensorBoard
        writer.add_scalar('train loss total', loss_train, epoch)
        writer.add_scalar('train loss mano', loss_train_mano, epoch)

        if use_unet:
            writer.add_scalar('train loss mask', loss_train_mask, epoch)

        writer.add_scalar('valid loss total', loss_valid, epoch)
        writer.add_scalar('valid loss mano', loss_valid_mano, epoch)

        if use_unet:
            writer.add_scalar('valid loss mask', loss_valid_mask, epoch)

        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)

        # check if best epoch
        if loss_valid < loss_valid_best:
            epoch_best = epoch
            loss_valid_best = loss_valid

        logging.info(f'''Epoch stats:
      Epoch:                {epoch}
      Best epoch:           {epoch_best}
      Best validation loss: {loss_valid_best}
      Training loss:        {loss_train}
      Validation loss:      {loss_valid}
      Learning rate:        {optimizer.param_groups[0]['lr']}
      Global step:          {global_step}
            ''')

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

            checkpoint = {'epoch': epoch, 'epoch_best': epoch_best, 'global_step': global_step,
                          'loss_valid_best': loss_valid_best, 'model': net.state_dict(),
                          'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}

            torch.save(checkpoint, os.path.join(dir_checkpoint, 'current.pth'))
            logging.info(f'checkpoint {epoch} saved')

            if epoch == epoch_best:
                copyfile(os.path.join(dir_checkpoint, 'current.pth'), os.path.join(dir_checkpoint, 'best.pth'))

    writer.close()


# parse arguments
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on LNES and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


# main function
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = TEHNet(use_unet=use_unet)
    checkpoint = None

    if args.load:
        checkpoint = torch.load(args.load, map_location=device)

        net.load_state_dict(checkpoint['model'])
        logging.info(f'Checkpoint loaded from {args.load}')

    net.to(device=device)
    # net = nn.DataParallel(net)

    train_net(net=net, device=device, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr,
              val_percent=args.val / 100, checkpoint=checkpoint)
