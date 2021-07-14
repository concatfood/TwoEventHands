import argparse
import logging
import os
from shutil import copyfile

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader

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
def train_net(net, device, epochs=100, batch_size=16, lr=0.001, val_percent=0.1, save_cp=True, img_scale=1.0,
              checkpoint=None):

    # setup data loader
    dataset = BasicDataset(dir_events, dir_mano, dir_mask, l_lnes, img_scale)

    # split such that validation set consists of the midmost parts of all sequences
    list_val = [list(range(int(round(len(sequence) / 2 * (1 - val_percent) + l_lnes - 1)),
                           int(round(len(sequence) / 2 * (1 + val_percent))))) for sequence in dataset.events]

    list_train = [list(range(l_lnes - 1, int(round(len(sequence) / 2 * (1 - val_percent))))) +
                  list(range(int(round(len(sequence) / 2 * (1 + val_percent) + l_lnes - 1)), len(sequence)))
                  for sequence in dataset.events]

    for s in range(1, len(list_val)):
        len_total = 0

        for i in range(0, s):
            len_total += len(dataset.events[i])

        list_val[s] = [item + len_total for item in list_val[s]]

    for s in range(1, len(list_train)):
        len_total = 0

        for i in range(0, s):
            len_total += len(dataset.events[i])

        list_train[s] = [item + len_total for item in list_train[s]]

    list_val = [item for sublist in list_val for item in sublist]
    list_train = [item for sublist in list_train for item in sublist]

    train = torch.utils.data.Subset(dataset, list_train)
    val = torch.utils.data.Subset(dataset, list_val)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    n_train = len(train_loader)
    n_val = len(val_loader)

    # TensorBoard
    writer = SummaryWriter('runs/TEHNet')

    # optimization and scheduling
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

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
    epoch_start = 0 if checkpoint is None else checkpoint['epoch']
    lr = lr if checkpoint is None else optimizer.param_groups[0]['lr']

    logging.info(f'''Starting training:
      Start epoch:     {epoch_start}
      Maximum epochs:  {epochs}
      Batch size:      {batch_size}
      Learning rate:   {lr}
      Training size:   {len(list_train)}
      Validation size: {len(list_val)}
      Checkpoints:     {save_cp}
      Device:          {device.type}
      Scaling:         {img_scale}
    ''')

    global_step = 0

    # dataset loop
    for epoch in range(epoch_start, epochs):
        net.train()
        loss_train = 0

        # epoch loop
        # with tqdm(total=n_train, desc='training phase') as pbar:
        for it, batch in enumerate(train_loader):
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

            writer.add_text('phase, epoch, iteration', 'training, ' + str(epoch) + ', ' + str(it), global_step)

            global_step += 1

            # pbar.update()

        # validation phase
        net.eval()
        loss_valid = eval_net(net, val_loader, device, writer, epoch)
        scheduler.step(loss_valid)

        # log to TensorBoard
        writer.add_scalar('train loss', loss_train / n_train)
        writer.add_scalar('valid loss', loss_valid / n_val)
        writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'])

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

            checkpoint = {'epoch': epoch + 1, 'epoch_best': epoch_best, 'loss_valid_best': loss_valid_best,
                          'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict()}

            torch.save(checkpoint, dir_checkpoint + f'CP_epoch_' + str(epoch).zfill(len(str(epochs - 1))) + '.pth')
            logging.info(f'Checkpoint {epoch} saved')

            if epoch == epoch_best:
                copyfile(dir_checkpoint + f'CP_epoch_' + str(epoch).zfill(len(str(epochs - 1))) + '.pth',
                         dir_checkpoint + f'CP_epoch_best.pth')

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
    checkpoint = None

    if args.load:
        checkpoint = torch.load(args.load, map_location=device)

        net.load_state_dict(checkpoint['model'])
        logging.info(f'Checkpoint loaded from {args.load}')

    net.to(device=device)
    # net = nn.DataParallel(net)

    train_net(net=net, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, device=device, img_scale=args.scale,
              val_percent=args.val / 100, checkpoint=checkpoint)
