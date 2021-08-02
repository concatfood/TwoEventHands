import argparse
import queue

from eval import eval_net
import math
import logging
import os
from queue import Queue
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
dir_events = 'data/events/'
dir_mano = 'data/mano/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'

# early stopping
patience = 100

# LNES window size
l_lnes = 200

# interval between frames
interval_data = int(round(l_lnes / 2))
level_split_max = int(math.floor(math.log2(interval_data)))

# weights
weight_mano = 1.0
weight_mask = 0.1


# training function
def train_net(net, device, epochs=1000, batch_size=16, lr=0.001, val_percent=0.1, save_cp=True, checkpoint=None):
    # setup data loader
    dataset = BasicDataset(dir_events, dir_mano, dir_mask, l_lnes)

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
    level_split = 0 if checkpoint is None else checkpoint['level_split']
    loss_mano_inter = 0 if checkpoint is None else checkpoint['loss_mano_inter']
    loss_mask_inter = 0 if checkpoint is None else checkpoint['loss_mask_inter']
    loss_train_inter = 0 if checkpoint is None else checkpoint['loss_train_inter']
    num_steps_inter = 0 if checkpoint is None else checkpoint['num_steps_inter']
    num_steps_val_inter = 0 if checkpoint is None else checkpoint['num_steps_val_inter']
    num_val_phases_inter = 10 if checkpoint is None else checkpoint['num_val_phases_inter']
    starts_split = Queue() if checkpoint is None else Queue(checkpoint['starts_split'])

    logging.info(f'''Starting training:
      Start epoch:     {epoch_start}
      Maximum epochs:  {epochs}
      Batch size:      {batch_size}
      Learning rate:   {lr}
      Device:          {device.type}
    ''')

    # TensorBoard
    writer = SummaryWriter('runs/TEHNet')

    # dataset loop
    for epoch in range(epoch_start, epochs):
        # split such that validation set consists of the midmost parts of all sequences
        if starts_split.empty():
            if level_split < level_split_max:
                level_split += 1
                start_split = interval_data / 2 ** level_split
                interval_split = interval_data / 2 ** (level_split - 1)
            else:
                level_split = 1
                start_split = interval_data / 2
                interval_split = interval_data

            starts_split = Queue()

            for i in range(2 ** (level_split - 1)):
                starts_split.put(start_split + interval_split * i)

        start_split = int(round(starts_split.get())) - 1

        list_train = [list(range(start_split,
                                 int(round(len(sequence) / 2 * (1 - val_percent))),
                                 interval_data)) +
                      list(range(int(round(len(sequence) / 2 * (1 + val_percent))) + start_split,
                                 len(sequence),
                                 interval_data))
                      for sequence in dataset.events]

        list_val = [list(range(int(round(len(sequence) / 2 * (1 - val_percent))) + start_split,
                               int(round(len(sequence) / 2 * (1 + val_percent))),
                               interval_data))
                    for sequence in dataset.events]

        for s in range(1, len(list_train)):
            len_total = 0

            for i in range(0, s):
                len_total += len(dataset.events[i])

            list_train[s] = [item + len_total for item in list_train[s]]

        for s in range(1, len(list_val)):
            len_total = 0

            for i in range(0, s):
                len_total += len(dataset.events[i])

            list_val[s] = [item + len_total for item in list_val[s]]

        list_train = [item for sublist in list_train for item in sublist]
        list_val = [item for sublist in list_val for item in sublist]

        train = torch.utils.data.Subset(dataset, list_train)
        val = torch.utils.data.Subset(dataset, list_val)
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        val_loader_mini = DataLoader(val, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

        n_train = len(train_loader)
        n_val = len(val_loader)
        iters_val = int(round(n_val / (num_val_phases_inter - 1)))

        step_inter = 0.01 * len(train_loader)
        step_val_inter = 0.1 * len(train_loader)
        loss_train = 0

        net.train()

        # epoch loop
        # with tqdm(total=n_train, desc='training phase') as pbar:
        for it, batch in enumerate(train_loader):
            # log phase, epoch and iteration
            with open('phase_epoch_iteration.txt', "w") as f:
                f.write('training, ' + str(epoch) + ', ' + str(it))

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
            loss_mano = criterion_mano(mano_pred, true_mano)
            loss_mask = criterion_mask(masks_pred, true_masks)
            loss_total = weight_mask * loss_mask + weight_mano * loss_mano
            loss_total_item = loss_total.item()
            loss_train_inter += loss_total_item
            loss_mano_inter += loss_mano.item()
            loss_mask_inter += loss_mask.item()
            loss_train += loss_total_item

            # backward propagation
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            if global_step >= (num_steps_inter + 1) * step_inter:
                num_steps_inter += 1

                writer.add_scalar('mano loss (inter)', loss_mano_inter / step_inter, global_step)
                writer.add_scalar('mask loss (inter)', loss_mask_inter / step_inter, global_step)
                writer.add_scalar('train loss (inter)', loss_train_inter / step_inter, global_step)

                loss_mano_inter = 0
                loss_mask_inter = 0
                loss_train_inter = 0

            if global_step >= (num_steps_val_inter + 1) * step_val_inter:
                num_steps_val_inter += 1

                # inter validation phase
                net.eval()
                loss_valid = eval_net(net, val_loader_mini, device, epoch, iters_val)
                net.train()
                loss_valid /= iters_val

                writer.add_scalar('valid loss (inter)', loss_valid, global_step)

            global_step += 1

            # pbar.update()

        loss_train /= n_train

        # validation phase
        net.eval()
        loss_valid = eval_net(net, val_loader, device, epoch)
        loss_valid /= n_val
        scheduler.step(loss_valid)

        # log to TensorBoard
        writer.add_scalar('train loss', loss_train, epoch)
        writer.add_scalar('valid loss', loss_valid, epoch)
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
                          'level_split': level_split, 'loss_valid_best': loss_valid_best,
                          'loss_mano_inter': loss_mano_inter, 'loss_mask_inter': loss_mask_inter,
                          'loss_train_inter': loss_train_inter, 'model': net.state_dict(),
                          'num_steps_inter': num_steps_inter,
                          'num_steps_val_inter': num_steps_val_inter, 'optimizer': optimizer.state_dict(),
                          'num_val_phases_inter': num_val_phases_inter,
                          'scheduler': scheduler.state_dict(), 'starts_split': list(starts_split.queue)}

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
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.001,
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

    net = TEHNet()
    checkpoint = None

    if args.load:
        checkpoint = torch.load(args.load, map_location=device)

        net.load_state_dict(checkpoint['model'])
        logging.info(f'Checkpoint loaded from {args.load}')

    net.to(device=device)
    # net = nn.DataParallel(net)

    train_net(net=net, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, device=device,
              val_percent=args.val / 100, checkpoint=checkpoint)
