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


# training function
def train_net(net,
              device,
              epochs=150,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    # setup data loader
    dataset = BasicDataset(dir_events, dir_mano, dir_mask, 100, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    # TensorBoard
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Scaling:  {img_scale}
    ''')

    # optimization and scheduling
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    # losses
    criterion_mask = nn.CrossEntropyLoss()
    criterion_mano = nn.MSELoss()

    # early stopping values
    epoch_best = -1
    val_score_best = float('inf')

    # dataset loop
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0

        # epoch loop
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
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
                loss_total = loss_mask + loss_mano
                epoch_loss += loss_total
                writer.add_scalar('Loss/train', loss_total.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss_total.item()})

                # backward propagation
                optimizer.zero_grad()
                loss_total.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(lnes.shape[0])
                global_step += 1

                # evaluation (intra-epoch)
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    logging.info('Validation cross entropy: {}'.format(val_score))
                    writer.add_scalar('Loss/test', val_score, global_step)

        # evaluation (inter-epoch)
        val_score_epoch = eval_net(net, val_loader, device)

        # check if best epoch
        if val_score_epoch < val_score_best:
            epoch_best = epoch
            val_score_best = val_score_epoch

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
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

            if epoch == epoch_best:
                copyfile(dir_checkpoint + f'CP_epoch{epoch + 1}.pth', dir_checkpoint + f'CP_epoch_best.pth')

    writer.close()


# parse arguments
def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on LNES and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
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
