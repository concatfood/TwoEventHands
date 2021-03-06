import argparse
from eval import eval_net
import logging
import math
from mesh_intersection.bvh_search_tree import BVH
import mesh_intersection.loss as collisions_loss
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
dir_masks = os.path.join('data', 'train', 'masks')
dir_checkpoint = 'checkpoints'

# resolution
res = (240, 180)

# LNES window size
l_lnes = 200

# percentage of data used for training
percentage_scale = 0.1
percentage_data = 0.01 * percentage_scale

# optimization parameters
epochs_max = int(round(100 / percentage_scale))
batch_size = 64
learning_rate = 0.0001
patience = int(round(20 / percentage_scale))
step_size = int(round(10 / percentage_scale))
weight_decay = 0.0

# weights
weight_hands = 1.0
weight_rot = 1.0
weight_root2d = math.sqrt(0.01)
weight_dscale = math.sqrt(1000.0)
weight_masks = 1.0
weight_3d = 100.0
weight_2d = 0.0
weight_pen = 0.0    # 0.0001
weights_hands = torch.cat((weight_rot * torch.ones(96), weight_root2d * torch.ones(2), weight_dscale * torch.ones(1),
                           weight_rot * torch.ones(96), weight_root2d * torch.ones(2), weight_dscale * torch.ones(1)))\
    .cuda()
linear_max = 0.005          # for distance field penetration loss
penalize_outside = False    # for distance field penetration loss
sigma = 0.005               # for distance field penetration loss
max_collisions = 32         # for BVH search tree

# additional losses
cross_entropy = nn.CrossEntropyLoss()
pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=sigma, point2plane=False, vectorized=True,
                                                            penalize_outside=penalize_outside, linear_max=linear_max)


# training function
def train_net(net, device, save_cp=True, checkpoint=None):
    # setup data loader
    dataset = BasicDataset(dir_events, dir_mano, dir_masks, res, l_lnes)

    # split such that one sequences with all camera angles is both the test and validation dataset
    sequences_all = 3 * 21
    sequences_val_start = 1 * 21
    sequences_val_end = 2 * 21
    sequences_train = [s for s in list(range(sequences_all)) if not sequences_val_start <= s < sequences_val_end]
    sequences_val = list(range(sequences_val_start, sequences_val_end))
    list_train = [list(range(dataset.len_until[s], dataset.len_until[s + 1])) if s < sequences_all - 1
                  else list(range(dataset.len_until[s], len(dataset))) for s in sequences_train]
    list_val = [list(range(dataset.len_until[s], dataset.len_until[s + 1])) if s < sequences_all - 1
                else list(range(dataset.len_until[s], len(dataset))) for s in sequences_val]

    list_train = [item for sublist in list_train for item in sublist]

    list_val_np = np.array([item for sublist in list_val for item in sublist])
    indices_list_val_np = np.rint(np.arange(0, len(list_val_np), 1 / percentage_data)).astype(int)
    list_val = list_val_np[indices_list_val_np].tolist()

    train = torch.utils.data.Subset(dataset, list_train)
    val = torch.utils.data.Subset(dataset, list_val)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # optimization and scheduling
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size)

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    # early stopping values
    epoch_best = -1 if checkpoint is None else checkpoint['epoch_best']
    loss_valid_best = float('inf') if checkpoint is None else checkpoint['loss_valid_best']

    # start where we left of last time
    epoch_start = 0 if checkpoint is None else checkpoint['epoch'] + 1
    lr = learning_rate if checkpoint is None else optimizer.param_groups[0]['lr']

    global_step = 0 if checkpoint is None else checkpoint['global_step']

    logging.info(f'''Starting training:
      Start epoch:     {epoch_start}
      Maximum epochs:  {epochs_max}
      Batch size:      {batch_size}
      Learning rate:   {lr}
      Device:          {device.type}
    ''')

    n_train = len(train_loader)
    n_val = len(val_loader)
    iters_train = int(round(percentage_data * n_train))
    iters_val = n_val

    # TensorBoard
    writer = SummaryWriter('runs/TEHNet')

    # dataset loop
    for epoch in range(epoch_start, epochs_max):
        loss_train_hands = 0.0
        loss_train_masks = 0.0
        loss_train_3d = 0.0
        loss_train_2d = 0.0
        loss_train_pen = 0.0
        loss_train = 0.0

        distance_train_3d = 0.0
        distance_train_2d = 0.0

        net.train()

        # epoch loop
        # with tqdm(total=iters_train, desc='training phase') as pbar:
        for it, batch in enumerate(train_loader):
            if it == iters_train:
                break

            # log phase, epoch and iteration
            with open('phase_epoch_iteration.txt', 'w') as f:
                f.write('training, ' + str(epoch) + ', ' + str(it))

            # load data
            lnes = batch['lnes']
            true_hands, true_masks, true_joints_3d, true_joints_2d\
                = batch['hands'], batch['masks'], batch['joints_3d'], batch['joints_2d']
            # true_hands, true_joints_3d, true_joints_2d = batch['hands'], batch['joints_3d'], batch['joints_2d']

            lnes = lnes.to(device=device, dtype=torch.float32)
            true_hands = true_hands.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            true_joints_3d = true_joints_3d.to(device=device, dtype=torch.float32)
            true_joints_2d = true_joints_2d.to(device=device, dtype=torch.float32)

            # forward and loss computation
            hands_pred, verts, masks_pred, joints_3d_pred, joints_2d_pred = net(lnes)
            # hands_pred, verts, joints_3d_pred, joints_2d_pred = net(lnes)

            triangles = verts[:, dataset.faces]
            search_tree = BVH(max_collisions=max_collisions)

            with torch.no_grad():
                collision_idxs = search_tree(triangles)

            diff_hands = weights_hands * (hands_pred - true_hands)
            diff_joints_3d = joints_3d_pred - true_joints_3d
            diff_joints_3d = diff_joints_3d.reshape((-1, diff_joints_3d.shape[2]))
            diff_joints_2d = joints_2d_pred - true_joints_2d
            diff_joints_2d = diff_joints_2d.reshape((-1, diff_joints_2d.shape[2]))

            norm_hands = torch.abs(diff_hands)
            norm_joints_3d = torch.norm(diff_joints_3d, dim=1)
            norm_joints_2d = torch.norm(diff_joints_2d, dim=1)
            distance_train_3d += torch.mean(norm_joints_3d)
            distance_train_2d += torch.mean(norm_joints_2d)

            norm_squared_hands = 0.5 * norm_hands.pow(2)
            norm_squared_joints_3d = 0.5 * norm_joints_3d.pow(2)
            norm_squared_joints_2d = 0.5 * norm_joints_2d.pow(2)
            loss_hands = torch.mean(norm_squared_hands)
            loss_masks = cross_entropy(masks_pred, true_masks)
            loss_3d = torch.mean(norm_squared_joints_3d)
            loss_2d = torch.mean(norm_squared_joints_2d)
            loss_pen = torch.mean(pen_distance(triangles, collision_idxs))

            loss_total = weight_hands * loss_hands + weight_masks * loss_masks + weight_3d * loss_3d\
                         + weight_2d * loss_2d + weight_pen * loss_pen
            # loss_total = weight_hands * loss_hands + weight_3d * loss_3d + weight_2d * loss_2d + weight_pen * loss_pen

            loss_train += loss_total.item()
            loss_train_hands += loss_hands.item()
            loss_train_masks += loss_masks.item()
            loss_train_3d += loss_3d.item()
            loss_train_2d += loss_2d.item()
            loss_train_pen += loss_pen.item()

            # backward propagation
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()

            global_step += 1

            # pbar.update()

        loss_train /= iters_train
        loss_train_hands /= iters_train
        loss_train_masks /= iters_train
        loss_train_3d /= iters_train
        loss_train_2d /= iters_train
        loss_train_pen /= iters_train
        distance_train_3d /= iters_train
        distance_train_2d /= iters_train

        # validation phase
        net.eval()
        loss_valid, loss_valid_hands, loss_valid_masks, loss_valid_3d, loss_valid_2d, loss_valid_pen,\
        distance_valid_3d, distance_valid_2d = eval_net(net, dataset, val_loader, device, epoch, iters_val)
        # loss_valid, loss_valid_hands, loss_valid_3d, loss_valid_2d, loss_valid_pen, distance_valid_3d, distance_valid_2d\
        #     = eval_net(net, dataset, val_loader, device, epoch, iters_val)
        loss_valid /= iters_val
        loss_valid_hands /= iters_val
        loss_valid_masks /= iters_val
        loss_valid_3d /= iters_val
        loss_valid_2d /= iters_val
        loss_valid_pen /= iters_val
        distance_valid_3d /= iters_val
        distance_valid_2d /= iters_val

        scheduler.step()

        # log to TensorBoard
        writer.add_scalar('train loss total', loss_train, epoch)
        writer.add_scalar('train loss hands', loss_train_hands, epoch)
        writer.add_scalar('train loss masks', loss_train_masks, epoch)
        writer.add_scalar('train loss 3d', loss_train_3d, epoch)
        writer.add_scalar('train loss 2d', loss_train_2d, epoch)
        writer.add_scalar('train loss pen', loss_train_pen, epoch)
        writer.add_scalar('train distance 3d', distance_train_3d, epoch)
        writer.add_scalar('train distance 2d', distance_train_2d, epoch)

        writer.add_scalar('valid loss total', loss_valid, epoch)
        writer.add_scalar('valid loss hands', loss_valid_hands, epoch)
        writer.add_scalar('valid loss masks', loss_valid_masks, epoch)
        writer.add_scalar('valid loss 3d', loss_valid_3d, epoch)
        writer.add_scalar('valid loss 2d', loss_valid_2d, epoch)
        writer.add_scalar('valid loss pen', loss_valid_pen, epoch)
        writer.add_scalar('valid distance 3d', distance_valid_3d, epoch)
        writer.add_scalar('valid distance 2d', distance_valid_2d, epoch)

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
    parser = argparse.ArgumentParser(description='Train the UNet on LNES',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', dest='model', type=str, default=False, help='Load model from a .pth file')

    return parser.parse_args()


# main function
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = TEHNet()
    checkpoint = None

    if args.model:
        checkpoint = torch.load(os.path.join(dir_checkpoint, args.model), map_location=device)

        net.load_state_dict(checkpoint['model'])
        logging.info(f'Checkpoint loaded from {args.model}')

    net.to(device=device)
    # net = nn.DataParallel(net)

    train_net(net=net, device=device, checkpoint=checkpoint)
