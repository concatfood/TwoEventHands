import torch
import torch.nn.functional as F
from tqdm import tqdm


# weights
weight_mano = 1.0
weight_rot = 1.0
weight_trans = 500
weight_3d = 0.1
weight_2d = 0.004602373**2 * weight_3d
threshold_3d = 0.05     # 5cm (per dimension)
threshold_2d = 10       # 10px (per dimension)


# evaluate network
def eval_net(net, loader, device, epoch):
    loss_valid_mano = 0
    loss_valid_3d = 0
    loss_valid_2d = 0
    loss_valid = 0

    distance_valid_mano = 0
    distance_valid_3d = 0
    distance_valid_2d = 0

    global_step = 0

    # with tqdm(total=num_it, desc='validation phase') as pbar:
    for it, batch in enumerate(loader):
        lnes = batch['lnes']
        true_mano, true_joints_3d, true_joints_2d = batch['mano'], batch['joints_3d'], batch['joints_2d']
        lnes = lnes.to(device=device, dtype=torch.float32)
        true_mano = true_mano.to(device=device, dtype=torch.float32)
        true_joints_3d = true_joints_3d.to(device=device, dtype=torch.float32)
        true_joints_2d = true_joints_2d.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            mano_pred, joints_3d_pred, joints_2d_pred = net(lnes)

        # weighting
        loss_rot = F.mse_loss(torch.cat((mano_pred[:, 0:96], mano_pred[:, 99:195]), 1),
                              torch.cat((true_mano[:, 0:96], true_mano[:, 99:195]), 1))
        loss_trans = F.mse_loss(torch.cat((mano_pred[:, 96:99], mano_pred[:, 195:198]), 1),
                                torch.cat((true_mano[:, 96:99], true_mano[:, 195:198]), 1))
        loss_mano = weight_rot * loss_rot + weight_trans * loss_trans
        loss_3d = threshold_3d * F.smooth_l1_loss(joints_3d_pred, true_joints_3d)
        loss_2d = threshold_2d * F.smooth_l1_loss(joints_2d_pred, true_joints_2d)
        loss_valid_mano += loss_mano.item()
        loss_valid_3d += loss_3d.item()
        loss_valid_2d += loss_2d.item()
        loss_total = weight_mano * loss_mano + weight_3d * loss_3d + weight_2d * loss_2d

        loss_valid += loss_total.item()

        # average accuracies calculated using l2 norm
        diff_mano = mano_pred - true_mano
        diff_mano = diff_mano.reshape((diff_mano.shape[0], diff_mano.shape[1]))
        diff_joints_3d = joints_3d_pred - true_joints_3d
        diff_joints_3d = diff_joints_3d.reshape((diff_joints_3d.shape[0] * diff_joints_3d.shape[1],
                                                 diff_joints_3d.shape[2]))
        diff_joints_2d = joints_2d_pred - true_joints_2d
        diff_joints_2d = diff_joints_2d.reshape((diff_joints_2d.shape[0] * diff_joints_2d.shape[1],
                                                 diff_joints_2d.shape[2]))
        distance_valid_mano += torch.mean(torch.norm(diff_mano, dim=1))
        distance_valid_3d += torch.mean(torch.norm(diff_joints_3d, dim=1))
        distance_valid_2d += torch.mean(torch.norm(diff_joints_2d, dim=1))

        # log phase, epoch and iteration
        with open('phase_epoch_iteration.txt', "w") as f:
            f.write('validation, ' + str(epoch) + ', ' + str(it))

        global_step += 1

        # pbar.update()

    return loss_valid, loss_valid_mano, loss_valid_3d, loss_valid_2d, distance_valid_mano, distance_valid_3d,\
           distance_valid_2d
