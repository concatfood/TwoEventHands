import torch
import torch.nn.functional as F
from tqdm import tqdm


# weights
weight_mano = 1.0
weight_3d = 0.0
weight_2d = 0.0


# evaluate network
def eval_net(net, loader, device, epoch, num_it=-1):
    loss_valid_mano = 0
    loss_valid_3d = 0
    loss_valid_2d = 0
    loss_valid = 0
    global_step = 0

    # with tqdm(total=num_it, desc='validation phase') as pbar:
    for it, batch in enumerate(loader):
        if it >= num_it:
            break

        lnes = batch['lnes']
        true_mano, true_joints_3d, true_joints_2d = batch['mano'], batch['joints_3d'], batch['joints_2d']
        lnes = lnes.to(device=device, dtype=torch.float32)
        true_mano = true_mano.to(device=device, dtype=torch.float32)
        true_joints_3d = true_joints_3d.to(device=device, dtype=torch.float32)
        true_joints_2d = true_joints_2d.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            mano_pred, joints_3d_pred, joints_2d_pred = net(lnes)

        loss_mano = weight_mano * F.mse_loss(mano_pred, true_mano)
        loss_3d = weight_3d * F.mse_loss(joints_3d_pred, true_joints_3d)
        loss_2d = weight_2d * F.mse_loss(joints_2d_pred, true_joints_2d)
        loss_valid_mano += loss_mano.item()
        loss_valid_3d += loss_3d.item()
        loss_valid_2d += loss_2d.item()
        loss_total = loss_mano + loss_3d + loss_2d

        loss_valid += loss_total.item()

        # log phase, epoch and iteration
        with open('phase_epoch_iteration.txt', "w") as f:
            f.write('validation, ' + str(epoch) + ', ' + str(it))

        global_step += 1

        # pbar.update()

    return loss_valid, loss_valid_mano, loss_valid_3d, loss_valid_2d
