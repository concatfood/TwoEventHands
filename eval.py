import torch
from tqdm import tqdm


# weights
weight_mano = 1.0
weight_rot = 1.0
weight_trans = 500
weight_3d = weight_trans
weight_2d = 0.0
weights_mano = torch.cat((weight_rot * torch.ones(96), weight_trans * torch.ones(3),
                          weight_rot * torch.ones(96), weight_trans * torch.ones(3))).cuda()


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

        diff_mano = weights_mano * (mano_pred - true_mano)
        diff_joints_3d = joints_3d_pred - true_joints_3d
        diff_joints_3d = diff_joints_3d.reshape((-1, diff_joints_3d.shape[2]))
        diff_joints_2d = joints_2d_pred - true_joints_2d
        diff_joints_2d = diff_joints_2d.reshape((-1, diff_joints_2d.shape[2]))

        norm_mano = torch.abs(diff_mano)
        norm_joints_3d = torch.norm(diff_joints_3d, dim=1)
        norm_joints_2d = torch.norm(diff_joints_2d, dim=1)
        distance_valid_mano += torch.mean(norm_mano)
        distance_valid_3d += torch.mean(norm_joints_3d)
        distance_valid_2d += torch.mean(norm_joints_2d)

        norm_squared_mano = 0.5 * norm_mano.pow(2)
        norm_squared_joints_3d = 0.5 * norm_joints_3d.pow(2)
        norm_squared_joints_2d = 0.5 * norm_joints_2d.pow(2)
        loss_mano = torch.mean(norm_squared_mano)
        loss_3d = torch.mean(norm_squared_joints_3d)
        loss_2d = torch.mean(norm_squared_joints_2d)

        loss_total = weight_mano * loss_mano + weight_3d * loss_3d + weight_2d * loss_2d

        loss_valid += loss_total.item()
        loss_valid_mano += loss_mano.item()
        loss_valid_3d += loss_3d.item()
        loss_valid_2d += loss_2d.item()

        # log phase, epoch and iteration
        with open('phase_epoch_iteration.txt', "w") as f:
            f.write('validation, ' + str(epoch) + ', ' + str(it))

        global_step += 1

        # pbar.update()

    return loss_valid, loss_valid_mano, loss_valid_3d, loss_valid_2d, distance_valid_mano, distance_valid_3d,\
           distance_valid_2d
