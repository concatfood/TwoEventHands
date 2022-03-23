import math
from mesh_intersection.bvh_search_tree import BVH
import mesh_intersection.loss as collisions_loss
import torch
import torch.nn as nn
from tqdm import tqdm


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

# search tree for penetration loss

# additional losses
cross_entropy = nn.CrossEntropyLoss()
pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=sigma, point2plane=False, vectorized=True,
                                                            penalize_outside=penalize_outside, linear_max=linear_max)


# evaluate network
def eval_net(net, dataset, loader, device, epoch, num_it):
    loss_valid_hands = 0.0
    loss_valid_masks = 0.0
    loss_valid_3d = 0.0
    loss_valid_2d = 0.0
    loss_valid_pen = 0.0
    loss_valid = 0.0

    distance_valid_3d = 0.0
    distance_valid_2d = 0.0

    global_step = 0

    # with tqdm(total=num_it, desc='validation phase') as pbar:
    for it, batch in enumerate(loader):
        lnes = batch['lnes']

        true_hands, true_masks, true_joints_3d, true_joints_2d\
            = batch['hands'], batch['masks'], batch['joints_3d'], batch['joints_2d']
        # true_hands, true_joints_3d, true_joints_2d = batch['hands'], batch['joints_3d'], batch['joints_2d']

        lnes = lnes.to(device=device, dtype=torch.float32)
        true_hands = true_hands.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)
        true_joints_3d = true_joints_3d.to(device=device, dtype=torch.float32)
        true_joints_2d = true_joints_2d.to(device=device, dtype=torch.float32)

        search_tree = BVH(max_collisions=max_collisions)

        with torch.no_grad():
            hands_pred, verts, masks_pred, joints_3d_pred, joints_2d_pred = net(lnes)
            # hands_pred, verts, joints_3d_pred, joints_2d_pred = net(lnes)
            triangles = verts[:, dataset.faces]
            collision_idxs = search_tree(triangles)

        diff_hands = weights_hands * (hands_pred - true_hands)
        diff_joints_3d = joints_3d_pred - true_joints_3d
        diff_joints_3d = diff_joints_3d.reshape((-1, diff_joints_3d.shape[2]))
        diff_joints_2d = joints_2d_pred - true_joints_2d
        diff_joints_2d = diff_joints_2d.reshape((-1, diff_joints_2d.shape[2]))

        norm_hands = torch.abs(diff_hands)
        norm_joints_3d = torch.norm(diff_joints_3d, dim=1)
        norm_joints_2d = torch.norm(diff_joints_2d, dim=1)
        distance_valid_3d += torch.mean(norm_joints_3d)
        distance_valid_2d += torch.mean(norm_joints_2d)

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

        loss_valid += loss_total.item()
        loss_valid_hands += loss_hands.item()
        loss_valid_masks += loss_masks.item()
        loss_valid_3d += loss_3d.item()
        loss_valid_2d += loss_2d.item()
        loss_valid_pen += loss_pen.item()

        # log phase, epoch and iteration
        with open('phase_epoch_iteration.txt', "w") as f:
            f.write('validation, ' + str(epoch) + ', ' + str(it))

        global_step += 1

        # pbar.update()

    return loss_valid, loss_valid_hands, loss_valid_masks, loss_valid_3d, loss_valid_2d, loss_valid_pen,\
           distance_valid_3d, distance_valid_2d
    # return loss_valid, loss_valid_hands, loss_valid_3d, loss_valid_2d, loss_valid_pen, distance_valid_3d,\
    #        distance_valid_2d
