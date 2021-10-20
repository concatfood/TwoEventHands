import argparse
# import cv2 as cv
from manopth.manolayer import ManoLayer
import math
from mesh_intersection.bvh_search_tree import BVH
import mesh_intersection.loss as collisions_loss
import numpy as np
import os
from pathlib import Path
import pickle
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.transforms import quaternion_to_axis_angle
from pytorch3d.transforms import rotation_6d_to_matrix
from TEHNet import TEHNet
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
# import torch.nn.functional as F
from utils.dataset import BasicDataset
from utils.one_euro_filter import OneEuroFilter


# relative directories
dir_events = os.path.join('data', 'test', 'events')
dir_mano = os.path.join('data', 'train', 'mano')
dir_output = 'output'

# LNES window length
l_lnes = 200

# camera parameters
res = (240, 180)
fov = 45.0
fovy = np.radians(fov)
focal = 0.5 * res[1] / math.tan(fovy / 2.0)
mat_cam = torch.from_numpy(np.array([[focal, 0.0, -res[0] / 2.0],
                                     [0.0, -focal, -res[1] / 2.0],
                                     [0.0, 0.0, -1.0]])).float().cuda()
shape = torch.zeros((1, 10)).cuda()

# framerates
fps_in = 1000
fps_out = 30

linear_max = 0.005          # for distance field penetration loss
penalize_outside = False    # for distance field penetration loss
sigma = 0.005               # for distance field penetration loss
max_collisions = 32         # for BVH search tree

pen_distance = collisions_loss.DistanceFieldPenetrationLoss(sigma=sigma, point2plane=False, vectorized=True,
                                                            penalize_outside=penalize_outside, linear_max=linear_max)

# temporal filtering
one_euro_filter = None


# evaluate l2 loss
def evaluate_l2(out_1, out_2):
    diff = out_1 - out_2
    norm = np.linalg.norm(diff, axis=1)

    return np.mean(norm)


# load all events
def load_events(name_sequence):
    # load events
    files_events = sorted([name for name in os.listdir(os.path.join(dir_events, name_sequence))
                           if os.path.isfile(os.path.join(dir_events, name_sequence, name))
                           and name.endswith('.pkl')])

    frames_events_total = []

    for events_file in files_events:
        path_events = os.path.join(dir_events, name_sequence, events_file)

        with open(path_events, 'rb') as file:
            frames_events = pickle.load(file)

        frames_events_total.extend(frames_events)

    return frames_events_total


# load all mano frames
def load_mano(name_gt):
    files_mano = sorted([name for name in os.listdir(os.path.join(dir_mano, name_gt))
                         if os.path.isfile(os.path.join(dir_mano, name_gt, name))
                         and name.endswith('.pkl')])

    frames_mano_total = {}

    for file_mano in files_mano:
        path_mano = os.path.join(dir_mano, name_gt, file_mano)

        with open(path_mano, 'rb') as file:
            frames_mano = pickle.load(file)
            frames_mano_total.update(frames_mano)

    return frames_mano_total


# predict and MANO parameters
def predict(net, lnes, device, t, mano_gt):
    global one_euro_filter

    net.eval()

    vertices_gt, joints_3d_gt, joints_2d_gt = None, None, None

    # compute ground truth joint positions
    if mano_gt is not None:
        mano_gt = np.concatenate((mano_gt[0]['pose'], mano_gt[0]['trans'], mano_gt[1]['pose'], mano_gt[1]['trans']))

        params_axisangle_gt_pth = torch.unsqueeze(torch.from_numpy(mano_gt), 0).float().cuda()
        vertices_gt_right, joints_gt_right = net.layer_mano_right(params_axisangle_gt_pth[:, 0:48], th_betas=shape,
                                                                  th_trans=params_axisangle_gt_pth[:, 48:51])
        vertices_gt_left, joints_gt_left = net.layer_mano_left(params_axisangle_gt_pth[:, 51:99], th_betas=shape,
                                                               th_trans=params_axisangle_gt_pth[:, 99:102])

        vertices_gt = torch.cat((vertices_gt_right, vertices_gt_left), 1)
        joints_3d_gt = torch.cat((joints_gt_right, joints_gt_left), 1)
        joints_intrinsic_gt = torch.matmul(joints_3d_gt, torch.transpose(mat_cam, 0, 1))
        joints_2d_gt = joints_intrinsic_gt[..., 0:2] / joints_intrinsic_gt[..., [2]]
        joints_3d_gt = joints_3d_gt.squeeze(0).cpu().numpy()
        joints_2d_gt = joints_2d_gt.squeeze(0).cpu().numpy()
    else:
        joints_3d_gt = None
        joints_2d_gt = None

    lnes = torch.from_numpy(lnes)
    lnes = lnes.unsqueeze(0)
    lnes = lnes.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        # mano_output, vertices_output, mask_output, joints_3d_output, joints_2d_output = net(lnes)
        mano_output, vertices_output, joints_3d_output, joints_2d_output = net(lnes)
        params = mano_output.squeeze(0)
        # mask_output = mask_output.squeeze(0)
        # mask_output = F.softmax(mask_output, dim=0)
        joints_3d_output = joints_3d_output.squeeze(0)
        joints_2d_output = joints_2d_output.squeeze(0)
        params = params.cpu().numpy()
        # mask_output = mask_output.cpu().numpy()
        # indices_max = np.argmax(mask_output, axis=0)
        # mask_output = 255 * (np.arange(3) == indices_max[..., None]).astype(int)
        joints_3d_output = joints_3d_output.cpu().numpy()
        joints_2d_output = joints_2d_output.cpu().numpy()

    if one_euro_filter is None:
        one_euro_filter = OneEuroFilter(t, params, dx0=np.array([0.0]), min_cutoff=np.array([1.0]),
                                        beta=np.array([0.0]), d_cutoff=np.array([1.0]))
        params_smoothed = params
    else:
        params_smoothed = one_euro_filter(t, params)

    params_axisangle = np.zeros(102)
    params_axisangle_smoothed = np.zeros(102)

    for h in range(2):
        params_axisangle[h * 51 + 0:h * 51 + 48] = quaternion_to_axis_angle(matrix_to_quaternion(rotation_6d_to_matrix(
            torch.from_numpy(params[h * 99 + 0:h * 99 + 96].reshape(16, 6))))).reshape(48)
        params_axisangle[h * 51 + 48:h * 51 + 51] = params[h * 99 + 96:h * 99 + 99]

        params_axisangle_smoothed[h * 51 + 0:h * 51 + 48] = quaternion_to_axis_angle(matrix_to_quaternion(
            rotation_6d_to_matrix(torch.from_numpy(params_smoothed[h * 99 + 0:h * 99 + 96].reshape(16, 6)))))\
            .reshape(48)
        params_axisangle_smoothed[h * 51 + 48:h * 51 + 51] = params_smoothed[h * 99 + 96:h * 99 + 99]

    # compute smoothed joints positions
    params_axisangle_smoothed_pth = torch.unsqueeze(torch.from_numpy(params_axisangle_smoothed), 0).float().cuda()
    vertices_smoothed_right, joints_smoothed_right = net.layer_mano_right(params_axisangle_smoothed_pth[:, 0:48],
                                                                          th_betas=shape,
                                                                          th_trans=
                                                                          params_axisangle_smoothed_pth[:, 48:51])
    vertices_smoothed_left, joints_smoothed_left = net.layer_mano_left(params_axisangle_smoothed_pth[:, 51:99],
                                                                       th_betas=shape,
                                                                       th_trans=
                                                                       params_axisangle_smoothed_pth[:, 99:102])

    vertices_smoothed = torch.cat((vertices_smoothed_right, vertices_smoothed_left), 1)
    joints_3d_smoothed = torch.cat((joints_smoothed_right, joints_smoothed_left), 1)
    joints_intrinsic_smoothed = torch.matmul(joints_3d_smoothed, torch.transpose(mat_cam, 0, 1))
    joints_2d_smoothed = joints_intrinsic_smoothed[..., 0:2] / joints_intrinsic_smoothed[..., [2]]
    joints_3d_smoothed = torch.unsqueeze(joints_3d_smoothed, 0).cpu().numpy()
    joints_2d_smoothed = torch.unsqueeze(joints_2d_smoothed, 0).cpu().numpy()

    # out_net = (params_axisangle, vertices_output, joints_3d_output, joints_2d_output, mask_output)
    out_net = (params_axisangle, vertices_output, joints_3d_output, joints_2d_output)
    out_smoothed = (params_axisangle_smoothed, vertices_smoothed, joints_3d_smoothed, joints_2d_smoothed)
    out_gt = (mano_gt, vertices_gt, joints_3d_gt, joints_2d_gt)

    return out_net, out_smoothed, out_gt


# parse arguments
def get_args():
    parser = argparse.ArgumentParser(description='Predict mano parameters from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='best.pth', metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', help='Name of input sequence', required=True)
    parser.add_argument('--gt', '-g', default='', metavar='INPUT', help='Name of ground truth sequence')

    return parser.parse_args()


# main function
if __name__ == "__main__":
    args = get_args()
    name_sequence = args.input
    name_gt = args.gt

    events = load_events(name_sequence)
    mano_gt_all = load_mano(name_gt) if name_gt != '' else None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = TEHNet()
    net.to(device=device)
    net.load_state_dict(torch.load(os.path.join('checkpoints', args.model), map_location=device)['model'])

    layer_mano_right = ManoLayer(flat_hand_mean=True, side='right', mano_root='mano', use_pca=False,
                                 root_rot_mode='axisang', joint_rot_mode='axisang')
    layer_mano_left = ManoLayer(flat_hand_mean=True, side='left', mano_root='mano', use_pca=False,
                                root_rot_mode='axisang', joint_rot_mode='axisang')

    pose_zero = torch.zeros((1, 48))
    shape_zero = torch.zeros((1, 10))
    trans_zero = torch.zeros((1, 3))

    vertices_right_zero, joints_right_zero = layer_mano_right(pose_zero, th_betas=shape_zero, th_trans=trans_zero)
    vertices_left_zero, joints_left_zero = layer_mano_left(pose_zero, th_betas=shape_zero, th_trans=trans_zero)

    faces = torch.cat((layer_mano_right.th_faces, layer_mano_left.th_faces + vertices_right_zero.shape[1]), 0)\
        .to(device='cuda', dtype=torch.int64)

    dir_sequence = os.path.join(dir_output, name_sequence)
    Path(dir_sequence).mkdir(parents=True, exist_ok=True)

    # Path(os.path.join(dir_output, name_sequence, 'masks')).mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(os.path.join(dir_sequence, 'metrics'))

    mano_pred_seq = {}
    mano_pred_seq_smoothed = {}

    distance_joints_2d_mean = 0.0
    distance_joints_2d_smoothed_mean = 0.0
    distance_joints_3d_mean = 0.0
    distance_joints_3d_smoothed_mean = 0.0
    loss_pen_mean = 0.0
    loss_pen_smoothed_mean = 0.0

    for i_f, f_float in enumerate(tqdm(np.arange(0, len(events), fps_in / fps_out))):
        f = int(round(f_float))     # in milliseconds

        mano_gt_f = mano_gt_all[f] if mano_gt_all is not None else None

        frames = events[max(0, f - l_lnes + 1):f + 1]
        lnes = BasicDataset.preprocess_events(frames, f - l_lnes + 1, res)
        out_net, out_smoothed, out_gt = predict(net=net, lnes=lnes, device=device, t=f / 1000, mano_gt=mano_gt_f)
        # mano_pred, joints_3d_pred, joints_2d_pred, mask_out = out_net
        mano_pred, verts_pred, joints_3d_pred, joints_2d_pred = out_net
        mano_pred_smoothed, verts_pred_smoothed, joints_3d_pred_smoothed, joints_2d_pred_smoothed = out_smoothed
        mano_gt, verts_gt, joints_3d_gt, joints_2d_gt = out_gt

        seq_dict = {i_f: [{'pose': mano_pred[0:48],
                           'shape': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                           'trans': mano_pred[48:51],
                           'hand_type': 'right'},
                          {'pose': mano_pred[51:99],
                           'shape': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                           'trans': mano_pred[99:102],
                           'hand_type': 'left'}]}

        seq_dict_smoothed = {i_f: [{'pose': mano_pred_smoothed[0:48],
                                    'shape': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                    'trans': mano_pred_smoothed[48:51],
                                    'hand_type': 'right'},
                                   {'pose': mano_pred_smoothed[51:99],
                                    'shape': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                                    'trans': mano_pred_smoothed[99:102],
                                    'hand_type': 'left'}]}

        mano_pred_seq.update(seq_dict)
        mano_pred_seq_smoothed.update(seq_dict_smoothed)

        triangles = verts_pred[:, faces]
        search_tree = BVH(max_collisions=max_collisions)

        with torch.no_grad():
            collision_idxs = search_tree(triangles)

        loss_pen = torch.mean(pen_distance(triangles, collision_idxs))

        triangles_smoothed = verts_pred_smoothed[:, faces]
        search_tree_smoothed = BVH(max_collisions=max_collisions)

        with torch.no_grad():
            collision_idxs_smoothed = search_tree_smoothed(triangles_smoothed)

        loss_pen_smoothed = torch.mean(pen_distance(triangles_smoothed, collision_idxs_smoothed))

        writer.add_scalar('penetration loss', loss_pen, f)
        writer.add_scalar('penetration loss (smoothed)', loss_pen_smoothed, f)

        loss_pen_mean += loss_pen.item()
        loss_pen_smoothed_mean += loss_pen_smoothed.item()

        if mano_gt_all is not None:
            distance_joints_3d = evaluate_l2(joints_3d_pred, joints_3d_gt)
            distance_joints_3d_smoothed = evaluate_l2(joints_3d_pred_smoothed, joints_3d_gt)
            distance_joints_2d = evaluate_l2(joints_2d_pred, joints_2d_gt)
            distance_joints_2d_smoothed = evaluate_l2(joints_2d_pred_smoothed, joints_2d_gt)

            writer.add_scalar('distance 3d', distance_joints_3d, f)
            writer.add_scalar('distance 3d (smoothed)', distance_joints_3d_smoothed, f)
            writer.add_scalar('distance 2d', distance_joints_2d, f)
            writer.add_scalar('distance 2d (smoothed)', distance_joints_2d_smoothed, f)

            distance_joints_3d_mean += distance_joints_3d
            distance_joints_3d_smoothed_mean += distance_joints_3d_smoothed
            distance_joints_2d_mean += distance_joints_2d
            distance_joints_2d_smoothed_mean += distance_joints_2d_smoothed

        # cv.imwrite(os.path.join(dir_output, name_sequence, 'masks', 'frame_'
        #                         + str(i_f + 1).zfill(len(str(int(round(len(events) * fps_out / fps_in)) - 1))))
        #            + '.png', mask_out[:, :, ::-1])

    num_iterations = int(math.floor(len(events) * fps_out / fps_in))

    loss_pen_mean /= num_iterations
    loss_pen_smoothed_mean /= num_iterations

    print('mean penetration loss', loss_pen_mean)
    print('mean penetration loss (smoothed)', loss_pen_smoothed_mean)

    if mano_gt_all is not None:
        distance_joints_3d_mean /= num_iterations
        distance_joints_3d_smoothed_mean /= num_iterations
        distance_joints_2d_mean /= num_iterations
        distance_joints_2d_smoothed_mean /= num_iterations

        print('mean distance of 3d joints', distance_joints_3d_mean)
        print('mean distance of 3d joints (smoothed)', distance_joints_3d_smoothed_mean)
        print('mean distance of 2d joints', distance_joints_2d_mean)
        print('mean distance of 2d joints (smoothed)', distance_joints_2d_smoothed_mean)

    out_mano = os.path.join(dir_sequence, 'sequence_mano.pkl')
    out_mano_smoothed = os.path.join(dir_sequence, 'sequence_mano_smoothed.pkl')

    with open(out_mano, 'wb') as file:
        pickle.dump(mano_pred_seq, file)

    with open(out_mano_smoothed, 'wb') as file:
        pickle.dump(mano_pred_seq_smoothed, file)
