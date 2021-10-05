import ffmpeg
import logging
from manopth.manolayer import ManoLayer
import math
import numpy as np
import os
import os.path
from pathlib import Path
import pickle
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.transforms import axis_angle_to_quaternion
from pytorch3d.transforms import matrix_to_quaternion
from pytorch3d.transforms import matrix_to_rotation_6d
from pytorch3d.transforms import quaternion_multiply
from pytorch3d.transforms import quaternion_to_axis_angle
from pytorch3d.transforms import quaternion_to_matrix
from pytorch3d.transforms import rotation_6d_to_matrix
import torch
from torch.utils.data import Dataset


# name of dataset
prefix_dataset = 'raw_sequence'

# numpy data type
dtype_events = np.dtype([('t', np.int64), ('x', np.int16), ('y', np.int16), ('p', np.int8)])

# maximum number of events and mano parameter vectors per file
interval = 1000
interval_masks = 10

# camera parameters
res = (240, 180)
far = 1.0
fov = 45.0
fovy = np.radians(fov)
focal = 0.5 * res[1] / math.tan(fovy / 2.0)
mat_cam = np.array([[focal, 0.0, -res[0] / 2.0],    # y and z negative because of different coordinate systems
                    [0.0, -focal, -res[1] / 2.0],
                    [0.0, 0.0, -1.0]])

# camera position relative to hands in MANO space
hands_avg = np.array([0.0, 0.0, -0.5])

angles_augmentation = {'0': 5.0 / 360.0 * (2.0 * math.pi), '1': 10.0 / 360.0 * (2.0 * math.pi),
                       '2': 15.0 / 360.0 * (2.0 * math.pi), '3': 20.0 / 360.0 * (2.0 * math.pi),
                       '4': 25.0 / 360.0 * (2.0 * math.pi)}
angles_position = {'0': None, '1': 45.0 / 360.0 * (2 * math.pi), '2': 135.0 / 360.0 * (2 * math.pi),
                   '3': 225.0 / 360.0 * (2 * math.pi), '4': 315.0 / 360.0 * (2 * math.pi)}

# lnes augmentation parameters
percentage_polarity_changes = 0.05
percentage_noisy_pixels_n = 0.01
percentage_noisy_pixels_p = 0.01


# dataset loader
class BasicDataset(Dataset):
    def __init__(self, events_dir, mano_dir, mask_dir, res, l_lnes):
        self.events_dir = events_dir
        self.mano_dir = mano_dir
        self.mask_dir = mask_dir
        self.res = res
        self.l_lnes = l_lnes
        self.num_frames = []

        self.events_dirs = sorted([directory for directory in os.listdir(self.events_dir)
                                   if os.path.isdir(os.path.join(self.events_dir, directory))])

        self.ids = []
        idx = 0

        for d, directory in enumerate(self.events_dirs):
            files_events = sorted([name for name in os.listdir(os.path.join(self.events_dir, directory))
                                   if os.path.isfile(os.path.join(self.events_dir, directory, name))
                                   and name.endswith('.pkl')])

            with open(os.path.join(os.path.join(self.events_dir, directory, files_events[0])), 'rb') as file:
                length_first = len(pickle.load(file))

                if length_first != interval:
                    print(f'dataset deployment interval is {length_first}, but should be {interval}')

            with open(os.path.join(os.path.join(self.events_dir, directory, files_events[-1])), 'rb') as file:
                length_last = len(pickle.load(file))

            self.num_frames.append(length_first * (len(files_events) - 1) + length_last)

            for f in range(length_first * (len(files_events) - 1) + length_last):
                self.ids.append(idx)
                idx += 1

        self.len_until = np.zeros(len(self.events_dirs), dtype=np.int32)

        for n in range(len(self.num_frames)):
            self.len_until[n + 1:] += self.num_frames[n]

        self.len_until_reverse = self.len_until[::-1]

        self.layer_mano_right = ManoLayer(flat_hand_mean=True, side='right', mano_root='mano', use_pca=False,
                                          root_rot_mode='axisang', joint_rot_mode='axisang')
        self.layer_mano_left = ManoLayer(flat_hand_mean=True, side='left', mano_root='mano', use_pca=False,
                                         root_rot_mode='axisang', joint_rot_mode='axisang')

        # assume constant roots because of constant shapes
        pose_zero = torch.zeros((1, 48))
        shape_zero = torch.zeros((1, 10))
        trans_zero = torch.zeros((1, 3))

        _, joints_right = self.layer_mano_right(pose_zero, th_betas=shape_zero, th_trans=trans_zero)
        _, joints_left = self.layer_mano_left(pose_zero, th_betas=shape_zero, th_trans=trans_zero)
        self.roots = [joints_right[0, 0, ...].numpy(), joints_left[0, 0, ...].numpy()]

        logging.info(f'The dataset contains {len(self.ids)} frames.')

    # length equals to number of sequences time number of (frames - LNES window length)
    def __len__(self):
        return len(self.ids)

    # compute intrinsic matrix
    @classmethod
    def compute_camera_matrix(cls, res):
        f = 0.5 * res[1] / math.tan(fovy / 2.0)

        mat_cam = np.array([[f, 0.0, -res[0] / 2.0],  # y and z negative because of different coordinate systems
                            [0.0, -f, -res[1] / 2.0],
                            [0.0, 0.0, -1.0]])

        return mat_cam

    # compute vertices and joints given mano parameters with axis-angle rotations
    @classmethod
    def compute_vertices_and_joints(cls, mano_params, layer_mano_right, layer_mano_left):
        mano_params = torch.from_numpy(mano_params[np.newaxis, ...]).float()

        axisangle_right = quaternion_to_axis_angle(matrix_to_quaternion(rotation_6d_to_matrix(
            mano_params[:, 0:96].reshape(16, 6)))).reshape(1, 48)
        trans_right = mano_params[:, 96:99]
        axisangle_left = quaternion_to_axis_angle(matrix_to_quaternion(rotation_6d_to_matrix(
            mano_params[:, 99:195].reshape(16, 6)))).reshape(1, 48)
        trans_left = mano_params[:, 195:198]
        shape = torch.zeros((mano_params.shape[0], 10))

        vertices_right, joints_right = layer_mano_right(axisangle_right, th_betas=shape, th_trans=trans_right)
        vertices_left, joints_left = layer_mano_left(axisangle_left, th_betas=shape, th_trans=trans_left)

        vertices_right = vertices_right.numpy().squeeze()
        vertices_left = vertices_left.numpy().squeeze()
        joints_right = joints_right.numpy().squeeze()
        joints_left = joints_left.numpy().squeeze()

        return np.concatenate((vertices_right, vertices_left), 0), np.concatenate((joints_right, joints_left), 0)

    # convert events to LNES frames
    @classmethod
    def preprocess_events(cls, frames, f_start, size):
        lnes = np.zeros((2, size[1], size[0]))

        for t in range(len(frames)):
            ts, xs, ys, ps = frames[t]['t'], frames[t]['x'].astype(int), frames[t]['y'].astype(int),\
                             frames[t]['p'].astype(int)
            lnes[ps, ys, xs] = (ts - f_start) / len(frames)

        return lnes

    # convert events to LNES frames
    @classmethod
    def preprocess_events_noisy(cls, frames, f_start, size):
        num_pixels_total = size[0] * size[1]
        lnes = np.zeros((2, size[1], size[0]))

        num_pixels_polarity_changes = int(round(num_pixels_total * percentage_polarity_changes))
        xs_polarity_changes = np.random.randint(low=0, high=size[0], size=num_pixels_polarity_changes)
        ys_polarity_changes = np.random.randint(low=0, high=size[1], size=num_pixels_polarity_changes)

        for t in range(len(frames)):
            ts, xs, ys, ps = frames[t]['t'], frames[t]['x'].astype(int), frames[t]['y'].astype(int),\
                             frames[t]['p'].astype(int)
            lnes[ps, ys, xs] = (ts - f_start) / len(frames)
            lnes[:, ys_polarity_changes, xs_polarity_changes] = lnes[::-1, ys_polarity_changes, xs_polarity_changes]

        num_pixels_noisy_n = int(round(num_pixels_total * percentage_noisy_pixels_n))
        num_pixels_noisy_p = int(round(num_pixels_total * percentage_noisy_pixels_p))

        xs_noisy_n = np.random.randint(low=0, high=size[0], size=num_pixels_noisy_n)
        ys_noisy_n = np.random.randint(low=0, high=size[1], size=num_pixels_noisy_n)
        xs_noisy_p = np.random.randint(low=0, high=size[0], size=num_pixels_noisy_p)
        ys_noisy_p = np.random.randint(low=0, high=size[1], size=num_pixels_noisy_p)

        noise_n_full = np.random.rand(size[1], size[0])
        noise_p_full = np.random.rand(size[1], size[0])

        lnes[0, ys_noisy_n, xs_noisy_n] = noise_n_full[ys_noisy_n, xs_noisy_n]
        lnes[1, ys_noisy_p, xs_noisy_p] = noise_p_full[ys_noisy_p, xs_noisy_p]

        return lnes

    # convert mask images to PyTorch masks
    @classmethod
    def preprocess_mask(cls, img):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)

        # HWC to CHW
        img_trans = img.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return np.argmax(img_trans, axis=0)

    # compute left hand parameters in right hand frame and right hand parameters in camera frame
    @classmethod
    def preprocess_mano(cls, params_mano, aa, ap, roots):
        angle_augmentation, angle_position = angles_augmentation[aa], angles_position[ap]

        camera_relative = np.array([0.0, 0.0, 0.0])
        forward = np.array([0.0, 0.0, -1.0])

        view_matrix = np.eye(4)

        if angle_position is not None:
            rot_aa = np.array([[math.cos(angle_augmentation), 0.0, math.sin(angle_augmentation)],
                               [0.0, 1.0, 0.0],
                               [-math.sin(angle_augmentation), 0.0, math.cos(angle_augmentation)]])
            rot_ap = np.array([[math.cos(angle_position), -math.sin(angle_position), 0.0],
                               [math.sin(angle_position), math.cos(angle_position), 0.0],
                               [0.0, 0.0, 1.0]])

            camera_relative_transformed = rot_aa.dot(camera_relative - hands_avg) + hands_avg
            forward_transformed = rot_aa.dot(forward)
            camera_relative = rot_ap.dot(camera_relative_transformed)
            forward = rot_ap.dot(forward_transformed)
            line_target_2d = np.array([-forward[2], forward[0]])
            line_target_2d /= np.linalg.norm(line_target_2d)
            right_horizontal = np.array([line_target_2d[0], 0.0, line_target_2d[1]])
            up = np.cross(-forward, right_horizontal)

            view_matrix = np.zeros((4, 4))
            view_matrix[0, :3] = np.array(right_horizontal)
            view_matrix[1, :3] = np.array(up)
            view_matrix[2, :3] = np.array(-forward)
            view_matrix[:3, 3] = -view_matrix[:3, :3].dot(hands_avg + camera_relative)
            view_matrix[3, 3] = 1.0

        rot_vm = view_matrix[:3, :3]

        params_new = np.zeros(198)

        for h, hand in enumerate(params_mano):
            # global MANO rotation in MANO camera frame
            rot_mano = hand['pose'][:3]
            trans_mano = hand['trans']
            trans_manocam = trans_mano - camera_relative

            params_new[h * 99 + 0:h * 99 + 6] = matrix_to_rotation_6d(quaternion_to_matrix(quaternion_multiply(
                matrix_to_quaternion(torch.from_numpy(rot_vm)), axis_angle_to_quaternion(torch.from_numpy(rot_mano)))))\
                .numpy().reshape(6)
            params_new[h * 99 + 6:h * 99 + 96] = matrix_to_rotation_6d(axis_angle_to_matrix(torch.from_numpy(
                hand['pose'][3:48].reshape(15, 3)))).numpy().reshape(90)

            params_new[h * 99 + 96:h * 99 + 99] = -roots[h] + rot_vm.dot(roots[h] + trans_manocam)

        return params_new

    # project vertices given an intrinsic camera matrix
    @classmethod
    def project_vertices(cls, vertices, mat_cam):
        vertices_intrinsic = vertices.dot(mat_cam.transpose())
        vertices_projected = vertices_intrinsic[:, 0:2] / vertices_intrinsic[:, [2]]

        return vertices_projected

    # get one item for a batch
    def __getitem__(self, i):
        idx = self.ids[i]

        smaller_than = idx < self.len_until_reverse
        n = len(smaller_than) - np.argmin(smaller_than) - 1

        directory_noprefix = self.events_dirs[n][len('raw_sequence'):]
        split_directory_noprefix = directory_noprefix.split('_')
        name = split_directory_noprefix[0]
        aa = split_directory_noprefix[1]
        ap = split_directory_noprefix[2]

        l_lnes_lb = 0.3 * self.l_lnes
        l_lnes_ub = 3.0 * self.l_lnes

        def log_uniform_int(a, b):
            l = np.random.uniform(np.log(a), np.log(b))
            return int(round(np.exp(l)))

        l_lnes_aug = max(1, log_uniform_int(l_lnes_lb, l_lnes_ub))

        f = idx - self.len_until[n]
        f_start = max(0, f - l_lnes_aug + 1)
        f_finish = min(f + 1, self.num_frames[n])

        offset_start = f_start % interval
        offset_finish = f_finish % interval
        i_file_start = f_start // interval
        i_file_finish = f_finish // interval

        len_digits_interval = len(str((self.num_frames[n] - 1) // interval))
        len_digits_interval_masks = len(str((self.num_frames[n] - 1) // interval_masks))

        # start file
        file_events_start = os.path.join(self.events_dir, prefix_dataset + name + '_' + aa + '_' + ap,
                                         str(i_file_start).zfill(len_digits_interval) + '.pkl')

        with open(file_events_start, 'rb') as file:
            frames_events = pickle.load(file)[offset_start:min(offset_finish, interval)]

        # middle files
        for i_file in range(i_file_start + 1, i_file_finish):
            file_events = os.path.join(self.events_dir, prefix_dataset + name + '_' + aa + '_' + ap,
                                       str(i_file).zfill(len_digits_interval) + '.pkl')

            with open(file_events, 'rb') as file:
                frames_events_next = pickle.load(file)
                frames_events.extend(frames_events_next)

        # finish file
        if i_file_start < i_file_finish:
            file_events_finish = os.path.join(self.events_dir, prefix_dataset + name + '_' + aa + '_' + ap,
                                              str(i_file_finish).zfill(len_digits_interval) + '.pkl')

            with open(file_events_finish, 'rb') as file:
                frames_events_next = pickle.load(file)[:offset_finish]
                frames_events.extend(frames_events_next)

        lnes = self.preprocess_events_noisy(frames_events, f - len(frames_events) + 1, self.res)

        file_mano = os.path.join(self.mano_dir, prefix_dataset + name,
                                 str(f // interval).zfill(len_digits_interval) + '.pkl')

        with open(file_mano, 'rb') as file:
            frame_mano = pickle.load(file)[f]

        mano_params = self.preprocess_mano(frame_mano, aa, ap, self.roots)

        file_mask = os.path.join(self.mask_dir, prefix_dataset + name + '_' + aa + '_' + ap,
                                 str(f // interval_masks).zfill(len_digits_interval_masks) + '.npz')

        with np.load(file_mask) as data:
            masks = data['arr_0'][f % interval_masks]

        masks = self.preprocess_mask(masks)

        vertices, joints_3d = self.compute_vertices_and_joints(mano_params, self.layer_mano_right, self.layer_mano_left)
        joints_2d = self.project_vertices(joints_3d, mat_cam)

        return {
            'lnes': torch.from_numpy(lnes).type(torch.FloatTensor),
            'mano': torch.from_numpy(mano_params).type(torch.FloatTensor),
            'masks': torch.from_numpy(masks).type(torch.FloatTensor),
            'joints_3d': torch.from_numpy(joints_3d).type(torch.FloatTensor),
            'joints_2d': torch.from_numpy(joints_2d).type(torch.FloatTensor)
        }


def deploy_dataset(events_dir, mano_dir, masks_dir):
    # events
    print('deploying events')

    for file in sorted(os.listdir(events_dir)):
        if file.endswith('.npz'):
            print(file)

            events_raw = np.load(os.path.join(events_dir, file))['arr_0']
            events_raw['t'] = events_raw['t'] / 1000000

            events_list = [[] for i in range(math.ceil(events_raw[-1]['t']))]

            for event_raw in events_raw:
                events_list[min(math.floor(event_raw['t']), len(events_list) - 1)].append(event_raw)

            events = []

            for frame in events_list:
                events_frame = np.zeros(len(frame), dtype_events)

                for e, event in enumerate(frame):
                    events_frame[e] = event

                events.append(events_frame)

            Path(os.path.join(events_dir, file[:-4])).mkdir(parents=True, exist_ok=True)
            num_files = math.ceil(len(events) / interval)

            for f in range(num_files):
                low = min(max(0, f * interval), len(events))
                high = min(max(0, (f + 1) * interval), len(events))

                with open(os.path.join(events_dir, file[:-4], str(f).zfill(len(str(num_files - 1))) + '.pkl'), 'wb')\
                        as file_out:
                    pickle.dump(events[low:high], file_out)

            os.remove(os.path.join(events_dir, file))

    # MANO
    print('deploying MANO')

    for file in sorted(os.listdir(mano_dir)):
        if file.endswith('.pkl'):
            print(file)

            with open(os.path.join(mano_dir, file), 'rb') as file_in:
                seq_dict = pickle.load(file_in)

                Path(os.path.join(mano_dir, file[:-4])).mkdir(parents=True, exist_ok=True)
                num_files = math.ceil(len(seq_dict) / interval)

                for f in range(num_files):
                    low = min(max(0, f * interval), len(seq_dict))
                    high = min(max(0, (f + 1) * interval), len(seq_dict))

                    output_pickle = {}

                    for i in range(low, high):
                        output_pickle[i] = seq_dict[i]

                    with open(os.path.join(os.path.join(mano_dir, file[:-4]),
                                           str(f).zfill(len(str(num_files - 1))) + '.pkl'), 'wb') as file_out:
                        pickle.dump(output_pickle, file_out)

            os.remove(os.path.join(mano_dir, file))

    # masks
    print('deploying masks')

    for file in sorted(os.listdir(masks_dir)):
        if file.endswith('.mp4'):
            print(file)

            process = (
                ffmpeg.input(os.path.join(masks_dir, file))
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
                .run_async(pipe_stdout=True)
            )

            num_frames_appr = 300000    # assume 6-digit number of frames
            num_bundles_appr = int(math.ceil(num_frames_appr / interval_masks))
            num_bundle = 0
            bundle_frames = []

            Path(os.path.join(masks_dir, file[:-4])).mkdir(parents=True, exist_ok=True)

            while True:
                in_bytes = process.stdout.read(240 * 180 * 3)

                if not in_bytes:
                    break

                in_frame = np.frombuffer(in_bytes, np.uint8).reshape([180, 240, 3])
                bundle_frames.append(in_frame)

                if len(bundle_frames) == interval_masks:
                    np.savez_compressed(os.path.join(masks_dir, file[:-4], str(num_bundle)
                                                     .zfill(len(str(num_bundles_appr - 1))) + '.npz'),
                                        np.stack(bundle_frames, axis=0))
                    num_bundle += 1
                    bundle_frames = []

            if len(bundle_frames) > 0:
                np.savez_compressed(os.path.join(masks_dir, file[:-4], str(num_bundle)
                                                 .zfill(len(str(num_bundles_appr - 1))) + '.npz'),
                                    np.stack(bundle_frames, axis=0))

            os.remove(os.path.join(masks_dir, file))


if __name__ == '__main__':
    deploy_dataset('data/train/events/', 'data/train/mano/', 'data/train/masks/')
