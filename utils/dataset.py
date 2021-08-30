import ffmpeg
import glm
import logging
import math
import numpy as np
import os
import os.path
from pathlib import Path
import pickle
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset

# name of dataset
prefix_dataset = 'raw_sequence'

# numpy data type
dtype_events = np.dtype([('t', np.int64), ('x', np.int16), ('y', np.int16), ('p', np.int8)])

# maximum number of events and mano parameter vectors per file
interval = 1000
interval_masks = 10

# average position of hands
t_hands_avg = {'0': np.array([1.15030333, -0.26061168, 0.78577989]),
               '1': np.array([1.12174921, -0.20740417, 0.81597976]),
               '2': np.array([1.16503733, -0.31407652, 0.81827346]),
               '3': np.array([1.03878048, -0.27550721, 0.82758726]),
               '4': np.array([1.03542053, -0.1853186, 0.77306902]),
               '5': np.array([0.98415266, -0.42346881, 0.76287726]),
               '6': np.array([0.99070947, -0.40857825, 0.75110521]),
               '7': np.array([1.00070618, -0.40154313, 0.77840039])}

# camera parameters
far = 1.0
fov = 45.0
fovy = glm.radians(fov)

# camera position relative to hands in MANO space
t_cam_rel = np.array([0.5 * far, 0.0, 0.0])

angles_augmentation = {'0': 5.0 / 360.0 * (2.0 * math.pi), '1': 10.0 / 360.0 * (2.0 * math.pi),
                       '2': 15.0 / 360.0 * (2.0 * math.pi), '3': 20.0 / 360.0 * (2.0 * math.pi),
                       '4': 25.0 / 360.0 * (2.0 * math.pi)}
angles_position = {'0': None, '1': 45.0 / 360.0 * (2 * math.pi), '2': 135.0 / 360.0 * (2 * math.pi),
                   '3': 225.0 / 360.0 * (2 * math.pi), '4': 315.0 / 360.0 * (2 * math.pi)}


# dataset loader
class BasicDataset(Dataset):
    def __init__(self, events_dir, mano_dir, masks_dir, res, l_lnes, use_unet=True):
        self.events_dir = events_dir
        self.mano_dir = mano_dir
        self.masks_dir = masks_dir
        self.res = res
        self.l_lnes = l_lnes
        self.num_frames = []
        self.use_unet = use_unet

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

        logging.info(f'The dataset contains {len(self.ids)} frames.')

    # length equals to number of sequences time number of (frames - LNES window length)
    def __len__(self):
        return len(self.ids)

    # compute intrinsic and extrinsic matrices
    @classmethod
    def compute_camera_matrices(cls, name, aa, ap, res):
        f = 0.5 * res[1] / math.tan(fovy / 2.0)

        mat_intrinsic = np.array([[f, 0.0, -res[0] / 2.0],  # y and z negative because of different coordinate systems
                                  [0.0, -f, -res[1] / 2.0],
                                  [0.0, 0.0, -1.0]])

        camera_relative = glm.vec3(0.5 * far, 0.0, 0.0)
        forward = glm.vec3(-1.0, 0.0, 0.0)
        up = glm.vec3(0.0, 0.0, 1.0)

        if angles_position[ap] is not None:
            camera_relative_transformed = glm.rotateZ(camera_relative, angles_augmentation[aa])
            forward_transformed = glm.rotateZ(forward, angles_augmentation[aa])
            camera_relative = glm.rotateX(camera_relative_transformed, angles_position[ap])
            forward = glm.rotateX(forward_transformed, angles_position[ap])
            line_target_2d = np.array([forward.y, -forward.x])
            line_target_2d /= np.linalg.norm(line_target_2d)
            right_horizontal = glm.vec3(line_target_2d[0], line_target_2d[1], 0.0)
            up = glm.cross(-forward_transformed, right_horizontal)  # actually -forward, but mistake in renderer

        mat_extrinsic = np.array(glm.lookAt(glm.vec3(t_hands_avg[name]) + camera_relative,
                                            glm.vec3(t_hands_avg[name]) + camera_relative + forward,
                                            up))

        return mat_intrinsic, mat_extrinsic

    # convert events to LNES frames
    @classmethod
    def preprocess_events(cls, frames, f_start, size):
        lnes = np.zeros((2, size[1], size[0]))

        for t in range(len(frames)):
            ts, xs, ys, ps = frames[t]['t'], frames[t]['x'].astype(int), frames[t]['y'].astype(int),\
                             frames[t]['p'].astype(int)
            lnes[ps, ys, xs] = (ts - f_start) / len(frames)

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
    def preprocess_mano(cls, params_mano, name, aa, ap):
        angle_aa, angle_ap = angles_augmentation[aa], angles_position[ap]

        rot_aug_inv = np.eye(3)

        if angle_ap is not None:
            rot_aa = np.array([[math.cos(angle_aa), -math.sin(angle_aa), 0.0],
                               [math.sin(angle_aa), math.cos(angle_aa), 0.0],
                               [0.0, 0.0, 1.0]])
            rot_ap = np.array([[1.0, 0.0, 0.0],
                               [0.0, math.cos(angle_ap), -math.sin(angle_ap)],
                               [0.0, math.sin(angle_ap), math.cos(angle_ap)]])
            rot_aug = rot_ap.dot(rot_aa)
            rot_aug_inv = rot_aug.transpose()

        t_cam = t_hands_avg[name] + t_cam_rel

        params_new = np.zeros(134)

        for h, hand in enumerate(params_mano):
            # global MANO rotation in MANO camera frame
            params_new[h * 67 + 0:h * 67 + 4] = (R.from_matrix(rot_aug_inv) * R.from_rotvec(hand['pose'][:3])).as_quat()
            params_new[h * 67 + 4:h * 67 + 64] = R.from_rotvec(hand['pose'][3:48].reshape(15, 3)).as_quat()\
                .reshape(1, 60)
            params_new[h * 67 + 64:h * 67 + 67] = rot_aug_inv.dot(hand['trans'] - t_hands_avg[name]) + t_hands_avg[name]

        # left hand position relative to right hand position
        # right hand position relative to camera position
        params_new[1 * 67 + 64:1 * 67 + 67] -= params_new[0 * 67 + 64:0 * 67 + 67]
        params_new[0 * 67 + 64:0 * 67 + 67] -= t_cam

        return params_new

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

        f = idx - self.len_until[n]
        f_start = max(0, f - self.l_lnes + 1)
        f_finish = min(f + 1, self.num_frames[n])

        offset_start = f_start % interval
        offset_finish = f_finish % interval
        i_file_start = f_start // interval
        i_file_finish = f_finish // interval

        len_digits_interval = 3     # len(str((self.num_frames[n] - 1) // interval))
        len_digits_interval_masks = 5   # len(str(self.num_frames[n] - 1) // interval_masks)

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

        lnes = self.preprocess_events(frames_events, f - self.l_lnes + 1, self.res)

        file_mano = os.path.join(self.mano_dir, prefix_dataset + name,
                                 str(f // interval).zfill(len_digits_interval) + '.pkl')

        # mat_intrinsic, mat_extrinsic = self.compute_camera_matrices(name, aa, ap, self.res)

        with open(file_mano, 'rb') as file:
            frame_mano = pickle.load(file)[f]

        mano_params = self.preprocess_mano(frame_mano, name, aa, ap)

        mask = None

        if self.use_unet:
            file_mask = os.path.join(self.masks_dir, prefix_dataset + name + '_' + aa + '_' + ap,
                                     str(f // interval_masks).zfill(len_digits_interval_masks) + '.npz')

            with np.load(file_mask) as data:
                mask = data['arr_0'][f % interval_masks]

            mask = self.preprocess_mask(mask)

        if self.use_unet:
            return {
                'lnes': torch.from_numpy(lnes).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask).type(torch.FloatTensor),
                'mano': torch.from_numpy(mano_params).type(torch.FloatTensor)
            }
        else:
            return {
                'lnes': torch.from_numpy(lnes).type(torch.FloatTensor),
                'mano': torch.from_numpy(mano_params).type(torch.FloatTensor)
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
