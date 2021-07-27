import numpy as np
from glob import glob
import pickle
from scipy.spatial.transform import Rotation as R
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import math

# hands_avg + np.array([0.5 * far, 0, 0])   # far = 1.0
pos_cam = [np.array([1.63104033, -0.23615411, 0.77483138]),
           np.array([1.57374847, -0.24665557, 0.96311718]),
           np.array([1.65183896, -0.36861104, 0.85237268]),
           np.array([1.55133876, -0.24436227, 0.73017362]),
           np.array([1.52926517, -0.19142124, 0.76753855])]


# dataset loader
class BasicDataset(Dataset):
    def __init__(self, events_dir, mano_dir, masks_dir, l_lnes):
        self.events_dir = events_dir
        self.mano_dir = mano_dir
        self.masks_dir = masks_dir
        self.l_lnes = l_lnes

        # load mano frames
        mano_files = sorted(glob(self.mano_dir + '*.pkl'))

        # list of numpy arrays per sequence
        self.mano_params = []

        for mf, mano_file in enumerate(mano_files):
            with open(mano_file, 'rb') as f:
                seq_dict = pickle.load(f)
                # entries = np.zeros((len(seq_dict), 102))  # axis-angle
                entries = np.zeros((len(seq_dict), 134))    # quaternion

                for e in range(len(seq_dict)):
                    frame = seq_dict[e]

                    for h, hand in enumerate(frame):
                        # entries[e, h * 51 + 0:h * 51 + 48] = hand['pose']
                        for j in range(16):
                            entries[e, h * 67 + 0:h * 67 + 64] = R.from_rotvec(hand['pose'][3 * j]).as_quat()

                        # entries[e, h * 51 + 48:h * 51 + 51] = hand['trans']
                        entries[e, h * 67 + 64:h * 67 + 67] = hand['trans']

                    # entries[e, 1 * 51 + 48:1 * 51 + 51] -= entries[e, 0 * 51 + 48:0 * 51 + 51]
                    # entries[e, 0 * 51 + 48:0 * 51 + 51] -= pos_cam[mf]
                    entries[e, 1 * 67 + 64:1 * 67 + 67] -= entries[e, 0 * 67 + 64:0 * 67 + 67]
                    entries[e, 0 * 67 + 64:0 * 67 + 67] -= pos_cam[mf]

                self.mano_params.append(entries)

        # load events
        events_files = sorted(glob(self.events_dir + '*.txt'))
        events_list = []

        for events_file in events_files:
            with open(events_file) as f:
                content = f.readlines()

            events_raw = []

            for line in content:
                event = line.split()
                time = int(event[0]) / 1000000
                x = int(event[1])
                y = int(event[2])
                polarity = int(event[3])

                events_raw.append((time, x, y, polarity))

            events_frames = [[] for i in range(math.ceil(events_raw[-1][0]))]

            for event in events_raw:
                events_frames[math.floor(event[0])].append(event)

            events_list.append(events_frames)

        # list (sequences) of lists (frames) of numpy arrays (events (t, x, y, p))
        self.events = []

        for f, file in enumerate(events_list):
            self.events.append([])

            for t, frame in enumerate(file):
                self.events[f].append(np.zeros((len(frame), 4)))

                for e, event in enumerate(frame):
                    self.events[f][t][e, :] = np.array(events_list[f][t][e])

        # two-dimensional ids (sequence, frame) with frame >= LNES window length
        self.ids = [(s, f) for s, sequence in enumerate(self.events) for f, frame in enumerate(sequence)]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    # length equals to number of sequences time number of (frames - LNES window length)
    def __len__(self):
        return len(self.ids)

    # convert events to LNES frames
    @classmethod
    def preprocess_events(cls, frames, f_start, size):
        lnes = np.zeros((2, size[1], size[0]))

        for t in range(len(frames)):
            ts, xs, ys, ps = frames[t][:, 0], frames[t][:, 1].astype(int), frames[t][:, 2].astype(int),\
                             frames[t][:, 3].astype(int)
            lnes[ps, ys, xs] = (ts - f_start) / len(frames)

        return lnes

    # convert mask images to PyTorch masks
    @classmethod
    def preprocess_mask(cls, pil_img):
        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return np.argmax(img_trans, axis=0)

    # get one item for a batch
    def __getitem__(self, i):
        idx = self.ids[i]
        s = idx[0]
        f = idx[1]
        mask_file = glob(self.masks_dir + str(s).zfill(1) + '/frame_' + str(f + 1).zfill(6) + '.png')

        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'

        mask = Image.open(mask_file[0])
        frames = self.events[s][f - self.l_lnes + 1:f + 1]

        lnes = self.preprocess_events(frames, f - self.l_lnes + 1, mask.size)
        mask = self.preprocess_mask(mask)
        mano_params = self.mano_params[s][f, :]

        return {
            'lnes': torch.from_numpy(lnes).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'mano': torch.from_numpy(mano_params).type(torch.FloatTensor)
        }
