import numpy as np
from glob import glob
import pickle
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import math


# dataset loader
class BasicDataset(Dataset):
    def __init__(self, events_dir, mano_dir, masks_dir, l_lnes, scale=1):
        self.events_dir = events_dir
        self.mano_dir = mano_dir
        self.masks_dir = masks_dir
        self.l_lnes = l_lnes
        self.scale = scale
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        # load mano frames
        mano_files = sorted(glob(self.mano_dir + '*.pkl'))

        # list of numpy arrays per sequence
        self.mano_params = []

        for mano_file in mano_files:
            with open(mano_file, 'rb') as f:
                seq_dict = pickle.load(f)
                entries = np.zeros((len(seq_dict), 102))

                for e in range(len(seq_dict)):
                    frame = seq_dict[e]

                    for h, hand in enumerate(frame):
                        entries[e, h * 51 + 0:h * 51 + 48] = hand['pose']
                        entries[e, h * 51 + 48:h * 51 + 51] = hand['trans']

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
        self.ids = [(s, f - self.l_lnes) for s, sequence in enumerate(self.events) for f, frame in enumerate(sequence)
                    if f >= self.l_lnes]

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    # length equals to number of sequences time number of (frames - LNES window length)
    def __len__(self):
        return len(self.ids)

    # convert events to LNES frames
    @classmethod
    def preprocess_events(cls, frames, f, l_lnes, size, scale):
        w, h = size
        newW, newH = int(scale * w), int(scale * h)

        lnes = np.zeros((2, newH, newW))

        for t in range(l_lnes):
            ts, xs, ys, ps = frames[t][:, 0], frames[t][:, 1].astype(int), frames[t][:, 2].astype(int),\
                             frames[t][:, 3].astype(int)
            lnes[ps, ys, xs] = (ts - f) / l_lnes

        return lnes

    # convert mask images to PyTorch masks
    @classmethod
    def preprocess_mask(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

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
        mask_file = glob(self.masks_dir + str(s).zfill(1) + '/frame_' + str(f + self.l_lnes - 1).zfill(4) + '.png')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        mask = Image.open(mask_file[0])
        frames = self.events[s][f:f + self.l_lnes]

        lnes = self.preprocess_events(frames, f, self.l_lnes, mask.size, self.scale)
        mask = self.preprocess_mask(mask, self.scale)
        mano_params = self.mano_params[s][f, :]

        return {
            'lnes': torch.from_numpy(lnes).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor),
            'mano': torch.from_numpy(mano_params).type(torch.FloatTensor)
        }
