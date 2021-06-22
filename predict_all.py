import argparse
import logging
import os
from glob import glob

import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.dataset import BasicDataset

# LNES window length
l_lnes = 100
res = (240, 180)


# load all events
def load_events(events_dir):
    # load events
    events_files = sorted(glob(events_dir + '/*.txt'))
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
    events = []

    for f, file in enumerate(events_list):
        events.append([])

        for t, frame in enumerate(file):
            events[f].append(np.zeros((len(frame), 4)))

            for e, event in enumerate(frame):
                events[f][t][e, :] = np.array(events_list[f][t][e])

    return events


# predict output for all LNES frames
def predict_lnes(net,
                lnes,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    lnes = torch.from_numpy(lnes)
    lnes = lnes.unsqueeze(0)
    lnes = lnes.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(lnes)

        probs = F.softmax(output, dim=1)
        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(lnes.shape[2]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


# parse arguments
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


# PyTorch mask to mask image
def mask_to_image(mask):
    mask = mask.transpose((1, 2, 0))
    return Image.fromarray((mask * 255).astype(np.uint8))


# main function
if __name__ == "__main__":
    args = get_args()
    in_files = args.input

    events = load_events(in_files)

    net = UNet()

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    for s, sequence in enumerate(events):
        for f in range(len(sequence) - l_lnes + 1):
            print(s, f)
            logging.info("\nPredicting sequence {} ...".format(s))

            frames = events[s][f:f + l_lnes]
            lnes = BasicDataset.preprocess_events(frames, f, l_lnes, res, 1)

            mask = predict_lnes(net=net,
                               lnes=lnes,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)

            if not args.no_save:
                out_fn = 'output/' + str(s) + '/' + str(f + l_lnes - 1).zfill(4) + '.png'
                result = mask_to_image(mask)
                result.save(out_fn)

                logging.info("Mask saved to {}".format(out_fn))
