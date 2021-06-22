import argparse
import logging
import os

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


# load events for one LNES frame
def load_events(events_file):
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

    events = []

    for t, frame in enumerate(events_frames):
        events.append(np.zeros((len(frame), 4)))

        for e, event in enumerate(frame):
            events[t][e, :] = np.array(events_frames[t][e])

    return events


# predict output for one LNES frame
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
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('--frame', '-f', metavar='INPUT', nargs='+',
                        help='frame in the sequence', required=True)

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


# get name for output file names
def get_output_filenames(args):
    in_files = args.input
    frame = int(args.frame[0])
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_{}_OUT{}".format(pathsplit[0], frame, '.png'))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


# PyTorch mask to mask image
def mask_to_image(mask):
    mask = mask.transpose((1, 2, 0))
    return Image.fromarray((mask * 255).astype(np.uint8))


# main function
if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    frame = int(args.frame[0])
    out_files = get_output_filenames(args)

    net = UNet()

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    # typically one file
    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        # events for one LNES window
        frames = load_events(fn)[frame - l_lnes + 1:frame + 1]
        lnes = BasicDataset.preprocess_events(frames, frame - l_lnes + 1, l_lnes, res, 1)

        mask = predict_lnes(net=net,
                           lnes=lnes,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        # save output
        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))
