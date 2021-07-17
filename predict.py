import argparse
import logging
import os

import math
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from TEHNet import TEHNet
from utils.dataset import BasicDataset

# LNES window length
l_lnes = 200
res = (240, 180)


# loads mano parameters
def load_mano(mano_file):
    with open(mano_file, 'rb') as f:
        seq_dict = pickle.load(f)
        entries = np.zeros((len(seq_dict), 102))

        for e in range(len(seq_dict)):
            frame = seq_dict[e]

            for h, hand in enumerate(frame):
                entries[e, h * 51 + 0:h * 51 + 48] = hand['pose']
                entries[e, h * 51 + 48:h * 51 + 51] = hand['trans']

    return entries


# loads events
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


# predict segmentation mask and MANO parameters
def predict_mask(net, lnes, device):
    net.eval()

    lnes = torch.from_numpy(lnes)
    lnes = lnes.unsqueeze(0)
    lnes = lnes.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        mask_output, _ = net(lnes)

        probs = F.softmax(mask_output, dim=1)
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

    indices_max = np.argmax(full_mask, axis=0)
    prediction = np.zeros(full_mask.shape)
    prediction[indices_max] = full_mask[indices_max]

    return prediction


# predict MANO parameters
def predict_mano(net,
                 lnes,
                 device):
    net.eval()

    lnes = torch.from_numpy(lnes)
    lnes = lnes.unsqueeze(0)
    lnes = lnes.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        _, mano_output = net(lnes)

        params = mano_output.squeeze(0)
        params = params.cpu().numpy()

    return params


# predict segmentation mask and MANO parameters
def predict(net, lnes, device):
    net.eval()

    lnes = torch.from_numpy(lnes)
    lnes = lnes.unsqueeze(0)
    lnes = lnes.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        mask_output, mano_output = net(lnes)

        probs = F.softmax(mask_output, dim=1)
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

        params = mano_output.squeeze(0)
        params = params.cpu().numpy()

    indices_max = np.argmax(full_mask, axis=0)
    prediction = np.zeros(full_mask.shape)
    prediction[indices_max] = full_mask[indices_max]

    return prediction, params


# parse arguments
def get_args():
    parser = argparse.ArgumentParser(description='Predict mask from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)
    parser.add_argument('--frame', '-f', metavar='INPUT', nargs='+',
                        help='frame in the sequence', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output mask",
                        default=False)

    return parser.parse_args()


# get name for output file names
def get_output_filenames(args):
    in_files = args.input
    frame = int(args.frame[0])
    out_files = []

    if not args.output:
        pathsplit = os.path.splitext(in_files[0])
        out_files.append("{}_{}_OUT{}".format(pathsplit[0], frame, '.png'))
        out_files.append("{}_{}_OUT{}".format(pathsplit[0], frame, '.pkl'))
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
    in_events = args.input[0]
    in_mano = args.input[1]
    frame = int(args.frame[0])
    out_files = get_output_filenames(args)

    net = TEHNet()

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device)['model'])

    logging.info("Model loaded !")
    logging.info("\nPredicting image {} ...".format(in_events))

    # events for one LNES window
    frames = load_events(in_events)[frame - l_lnes + 1:frame + 1]
    mano_true = load_mano(in_mano)[frame, :]
    lnes = BasicDataset.preprocess_events(frames, frame - l_lnes + 1, res)

    mask_pred, mano_pred = predict(net=net, lnes=lnes, device=device)

    # save output
    if not args.no_save:
        out_fn = out_files[0]
        result = mask_to_image(mask_pred)
        result.save(out_files[0])

        logging.info("Mask saved to {}".format(out_files[0]))

        seq_dict = {frame: [{'pose': mano_pred[3:51],
                             'shape': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                             'trans': mano_pred[0:3],
                             'hand_type': 'right'},
                            {'pose': mano_pred[54:102],
                             'shape': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                             'trans': mano_pred[51:54],
                             'hand_type': 'left'}]}

        with open(out_files[1], 'wb') as f:
            pickle.dump(seq_dict, f)

        logging.info("MANO parameters saved to {}".format(out_files[1]))
