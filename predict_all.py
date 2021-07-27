import argparse
from glob import glob
import logging
import math
import numpy as np
import pickle
from PIL import Image
from scipy.spatial.transform import Rotation as R
from TEHNet import TEHNet
import torch
import torch.nn.functional as F
from torchvision import transforms
from utils.dataset import BasicDataset

# LNES window length
l_lnes = 200
res = (240, 180)

# framerates
fps_in = 1000
fps_out = 30


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
def predict_mano(net, lnes, device):
    net.eval()

    lnes = torch.from_numpy(lnes)
    lnes = lnes.unsqueeze(0)
    lnes = lnes.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        _, mano_output = net(lnes)

        params = mano_output.squeeze(0)
        params = params.cpu().numpy()

    params_axisangle = np.zeros(102)

    for h in range(2):
        for j in range(16):
            params_axisangle[h * 51 + j * 3:h * 51 + (j + 1) * 3] =\
                R.from_quat(params_axisangle[h * 67 + j * 4:h * 67 + (j + 1) * 4]).as_rotvec()

        params_axisangle[h * 51 + 48:h * 51 + 51] = params[h * 67 + 64:h * 67 + 67]

    params = params_axisangle

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

    params_axisangle = np.zeros(102)

    for h in range(2):
        for j in range(16):
            params_axisangle[h * 51 + j * 3:h * 51 + (j + 1) * 3] =\
                R.from_quat(params_axisangle[h * 67 + j * 4:h * 67 + (j + 1) * 4]).as_rotvec()

        params_axisangle[h * 51 + 48:h * 51 + 51] = params[h * 67 + 64:h * 67 + 67]

    params = params_axisangle

    indices_max = np.argmax(full_mask, axis=0)
    prediction = (np.arange(indices_max.max()+1) == indices_max[...,None]).astype(int)

    return prediction, params


# parse arguments
def get_args():
    parser = argparse.ArgumentParser(description='Predict mask from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output mask",
                        default=False)

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

    net = TEHNet()

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device)['model'])

    logging.info("Model loaded !")

    for s, sequence in enumerate(events):
        mano_pred_seq = {}

        for i_f, f_float in enumerate(np.arange(0, len(sequence), fps_in / fps_out)):
            f = int(round(f_float))
            print(s, f)
            logging.info("\nPredicting sequence {} ...".format(s))

            frames = events[s][max(0, f - l_lnes + 1):f + 1]
            lnes = BasicDataset.preprocess_events(frames, f - l_lnes + 1, res)

            mask_pred, mano_pred = predict(net=net, lnes=lnes, device=device)

            seq_dict = {i_f: [{'pose': mano_pred[0:48],
                               'shape': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                               'trans': mano_pred[48:51],
                               'hand_type': 'right'},
                              {'pose': mano_pred[51:99],
                               'shape': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                               'trans': mano_pred[99:102] + mano_pred[48:51],
                               'hand_type': 'left'}]}

            mano_pred_seq.update(seq_dict)

            if not args.no_save:
                out_fn = 'output/' + str(s) + '/frame_' +\
                         str(i_f + 1).zfill(len(str(int(round(len(sequence) * fps_out / fps_in))))) + '.png'
                # result = mask_to_image(mask_pred)
                result = Image.fromarray((mask_pred * 255).astype(np.uint8))
                result.save(out_fn)

                logging.info("Mask saved to {}".format(out_fn))

        if not args.no_save:
            out_fn = 'output/' + str(s) + '.pkl'

            with open(out_fn, 'wb') as f:
                pickle.dump(mano_pred_seq, f)

                logging.info("MANO parameters saved to {}".format(out_fn))
