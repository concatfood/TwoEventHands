import argparse
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
from utils.dataset import BasicDataset
from utils.one_euro_filter import OneEuroFilter


# relative directories
dir_events = os.path.join('data', 'test', 'events')
dir_output = 'output'

# LNES window length
l_lnes = 200
res = (240, 180)

# framerates
fps_in = 1000
fps_out = 30

# temporal filtering
one_euro_filter = None


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


# predict segmentation mask and MANO parameters
def predict(net, lnes, device, t):
    global one_euro_filter

    net.eval()

    lnes = torch.from_numpy(lnes)
    lnes = lnes.unsqueeze(0)
    lnes = lnes.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        mano_output, joints_3d_output, joints_2d_output = net(lnes)
        params = mano_output.squeeze(0)
        params = params.cpu().numpy()

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

    return params_axisangle, params_axisangle_smoothed


# parse arguments
def get_args():
    parser = argparse.ArgumentParser(description='Predict mask from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='best.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='Name of input sequence', required=True)

    return parser.parse_args()


# main function
if __name__ == "__main__":
    args = get_args()
    name_sequence = args.input

    events = load_events(name_sequence)

    net = TEHNet()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.to(device=device)
    net.load_state_dict(torch.load(os.path.join('checkpoints', args.model), map_location=device)['model'])

    mano_pred_seq = {}
    mano_pred_seq_smoothed = {}

    for i_f, f_float in enumerate(tqdm(np.arange(0, len(events), fps_in / fps_out))):
        f = int(round(f_float))     # in milliseconds

        frames = events[max(0, f - l_lnes + 1):f + 1]
        lnes = BasicDataset.preprocess_events(frames, f - l_lnes + 1, res)
        mano_pred, mano_pred_smoothed = predict(net=net, lnes=lnes, device=device, t=f / 1000)

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

    dir_sequence_mano = os.path.join(dir_output, name_sequence)
    Path(dir_sequence_mano).mkdir(parents=True, exist_ok=True)
    out_mano = os.path.join(dir_sequence_mano, 'sequence_mano.pkl')
    out_mano_smoothed = os.path.join(dir_sequence_mano, 'sequence_mano_smoothed.pkl')

    with open(out_mano, 'wb') as file:
        pickle.dump(mano_pred_seq, file)

    with open(out_mano_smoothed, 'wb') as file:
        pickle.dump(mano_pred_seq_smoothed, file)
