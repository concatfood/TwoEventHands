import argparse
# import ffmpeg
import numpy as np
import os
from pathlib import Path
import pickle
from PIL import Image
from scipy.spatial.transform import Rotation as R
from TEHNet import TEHNet
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
from utils.dataset import BasicDataset

# relative directories
dir_events = os.path.join('data', 'test', 'events')
dir_output = 'output'

# LNES window length
l_lnes = 200
res = (240, 180)

# framerates
fps_in = 1000
fps_out = 30


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
def predict(net, lnes, device, use_unet=True):
    net.eval()

    lnes = torch.from_numpy(lnes)
    lnes = lnes.unsqueeze(0)
    lnes = lnes.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        if use_unet:
            mask_output, mano_output = net(lnes)
        else:
            mano_output = net(lnes)

        if use_unet:
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
        params_axisangle[h * 51 + 0:h * 51 + 48] = R.from_quat(params[h * 67 + 0:h * 67 + 64].reshape(16, 4))\
            .as_rotvec().reshape(1, 48)
        params_axisangle[h * 51 + 48:h * 51 + 51] = params[h * 67 + 64:h * 67 + 67]

    params = params_axisangle

    if use_unet:
        indices_max = np.argmax(full_mask, axis=0)
        prediction = (np.arange(3) == indices_max[..., None]).astype(int)

        return prediction, params
    else:
        return params


# parse arguments
def get_args():
    parser = argparse.ArgumentParser(description='Predict mask from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='best.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT',
                        help='Name of input sequence', required=True)
    parser.add_argument('--use_unet', '-u', default=True, type=bool,
                        help='Use U-Net for mask prediction')

    return parser.parse_args()


# main function
if __name__ == "__main__":
    args = get_args()
    name_sequence = args.input
    use_unet = args.use_unet

    events = load_events(name_sequence)

    net = TEHNet(use_unet=use_unet)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.to(device=device)
    net.load_state_dict(torch.load(os.path.join('checkpoints', args.model), map_location=device)['model'])

    # dir_masks = os.path.join(dir_output, name_sequence)
    dir_masks = os.path.join(dir_output, name_sequence, 'masks')
    Path(dir_masks).mkdir(parents=True, exist_ok=True)

    # process = (ffmpeg
    #            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(res[0], res[1]), framerate=fps_in)
    #            .output(os.path.join(dir_masks, 'masks.mp4'), pix_fmt='yuv444p', vcodec='libx264', preset='veryslow',
    #                    crf=0, r=fps_out, movflags='faststart')
    #            .overwrite_output()
    #            .run_async(pipe_stdin=True))

    mano_pred_seq = {}

    for i_f, f_float in enumerate(tqdm(np.arange(0, len(events), fps_in / fps_out))):
        f = int(round(f_float))

        frames = events[max(0, f - l_lnes + 1):f + 1]
        lnes = BasicDataset.preprocess_events(frames, f - l_lnes + 1, res)

        mask_pred = None

        if use_unet:
            mask_pred, mano_pred = predict(net=net, lnes=lnes, device=device, use_unet=use_unet)
        else:
            mano_pred = predict(net=net, lnes=lnes, device=device, use_unet=use_unet)

        seq_dict = {i_f: [{'pose': mano_pred[0:48],
                           'shape': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                           'trans': mano_pred[48:51],
                           'hand_type': 'right'},
                          {'pose': mano_pred[51:99],
                           'shape': np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                           'trans': mano_pred[99:102] + mano_pred[48:51],
                           'hand_type': 'left'}]}

        mano_pred_seq.update(seq_dict)

        if use_unet:
            # process.stdin.write((mask_pred * 255).astype(np.uint8).tobytes())
            out_fn = os.path.join(dir_masks, 'frame_'
                                  + str(i_f + 1).zfill(len(str(int(round(len(events) * fps_out / fps_in))))) + '.png')
            result = Image.fromarray((mask_pred * 255).astype(np.uint8))
            result.save(out_fn)

    # process.stdin.close()

    dir_sequence_mano = os.path.join(dir_output, name_sequence)
    Path(dir_sequence_mano).mkdir(parents=True, exist_ok=True)
    out_mano = os.path.join(dir_sequence_mano, 'sequence_mano.pkl')

    with open(out_mano, 'wb') as file:
        pickle.dump(mano_pred_seq, file)
