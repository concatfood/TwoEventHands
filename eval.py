import torch
import torch.nn.functional as F
from tqdm import tqdm


# weights
weight_mask = 1.0
weight_mano = 1.0


# evaluate network
def eval_net(net, loader, device):
    mask_type = torch.long
    mano_type = torch.float32

    n_val = len(loader)
    tot = 0

    with tqdm(total=n_val, desc='validation phase') as pbar:
        for batch in loader:
            lnes, true_masks, true_mano = batch['lnes'], batch['mask'], batch['mano']
            lnes = lnes.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)
            true_mano = true_mano.to(device=device, dtype=mano_type)

            with torch.no_grad():
                mask_pred, mano_pred = net(lnes)

            tot += weight_mask * F.cross_entropy(mask_pred, true_masks).item()
            tot += weight_mano * F.mse_loss(mano_pred, true_mano).item()

            pbar.update()

    return tot
