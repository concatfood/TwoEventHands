import torch
import torch.nn.functional as F
from tqdm import tqdm


# evaluate network
def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            lnes, true_masks = batch['lnes'], batch['mask']
            lnes = lnes.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(lnes)

            tot += F.cross_entropy(mask_pred, true_masks).item()

            pbar.update()

    net.train()
    return tot / n_val
