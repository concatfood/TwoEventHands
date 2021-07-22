import torch
import torch.nn.functional as F
from tqdm import tqdm


# weights
weight_mano = 1.0
weight_mask = 0.1


# evaluate network
def eval_net(net, loader, device, writer, epoch):
    mask_type = torch.long
    mano_type = torch.float32

    n_val = len(loader)
    loss_mano_total = 0
    loss_mask_total = 0
    loss_total = 0
    global_step = 0

    # with tqdm(total=n_val, desc='validation phase') as pbar:
    for it, batch in enumerate(loader):
        lnes, true_masks, true_mano = batch['lnes'], batch['mask'], batch['mano']
        lnes = lnes.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)
        true_mano = true_mano.to(device=device, dtype=mano_type)

        with torch.no_grad():
            mask_pred, mano_pred = net(lnes)

        loss_mano = F.mse_loss(mano_pred, true_mano).item()
        loss_mask = F.cross_entropy(mask_pred, true_masks).item()
        loss_mano_total += loss_mano
        loss_mask_total += loss_mask
        loss_total += weight_mano * loss_mano + weight_mask * loss_mask

        writer.add_text('phase, epoch, iteration', 'validation, ' + str(epoch) + ', ' + str(it), global_step)

        global_step += 1

        # pbar.update()

    return loss_total, loss_mano_total, loss_mask_total
