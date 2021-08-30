import torch
import torch.nn.functional as F
from tqdm import tqdm


# weights
weight_mano = 1.0
weight_mask = 0.1


# evaluate network
def eval_net(net, loader, device, epoch, num_it=-1, use_unet=True):
    mask_type = torch.long
    mano_type = torch.float32

    loss_valid_mano = 0
    loss_valid_mask = 0
    loss_valid = 0
    global_step = 0

    # with tqdm(total=num_it, desc='validation phase') as pbar:
    for it, batch in enumerate(loader):
        if it >= num_it:
            break

        if use_unet:
            lnes, true_masks, true_mano = batch['lnes'], batch['mask'], batch['mano']
        else:
            lnes, true_mano = batch['lnes'], batch['mano']

        lnes = lnes.to(device=device, dtype=torch.float32)

        if use_unet:
            true_masks = true_masks.to(device=device, dtype=mask_type)

        true_mano = true_mano.to(device=device, dtype=mano_type)

        with torch.no_grad():
            if use_unet:
                mask_pred, mano_pred = net(lnes)
            else:
                mano_pred = net(lnes)

        loss_mano = F.mse_loss(mano_pred, true_mano)
        loss_valid_mano += loss_mano.item()

        if use_unet:
            loss_mask = F.cross_entropy(mask_pred, true_masks)
            loss_valid_mask += loss_mask.item()
            loss_total = weight_mano * loss_mano + weight_mask * loss_mask
        else:
            loss_total = weight_mano * loss_mano

        loss_valid += loss_total.item()

        # log phase, epoch and iteration
        with open('phase_epoch_iteration.txt', "w") as f:
            f.write('validation, ' + str(epoch) + ', ' + str(it))

        global_step += 1

        # pbar.update()

    return loss_valid, loss_valid_mano, loss_valid_mask
