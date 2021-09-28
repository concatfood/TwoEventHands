import torch


def huber_loss(diff, reduction='mean', delta=1.0):
    errors = torch.abs(diff)
    mask = errors < delta
    loss_unreduced = mask * (0.5 * (errors ** 2)) + ~mask * (delta * (errors - 0.5 * delta))
    loss_reduced = torch.mean(loss_unreduced) if reduction == 'mean' else torch.sum(loss_unreduced)

    return loss_reduced


class HuberLoss:
    def __init__(self, reduction: str = 'mean', delta: float = 1.0):
        self.reduction = reduction
        self.delta = delta

    def __call__(self, diff):
        return huber_loss(diff, reduction=self.reduction, delta=self.delta)
