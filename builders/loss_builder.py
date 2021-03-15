import torch

__AVAILABLE_LOSS = ['smoothl1']


def build_loss(cfg):
    assert cfg.loss.name in __AVAILABLE_LOSS
    if cfg.loss.name == 'smoothl1':
        loss = torch.nn.SmoothL1Loss(reduction='mean', beta=1.0)
    return loss
