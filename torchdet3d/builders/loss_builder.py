import torch

__AVAILABLE_LOSS = ['smoothl1', 'cross_entropy']


def build_loss(cfg):
    "build losses in right order"
    criterions = []
    for loss_name in cfg.loss.names:
        assert loss_name in __AVAILABLE_LOSS

    if 'smoothl1' in cfg.loss.names:
        # criterions.append(torch.nn.SmoothL1Loss(reduction='mean', beta=1.0))
        criterions.append(torch.nn.MSELoss())

    if 'cross_entropy' in cfg.loss.names:
        criterions.append(torch.nn.CrossEntropyLoss())
    else:
        criterions.append(None)

    return criterions
