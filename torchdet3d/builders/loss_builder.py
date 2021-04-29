import torch
from torchdet3d.losses import DiagLoss, ADD_loss, WingLoss

AVAILABLE_LOSS = ['smoothl1', 'l1', 'cross_entropy', 'diag_loss', 'mse', 'add_loss', 'wing']


def build_loss(cfg):
    "build losses in right order"
    regress_criterions = []
    class_criterions = []
    for loss_name in cfg.loss.names:
        assert loss_name in AVAILABLE_LOSS
        if loss_name == 'cross_entropy':
            class_criterions.append(torch.nn.CrossEntropyLoss())
        elif loss_name == 'smoothl1':
            regress_criterions.append(torch.nn.SmoothL1Loss(reduction='mean', beta=cfg.loss.smoothl1_beta))
        elif loss_name == 'l1':
            regress_criterions.append(torch.nn.L1Loss(reduction='mean'))
        elif loss_name == 'mse':
            regress_criterions.append(torch.nn.MSELoss())
        elif loss_name == 'wing':
            regress_criterions.append(WingLoss(w=cfg.loss.w, eps=cfg.loss.eps))
        elif loss_name == 'add_loss':
            regress_criterions.append(ADD_loss())
        elif loss_name == 'diag_loss':
            regress_criterions.append(DiagLoss())

    return regress_criterions, class_criterions
