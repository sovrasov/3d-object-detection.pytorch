import torch

AVAILABLE_OPTIMS = ['sgd', 'rmsprop', 'adam', 'adadelta']

def build_optimizer(cfg, net):
    assert cfg.optim.name in AVAILABLE_OPTIMS
    if cfg.optim.name == 'adadelta':
        optim = torch.optim.Adadelta(net.parameters(), lr=cfg.optim.lr, rho=cfg.optim.rho,
                                     weight_decay=cfg.optim.wd)
    elif cfg.optim.name == 'adam':
        optim = torch.optim.AdamW(net.parameters(), lr=cfg.optim.lr, betas=cfg.optim.betas,
                                 weight_decay=cfg.optim.wd)
    elif cfg.optim.name == 'rmsprop':
        optim = torch.optim.RMSprop(net.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.wd, alpha=cfg.optim.alpha)
    elif cfg.optim.name == 'sgd':
        optim = torch.optim.SGD(net.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.wd,
                                momentum=cfg.optim.momentum, nesterov=cfg.optim.nesterov)

    return optim
