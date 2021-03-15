import torch

__AVAILABLE_OPTIMS = ['sgd', 'rmsprop', 'adam', 'adadelta']

def build_optimizer(cfg, net):
    assert cfg.optim.name in __AVAILABLE_OPTIMS
    if cfg.optim.name == 'adadelta':
        optim = torch.optim.Adadelta(net.parameters(), lr=cfg.optim.lr, rho=cfg.optim.rho,
                                     weight_decay=cfg.otim.wd)
    elif cfg.optim.name == 'adam':
        optim = torch.optim.Adam(net.parameters(), lr=cfg.optim.lr, betas=cfg.optim.betas,
                                 weight_decay=cfg.otim.wd)
    elif cfg.optim.name == 'rmsprob':
        optim = torch.optim.RMSprop(net.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.wd,
                                    centered=cfg.optim.centered, alpha=cfg.optim.alpha)
    elif cfg.optim.name == 'sgd':
        optim = torch.optim.SGD(net.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.wd,
                                momentum=cfg.optim.momentum)

    return optim
