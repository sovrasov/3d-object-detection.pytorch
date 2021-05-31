import torch
import torch.nn as nn

AVAILABLE_OPTIMS = ['sgd', 'rmsprop', 'adam', 'adadelta']

def build_optimizer(cfg, net):
    assert cfg.optim.name in AVAILABLE_OPTIMS

    if cfg.optim.no_bias_decay:
        trainable_parameters = no_bias_decay(net, cfg.optim.lr, cfg.optim.wd)
    else:
        trainable_parameters = net.parameters()

    if cfg.optim.name == 'adadelta':
        optim = torch.optim.Adadelta(trainable_parameters, lr=cfg.optim.lr, rho=cfg.optim.rho,
                                     weight_decay=cfg.optim.wd)
    elif cfg.optim.name == 'adam':
        optim = torch.optim.AdamW(trainable_parameters, lr=cfg.optim.lr, betas=cfg.optim.betas,
                                 weight_decay=cfg.optim.wd)
    elif cfg.optim.name == 'rmsprop':
        optim = torch.optim.RMSprop(trainable_parameters, lr=cfg.optim.lr, weight_decay=cfg.optim.wd,
                                    alpha=cfg.optim.alpha)
    elif cfg.optim.name == 'sgd':
        optim = torch.optim.SGD(trainable_parameters, lr=cfg.optim.lr, weight_decay=cfg.optim.wd,
                                momentum=cfg.optim.momentum, nesterov=cfg.optim.nesterov)

    return optim


def no_bias_decay(model, lr, weight_decay):
    """
    no bias decay : only apply weight decay to the weights in convolution and fully-connected layers
    In paper [Bag of Tricks for Image Classification with Convolutional
    Neural Networks](https://arxiv.org/abs/1812.01187)
    """

    decay, bias_no_decay, weight_no_decay = [], [], []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            decay.append(m.weight)
            if m.bias is not None:
                bias_no_decay.append(m.bias)
        else: # BNs are intended to fall into this branch
            if hasattr(m, 'weight'):
                weight_no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                bias_no_decay.append(m.bias)

    assert len(list(model.parameters())) == len(decay) + len(bias_no_decay) + len(weight_no_decay)

    return [{'params': bias_no_decay, 'lr': 2*lr, 'weight_decay': 0.0},
            {'params': weight_no_decay, 'lr': lr, 'weight_decay': 0.0},
            {'params': decay, 'lr': lr, 'weight_decay': weight_decay}]
