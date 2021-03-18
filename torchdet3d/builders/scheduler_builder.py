import torch

__AVAILABLE_SCHEDS = ['cosine']

def build_scheduler(cfg, optimizer):
    assert cfg.scheduler.name in __AVAILABLE_SCHEDS
    if cfg.scheduler.name == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.data.max_epochs, eta_min=5e-6)
    return sched
