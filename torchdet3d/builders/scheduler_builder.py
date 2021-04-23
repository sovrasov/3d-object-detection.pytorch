import torch

AVAILABLE_SCHEDS = ['cosine', 'exp', 'stepLR', 'multistepLR']

def build_scheduler(cfg, optimizer):
    if cfg.scheduler.name is None:
        return None

    assert cfg.scheduler.name in AVAILABLE_SCHEDS
    if cfg.scheduler.name == 'cosine':
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=cfg.data.max_epochs,
                                                           eta_min=5e-6)
    if cfg.scheduler.name == 'exp':
        sched = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                       gamma=cfg.scheduler.exp_gamma)
    if cfg.scheduler.name == 'stepLR':
        sched = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=cfg.scheduler.steps[0],
                                                gamma=cfg.scheduler.gamma)
    if cfg.scheduler.name == 'multistepLR':
        sched = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=cfg.scheduler.steps,
                                                     gamma=cfg.scheduler.gamma)
    return sched
