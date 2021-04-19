import math

import torch
from torch.nn.modules.loss import _Loss

__all__ = ['DiagLoss', 'ADD_loss', 'WingLoss', 'LossManager']

class DiagLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.l1_loss = torch.nn.SmoothL1Loss(beta=.4)

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diag_pr = compute_diag(input_)
        diag_tr = compute_diag(target)
        diag_diff = self.l1_loss(diag_pr, diag_tr)

        return diag_diff

class ADD_loss(_Loss):
    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # find distance between each point of the input and target. Sum it for each
        # instance and mean it over all instances
        return torch.mean(torch.sum(torch.linalg.norm(input_-target, dim=2), dim=1))

class WingLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, w=0.05, eps=2, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.w = w
        self.eps = eps

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        wing_const = self.w - self.wing_core(self.w, self.w, self.eps)
        loss = torch.abs(input_ - target)
        loss[loss < self.w] = self.wing_core(loss[loss < self.w], self.w, self.eps)
        loss[loss >= self.w] -= wing_const
        # diag_dist = compute_diag(target)
        # loss /= diag_dist.view(input_.size(0),1,1)

        return torch.mean(loss)

    @staticmethod
    def wing_core(x, w, eps):
        """Calculates the wing function from https://arxiv.org/pdf/1711.06753.pdf"""
        if isinstance(x, float):
            return w*math.log(1. + x / eps)
        return w*torch.log(1. + x / eps)

def compute_diag(input_: torch.Tensor):
    x0 = torch.min(input_[:,:,0], dim=1).values
    y0 = torch.min(input_[:,:,1], dim=1).values
    x1 = torch.max(input_[:,:,0], dim=1).values
    y1 = torch.max(input_[:,:,1], dim=1).values
    diag = torch.sqrt((x1 - x0)**2 + (y1 - y0)**2)

    return diag

class LossManager:
    def __init__(self, criterions, coefficients, alwa):
        self.reg_criterions, self.class_criterions = criterions
        self.reg_coeffs, self.class_coeffs = coefficients
        assert len(self.reg_coeffs) == len(self.reg_criterions)
        assert len(self.class_coeffs) == len(self.class_criterions)
        assert self.reg_criterions
        self.use_alwa = alwa.use
        if alwa.use:
            assert self.class_criterions
            assert self.reg_coeffs[0] == self.class_coeffs[0] == 1.
        # init lambdas for alwa algorithm
        self.lam_cls = alwa.lam_cls
        self.lam_reg = alwa.lam_reg
        self.s_cls = list()
        self.s_reg = list()
        self.C = alwa.C
        self.alwa_version = 'ver_1' if alwa.compute_std else 'ver_2'

    def parse_losses(self, pred_kp, gt_kp,
                        pred_cats, gt_cats, iter_):
        class_loss = []
        regress_loss = []
        # compute losses
        if self.class_criterions:
            for k, cr in zip(self.class_coeffs, self.class_criterions):
                class_loss.append(cr(pred_cats, gt_cats) * k)
        else:
            class_loss = torch.zeros(1, requires_grad=True)
        for k, cr in zip(self.reg_coeffs, self.reg_criterions):
            regress_loss.append(cr(pred_kp, gt_kp) * k)
        reg_loss = sum(regress_loss)
        cls_loss = sum(class_loss)
        # compute alwa algo or just return sum of losses
        if not self.use_alwa:
            return sum(regress_loss) + sum(class_loss)
        self.s_cls.append(self.lam_cls*cls_loss)
        self.s_reg.append(self.lam_reg*reg_loss)
        if iter_ % self.C == 0 and iter_ != 0:
            cls_mean = torch.mean(torch.stack(self.s_cls))
            cls_std = torch.std(torch.stack(self.s_cls))
            reg_mean = torch.mean(torch.stack(self.s_reg))
            reg_std = torch.std(torch.stack(self.s_reg))
            self.s_cls.clear()
            self.s_reg.clear()
            if self.alwa_version == 'ver_1':
                cls = cls_mean + cls_std
                reg = reg_mean + reg_std
            else:
                cls = cls_mean
                reg = reg_mean
            if cls > reg:
                self.lam_cls = (1 - (cls - reg)/cls).item()
                print(f"classification coefficient changed : {self.lam_cls}")

        return self.lam_reg * sum(regress_loss) + self.lam_cls * sum(class_loss)
