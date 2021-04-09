import math

import torch
from icecream import ic
from torch.nn.modules.loss import _Loss

__all__ = ['DiagLoss', 'ADD_loss', 'WingLoss']

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
        diag_dist = compute_diag(target)
        loss /= diag_dist.view(input_.size(0),1,1)

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

def test():
    import torch.nn.functional as F
    for loss in [WingLoss()]:
        input_ = F.sigmoid(torch.randn(3, 9, 2, requires_grad=True))
        target = F.sigmoid(torch.randn(3, 9, 2))
        output = loss(input_, target)
        ic(output)
        output.backward()

if __name__ == '__main__':
    test()
