import torch
from icecream import ic
from torch.nn.modules.loss import _Loss

__all__ = ['DiagLoss', 'ADD_loss']

class DiagLoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)
        self.l1_loss = torch.nn.SmoothL1Loss(beta=.4)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x0_pr, x0_tr = torch.min(input[:,:,0], dim=1).values, torch.min(target[:,:,0], dim=1).values
        y0_pr, y0_tr = torch.min(input[:,:,1], dim=1).values, torch.min(target[:,:,1], dim=1).values
        x1_pr, x1_tr = torch.max(input[:,:,0], dim=1).values, torch.max(target[:,:,0], dim=1).values
        y1_pr, y1_tr = torch.max(input[:,:,1], dim=1).values, torch.max(target[:,:,1], dim=1).values
        diag_pr = torch.sqrt((x1_pr - x0_pr)**2 + (y1_pr - y0_pr)**2)
        diag_tr = torch.sqrt((x1_tr - x0_tr)**2 + (y1_tr - y0_tr)**2)

        diag_diff = self.l1_loss(diag_pr, diag_tr)

        return diag_diff

class ADD_loss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(size_average, reduce, reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # find distance between each point of the input and target. Sum it for each
        # instance and mean it over all instances
        return torch.mean(torch.sum(torch.linalg.norm(input-target, dim=2), dim=1))

def test():
    import torch.nn.functional as F
    for loss in [DiagMSE(), ADD_loss()]:
        input = F.sigmoid(torch.randn(3, 9, 2, requires_grad=True))
        target = F.sigmoid(torch.randn(3, 9, 2))
        output = loss(input, target)
        ic(output)
        output.backward()

if __name__ == '__main__':
    test()
