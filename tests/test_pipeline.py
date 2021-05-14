import torch

from torchdet3d.evaluation import compute_metrics_per_cls

from torchdet3d.losses import WingLoss, ADD_loss, DiagLoss
from torchdet3d.builders import (build_loss, build_optimizer, build_scheduler,
                                    build_model, AVAILABLE_LOSS, AVAILABLE_OPTIMS, AVAILABLE_SCHEDS)
from torchdet3d.utils import read_py_config


class TestCasesPipeline:
    gt_kps = torch.rand(128,9,2)
    test_kps = torch.rand(128,9,2, requires_grad=True)
    gt_cats = torch.randint(0,9,(128,))
    test_cats = torch.rand(128,9)
    config = read_py_config("./configs/default_config.py")

    def test_metrics(self):
        cls_metrics, ADD, SADD, IOU, acc = compute_metrics_per_cls(self.test_kps, self.gt_kps,
                                                                   self.test_cats, self.gt_cats)
        assert 0 <= ADD <= 1 and 0 <= SADD <= 1 and 0 <= IOU <= 1 and 0 <= acc <= 1
        assert len(cls_metrics) == 9 and len(cls_metrics[0]) == 5

    def test_losses(self):
        for loss in [WingLoss(), ADD_loss(), DiagLoss()]:
            input_ = torch.sigmoid(torch.randn(512, 9, 2, requires_grad=True))
            target = torch.sigmoid(torch.randn(512, 9, 2))
            output = loss(input_, target)
            assert not torch.any(torch.isnan(output))
            output.backward()

    def test_builders(self):
        for loss_ in AVAILABLE_LOSS:
            if loss_ != 'cross_entropy':
                self.config['loss']['names']=[loss_, 'cross_entropy']
                self.config.loss.coeffs=([1.],[1.])
                regress_criterions, class_criterions = build_loss(self.config)
                assert len(regress_criterions) == 1 and len(class_criterions) == 1
        model = build_model(self.config)
        assert model is not None
        for optim_ in AVAILABLE_OPTIMS:
            self.config['optim']['name'] = optim_
            optimizer = build_optimizer(self.config, model)
            assert optimizer is not None
            for schd in AVAILABLE_SCHEDS:
                self.config['scheduler']['name'] = schd
                scheduler = build_scheduler(self.config, optimizer)
                assert scheduler is not None

    def test_random_inference(self):
        model = build_model(self.config)
        image = torch.rand(128,3,224,224)
        kp, cat = model(image, self.gt_cats)
        assert kp.shape == (128,9,2)
        assert cat.shape == (128,9)
