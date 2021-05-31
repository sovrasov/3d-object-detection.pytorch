from torchdet3d.losses import build_prtr_loss


def build_postprocessor(cfg):
    if 'prtr' in cfg.model.name:
        return PRTRPostprocessor()

    return lambda x, _: x


class PRTRPostprocessor:
    def __init__(self):
        self.criterion = build_prtr_loss()

    def __call__(self, outputs, gt_kpts):
        return self.criterion.get_val_predictions(outputs, gt_kpts)
