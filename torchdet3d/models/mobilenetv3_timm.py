import torch
from timm.models.mobilenetv3 import mobilenetv3_large_100


class MobileNetV3_large_100(torch.nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.model = mobilenetv3_large_100(pretrained)
        self.model.classifier = None

    def extract_features(self, x):
        return self.model.forward_features(x)
