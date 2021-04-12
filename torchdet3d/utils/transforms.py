import cv2 as cv
import numpy as np
import torch
from albumentations.core.transforms_interface import BasicTransform, ImageOnlyTransform

from .utils import normalize

class ConvertColor(ImageOnlyTransform):
    """Converting color of the image
    """
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)

class ToTensor(BasicTransform):
    """Converting color of the image
    """
    def __init__(self, img_shape, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.img_final_shape = img_shape

    @property
    def targets(self):
        return {"image": self.apply, "keypoints": self.apply_to_keypoints}

    def apply(self, img, **params):  # skipcq: PYL-W0613
        if len(img.shape) not in [2, 3]:
            raise ValueError("Albumentations only supports images in HW or HWC format")

        if len(img.shape) == 2:
            img = np.expand_dims(img, 2)

        return torch.from_numpy(img.transpose(2, 0, 1)).float()

    def apply_to_keypoints(self, keypoints, **params):
        keypoints = normalize(self.img_final_shape, keypoints)
        return  torch.tensor(keypoints, dtype=torch.float)
