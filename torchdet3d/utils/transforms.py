import math
import random
import cv2 as cv
import numpy as np
import torch
from albumentations.core.transforms_interface import BasicTransform, ImageOnlyTransform, DualTransform, to_tuple

from .utils import normalize

class ConvertColor(ImageOnlyTransform):
    """Converting color of the image
    """
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, img, **params):
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)


class RandomRescale(DualTransform):
    """Rescaling image and keypoints
    """
    def __init__(self, scale_limit=0.1, interpolation=cv.INTER_LINEAR, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.scale_limit = to_tuple(scale_limit, bias=0)
        self.interpolation = interpolation

    def get_params(self):
        return {"scale": random.uniform(self.scale_limit[0], self.scale_limit[1])}

    def apply(self, img, scale=0, interpolation=cv.INTER_LINEAR, **params):
        h, w = img.shape[:2]
        rot_mat = cv.getRotationMatrix2D((w*0.5, h*0.5), 0, scale)
        image = cv.warpAffine(img, rot_mat, (w, h), flags=interpolation)
        return image

    def apply_to_bbox(self, bbox, scale=0, **params):
        pass

    def apply_to_keypoint(self, keypoint, scale=0, cols=0, rows=0, **params):
        x, y, angle, s = keypoint[:4]
        rot_mat_l = cv.getRotationMatrix2D((0.5*cols, 0.5*rows), 0, scale)
        new_keypoint = cv.transform(np.array([x, y]).reshape((1, 1, 2)), rot_mat_l).reshape(-1)
        return new_keypoint[0], new_keypoint[1], angle, s * scale

    def get_transform_init_args(self):
        return {"interpolation": self.interpolation, "scale_limit": to_tuple(self.scale_limit, bias=-1.0)}


class RandomRotate(DualTransform):
    """Rotate image and keypoints
    """
    def __init__(self, angle_limit=0.1, interpolation=cv.INTER_LINEAR, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.angle_limit = to_tuple(angle_limit)
        self.interpolation = interpolation

    def get_params(self):
        return {"angle": random.uniform(self.angle_limit[0], self.angle_limit[1])}

    def apply(self, img, angle=0, interpolation=cv.INTER_LINEAR, **params):
        h, w = img.shape[:2]
        scale = self._get_scale_by_angle(angle, h, w)
        rot_mat = cv.getRotationMatrix2D((w*0.5, h*0.5), angle, scale)
        image = cv.warpAffine(img, rot_mat, (w, h), flags=interpolation)
        return image

    def apply_to_bbox(self, bbox, scale=0, **params):
        pass

    @staticmethod
    def _get_scale_by_angle(angle, h, w):
        rad_angle = math.radians(angle)
        cos = math.cos(rad_angle) - 1
        sin = math.sin(rad_angle)
        delta_h = w / 2 * cos + h / 2 * sin
        delta_w = w / 2 * sin + h / 2 * cos
        return max(w / (w + 2 * abs(delta_w)), h / (h + 2 * abs(delta_h)))

    def apply_to_keypoint(self, keypoint, angle=0, cols=0, rows=0, **params):
        x, y, phi, s = keypoint[:4]
        w, h = cols, rows
        scale = self._get_scale_by_angle(angle, h, w)
        rot_mat_l = cv.getRotationMatrix2D((w*0.5, h*0.5), angle, scale)
        new_keypoint = cv.transform(np.array([x, y]).reshape((1, 1, 2)), rot_mat_l).reshape(-1)
        return new_keypoint[0], new_keypoint[1], phi + math.radians(angle), s * scale

    def get_transform_init_args(self):
        return {"interpolation": self.interpolation, "scale_limit": to_tuple(self.angle_limit)}


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
