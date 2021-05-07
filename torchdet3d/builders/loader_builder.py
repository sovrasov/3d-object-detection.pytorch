from torch.utils.data import DataLoader
import albumentations as A
import numpy as np

from torchdet3d.dataloaders import Objectron
from torchdet3d.utils import ConvertColor, ToTensor, RandomRescale

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def build_loader(config, mode='train'):

    train_transform, test_transform = build_augmentations(cfg=config)

    train_dataset = Objectron(config.data.root, mode='train', transform=train_transform,
                                category_list=config.data.category_list)
    train_loader = DataLoader(train_dataset, batch_size=config.data.train_batch_size,
                                shuffle=True, num_workers=config.data.num_workers,
                                worker_init_fn=worker_init_fn)

    val_dataset = Objectron(config.data.root, mode='val', transform=test_transform,
                                category_list=config.data.category_list)
    val_loader = DataLoader(val_dataset, batch_size=config.data.val_batch_size, shuffle=True,
                                num_workers=config.data.num_workers,
                                worker_init_fn=worker_init_fn)

    test_dataset = Objectron(config.data.root, mode='test', transform=test_transform,
                                category_list=config.data.category_list)
    test_loader = DataLoader(test_dataset, batch_size=config.data.val_batch_size, shuffle=False,
                                num_workers=config.data.num_workers,
                                worker_init_fn=worker_init_fn)

    return train_loader, val_loader, test_loader

def build_augmentations(cfg):
    normalize = A.augmentations.transforms.Normalize (**cfg.data.normalization)

    train_transform = A.Compose([
                            ConvertColor(),
                            RandomRescale(scale_limit=(0.8, 0.95), p=0.4),
                            A.Resize(*cfg.data.resize),
                            A.HorizontalFlip(p=0.35),
                            # A.Rotate(limit=30, p=0.3),
                            A.OneOf([
                                        A.HueSaturationValue(p=0.3),
                                        A.RGBShift(p=0.3),
                                        A.RandomBrightnessContrast(p=0.3),
                                        A.ColorJitter(p=0.3),
                                    ], p=1),
                            A.Blur(blur_limit=5, p=0.15),
                            # A.IAAPiecewiseAffine(p=0.3),
                            normalize,
                            ToTensor(cfg.data.resize)
                            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    test_transform = A.Compose([
                            ConvertColor(),
                            A.Resize(*cfg.data.resize),
                            normalize,
                            ToTensor(cfg.data.resize)
                            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    return train_transform, test_transform
