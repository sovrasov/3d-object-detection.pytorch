import os

from torch.utils.data import DataLoader
import albumentations as A

from torchdet3d.dataloaders import Objectron
from torchdet3d.utils import ToTensor, ConvertColor

def build_loader(config, mode='train'):

    train_transform, test_transform = build_augmentations(cfg=config)

    train_dataset = Objectron(config.data.root, mode='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=config.data.batch_size,
                              shuffle=True, num_workers=config.data.num_workers)

    val_dataset = Objectron(config.data.root, mode='val', transform=test_transform)
    val_loader = DataLoader(val_dataset, batch_size=config.data.batch_size, shuffle=True,
                            num_workers=config.data.num_workers)

    test_dataset = Objectron(config.data.root, mode='test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.data.batch_size, shuffle=False,
                            num_workers=config.data.num_workers)

    return train_loader, val_loader, test_loader

def build_augmentations(cfg):
    normalize = A.augmentations.transforms.Normalize (**cfg.data.normalization)

    train_transform = A.Compose([
                            ConvertColor(),
                            A.Resize(*cfg.data.resize),
                            A.HorizontalFlip(p=0.3),
                            A.RandomBrightnessContrast(p=0.2),
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
