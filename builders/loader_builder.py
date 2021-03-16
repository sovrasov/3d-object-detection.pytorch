from torch.utils.data import DataLoader
import albumentations as A

import os
import sys
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
from dataloaders import Objectron


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
                            A.Resize(*cfg.data.resize),
                            A.HorizontalFlip(p=0.5),
                            A.augmentations.transforms.ISONoise(color_shift=(0.15,0.35),
                                                                intensity=(0.2, 0.5), p=0.2),
                            normalize
                            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    test_transform = A.Compose([
                            A.Resize(*cfg.data.resize),
                            normalize
                            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    return train_transform, test_transform
