from torch.utils.data import DataLoader
import albumentations as A
import cv2

from torchdet3d.dataloaders import Objectron
from torchdet3d.utils import ConvertColor, ToTensor

def build_loader(config, mode='train'):

    train_transform, test_transform = build_augmentations(cfg=config)

    train_dataset = Objectron(config.data.root, mode='train', transform=train_transform,
                                category_list=config.data.category_list)
    train_loader = DataLoader(train_dataset, batch_size=config.data.train_batch_size,
                                shuffle=True, num_workers=config.data.num_workers)

    val_dataset = Objectron(config.data.root, mode='val', transform=test_transform,
                                category_list=config.data.category_list)
    val_loader = DataLoader(val_dataset, batch_size=config.data.val_batch_size, shuffle=True,
                                num_workers=config.data.num_workers)

    test_dataset = Objectron(config.data.root, mode='test', transform=test_transform,
                                category_list=config.data.category_list)
    test_loader = DataLoader(test_dataset, batch_size=config.data.val_batch_size, shuffle=False,
                                num_workers=config.data.num_workers)

    return train_loader, val_loader, test_loader

def build_augmentations(cfg):
    normalize = A.augmentations.transforms.Normalize (**cfg.data.normalization)

    train_transform = A.Compose([
                            ConvertColor(),
                            A.Resize(*cfg.data.resize, interpolation=cv2.INTER_CUBIC),
                            A.HorizontalFlip(p=0.3),
                            # A.Rotate(limit=30, p=0.3, interpolation=cv2.INTER_CUBIC),
                            A.RandomBrightnessContrast(p=0.2),
                            normalize,
                            ToTensor(cfg.data.resize)
                            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    test_transform = A.Compose([
                            ConvertColor(),
                            A.Resize(*cfg.data.resize,cv2.INTER_CUBIC),
                            normalize,
                            ToTensor(cfg.data.resize)
                            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    return train_transform, test_transform
