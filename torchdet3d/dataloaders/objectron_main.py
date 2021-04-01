''' Parse objectron data to PyTorch dataloader. Cereal box for now only with shuffled images.'''
from pathlib import Path

from torch.utils.data import Dataset, DataLoader
import torch
import cv2 as cv
import json
import numpy as np
from icecream import ic
import albumentations as A
import sys

from torchdet3d.utils import (draw_kp, normalize, unnormalize,
                                unnormalize_img, ToTensor, ConvertColor)
from objectron.dataset import graphics


class Objectron(Dataset):
    def __init__(self, root_folder, mode='train', transform=None, debug_mode=False, name='0', category_list='all'):
        self.root_folder = root_folder
        self.name=name
        self.transform = transform
        self.debug_mode = debug_mode
        self.mode = mode
        from itertools import islice
        def take(n, iterable):
            "Return first n items of the iterable as a list"
            return list(islice(iterable, n))

        if mode == 'train':
            ann_path = Path(root_folder).resolve() / 'annotations/objectron_train.json'
            with open(ann_path, 'r') as f:
                self.ann = json.load(f)
        elif mode in ['val', 'test']:
            ann_path = Path(root_folder).resolve() / 'annotations/objectron_test.json'
            with open(ann_path, 'r') as f:
                self.ann = json.load(f)
        else:
            raise RuntimeError("Unknown dataset mode")

        # filter categories
        if not category_list == 'all':
            self.annotations = list(filter(lambda x: x['category_id'] in category_list, self.ann['annotations']))
            images_id = {ann_obj['image_id'] for ann_obj in self.annotations}
            # create dict since ordering now different
            self.images = {img_obj['id']: img_obj for img_obj in filter(lambda x: x['id'] in images_id, self.ann['images'])}
            assert len(self.images) == len(images_id)
        else:
            self.annotations = self.ann['annotations']
            self.images = self.ann['images']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, indx):
        # get path to image from annotations
        raw_keypoints = self.annotations[indx]['keypoints']
        img_id = self.annotations[indx]['image_id']
        category = int(self.annotations[indx]['category_id']) - 1
        # get raw key points for bb from annotations
        img_path = self.root_folder + '/' + (self.images[img_id]['file_name'])
        # read image
        image = cv.imread(img_path)
        assert image is not None
        # The keypoints are [x, y] where `x` and `y` are unnormalized
        # transform raw key points to this representation
        unnormalized_keypoints = np.array(raw_keypoints).reshape(9, 2)
        # "print" image after crop with keypoints if needed

        if self.debug_mode:
            draw_kp(image, unnormalized_keypoints, f'image_before_pipeline_{self.name}.jpg',
                    normalized=False, RGB=False)
        # given unnormalized keypoints crop object on image
        cropped_keypoints, cropped_img, crop_cords = self.crop(image, unnormalized_keypoints)

        # do augmentations with keypoints
        if self.transform:
            transformed = self.transform(image=cropped_img, keypoints=cropped_keypoints)
            transformed_image = transformed['image']
            transformed_keypoints = transformed['keypoints']
            assert (transformed_keypoints.shape == (9,2) and
                    isinstance(transformed_image, torch.Tensor) and
                    isinstance(transformed_keypoints, torch.Tensor))
        else:
            transformed_image, transformed_keypoints = cropped_img, cropped_keypoints
        # "print" image after crop with keypoints if needed
        if self.debug_mode:
            draw_kp(unnormalize_img(transformed_image).numpy(), transformed_keypoints.numpy(),
                                    f'image_after_pipeline_{self.name}.jpg')

        if self.mode == 'test':
            return (image,
                    transformed_image,
                    transformed_keypoints,
                    category,
                    crop_cords)

        return transformed_image, transformed_keypoints, category

    def crop(self, image, keypoints):
        ''' fetch 2D bounding boxes from keypoints and crop the image '''
        real_h, real_w, _ = image.shape

        # clamp bbox coordinates according to image shape
        clipped_bb = self.clip_bb(keypoints, real_w, real_h)
        # crop 2D bounding box from image by given 3D keypoints
        # (min x, miny) - left lower corner;
        # (max x, max y) - upper right corner
        x0 = self.clamp(min(clipped_bb[:,0]) - 10, 0, real_w)
        y0 = self.clamp(min(clipped_bb[:,1]) - 10, 0, real_h)
        x1 = self.clamp(max(clipped_bb[:,0]) + 10, 0, real_w)
        y1 = self.clamp(max(clipped_bb[:,1]) + 10, 0, real_h)

        crop_cords = (x0,y0,x1,y1)
        # prepare transformation for image cropping and kp shifting
        transform_crop = A.Compose([
                        A.Crop(*crop_cords),
                        ], keypoint_params=A.KeypointParams(format='xy'))
        # do actual crop and kp shift
        transformed = transform_crop(
                                image=image,
                                keypoints=clipped_bb
                                )

        crop_img = transformed['image']
        bb = transformed['keypoints']
        assert len(bb) == 9

        return bb, crop_img, crop_cords

    def clip_bb(self, bbox, w, h):
        ''' clip offset bbox coordinates
        bbox: np.array, shape: [9,2], repr: [x,y]'''
        clipped_bbox = np.empty_like(bbox)
        clamped_x = list(map(lambda x: self.clamp(x, 3, w-3), bbox[:,0]))
        clamped_y = list(map(lambda y: self.clamp(y, 3, h-3), bbox[:,1]))
        clipped_bbox[:,0] = clamped_x;  clipped_bbox[:,1] = clamped_y
        return clipped_bbox

    @staticmethod
    def clamp(x, min_x, max_x):
        return min(max(x, min_x), max_x)


def test():
    "Perform dataloader test"
    def super_vision_test(root, mode='val', transform=None, index=7):
        ds = Objectron(root, mode=mode, transform=transform, debug_mode=True, name=str(index))
        print(len(ds))
        ds = Objectron(root, mode=mode, transform=transform, debug_mode=True, name=str(index), category_list=[1])
        print(len(ds))
        _, bbox, _ = ds[index]
        assert bbox.shape == (9,2)

    def dataset_test(root, mode='val', transform=None, batch_size=5):
        ds = Objectron(root, mode=mode, transform=transform)
        dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        iter_dt = iter(dataloader)
        img_tensor, bbox, cat = next(iter_dt)
        ic(mode)
        ic(cat)
        ic(img_tensor.shape)
        ic(bbox.shape)
        assert img_tensor.shape == (batch_size, 3, 290, 290)
        assert bbox.shape == (batch_size, 9, 2)

    root = './data_cereal_box'
    normalize = A.augmentations.transforms.Normalize(**dict(mean=[0.5931, 0.4690, 0.4229],
                                                            std=[0.2471, 0.2214, 0.2157]))
    transform = A.Compose([ ConvertColor(),
                            A.Resize(290, 290),
                            A.RandomBrightnessContrast(p=0.2),
                            A.HorizontalFlip(p=0.5),
                            A.augmentations.transforms.ISONoise(color_shift=(0.15,0.35),
                                                                intensity=(0.2, 0.5), p=0.2),
                            normalize,
                            ToTensor((290, 290)),
                          ],keypoint_params=A.KeypointParams(format='xy'))
    for index in np.random.randint(0,60000,1):
        super_vision_test(root, mode='train', transform=transform, index=index)
    dataset_test(root, mode='val', transform=transform, batch_size=256)
    dataset_test(root, mode='train', transform=transform, batch_size=256)

if __name__ == '__main__':
    test()
