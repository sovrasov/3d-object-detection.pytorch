''' Parse objectron data to PyTorch dataloader. Cereal box for now only with shuffled images.'''
from pathlib import Path
import os
from torch.utils.data import Dataset, DataLoader
import torch
import cv2 as cv
import json
import numpy as np
from icecream import ic
from numba import jit
import albumentations as A
import sys

sys.path.insert(1, '/home/prokofiev/Objectron')
from objectron.schema import features
from objectron.dataset import box, graphics


class Objectron(Dataset):
    def __init__(self, root_folder, mode='train', transform=None, debug_mode=False):
        self.root_folder = root_folder
        self.transform = transform
        self.debug_mode = debug_mode
        self.mode = mode
        if mode == 'train':
            ann_path = Path(root_folder).resolve() /  'annotations/objectron_train.json'
            with open(ann_path, 'r') as f:
                self.ann = json.load(f)
        elif mode in ['val', 'test']:
            ann_path = Path(root_folder).resolve() /  'annotations/objectron_test.json'
            with open(ann_path, 'r') as f:
                self.ann = json.load(f)
        else:
            raise RuntimeError("Unknown dataset mode")

    def __len__(self):
        return len(self.ann['annotations'])

    def __getitem__(self, indx):
        # get path to image from annotations
        raw_keypoints = self.ann['annotations'][indx]['keypoints']
        img_id = self.ann['annotations'][indx]['image_id']
        category = self.ann['annotations'][indx]['category_id']
        # get raw key points for bb from annotations
        img_path = self.root_folder + '/' + (self.ann['images'][img_id]['file_name'])
        # read image
        image = cv.imread(img_path, flags=1)
        assert image is not None
        # The keypoints are [x, y] where `x` and `y` are unnormalized
        # transform raw key points to this representation
        unnormalized_keypoints = np.array(raw_keypoints).reshape(9, 2)
        # "print" image after crop with keypoints if needed
        if self.debug_mode:
            imgh = image.copy()
            b = np.zeros((9,3))
            b[:,:2] = self.normalize(image, unnormalized_keypoints)
            graphics.draw_annotation_on_image(imgh, b , [9])
            cv.imwrite('image_before_pipeline.jpg', imgh)
        # given unnormalized keypoints crop object on image
        cropped_keypoints, cropped_img = self.crop(image, unnormalized_keypoints)

        # convert colors from BGR to RGB
        cropped_img = cv.cvtColor(cropped_img, cv.COLOR_BGR2RGB)
        # do augmentations with keypoints
        if self.transform:
            transformed = self.transform(image=cropped_img, keypoints=cropped_keypoints)
            assert np.array(transformed['keypoints']).shape == np.random.rand(9,2).shape

            transformed_image = transformed['image']
            transformed_bbox = transformed['keypoints']
        else:
            transformed_image, transformed_bbox = image, cropped_keypoints
        # given croped image and unnormalized key points: normilize it for cropped image
        transformed_bbox = self.normalize(transformed_image, transformed_bbox)
        # "print" image after crop with keypoints if needed
        if self.debug_mode:
            imgh = transformed_image.copy()
            b = np.zeros((9,3))
            b[:,:2] = transformed_bbox
            graphics.draw_annotation_on_image(imgh, b , [9])
            cv.imwrite('after_preprocessing.jpg', cv.cvtColor(imgh, cv.COLOR_RGB2BGR))

        transformed_image = np.transpose(transformed_image, (2,0,1)).astype(np.float32)
        transformed_bbox = torch.tensor(transformed_bbox, dtype=torch.float32)
        category = torch.tensor(category - 1, dtype=torch.float32)

        if self.mode == 'test':
            return (image,
                    torch.from_numpy(transformed_image),
                    transformed_bbox,
                    category.long())

        return (torch.from_numpy(transformed_image), transformed_bbox, category.long())

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

        # prepare transformation for image cropping and kp shifting
        transform_crop = A.Compose([
                        A.Crop(x0,y0,x1,y1),
                        ], keypoint_params=A.KeypointParams(format='xy'))
        # do actual crop and kp shift
        transformed = transform_crop(
                                image=image,
                                keypoints=clipped_bb
                                )

        crop_img = transformed['image']
        bb = transformed['keypoints']
        assert np.array(bb).shape == np.random.rand(9,2).shape

        return bb, crop_img

    def clip_bb(self, bbox, w, h):
        ''' clip offset bbox coordinates
        bbox: np.array, shape: [9,2], repr: [x,y]'''
        clipped_bbox = np.empty_like(bbox)
        clamped_x = list(map(lambda x: self.clamp(x, 3, w-3), bbox[:,0]))
        clamped_y = list(map(lambda y: self.clamp(y, 3, h-3), bbox[:,1]))
        clipped_bbox[:,0] = clamped_x;  clipped_bbox[:,1] = clamped_y
        return clipped_bbox

    @staticmethod
    def unnormalize(image, normalized_keypoints):
        ''' transform image to global pixel values '''
        h, w, _ = image.shape
        keypoints = np.multiply(normalized_keypoints, np.asarray([w, h], np.float32)).astype(int)
        return keypoints

    @staticmethod
    def normalize(image, unnormalized_keypoints):
        ''' normalize keypoints to image coordinates '''
        h, w, _ = image.shape
        keypoints = unnormalized_keypoints / np.asarray([w, h], np.float32)

        return keypoints

    @staticmethod
    def clamp(x, min_x, max_x):
        return min(max(x, min_x), max_x)


def test():
    "Perform dataloader test"
    def super_vision_test(root, mode='val', transform=None, index=7):
        ds = Objectron(root, mode=mode, transform=transform, debug_mode=True)
        _, bbox, _ = ds[index]
        assert bbox.shape == torch.empty((9,2)).shape

    def dataset_test(root, mode='val', transform=None, batch_size=5):
        ds = Objectron(root, mode=mode, transform=transform)
        dataloader = DataLoader(ds, batch_size=batch_size)
        iter_dt = iter(dataloader)
        img_tensor, bbox, cat = next(iter_dt)
        ic(mode)
        ic(cat)
        ic(img_tensor.shape)
        ic(bbox.shape)
        assert img_tensor.shape == torch.empty((batch_size, 3, 290, 128)).shape
        assert bbox.shape == torch.empty((batch_size, 9, 2)).shape

    root = '/home/prokofiev/3D-object-recognition/data'
    transform = A.Compose([
                            A.Resize(290, 128),
                            A.RandomBrightnessContrast(p=0.2),
                          ], keypoint_params=A.KeypointParams(format='xy'))

    super_vision_test(root, mode='train', transform=transform, index=180435)
    dataset_test(root, mode='val', transform=transform, batch_size=256)
    dataset_test(root, mode='train', transform=transform, batch_size=256)

if __name__ == '__main__':
    test()
