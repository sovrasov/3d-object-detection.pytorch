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
        train_root = Path(root_folder).resolve() / 'train'
        test_root = Path(root_folder).resolve() / 'test'
        self.transform = transform
        self.debug_mode = debug_mode
        if mode == 'train':
            self.fetch_ann(train_root)
            self.data_root = train_root
        else:
            assert mode == 'val'
            self.fetch_ann(test_root)
            self.data_root = test_root

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, indx):
        # get path to image from annotations
        img_path = self.data_root / (self.ann[str(indx)]['filename'] + '.jpg')
        # get raw key points for bb from annotations
        raw_keypoints = self.ann[str(indx)]['point_2d']
        # get num instatnces from annotations for current image
        num_istances = self.ann[str(indx)]['instance_num']
        # read image
        image = cv.imread(str(img_path), flags=1)
        assert image is not None
        # The keypoints are [x, y, d] where `x` and `y` are normalized
        # and `d` is the metric distance from the center of the camera.
        # transform raw key points to this representation
        normalized_keypoints = np.array(raw_keypoints).reshape(np.sum(num_istances), 9, 3)
        # "print" image after crop with keypoints if needed
        if self.debug_mode:
            imgh = image.copy()
            for obj in range(normalized_keypoints.shape[0]):
                graphics.draw_annotation_on_image(imgh, normalized_keypoints[obj] , [9])
            cv.imwrite('image_before_pipeline.jpg', imgh)
        # batch_keypoints = np.split(prefetched_bbox, np.array(np.cumsum(num_istances)))[0]
        # unnormilize keypoints to global representation
        unnormalized_keypoints = self.unnormalize(image, normalized_keypoints)
        # given unnormalized keypoints crop object on image
        cropped_keypoints, cropped_imgs = self.crop(image, unnormalized_keypoints, num_istances)

        # "print" image after crop with keypoints if needed
        if self.debug_mode:
            stacked = [np.zeros((9, 3)) for kp in cropped_keypoints]
            norm_kp = self.normalize(cropped_imgs, cropped_keypoints)
            for obj in range(len(cropped_imgs)):
                stacked[obj][:,:-1] = norm_kp[obj]
                graphics.draw_annotation_on_image(cropped_imgs[obj], stacked[obj] , [9])
                cv.imwrite(f'cropped_with_ann_{obj}.jpg', cropped_imgs[obj])

        # convert colors from BGR to RGB
        images = [cv.cvtColor(image, cv.COLOR_BGR2RGB) for image in cropped_imgs]
        # do augmentations with keypoints
        if self.transform:
            transformed_images, transformed_bbox = [], []
            for id in range(len(cropped_keypoints)):
                    transformed = self.transform(image=images[id], keypoints=cropped_keypoints[id])
                    assert np.array(transformed['keypoints']).shape == np.random.rand(9,2).shape

                    transformed_images.append(transformed['image'])
                    transformed_bbox.append(transformed['keypoints'])
        else:
            transformed_images, transformed_bbox = images, cropped_keypoints
        # given croped image and unnormalized key points: normilize it for cropped image
        transformed_bbox = self.normalize(transformed_images, transformed_bbox)
        # [batch, channels, height, width]
        return (transformed_images, transformed_bbox, np.sum(num_istances))

    def unnormalize(self, image, normalized_keypoints):
        ''' transform image to global pixel values '''
        keypoints = [keypoint.reshape(-1, 3) for keypoint in normalized_keypoints]
        h, w, _ = image.shape
        keypoints = np.array([ np.multiply(keypoint, np.asarray([w, h, 1.], np.float32)).astype(int)
                        for keypoint in keypoints ])
        return keypoints[:,:,:2]

    def crop(self, image, bbox, num_inctances):
        ''' fetch 2D bounding boxes from 3D and crop the image '''
        real_h, real_w, _ = image.shape
        cropped_imgs = []
        cropped_bbox = []

        for obj in range(np.sum(num_inctances)):
            # clamp bbox coordinates according to image shape
            clipped_bb = self.clip_bb(bbox[obj], real_w, real_h)
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
            cropped_imgs.append(crop_img)
            cropped_bbox.append(bb)

        return cropped_bbox, cropped_imgs

    def normalize(self, image, unnormalized_keypoints):
        ''' normalize keypoints to image coordinates '''
        normalized_keypoints = []
        for obj in range(len(image)):
            h, w, _ = image[obj].shape
            keypoints = unnormalized_keypoints[obj] / np.asarray([w, h], np.float32)
            normalized_keypoints.append(keypoints)

        return normalized_keypoints

    def fetch_ann(self, root):
        ann_path = list(map(lambda x: root / x,
                            filter(lambda x: x.endswith('.json'), os.listdir(root))))

        with open(ann_path[0], 'r') as f:
            self.ann = json.load(f)

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

def correct_bbox():
    from tqdm import tqdm
    broken_pipes = []
    root = '/home/prokofiev/3D-object-recognition/data'
    ds = Objectron(root, mode = 'val', transform=None)
    img_tensor, bbox, num_samples = ds[50000]
    for id in tqdm(range(len(ds))):
        try:
            img_tensor, bbox, num_samples = ds[id]
        except:
            broken_pipes.append(id)
            continue
    with open('broken_pipes_test.txt', 'w') as f:
        for item in broken_pipes:
            f.write("%s\n" % item)

    print(len(broken_pipes), broken_pipes)

def collate(batch):
    imgs = np.array([np.transpose(np.array(img), (2,0,1)).astype(np.float32)
                        for batch_inctances in batch for img in batch_inctances[0]])
    bbox = np.array([kp for batch_inctances in batch for kp in batch_inctances[1]])

    return torch.from_numpy(imgs), torch.from_numpy(bbox)

def super_vision_test(root, mode='val', transform=None, index=7):
    ds = Objectron(root, mode=mode, transform=transform, debug_mode=True)
    img_tensor, bbox, num_samples = ds[index]
    assert len(bbox) == num_samples
    assert len(img_tensor) == num_samples

def dataset_test(root, mode='val', transform=None, batch_size=5):
    ds = Objectron(root, mode=mode, transform=transform)
    dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=collate)
    iter_dt = iter(dataloader)
    img_tensor, bbox = next(iter_dt)
    ic(mode)
    ic(img_tensor.shape)
    ic(bbox.shape)
    assert img_tensor.shape[1:] == torch.empty((3, 224, 128)).shape
    assert bbox.shape[1:] == torch.empty((9, 2)).shape

def test():
    root = '/home/prokofiev/3D-object-recognition/data'
    transform = A.Compose([
                            A.Resize(224,128),
                            A.RandomBrightnessContrast(p=0.2),
                          ], keypoint_params=A.KeypointParams(format='xy'))

    super_vision_test(root, mode='train', transform=transform, index=21)
    # dataset_test(root, mode='val', transform=transform, batch_size=256)
    # dataset_test(root, mode='train', transform=transform, batch_size=256)

if __name__ == '__main__':
    test()
