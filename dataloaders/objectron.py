''' Parse objectron data to PyTorch dataloader. Cereal box for now only with shuffled images.'''
from pathlib import Path
import os
from torch.utils.data import Dataset
import torch
import cv2 as cv
import json
import numpy as np
from icecream import ic


class Objectron(Dataset):
    def __init__(self, root_folder, test_mode=False, transform=None):
        self.root_folder = root_folder
        self.train_root = Path(root_folder).resolve() / 'train'
        self.test_root = Path(root_folder).resolve() / 'test'
        self.transform = transform
        if not test_mode:
            self.fetch_ann(self.train_root)
        else:
            assert test_mode
            self.fetch_ann(self.test_root)

        self.num_instances = np.sum([anns['instance_num'] for anns in self.ann.values()])

    def __len__(self):
        return abs(self.num_instances)

    def __getitem__(self, indx):
        # get path to image from annotations
        img_path = self.train_root / (self.ann[str(indx)]['filename'] + '.jpg')
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
        # batch_keypoints = np.split(prefetched_bbox, np.array(np.cumsum(num_istances)))[0]
        # unnormilize keypoints to global representation
        unnormalized_keypoints = self.unnormalize(image, normalized_keypoints)
        # given unnormalized keypoints crop object on image
        cropped_imgs = self.crop(image, unnormalized_keypoints, num_istances)
        # convert colors from BGR to RGB
        images = np.array([cv.cvtColor(image, cv.COLOR_BGR2RGB) for image in cropped_imgs])
        images = cropped_imgs
        # given croped image and unnormalized key points: normilize it for cropped image
        bbox = self.normalize(cropped_imgs, unnormalized_keypoints)

        for bb, img in zip(bbox, images): # TO DO
            if self.transform:
                bb, img = self.transform(img, bb)

        # [batch, channels, height, width]
        ic(images.shape)
        bbox = np.expand_dims(bbox, axis=0)
        images = np.expand_dims(images, axis=0)
        image = np.transpose(images, (0, 3, 1, 2)).astype(np.float32)
        return (torch.tensor(image), torch.tensor(bbox), np.sum(num_istances))

    def unnormalize(self, image, normalized_keypoints):
        ''' transform image to global pixel values '''
        keypoints = normalized_keypoints.reshape(-1, 3)
        h, w, _ = image.shape
        keypoints = np.multiply(keypoints, np.asarray([w, h, 1.], np.float32)).astype(int)
        return keypoints[:,:2]

    def crop(self, image, bbox, num_inctances):
        ''' fetch 2D bounding boxes from 3D and crop the image '''
        real_h, real_w, _ = image.shape

        x0 = self.clamp(min(bbox[:,0]) - 3, 0, real_w)
        y0 = self.clamp(min(bbox[:,1]) - 3, 0, real_h)
        x1 = self.clamp(max(bbox[:,0]) + 3, 0, real_w)
        y1 = self.clamp(max(bbox[:,1]) + 3, 0, real_h)

        cropped_img = image[y0 : y1, x0 : x1]
        # cv.imwrite('./crop.jpg', cropped_img)
        # cv.imwrite('before.jpg', image)

        return cropped_img

    def normalize(self, image, unnormalized_keypoints):
        ''' normalize keypoints to image coordinates '''
        h, w, _ = image.shape
        keypoints = unnormalized_keypoints / np.asarray([w, h], np.float32)
        return keypoints

    def fetch_ann(self, root):
        ann_path = list(map(lambda x: root / x,
                            filter(lambda x: x.endswith('.json'), os.listdir(root))))

        with open(ann_path[0], 'r') as f:
            self.ann = json.load(f)

    @staticmethod
    def clamp(x, min_x, max_x):
        return min(max(x, min_x), max_x)


def test_dataset():
    root = '/home/prokofiev/3D-object-recognition/data'
    ds = Objectron(root)
    ic(len(ds))
    img_tensor, bbox, num_samples = ds[3]
    ic(img_tensor.shape)
    ic(bbox)
    ic(bbox.shape[0])
    assert bbox.shape[0] == num_samples
    assert img_tensor.shape[0] == num_samples

if __name__ == '__main__':
    test_dataset()