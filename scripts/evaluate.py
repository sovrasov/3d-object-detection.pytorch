import sys
import os

import torch
import numpy as np
from icecream import ic
from dataclasses import dataclass
import albumentations as A
from tqdm import tqdm
import cv2

from .metrics import compute_average_distance, compute_accuracy
from .utils import load_pretrained_weights, AverageMeter, mkdir_if_missing
from dataloaders import Objectron

module_path = os.path.abspath(os.path.join('/home/prokofiev/3D-object-recognition/3rdparty/Objectron'))
if module_path not in sys.path:
    sys.path.append(module_path)

from objectron.dataset import graphics

@dataclass
class Tester:
    model: object
    test_loader: object
    cfg: dict
    checkpoint_path: str = ''
    device: str = 'cuda'
    samples: object = 'random'
    num_samples: int = 10
    path_to_save_imgs: str = './testing_images'

    def evaluate(self):
        ds = Objectron(self.cfg.data.root, mode='test')
        mkdir_if_missing(self.path_to_save_imgs)
        if self.samples == 'random':
            indexes = np.random.choice(len(ds), self.num_samples, replace=False)
        else:
            assert isinstance(self.samples, list)
            indexes = self.samples

        for idx in tqdm(indexes):
            # pull next data
            prefetch_img, img, gt_kp, gt_cat = ds[idx]
            # feed forward net and put it on cpu
            pred_kp, pred_cat = self.model(torch.unsqueeze(img,0))
            pred_kp = pred_kp.to('cpu')
            pred_cat = pred_cat.to('cpu')
            # compute metrics for given sampled image
            ADD, SADD = compute_average_distance(pred_kp, torch.unsqueeze(gt_kp, 0))
            accuracy = compute_accuracy(pred_cat, torch.unsqueeze(gt_cat,0))

            print(f"\nimage №{idx}.\nComputed metrics:\n"
                  f"ADD ---> {ADD.item()}\n"
                  f"SADD ---> {SADD.item()}\n"
                  f"classification accuracy ---> {accuracy}")

            #translate keypoints from crop coordinates to original image
            img = img.detach().permute(1, 2, 0).numpy()
            pred_kp = ds.unnormalize(img, pred_kp[0].detach().numpy())
            pred_kp = A.Compose([
                                   A.Resize(height=prefetch_img.shape[0], width=prefetch_img.shape[1]),
                                ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))(image=img, keypoints=pred_kp)['keypoints']
            pred_kp = ds.normalize(prefetch_img, pred_kp)
            expanded_kp = np.zeros((9,3))
            expanded_kp[:,:2] = pred_kp
            graphics.draw_annotation_on_image(prefetch_img, expanded_kp, [9])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(prefetch_img, str(torch.argmax(pred_cat, dim=1).item()), (10,100), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite(f'{self.path_to_save_imgs}/image_{idx}.jpg', prefetch_img)

    def compute_metrics(self):
        ADD_meter = AverageMeter()
        SADD_meter = AverageMeter()
        ACC_meter = AverageMeter()

        loop = tqdm(enumerate(self.test_loader), total=len(self.test_loader), leave=False)
        for it, (prefetch_imgs, imgs, gt_kp, gt_cats) in loop:
            if any([obj is None for obj in (imgs, gt_kp, gt_cats)]):
                continue
            # put image and keypoints on the appropriate devuce
            imgs = imgs.to(self.device)
            gt_kp = gt_kp.to(self.device)
            gt_cats = gt_cats.to(self.device)
            # compute output and loss
            pred_kp, pred_cat = self.model(imgs)
            # measure metrics
            ADD, SADD = compute_average_distance(pred_kp, gt_kp)
            accuracy = compute_accuracy(pred_cat, gt_cats)
            ADD_meter.update(ADD.item(), imgs.size(0))
            SADD_meter.update(SADD.item(), imgs.size(0))
            ACC_meter.update(accuracy.item(), imgs.size(0))

            if self.cfg.debug_mode and it == 20:
                break

        print("\nMetrics on testing dataloader completed:\n"
              f"ADD ---> {ADD_meter.avg}\n"
              f"SADD ---> {SADD_meter.avg}\n"
              f"classification accuracy ---> {ACC_meter.avg}")

    def run_eval(self):
        print('.'*10,'Run evaluating protocol', '.'*10)
        self._init_model()
        self.compute_metrics()
        self.evaluate()

    def _init_model(self):
        if self.checkpoint_path:
            load_pretrained_weights(self.model, self.checkpoint_path)
