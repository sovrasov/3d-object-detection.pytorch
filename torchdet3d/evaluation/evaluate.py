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
from torchdet3d.utils import load_pretrained_weights, AverageMeter, mkdir_if_missing
from torchdet3d.dataloaders import Objectron

module_path = os.path.abspath(os.path.join('/3rdparty/Objectron'))
if module_path not in sys.path:
    sys.path.append(module_path)
from objectron.dataset import graphics

@dataclass
class Evaluater:
    model: object
    val_loader: object
    test_loader: object
    cfg: dict
    writer: object
    max_epoch: int
    checkpoint_path: str = ''
    device: str = 'cuda'
    samples: object = 'random'
    num_samples: int = 10
    path_to_save_imgs: str = './testing_images'
    debug: bool = False
    val_step: int = 0

    def visual_test(self):
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
            img = img.to(self.device)
            gt_kp = gt_kp.to(self.device)
            gt_cat = gt_cat.to(self.device)
            # feed forward net and put it on cpu
            pred_kp, pred_cat = self.model(torch.unsqueeze(img,0))
            # compute metrics for given sampled image
            ADD, SADD = compute_average_distance(pred_kp, torch.unsqueeze(gt_kp, 0))
            accuracy = compute_accuracy(pred_cat, torch.unsqueeze(gt_cat,0))

            print(f"\nimage №{idx}.\nComputed metrics:\n"
                  f"ADD ---> {ADD.item()}\n"
                  f"SADD ---> {SADD.item()}\n"
                  f"classification accuracy ---> {accuracy}")

            #translate keypoints from crop coordinates to original image
            img = img.detach().permute(1, 2, 0).cpu().numpy()
            pred_kp = ds.unnormalize(img, pred_kp[0].detach().cpu().numpy())
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

    def val(self, epoch=None):
        ''' procedure launching main validation'''
        ADD_meter = AverageMeter()
        SADD_meter = AverageMeter()
        ACC_meter = AverageMeter()

        # switch to eval mode
        self.model.eval()
        loop = tqdm(enumerate(self.val_loader), total=len(self.val_loader), leave=False)
        for it, (imgs, gt_kp, gt_cats) in loop:
            if any([obj is None for obj in (imgs, gt_kp, gt_cats)]):
                continue
            # put image and keypoints on the appropriate device
            imgs = imgs.to(self.device)
            gt_kp = gt_kp.to(self.device)
            gt_cats = gt_cats.to(self.device)
            # compute output and loss
            pred_kp, pred_cats = self.model(imgs)
            # measure metrics
            ADD, SADD = compute_average_distance(pred_kp, gt_kp)
            acc = compute_accuracy(pred_cats, gt_cats)
            # record loss
            ADD_meter.update(ADD.item(), imgs.size(0))
            SADD_meter.update(SADD.item(), imgs.size(0))
            ACC_meter.update(acc.item(), imgs.size(0))
            if epoch is not None:
                # write to writer for tensorboard
                self.writer.add_scalar('Val/ADD', ADD_meter.avg, global_step=self.val_step)
                self.writer.add_scalar('Val/SADD', SADD_meter.avg, global_step=self.val_step)
                self.writer.add_scalar('Val/ACC', ACC_meter.avg, global_step=self.val_step)
                self.val_step += 1
                # update progress bar
                loop.set_description(f'Val Epoch [{epoch}/{self.max_epoch}]')
                loop.set_postfix(ADD=ADD.item(), avr_ADD=ADD_meter.avg, SADD=SADD.item(),
                                    avr_SADD=SADD_meter.avg, acc=acc.item(), acc_avg = ACC_meter.avg)

            if self.debug and it == 10:
                break
        if epoch:
            print(f"val: epoch: {epoch}, ADD: {ADD_meter.avg},"
                  f" SADD: {SADD_meter.avg}, accuracy: {ACC_meter.avg}")
        else:
            print(f"\nComputed test metrics:\n"
                  f"ADD ---> {ADD_meter.avg}\n"
                  f"SADD ---> {SADD_meter.avg}\n"
                  f"classification accuracy ---> {ACC_meter.avg}")

    def run_eval_pipe(self):
        print('.'*10,'Run evaluating protocol', '.'*10)
        self._init_model()
        self.val()
        self.evaluate()

    def _init_model(self):
        if self.checkpoint_path:
            load_pretrained_weights(self.model, self.checkpoint_path)
