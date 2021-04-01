import sys
import os

import torch
import numpy as np
from dataclasses import dataclass
import albumentations as A
from tqdm import tqdm
from copy import deepcopy
import cv2

from objectron.dataset import graphics

from .metrics import compute_average_distance, compute_accuracy
from torchdet3d.utils import (load_pretrained_weights, AverageMeter,
                                mkdir_if_missing, unnormalize, draw_kp, unnormalize_img)
from torchdet3d.builders import build_augmentations
from torchdet3d.dataloaders import Objectron

module_path = os.path.abspath(os.path.join('/3rdparty/Objectron'))
if module_path not in sys.path:
    sys.path.append(module_path)

@dataclass
class Evaluator:
    model: object
    val_loader: object
    test_loader: object
    cfg: dict
    writer: object
    max_epoch: int
    device: str = 'cuda'
    samples: object = 'random'
    num_samples: int = 10
    path_to_save_imgs: str = './testing_images'
    debug: bool = False
    debug_steps: int = 30
    val_step: int = 0

    def visual_test(self):
        _, test_transform = build_augmentations(self.cfg)
        ds = Objectron(self.cfg.data.root, mode='test', transform=test_transform)
        mkdir_if_missing(self.path_to_save_imgs)
        if self.samples == 'random':
            indexes = np.random.choice(len(ds), self.num_samples, replace=False)
        else:
            assert isinstance(self.samples, list)
            indexes = self.samples

        self.model.eval()
        for idx in tqdm(indexes):
            # pull next data
            prefetch_img, img, gt_kp, gt_cat, crop_cords= ds[idx]
            # draw true key points on original image
            gt_kp_cp = self.transform_kp(deepcopy(gt_kp), crop_cords)
            draw_kp(prefetch_img, gt_kp_cp, f'{self.path_to_save_imgs}/tested_image_№_{idx}_true.jpg', RGB=False, normalized=False)
            img, gt_kp = self.put_on_device([img, gt_kp], self.device)
            # feed forward net
            pred_kp, pred_cat = self.model(torch.unsqueeze(img,0), torch.tensor(gt_cat).view(1,-1))
            # compute metrics for given sampled image
            ADD, SADD = compute_average_distance(pred_kp, torch.unsqueeze(gt_kp, 0))
            accuracy = compute_accuracy(pred_cat, gt_cat)

            print(f"\nimage №{idx}.\nComputed metrics:\n"
                  f"ADD ---> {ADD}\n"
                  f"SADD ---> {SADD}\n"
                  f"classification accuracy ---> {accuracy}")

            # draw key_points on original image
            pred_kp = self.transform_kp(pred_kp[0].detach().cpu().numpy(), crop_cords)
            label = str(torch.argmax(pred_cat, dim=1).item())
            draw_kp(prefetch_img, pred_kp, f'{self.path_to_save_imgs}/tested_image_№_{idx}_predicted.jpg', RGB=False, normalized=False, label=label)

    def val(self, epoch=None):
        ''' procedure launching main validation'''
        ADD_meter = AverageMeter()
        SADD_meter = AverageMeter()
        ACC_meter = AverageMeter()

        # switch to eval mode
        self.model.eval()
        loop = tqdm(enumerate(self.val_loader), total=len(self.val_loader), leave=False)
        for it, (imgs, gt_kp, gt_cats) in loop:
            # put image and keypoints on the appropriate device
            imgs, gt_kp, gt_cats = self.put_on_device([imgs, gt_kp, gt_cats], self.device)
            # compute output and loss
            pred_kp, pred_cats = self.model(imgs, gt_cats)
            # measure metrics
            ADD, SADD = compute_average_distance(pred_kp, gt_kp)
            acc = compute_accuracy(pred_cats, gt_cats)
            # record loss
            ADD_meter.update(ADD, imgs.size(0))
            SADD_meter.update(SADD, imgs.size(0))
            ACC_meter.update(acc, imgs.size(0))
            if epoch is not None:
                # write to writer for tensorboard
                self.writer.add_scalar('Val/ADD', ADD_meter.avg, global_step=self.val_step)
                self.writer.add_scalar('Val/SADD', SADD_meter.avg, global_step=self.val_step)
                self.writer.add_scalar('Val/ACC', ACC_meter.avg, global_step=self.val_step)
                self.val_step += 1
                # update progress bar
                loop.set_description(f'Val Epoch [{epoch}/{self.max_epoch}]')
                loop.set_postfix(ADD=ADD, avr_ADD=ADD_meter.avg, SADD=SADD,
                                    avr_SADD=SADD_meter.avg, acc=acc, acc_avg = ACC_meter.avg)

            if self.debug and it == self.debug_steps:
                break

        ep_mess = f"epoch : {epoch}\n" if epoch is not None else f""
        print(f"\nComputed val metrics:\n"
              f"{ep_mess}"
              f"ADD ---> {ADD_meter.avg}\n"
              f"SADD ---> {SADD_meter.avg}\n"
              f"classification accuracy ---> {ACC_meter.avg}")

    def run_eval_pipe(self, visual_only=False):
        print('.'*10,'Run evaluating protocol', '.'*10)
        if not visual_only:
            self.val()
        self.visual_test()

    @staticmethod
    def put_on_device(items, device):
        for i in range(len(items)):
            items[i] = items[i].to(device)
        return items

    @staticmethod
    def transform_kp(kp: np.array, crop_cords: tuple):
        x0,y0,x1,y1 = crop_cords
        crop_shape = (x1-x0,y1-y0)
        kp[:,0] = kp[:,0]*crop_shape[0]
        kp[:,1] = kp[:,1]*crop_shape[1]
        kp[:,0] += x0
        kp[:,1] += y0
        return kp
