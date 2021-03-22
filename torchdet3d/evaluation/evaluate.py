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
from torchdet3d.utils import (load_pretrained_weights, AverageMeter,
                                mkdir_if_missing, unnormalize, draw_kp, unnormalize_img)
from torchdet3d.builders import build_augmentations
from torchdet3d.dataloaders import Objectron

module_path = os.path.abspath(os.path.join('/3rdparty/Objectron'))
if module_path not in sys.path:
    sys.path.append(module_path)
from objectron.dataset import graphics

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
            prefetch_img, img, gt_kp, gt_cat = ds[idx]
            img_cp = unnormalize_img(img).numpy()
            draw_kp(img_cp, gt_kp.cpu(), f'{self.path_to_save_imgs}/tested_image_№_{idx}_tr.jpg', RGB=True, normalized=True)

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

            # pred_kp = unnormalize(img, pred_kp[0].detach().cpu().numpy())

            # pred_kp = A.Compose([
            #                        A.Resize(height=prefetch_img.shape[0], width=prefetch_img.shape[1]),
            #                     ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))(image=img, keypoints=pred_kp)['keypoints']
            # renormalize
            img = unnormalize_img(img).detach().cpu().numpy()
            pred_kp = pred_kp[0].detach().cpu().numpy()
            label = torch.argmax(pred_cat, dim=1).item()
            draw_kp(img, pred_kp, f'{self.path_to_save_imgs}/tested_image_№_{idx}_pr.jpg', RGB=True, normalized=True, label=label)

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
        # self.val()
        self.visual_test()

    @staticmethod
    def put_on_device(items, device):
        for i in range(len(items)):
            items[i] = items[i].to(device)
        return items
