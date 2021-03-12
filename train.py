import argparse

import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2 as cv
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from icecream import ic

from models import mobilenetv3_large
from utils import collate, AverageMeter
from dataloaders import Objectron
from metrics import compute_average_distance

TRAIN_STEP, VAL_STEP = 0,0
WRITER = SummaryWriter(args.log_path)

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='3D-object-detection training')
    parser.add_argument('--root', type=str, default='./data', help='path to root folder')
    parser.add_argument('--batch', type=int, default=128, help='specify which batch size to use')
    parser.add_argument('--epochs', type=int, default=50, help='specify how much epochs to use')
    parser.add_argument('--save_checkpoint', type=bool, default=True,
                        help='whether or not to save your model')
    parser.add_argument('--log-path', type=str, default='log_out', help='specify path to log folder')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'],
                        help='if you want to train model on cpu, pass "cpu" param')
    args = parser.parse_args()

    # preprocessing data
    normalize = A.Normalize(**dict(mean=[0.5931, 0.4690, 0.4229],
                    std=[0.2471, 0.2214, 0.2157]))
    train_transform = A.Compose([
                            A.Resize(224, 128),
                            A.HorizontalFlip(p=0.5),
                            A.augmentations.transforms.ISONoise(color_shift=(0.15,0.35),
                                                                intensity=(0.2, 0.5), p=0.2),
                            normalize
                            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    test_transform = A.Compose([
                            A.Resize(224, 128),
                            normalize
                            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    net = mobilenetv3_large(pretrained=True)
    net.to(device='cuda:0')

    train_dataset = Objectron(args.root, mode='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate)
    val_dataset = Objectron(args.root, mode='val', transform=test_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate)

    loss = nn.SmoothL1Loss(reduction='mean', beta=1.0)
    optimizer = torch.optim.Adadelta(net.parameters(), lr=1e-2, weight_decay=5e-4)


    for epoch in range(args.epochs):

        train(net,
              train_loader,
              optimizer,
              loss,
              epoch,
              args.epochs,
              args.log_path)

        val(net,
            val_loader,
            optimizer,
            epoch,
            args.epochs,
            args.log_path)

def train(model,
          train_loader,
          optimizer,
          criterion,
          epoch,
          max_epoch,
          log_path,
          device='cuda:0',
          save_chkpt=True):
    ''' procedure launching all main functions of training,
        validation and testing pipelines'''

    losses = AverageMeter()
    ADD_metr = AverageMeter()
    SADD_metr = AverageMeter()

    # switch to train mode and train one epoch
    model.train()
    loop = tqdm(train_loader, total=len(train_loader), leave=False)
    for imgs, gt_kp in loop:
        if (imgs is None or gt_kp is None):
            continue
        # put image and keypoints on the appropriate devuce
        imgs = imgs.to(device)
        gt_kp = gt_kp.to(device)
        # compute output and loss
        pred_kp = model(imgs)
        loss = criterion(pred_kp, gt_kp)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure accuracy
        ADD, SADD = compute_average_distance(pred_kp, gt_kp)
        # record loss
        losses.update(loss.item(), imgs.size(0))
        ADD_metr.update(ADD.item(), imgs.size(0))
        SADD_metr.update(SADD.item(), imgs.size(0))
        # write to writer for tensorboard
        WRITER.add_scalar('Train/loss', loss, global_step=TRAIN_STEP)
        WRITER.add_scalar('Train/ADD', ADD_metr.avg, global_step=TRAIN_STEP)
        WRITER.add_scalar('Train/SADD', SADD_metr.avg, global_step=TRAIN_STEP)
        TRAIN_STEP += 1
        # update progress bar
        loop.set_description(f'Epoch [{epoch}/{max_epoch}]')
        loop.set_postfix(loss=loss.item(), avr_loss = losses.avg,
                            ADD=ADD.item(), avr_ADD=ADD_metr.avg, SADD=SADD.item(),
                            avr_SADD=SADD_metr.avg,
                            lr=optimizer.param_groups[0]['lr'])

    if save_chkpt:
        checkpoint = {'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(), 'epoch': epoch}
        print('==> saving checkpoint')
        torch.save(checkpoint, f'{log_path}/chkpt.pth')

    print(f'train: epoch: {epoch} ADD: {ADD_metr.avg}, SADD: {SADD_metr.avg}, loss: {losses.avg}')

def val(model,
        val_loader,
        optimizer,
        epoch,
        max_epoch,
        log_path,
        device='cuda:0'):

    ADD_metr = AverageMeter()
    SADD_metr = AverageMeter()

    # switch to train mode and train one epoch
    model.eval()
    loop = tqdm(val_loader, total=len(val_loader), leave=False)
    for imgs, gt_kp in loop:
        if (imgs is None or gt_kp is None):
            continue
        # put image and keypoints on the appropriate devuce
        imgs = imgs.to(device)
        gt_kp = gt_kp.to(device)
        # compute output and loss
        pred_kp = model(imgs)
        # measure accuracy
        ADD, SADD = compute_average_distance(pred_kp, gt_kp)
        # record loss
        ADD_metr.update(ADD.item(), imgs.size(0))
        SADD_metr.update(SADD.item(), imgs.size(0))
        # write to writer for tensorboard
        WRITER.add_scalar('Val/ADD', ADD_metr.avg, global_step=VAL_STEP)
        WRITER.add_scalar('Val/SADD', SADD_metr.avg, global_step=VAL_STEP)
        VAL_STEP += 1
        # update progress bar
        loop.set_description(f'Val Epoch [{epoch}/{max_epoch}]')
        loop.set_postfix(ADD=ADD.item(), avr_ADD=ADD_metr.avg, SADD=SADD.item(),
                            avr_SADD=SADD_metr.avg,)

    print(f'val: epoch: {epoch} ADD: {ADD_metr.avg}, SADD: {SADD_metr.avg}')

def filter_kp(kp):
    kp_copy = kp.copy().view(1,-1).numpy().to_list()
    kp_copy.sort()
    for i in range(1, len(kp_copy)):
        if np.isclose(kp_copy[i-1], kp_copy[i]):
            return True
    return False

if __name__ == "__main__":
    main()
