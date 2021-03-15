import torch
from metrics import compute_average_distance
from utils import AverageMeter
from tqdm import tqdm
from icecream import ic
from dataclasses import dataclass

@dataclass
class Trainer:
    model: object
    train_loader: object
    val_loader: object
    optimizer: object
    criterion: object
    writer: object
    max_epoch : float
    log_path : str
    device : str ='cuda:0'
    save_chkpt: bool = True
    debug: bool = False
    train_step: int = 0
    val_step: int = 0
    debug_mode: bool = False

    def train(self, epoch):
        ''' procedure launching main training'''

        losses = AverageMeter()
        ADD_metr = AverageMeter()
        SADD_metr = AverageMeter()

        # switch to train mode and train one epoch
        self.model.train()
        loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
        for it, (imgs, gt_kp, cats) in loop:
            if any([obj is None for obj in (imgs, gt_kp, cats)]):
                continue
            # put image and keypoints on the appropriate devuce
            imgs = imgs.to(self.device)
            gt_kp = gt_kp.to(self.device)
            # compute output and loss
            pred_kp = self.model(imgs)
            loss = self.criterion(pred_kp.float(), gt_kp.float())
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # measure metrics
            ADD, SADD = compute_average_distance(pred_kp, gt_kp)
            # record loss
            losses.update(loss.item(), imgs.size(0))
            ADD_metr.update(ADD.item(), imgs.size(0))
            SADD_metr.update(SADD.item(), imgs.size(0))
            # write to writer for tensorboard
            self.writer.add_scalar('Train/loss', loss, global_step=self.train_step)
            self.writer.add_scalar('Train/ADD', ADD_metr.avg, global_step=self.train_step)
            self.writer.add_scalar('Train/SADD', SADD_metr.avg, global_step=self.train_step)
            self.train_step += 1
            # update progress bar
            loop.set_description(f'Epoch [{epoch}/{self.max_epoch}]')
            loop.set_postfix(loss=loss.item(), avr_loss = losses.avg,
                                ADD=ADD.item(), avr_ADD=ADD_metr.avg, SADD=SADD.item(),
                                avr_SADD=SADD_metr.avg,
                                lr=self.optimizer.param_groups[0]['lr'])

            if self.debug and it == 10:
                return

        if self.save_chkpt:
            checkpoint = {'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'epoch': epoch}

            print('==> saving checkpoint')
            torch.save(checkpoint, f'{self.log_path}/chkpt.pth')

        print(f'train: epoch: {epoch} ADD: {ADD_metr.avg}, SADD: {SADD_metr.avg}, loss: {losses.avg}')

    def val(self, epoch):

        ''' procedure launching main validation'''

        ADD_metr = AverageMeter()
        SADD_metr = AverageMeter()

        # switch to train mode and train one epoch
        self.model.eval()
        loop = tqdm(enumerate(self.val_loader), total=len(self.val_loader), leave=False)
        for it, (imgs, gt_kp, cats) in loop:
            if any([obj is None for obj in (imgs, gt_kp, cats)]):
                continue
            # put image and keypoints on the appropriate devuce
            imgs = imgs.to(self.device)
            gt_kp = gt_kp.to(self.device)
            # compute output and loss
            pred_kp = self.model(imgs)
            # measure metrics
            ADD, SADD = compute_average_distance(pred_kp, gt_kp)
            # record loss
            ADD_metr.update(ADD.item(), imgs.size(0))
            SADD_metr.update(SADD.item(), imgs.size(0))
            # write to writer for tensorboard
            self.writer.add_scalar('Val/ADD', ADD_metr.avg, global_step=self.val_step)
            self.writer.add_scalar('Val/SADD', SADD_metr.avg, global_step=self.val_step)
            self.val_step += 1
            # update progress bar
            loop.set_description(f'Val Epoch [{epoch}/{self.max_epoch}]')
            loop.set_postfix(ADD=ADD.item(), avr_ADD=ADD_metr.avg, SADD=SADD.item(),
                                avr_SADD=SADD_metr.avg,)

            if self.debug and it == 10:
                return

        print(f'val: epoch: {epoch} ADD: {ADD_metr.avg}, SADD: {SADD_metr.avg}')
