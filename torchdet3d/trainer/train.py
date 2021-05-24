import time
import datetime

from tqdm import tqdm
from dataclasses import dataclass

from torchdet3d.evaluation import compute_average_distance, compute_accuracy
from torchdet3d.utils import AverageMeter, save_snap, put_on_device

@dataclass(init=True)
class Trainer:
    model: object
    train_loader: object
    optimizer: object
    scheduler: object
    loss_manager: object
    writer: object
    max_epoch : int
    log_path : str
    device : str ='cuda'
    save_chkpt: bool = True
    debug: bool = False
    debug_steps: int = 30
    save_freq: int = 10
    print_freq: int = 10
    train_step: int = 0

    def train(self, epoch, is_last_epoch):
        ''' procedure launching main training'''

        losses = AverageMeter()
        ADD_meter = AverageMeter()
        SADD_meter = AverageMeter()
        ACC_meter = AverageMeter()
        batch_time = AverageMeter()

        # switch to train mode and train one epoch
        self.model.train()
        self.num_iters = len(self.train_loader)
        start = time.time()
        loop = tqdm(enumerate(self.train_loader), total=self.num_iters, leave=False)
        for it, (imgs, gt_kp, gt_cats) in loop:
            # put image and keypoints on the appropriate device
            imgs, gt_kp, gt_cats = put_on_device([imgs, gt_kp, gt_cats], self.device)
            # compute output and loss
            pred_kp, pred_cats = self.model(imgs, gt_cats)
            # get parsed loss
            loss = self.loss_manager.parse_losses(pred_kp, gt_kp, pred_cats, gt_cats, it)
            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # measure metrics
            ADD, SADD = compute_average_distance(pred_kp, gt_kp)
            acc = compute_accuracy(pred_cats, gt_cats)
            # record loss
            losses.update(loss.item(), imgs.size(0))
            ADD_meter.update(ADD, imgs.size(0))
            SADD_meter.update(SADD, imgs.size(0))
            ACC_meter.update(acc, imgs.size(0))
            # write to writer for tensorboard
            self.writer.add_scalar('Train/loss', loss.item(), global_step=self.train_step)
            self.writer.add_scalar('Train/ADD', ADD_meter.avg, global_step=self.train_step)
            self.writer.add_scalar('Train/SADD', SADD_meter.avg, global_step=self.train_step)
            self.writer.add_scalar('Train/ACC', ACC_meter.avg, global_step=self.train_step)
            self.train_step += 1
            # update progress bar
            loop.set_description(f'Epoch [{epoch}/{self.max_epoch}]')
            loop.set_postfix(loss=loss.item(),
                             avr_loss = losses.avg,
                             ADD=ADD, avr_ADD=ADD_meter.avg,
                             SADD=SADD,
                             avr_SADD=SADD_meter.avg,
                             acc=acc,
                             acc_avg = ACC_meter.avg,
                             lr=self.optimizer.param_groups[0]['lr'])
            # compute eta
            batch_time.update(time.time() - start)
            nb_this_epoch = self.num_iters - (it + 1)
            nb_future_epochs = (self.max_epoch - (epoch + 1)) * self.num_iters
            eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            if ((it % self.print_freq == 0) or (it == self.num_iters-1)):
                print(
                        'epoch: [{0}/{1}][{2}/{3}]\t'
                        'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'eta {eta}\t'
                        'cls acc {accuracy.val:.3f} ({accuracy.avg:.3f})\t'
                        'ADD {ADD.val:.4f} ({ADD.avg:.4f})\t'
                        'SADD {SADD.val:.4f} ({SADD.avg:.4f})\t'
                        'loss {losses.avg:.5f}\t'
                        'lr {lr:.6f}'.format(
                            epoch,
                            self.max_epoch,
                            it,
                            self.num_iters,
                            batch_time=batch_time,
                            eta=eta_str,
                            accuracy=ACC_meter,
                            ADD=ADD_meter,
                            SADD=SADD_meter,
                            losses=losses,
                            lr=self.optimizer.param_groups[0]['lr'])
                        )

            start = time.time()
            if (self.debug and it == self.debug_steps):
                break

        if self.save_chkpt and (epoch % self.save_freq == 0 or is_last_epoch) and not self.debug:
            save_snap(self.model, self.optimizer, self.scheduler, epoch, self.log_path)
        # do scheduler step
        if self.scheduler is not None:
            self.scheduler.step()
