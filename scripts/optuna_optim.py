import os.path as osp
import sys
import argparse
import datetime
import time

import torch
import optuna
from optuna.trial import TrialState
from functools import partial

from torchdet3d.builders import (build_loader, build_model, build_loss,
                                    build_optimizer, build_scheduler)
from torchdet3d.evaluation import compute_average_distance, compute_accuracy
from torchdet3d.losses import LossManager
from torchdet3d.utils import AverageMeter, read_py_config, Logger, set_random_seed

def put_on_device(items, device):
    for i, item in enumerate(items):
        items[i] = item.to(device)
    return items

def objective(cfg, args, trial):
    # build the model.
    model = build_model(cfg)
    model.to(args.device)
    if (torch.cuda.is_available() and args.device == 'cuda' and cfg.data_parallel.use_parallel):
        model = torch.nn.DataParallel(model, **cfg.data_parallel.parallel_params)
    # Generate the trials.
    eps = trial.suggest_float("eps", 0.01, 3)
    w = trial.suggest_float("w", 0.01, 10)

    cfg['loss']['w'] = w
    cfg['loss']['eps'] = eps
    print(f"\nnext trial with [w: {w}, epsilon: {eps}]")

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # Get the dataset.
    train_loader, val_loader, _ = build_loader(cfg)
    criterions = build_loss(cfg)
    loss_manager = LossManager(criterions, cfg.loss.coeffs, cfg.loss.alwa)

    # Training of the model.
    max_iter_train = len(train_loader)
    max_iter_val = len(val_loader)
    num_iters_train = min(max_iter_train, int(args.n_training_iterations * max_iter_train))
    num_iters_val = min(max_iter_val, int(args.n_validate_iterations * max_iter_val))

    for epoch in range(args.epochs):
        losses = AverageMeter()
        batch_time = AverageMeter()
        end = time.time()
        model.train()

        for batch_idx, (imgs, gt_kp, gt_cats) in enumerate(train_loader):
            # compute eta
            batch_time.update(time.time() - end)
            nb_this_epoch = num_iters_train - (batch_idx + 1)
            nb_future_epochs = (args.epochs - (epoch + 1)) * num_iters_train
            eta_seconds = batch_time.avg * (nb_this_epoch+nb_future_epochs)
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            # Limiting training data for faster epochs.
            if batch_idx > num_iters_train:
                break

            imgs, gt_kp, gt_cats = put_on_device([imgs, gt_kp, gt_cats], args.device)

            pred_kp, pred_cats = model(imgs, gt_cats)
            loss = loss_manager.parse_losses(pred_kp, gt_kp, pred_cats, gt_cats, batch_idx)
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record loss
            losses.update(loss.item(), imgs.size(0))

            if (((batch_idx + 1) % cfg.utils.print_freq == 0) or (batch_idx == num_iters_train-1)):
                print(
                        'epoch: [{0}/{1}][{2}/{3}]\t'
                        'time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'eta {eta}\t'
                        'loss {losses.avg:.5f}\t'
                        'lr {lr:.6f}'.format(
                            epoch,
                            args.epochs,
                            batch_idx,
                            num_iters_train,
                            batch_time=batch_time,
                            eta=eta_str,
                            losses=losses,
                            lr=optimizer.param_groups[0]['lr'])
                        )
            end = time.time()

        if scheduler is not None:
            scheduler.step()

        # Validation of the model.
        model.eval()
        ADD_meter = AverageMeter()
        SADD_meter = AverageMeter()
        ACC_meter = AverageMeter()

        with torch.no_grad():
            for batch_idx, (imgs, gt_kp, gt_cats) in enumerate(val_loader):
                if batch_idx > num_iters_val:
                    break
                # put image and keypoints on the appropriate device
                imgs, gt_kp, gt_cats = put_on_device([imgs, gt_kp, gt_cats], args.device)
                # compute output and loss
                pred_kp, pred_cats = model(imgs, gt_cats)
                # measure metrics
                ADD, SADD = compute_average_distance(pred_kp, gt_kp)
                acc = compute_accuracy(pred_cats, gt_cats)
                ADD_meter.update(ADD, imgs.size(0))
                SADD_meter.update(SADD, imgs.size(0))
                ACC_meter.update(acc, imgs.size(0))

            print("\nComputed val metrics:\n"
            f"ADD ---> {ADD_meter.avg}\n"
            f"SADD ---> {SADD_meter.avg}\n"
            f"classification accuracy ---> {ACC_meter.avg}")

        obj = SADD_meter.avg

        trial.report(obj, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return obj

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='3D-object-detection training')
    parser.add_argument('--root', type=str, default='', help='path to root folder')
    parser.add_argument('--disable_store_log', action='store_false')
    parser.add_argument('--config', type=str, default='./configs/default_config.py', help='path to config')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'],
                        help='choose device to train on')
    parser.add_argument('-e', '--epochs', type=int, default=150,
                        help='choose epochs to train with')
    parser.add_argument('--n_training_iterations', type=float, default=.5,
                        help='choose percentage of samples in each epoch to train with')
    parser.add_argument('--n_validate_iterations', type=float, default=.5,
                            help='choose percentage of samples in each epoch to validate with')

    args = parser.parse_args()
    cfg = read_py_config(args.config)

    # translate output to log file
    if args.disable_store_log:
        log_name = 'optuna.log'
        log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
        sys.stdout = Logger(osp.join(cfg.output_dir, log_name))
    set_random_seed(cfg.utils.random_seeds)

    study = optuna.create_study(study_name='regression task', direction="minimize")
    objective_partial = partial(objective, cfg, args)
    try:
        study.optimize(objective_partial, n_trials=100, timeout=None)
    finally:
        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

if __name__ == "__main__":
    main()
