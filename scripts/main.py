import argparse
import sys
import os.path as osp
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from torchdet3d.builders import (build_loader, build_model, build_loss,
                                    build_optimizer, build_scheduler)
from torchdet3d.evaluation import Evaluator
from torchdet3d.losses import LossManager
from torchdet3d.trainer import Trainer
from torchdet3d.utils import read_py_config, Logger, set_random_seed

def reset_config(cfg, args):
    if args.root:
        cfg['data']['root'] = args.root

def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='3D-object-detection training')
    parser.add_argument('--root', type=str, default='', help='path to root folder')
    parser.add_argument('--config', type=str, default='./configs/default_config.py', help='path to config')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'],
                        help='choose device to train on')
    parser.add_argument('--save_checkpoint', type=bool, default=True,
                        help='whether or not to save your model')
    args = parser.parse_args()
    cfg = read_py_config(args.config)
    reset_config(cfg, args)
    # translate output to log file
    log_name = 'train.log' if cfg.regime.type == 'training' else 'test.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.output_dir, log_name))
    set_random_seed(cfg.utils.random_seeds)

    # init main components
    net = build_model(cfg)
    # TO DO resume from chkpt opportunity
    net.to(args.device)
    if (torch.cuda.is_available() and args.device == 'cuda' and cfg.data_parallel.use_parallel):
        net = torch.nn.DataParallel(net, **cfg.data_parallel.parallel_params)

    criterions = build_loss(cfg)
    optimizer = build_optimizer(cfg, net)
    scheduler = build_scheduler(cfg, optimizer)
    loss_manager = LossManager(criterions, cfg.loss.coeffs, cfg.loss.alwa)
    train_loader, val_loader, test_loader = build_loader(cfg)
    writer = SummaryWriter(cfg.output_dir)

    trainer = Trainer(model=net,
                      train_loader=train_loader,
                      optimizer=optimizer,
                      loss_manager=loss_manager,
                      writer=writer,
                      max_epoch=cfg.data.max_epochs,
                      log_path=cfg.output_dir,
                      device=args.device,
                      save_chkpt=args.save_checkpoint,
                      debug=cfg.utils.debug_mode,
                      debug_steps=cfg.utils.debug_steps,
                      save_freq=cfg.utils.save_freq,
                      print_freq=cfg.utils.print_freq)

    evaluator = Evaluator(model=net,
                          val_loader=val_loader,
                          test_loader=test_loader,
                          cfg=cfg,
                          writer=writer,
                          device=args.device,
                          max_epoch=cfg.data.max_epochs,
                          path_to_save_imgs=cfg.output_dir,
                          debug=cfg.utils.debug_mode,
                          debug_steps=cfg.utils.debug_steps)
    # main loop
    if cfg.regime.type == "evaluation":
        evaluator.run_eval_pipe(cfg.regime.vis_only)
    else:
        assert cfg.regime.type == "training"
        for epoch in range(cfg.data.max_epochs):
            trainer.train(epoch)
            if scheduler is not None:
                scheduler.step()
            evaluator.val(epoch)
        evaluator.visual_test()


if __name__ == "__main__":
    main()
