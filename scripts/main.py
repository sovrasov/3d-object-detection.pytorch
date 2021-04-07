import argparse
import sys
import os.path as osp
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from torchdet3d.models import mobilenetv3_large
from torchdet3d.builders import build_loader, build_loss, build_optimizer, build_scheduler
from torchdet3d.evaluation import Evaluator
from torchdet3d.trainer import Trainer
from torchdet3d.utils import read_py_config, Logger


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='3D-object-detection training')
    parser.add_argument('--root', type=str, default='./data', help='path to root folder')
    parser.add_argument('--config', type=str, default='./configs/default_config.py', help='path to config')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda','cpu'],
                        help='choose device to train on')
    parser.add_argument('--save_checkpoint', type=bool, default=True,
                        help='whether or not to save your model')
    args = parser.parse_args()
    cfg = read_py_config(args.config)

    # translate output to log file
    log_name = 'train.log'
    log_name += time.strftime('-%Y-%m-%d-%H-%M-%S')
    sys.stdout = Logger(osp.join(cfg.output_dir, log_name))

    # init main components
    net = mobilenetv3_large(pretrained=True)
    net.to(args.device)
    if (torch.cuda.is_available() and args.device == 'cuda'):
        net = torch.nn.DataParallel(net, **cfg.data_parallel.parallel_params)

    criterions = build_loss(cfg)
    optimizer = build_optimizer(cfg, net)
    scheduler = build_scheduler(cfg, optimizer)
    train_loader, val_loader, test_loader = build_loader(cfg)
    writer = SummaryWriter(cfg.output_dir)

    trainer = Trainer(model=net,
                      train_loader=train_loader,
                      optimizer=optimizer,
                      criterions=criterions,
                      writer=writer,
                      max_epoch=cfg.data.max_epochs,
                      log_path=cfg.output_dir,
                      device=args.device,
                      save_chkpt=args.save_checkpoint,
                      debug=cfg.debug_mode)

    evaluator = Evaluator(model=net,
                          val_loader=val_loader,
                          test_loader=test_loader,
                          checkpoint_path=cfg.model.load_weights,
                          cfg=cfg,
                          writer=writer,
                          device=args.device,
                          max_epoch=cfg.data.max_epochs,
                          path_to_save_imgs=cfg.output_dir,
                          debug=cfg.debug_mode)
    # main loop
    if cfg.regime == "evaluation":
        evaluator.run_eval_pipe()
    else:
        assert cfg.regime == "training"
        for epoch in range(cfg.data.max_epochs):
            trainer.train(epoch)
            scheduler.step()
            evaluator.val(epoch)
        evaluator.visual_test()


if __name__ == "__main__":
    main()
