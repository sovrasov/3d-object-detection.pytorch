import argparse
import sys
import os.path as osp
import time

import albumentations as A
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from icecream import ic

from models import mobilenetv3_large
from builders import build_loader, build_loss, build_optimizer
from scripts import (Tester, Trainer, compute_average_distance,
                    compute_average_distance, read_py_config, Logger)


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
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net, **cfg.data_parallel.parallel_params)

    criterions = build_loss(cfg)
    optimizer = build_optimizer(cfg, net)
    train_loader, val_loader, test_loader = build_loader(cfg)

    writer = SummaryWriter(cfg.output_dir)
    trainer = Trainer(net,
                      train_loader,
                      val_loader,
                      optimizer,
                      criterions,
                      writer,
                      cfg.data.max_epochs,
                      cfg.output_dir,
                      args.device,
                      args.save_checkpoint,
                      cfg.debug_mode)
    # main loop
    for epoch in range(cfg.data.max_epochs):
        trainer.train(epoch)
        trainer.val(epoch)
    # test afterward
    tester = Tester(trainer.model,
                          test_loader,
                          cfg,
                          path_to_save_imgs=cfg.output_dir)
    tester.run_eval()

if __name__ == "__main__":
    main()
