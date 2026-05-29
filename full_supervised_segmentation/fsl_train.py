# import sys
# sys.path.append("..")
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import argparse
from loguru import logger
from data import build_train_loader
from trainer import Trainer
from utils.helpers import seed_torch, load_checkpoint
from losses.losses import *
from datetime import datetime
import wandb
from configs.config import get_config
from models import build_model
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
import os
import torch.backends.cudnn as cudnn
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist


def parse_option():
    parser = argparse.ArgumentParser("DIAS_training")
    parser.add_argument("--cfg", type=str, metavar="FILE", help="path to config file")
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("-wm", "--wandb_mode", default="offline")
    parser.add_argument("-mt", "--model_type", default="UNet")
    parser.add_argument(
        "-bs", "--batch-size", type=int, default=64, help="batch size for single GPU"
    )
    parser.add_argument(
        "-ed",
        "--enable_distributed",
        help="training without DDP",
        required=False,
        action="store_true",
    )
    parser.add_argument("-ws", "--world_size", type=int, help="process number for DDP")
    parser.add_argument(
        "--resume", type=str, default="", help="Path to the checkpoint to resume from"
    )
    args = parser.parse_args()
    config = get_config(args)

    return args, config


def main(config):
    if config.DIS:
        mp.spawn(
            main_worker,  # type: ignore
            args=(config,),
            nprocs=config.WORLD_SIZE,
        )
    else:
        main_worker(0, config)


def main_worker(local_rank, config):
    if local_rank == 0:
        config.defrost()
        if config.TRAIN.RESUME_PATH:
            config.EXPERIMENT_ID = os.path.basename(
                os.path.normpath(config.TRAIN.RESUME_PATH)
            )
        else:
            # Otherwise, make a new timestamp
            config.EXPERIMENT_ID = (
                f"{config.WANDB.TAG}_{datetime.now().strftime('%y%m%d_%H%M%S')}"
            )
        config.freeze()
        wandb.init(
            project=config.WANDB.PROJECT,
            name=config.EXPERIMENT_ID,
            config=config,
            mode=config.WANDB.MODE,
        )
    np.set_printoptions(formatter={"float": "{: 0.4f}".format}, suppress=True)
    torch.cuda.set_device(local_rank)
    if config.DIS:
        dist.init_process_group(
            "nccl", init_method="env://", rank=local_rank, world_size=config.WORLD_SIZE
        )
    seed = config.SEED + local_rank
    seed_torch(seed)
    cudnn.benchmark = True

    train_loader, val_loader = build_train_loader(config)
    model, is_2d = build_model(config)

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).cuda()
    if config.DIS:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )
    logger.info(f"\n{model}\n")
    # loss = CE_DiceLoss()
    # loss = SoftDiceLoss()
    loss = DC_and_CE_loss({}, {})
    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))

    # saving feature
    start_epoch = 1
    mnt_best = None

    if config.TRAIN.RESUME_PATH:
        logger.info(f"Resuming training from {config.TRAIN.RESUME_PATH} ...")

        checkpoint = load_checkpoint(config.TRAIN.RESUME_PATH, is_best=False)

        # restore states
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        # restore trackers
        start_epoch = checkpoint["epoch"] + 1
        if "monitor_best" in checkpoint:
            mnt_best = checkpoint["monitor_best"]

        logger.info(
            f"Successfully loaded checkpoint. Resuming from epoch {start_epoch}"
        )

    trainer = Trainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model.cuda(),
        is_2d=is_2d,
        loss=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        start_epoch=start_epoch,
        mnt_best=mnt_best,
    )
    trainer.train()


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "10000"
    _, config = parse_option()

    main(config)
