from __future__ import division, print_function

import sys

sys.path.append("core")

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger as loguru_logger

import core.datasets as datasets
import evaluate_FlowFormer_tile as evaluate
from core.FlowFormer import build_flowformer
from core.loss import amodal_sequence_loss, sequence_loss
from core.optimizer import fetch_optimizer
from core.utils.logger import Logger
from core.utils.misc import process_cfg

try:
    from torch.cuda.amp import GradScaler
except ImportError:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            optimizer.step()

        def update(self):
            pass


def on_load_checkpoint(state_dict, model_state_dict):
    is_changed = False
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print(
                    f"Skip loading parameter: {k}, "
                    f"required shape: {model_state_dict[k].shape}, "
                    f"loaded shape: {state_dict[k].shape}"
                )
                state_dict[k] = model_state_dict[k]
                is_changed = True
    return state_dict


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(cfg):
    loss_func = sequence_loss
    if cfg.use_smoothl1:
        raise ValueError("Smooth L1 loss not supported")

    model = nn.DataParallel(build_flowformer(cfg))
    loguru_logger.info("Parameter Count: %d" % count_parameters(model))

    if cfg.restore_ckpt is not None:
        loguru_logger.info(f"Loading ckpt from {cfg.restore_ckpt}")
        try:
            model.load_state_dict(torch.load(cfg.restore_ckpt), strict=True)
        except RuntimeError as e:
            loguru_logger.warning(f"Failed to load state dict in strict mode: {e}")
            loguru_logger.warning(f"Falling back to strict=False")
            model.load_state_dict(torch.load(cfg.restore_ckpt), strict=False)

    model.cuda()
    model.train()

    # if args.stage != 'chairs':
    #    model.module.freeze_bn()

    train_loader = datasets.fetch_dataloader(cfg)
    optimizer, scheduler = fetch_optimizer(model, cfg.trainer)

    total_steps = 0
    scaler = GradScaler(enabled=cfg.mixed_precision)
    logger = Logger(model, scheduler, cfg)

    # add_noise = True

    should_keep_training = True
    while should_keep_training:
        for i_batch, data_blob in enumerate(train_loader):
            t_start = time.time()

            optimizer.zero_grad()

            if cfg.amodal:
                image1, image2, flow, valid, sseg = data_blob

                image1, image2 = image1.cuda(), image2.cuda()
                flow = [flow[0].cuda(), flow[1].cuda()] + [
                    (f.cuda(), m.cuda()) for f, m in flow[2:]
                ]
                valid = [v.cuda() for v in valid]
                sseg = [s.cuda() for s in sseg]
            else:
                image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if cfg.add_noise:
                # print("[Adding noise]")
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(
                    0.0, 255.0
                )
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(
                    0.0, 255.0
                )

            output = {}
            flow_predictions = model(image1, image2, output)

            if cfg.amodal:
                loss, metrics = amodal_sequence_loss(flow_predictions, flow, valid, sseg, cfg)
            else:
                loss, metrics = sequence_loss(flow_predictions, flow, valid, cfg)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.clip)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()

            t_delta = time.time() - t_start
            loguru_logger.info(f"step: {total_steps}, time: {t_delta:.2}s")

            metrics.update(output)
            logger.push(metrics)

            if total_steps % cfg.val_freq == cfg.val_freq - 1:
                path = "%s/%d_%s.pth" % (cfg.log_dir, total_steps + 1, cfg.name)
                torch.save(model.state_dict(), path)

                results = {}
                for val_dataset in cfg.validation:
                    if val_dataset == "chairs":
                        results.update(evaluate.validate_chairs(model.module))
                    elif val_dataset == "sintel":
                        results.update(evaluate.validate_sintel(model.module))
                    elif val_dataset == "kitti":
                        results.update(evaluate.validate_kitti(model.module))
                    elif val_dataset == "amsynthdrive":
                        results.update(
                            evaluate.validate_amsynthdrive(
                                model.module, batch_size=cfg.batch_size
                            )
                        )
                    else:
                        raise ValueError(f"unknown validation dataset {val_dataset}")

                logger.write_dict(results)

                model.train()

            total_steps += 1

            if total_steps > cfg.trainer.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = cfg.log_dir + "/final"
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="flowformer", help="name your experiment")
    parser.add_argument("--stage", help="determines which dataset to use for training")
    parser.add_argument("--validation", type=str, nargs="+")

    parser.add_argument(
        "--mixed_precision", action="store_true", help="use mixed precision"
    )

    args = parser.parse_args()

    if args.stage == "chairs":
        from configs.default import get_cfg
    elif args.stage == "things":
        from configs.things import get_cfg
    elif args.stage == "sintel":
        from configs.sintel import get_cfg
    elif args.stage == "kitti":
        from configs.kitti import get_cfg
    elif args.stage == "amsynthdrive":
        from configs.amsynthdrive import get_cfg

    cfg = get_cfg()
    cfg.update(vars(args))
    process_cfg(cfg)
    loguru_logger.add(str(Path(cfg.log_dir) / "log.txt"), encoding="utf8")
    loguru_logger.info(cfg)

    torch.manual_seed(1234)
    np.random.seed(1234)

    if not os.path.isdir("checkpoints"):
        os.mkdir("checkpoints")

    train(cfg)
