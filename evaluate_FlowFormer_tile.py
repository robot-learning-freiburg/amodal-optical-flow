#!/usr/bin/env python
import sys

sys.path.append("core")

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import datasets
import epe
import flow_mb
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger as log
from PIL import Image
from tqdm import tqdm
from utils import flow_viz, frame_utils, labels
from utils.utils import InputPadder

from configs.amsynthdrive import get_cfg as get_amsynthdrive_cfg
from configs.submissions import get_cfg as get_submission_cfg
from core.FlowFormer import build_flowformer

TRAIN_SIZE = [432, 960]


def write_img(path: Path, img):
    img = (img + 1.0) / 2.0
    img = img.numpy()[:, :, ::-1]

    path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(path), img)


def write_mask_vis(path: Path, mask):
    rgba = torch.stack((mask, mask, mask, torch.ones_like(mask)), dim=-1)
    bgra = rgba_to_bgra(rgba)

    path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(path), bgra * 255)


def write_flow_vis(path: Path, uv, mask=None):
    rgba = flow_mb.flow_to_rgba(uv, mask)
    bgra = rgba_to_bgra(rgba)

    path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(path), bgra * 255)


def write_flow_png(path, uv, mask=None):
    MAX_FLOW = 695
    MIN_FLOW = -1687

    path = Path(path)

    if mask is None:
        mask = torch.ones(uv.shape[:-1])

    uv = ((uv - MIN_FLOW) / (MAX_FLOW - MIN_FLOW)) * np.iinfo(np.uint16).max
    mask = mask * (2**16 - 1)
    # mask = mask > 0.5
    uv = np.concatenate([uv, mask[..., None]], axis=-1).astype(np.uint16)

    path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(path), uv[..., ::-1])


def write_err_vis(path: Path, uv, target, mask=None):
    rgba = epe.end_point_error_abs(uv, target, mask)
    bgra = rgba_to_bgra(rgba)

    path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(path), bgra * 255)


def rgba_to_bgra(rgba):
    bgra = np.zeros_like(rgba)

    bgra[:, :, 0] = rgba[:, :, 2]
    bgra[:, :, 1] = rgba[:, :, 1]
    bgra[:, :, 2] = rgba[:, :, 0]
    bgra[:, :, 3] = rgba[:, :, 3]

    return bgra


class InputPadder:
    """Pads images such that dimensions are divisible by 8"""

    def __init__(self, dims, mode="sintel"):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == "sintel":
            self._pad = [
                pad_wd // 2,
                pad_wd - pad_wd // 2,
                pad_ht // 2,
                pad_ht - pad_ht // 2,
            ]
        elif mode == "kitti432":
            self._pad = [0, 0, 0, 432 - self.ht]
        elif mode == "kitti400":
            self._pad = [0, 0, 0, 400 - self.ht]
        elif mode == "kitti376":
            self._pad = [0, 0, 0, 376 - self.ht]
        else:
            self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="constant", value=0.0) for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]


def compute_weight(
    hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype="gaussian"
):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h**2 + w**2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h : h + patch_size[0], w : w + patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(
            weights[:, idx : idx + 1, h : h + patch_size[0], w : w + patch_size[1]]
        )

    return patch_weights


@torch.no_grad()
def create_sintel_submission(
    model, output_path="sintel_submission_multi8_768", sigma=0.05
):
    """Create submission for the Sintel leaderboard"""
    print("no warm start")
    # print(f"output path: {output_path}")
    IMAGE_SIZE = [436, 1024]

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    model.eval()
    for dstype in ["final", "clean"]:
        test_dataset = datasets.MpiSintel_submission(
            split="test", aug_params=None, dstype=dstype, root="./dataset/Sintel/test"
        )
        epe_list = []
        for test_id in range(len(test_dataset)):
            if (test_id + 1) % 100 == 0:
                print(f"{test_id} / {len(test_dataset)}")
                # break
            image1, image2, (sequence, frame) = test_dataset[test_id]
            image1, image2 = image1[None].cuda(), image2[None].cuda()

            flows = 0
            flow_count = 0

            for idx, (h, w) in enumerate(hws):
                image1_tile = image1[:, :, h : h + TRAIN_SIZE[0], w : w + TRAIN_SIZE[1]]
                image2_tile = image2[:, :, h : h + TRAIN_SIZE[0], w : w + TRAIN_SIZE[1]]
                flow_pre, flow_low = model(image1_tile, image2_tile)

                padding = (
                    w,
                    IMAGE_SIZE[1] - w - TRAIN_SIZE[1],
                    h,
                    IMAGE_SIZE[0] - h - TRAIN_SIZE[0],
                    0,
                    0,
                )
                flows += F.pad(flow_pre * weights[idx], padding)
                flow_count += F.pad(weights[idx], padding)

            flow_pre = flows / flow_count
            flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, "frame%04d.flo" % (frame + 1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)


@torch.no_grad()
def create_kitti_submission(model, output_path="kitti_submission", sigma=0.05):
    """Create submission for the Sintel leaderboard"""

    IMAGE_SIZE = [432, 1242]

    print(f"output path: {output_path}")
    print(f"image size: {IMAGE_SIZE}")
    print(f"training size: {TRAIN_SIZE}")

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, (432, 1242), TRAIN_SIZE, sigma)
    model.eval()
    test_dataset = datasets.KITTI(split="testing", aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        new_shape = image1.shape[1:]
        if (
            new_shape[1] != IMAGE_SIZE[1]
        ):  # fix the height=432, adaptive ajust the width
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[0] = 432
            IMAGE_SIZE[1] = new_shape[1]
            hws = compute_grid_indices(IMAGE_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

        padder = InputPadder(
            image1.shape, mode="kitti432"
        )  # padding the image to height of 432
        image1, image2 = padder.pad(image1[None].cuda(), image2[None].cuda())

        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h : h + TRAIN_SIZE[0], w : w + TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h : h + TRAIN_SIZE[0], w : w + TRAIN_SIZE[1]]
            flow_pre, _ = model(image1_tile, image2_tile)

            padding = (
                w,
                IMAGE_SIZE[1] - w - TRAIN_SIZE[1],
                h,
                IMAGE_SIZE[0] - h - TRAIN_SIZE[0],
                0,
                0,
            )
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = padder.unpad(flow_pre[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

        flow_img = flow_viz.flow_to_image(flow)
        image = Image.fromarray(flow_img)
        if not os.path.exists(f"vis_kitti_3patch"):
            os.makedirs(f"vis_kitti_3patch/flow")
            os.makedirs(f"vis_kitti_3patch/image")

        image.save(f"vis_kitti_3patch/flow/{test_id}.png")
        imageio.imwrite(
            f"vis_kitti_3patch/image/{test_id}_0.png",
            image1[0].cpu().permute(1, 2, 0).numpy(),
        )
        imageio.imwrite(
            f"vis_kitti_3patch/image/{test_id}_1.png",
            image2[0].cpu().permute(1, 2, 0).numpy(),
        )


@torch.no_grad()
def validate_kitti(model, sigma=0.05):
    IMAGE_SIZE = [376, 1242]
    TRAIN_SIZE = [288, 960]

    hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)
    model.eval()
    val_dataset = datasets.KITTI(split="training")

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        new_shape = image1.shape[1:]
        if new_shape[1] != IMAGE_SIZE[1] or new_shape[0] != IMAGE_SIZE[0]:
            print(f"replace {IMAGE_SIZE} with {new_shape}")
            IMAGE_SIZE[0] = new_shape[0]
            IMAGE_SIZE[1] = new_shape[1]
            hws = compute_grid_indices(IMAGE_SIZE, TRAIN_SIZE)
            weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

        image1, image2 = image1[None].cuda(), image2[None].cuda()

        flows = 0
        flow_count = 0

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h : h + TRAIN_SIZE[0], w : w + TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h : h + TRAIN_SIZE[0], w : w + TRAIN_SIZE[1]]
            flow_pre, flow_low = model(image1_tile, image2_tile)

            padding = (
                w,
                IMAGE_SIZE[1] - w - TRAIN_SIZE[1],
                h,
                IMAGE_SIZE[0] - h - TRAIN_SIZE[0],
                0,
                0,
            )
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        flow = flow_pre[0].cpu()
        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {"kitti-epe": epe, "kitti-f1": f1}


@torch.no_grad()
def validate_sintel(model, sigma=0.05):
    """Peform validation using the Sintel (train) split"""

    IMAGE_SIZE = [436, 1024]

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    model.eval()
    results = {}
    for dstype in ["final", "clean"]:
        val_dataset = datasets.MpiSintel(split="training", dstype=dstype)

        epe_list = []

        for val_id in range(len(val_dataset)):
            if val_id % 50 == 0:
                print(val_id)

            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            flows = 0
            flow_count = 0

            for idx, (h, w) in enumerate(hws):
                image1_tile = image1[:, :, h : h + TRAIN_SIZE[0], w : w + TRAIN_SIZE[1]]
                image2_tile = image2[:, :, h : h + TRAIN_SIZE[0], w : w + TRAIN_SIZE[1]]

                flow_pre, _ = model(image1_tile, image2_tile, flow_init=None)

                padding = (
                    w,
                    IMAGE_SIZE[1] - w - TRAIN_SIZE[1],
                    h,
                    IMAGE_SIZE[0] - h - TRAIN_SIZE[0],
                    0,
                    0,
                )
                flows += F.pad(flow_pre * weights[idx], padding)
                flow_count += F.pad(weights[idx], padding)

            flow_pre = flows / flow_count
            flow_pre = flow_pre[0].cpu()

            epe = torch.sum((flow_pre - flow_gt) ** 2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        epe_all = np.concatenate(epe_list)
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print(
            "Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f"
            % (dstype, epe, px1, px3, px5)
        )
        results[f"{dstype}_tile"] = np.mean(epe_list)

    return results


def compute_amodal_layer_weights(n=8, k=3):
    def g(x, b=np.e, k=3, t=0.25):
        x = (x - k) * (-np.log(t) / np.log(b)) / (n - 1 - k)
        x = np.maximum(x, 0.0)
        return x

    def f(x, b=np.e, k=3, t=0.25):
        return 1 / (b ** g(x, b=b, k=k, t=t))

    x = np.linspace(0, n - 1, n)

    return f(x, k=k)


class Stats:
    weights = compute_amodal_layer_weights()

    def __init__(self, max_pxl_dist=5, num_bins=100, device=None):
        self.data = {}

        self.w = 1.0 - torch.linspace(1, 100, num_bins, device=device) / 100.0
        self.w_norm = torch.sum(self.w)
        self.delta = np.linspace(1 / (num_bins / max_pxl_dist), max_pxl_dist, num_bins)

    def push(self, camera, sequence, frame, layer, epe, gt_mask=None, pred_mask=None):
        B, H, W = epe.shape

        if gt_mask is None:
            gt_mask = torch.ones((B, H, W), device=epe.device)

        if pred_mask is not None:
            pred_mask = pred_mask.squeeze(1)

        n = torch.sum(gt_mask, dim=(1, 2))

        # basic flow end-point-error-based metrics
        epe_mean = (epe * gt_mask).sum(dim=(1, 2)) / n
        epe_1px = ((epe < 1).float() * gt_mask).sum(dim=(1, 2)) / n
        epe_3px = ((epe < 3).float() * gt_mask).sum(dim=(1, 2)) / n
        epe_5px = ((epe < 5).float() * gt_mask).sum(dim=(1, 2)) / n

        # WAUC flow metric
        bins = torch.zeros((B, len(self.delta)), device=epe.device)
        for i, delta in enumerate(self.delta):
            bins[:, i] = ((epe < delta).float() * gt_mask).sum(dim=(1, 2))

        wauc = torch.sum((self.w[None, :] * bins / self.w_norm) / n[:, None], dim=1)

        # mask metrics
        if pred_mask is not None:
            gt_mask = gt_mask > 0.5
            pred_mask = pred_mask > 0.5

            tp = torch.sum(gt_mask & pred_mask, dim=(1, 2))
            fp = torch.sum(torch.logical_not(gt_mask) & pred_mask, dim=(1, 2))
            fn = torch.sum(gt_mask & torch.logical_not(pred_mask), dim=(1, 2))

            tp_fn = tp + fn

            iou = tp / (tp_fn + fp)
            stats_valid = tp_fn > 0
        else:
            iou, tp, fp, fn = None, None, None, None
            stats_valid = np.ones((B,))

        res = []
        for b in range(B):
            if pred_mask is not None:
                if not stats_valid[b]:
                    res.append((None, None, None, None, None, None))
                    continue

            d = (
                self.data.setdefault(camera[b], {})
                .setdefault(sequence[b], {})
                .setdefault(frame[b], {})
                .setdefault(layer, {})
            )

            epe_mean_b = epe_mean[b].item()
            epe_1px_b = epe_1px[b].item()
            epe_3px_b = epe_3px[b].item()
            epe_5px_b = epe_5px[b].item()
            wauc_b = wauc[b].item()

            if iou is not None:
                iou_b = iou[b].item()
                tp = tp[b].item()
                fp = fp[b].item()
                fn = fn[b].item()
            else:
                iou_b = np.nan
                tp = np.nan
                fp = np.nan
                fn = np.nan

            if np.isfinite(epe_mean_b):
                d["epe"] = epe_mean_b
            if np.isfinite(epe_1px_b):
                d["1px"] = epe_1px_b
            if np.isfinite(epe_3px_b):
                d["3px"] = epe_3px_b
            if np.isfinite(epe_5px_b):
                d["5px"] = epe_5px_b
            if np.isfinite(wauc_b):
                d["wauc"] = wauc_b
            if np.isfinite(iou_b):
                d["iou"] = iou_b
            if np.isfinite(tp):
                d["tp"] = tp
            if np.isfinite(fp):
                d["fp"] = fp
            if np.isfinite(fn):
                d["fn"] = fn

            res.append((epe_mean_b, epe_1px_b, epe_3px_b, epe_5px_b, wauc_b, iou_b))

        return res

    def total(self):
        flattened = {}

        for camera_data in self.data.values():
            for sequence_data in camera_data.values():
                for frame_data in sequence_data.values():
                    for layer_name, layer_data in frame_data.items():
                        for metric_name, metric_value in layer_data.items():
                            if np.isfinite(metric_value):
                                d = flattened.setdefault(layer_name, {})
                                d.setdefault(metric_name, list()).append(metric_value)

        layers = {}
        for layer, data in flattened.items():
            layers[layer] = {}

            for metric in ["epe", "1px", "3px", "5px", "wauc", "iou"]:
                if metric not in data:
                    continue

                layers[layer][metric] = np.mean(data[metric])

            for metric in ["tp", "fp", "fn"]:
                if metric not in data:
                    continue

                layers[layer][metric] = np.sum(data[metric]).item()

            if "tp" in layers[layer]:
                tp = layers[layer]["tp"]
                fp = layers[layer]["fp"]
                fn = layers[layer]["fn"]

                layers[layer]["iou"] = tp / (tp + fp + fn)

        m_wauc = [
            data["wauc"] for layer, data in layers.items() if layer not in {"full"}
        ]

        m_iou = [
            data["iou"]
            for layer, data in layers.items()
            if layer not in {"full", "empty"}
        ]

        total = {"layers": layers}

        n = min(8, len(m_wauc))

        if m_iou:
            m_iou = np.sum(m_iou[0 : n - 1] * self.weights[1:n]) / np.sum(
                self.weights[1:n]
            )
            total["m_iou"] = m_iou
        else:
            m_iou = None

        if m_wauc:
            m_wauc = np.sum(m_wauc[0:n] * self.weights[0:n]) / np.sum(self.weights[0:n])
            total["m_wauc"] = m_wauc
        else:
            m_wauc = None

        if m_iou is not None and m_wauc is not None:
            total["afq"] = np.sqrt(m_wauc * m_iou)

        return total

    def collect(self):
        return {
            "frames": self.data,
            "total": self.total(),
        }


@torch.no_grad()
def validate_amsynthdrive_modal(model, sigma=0.05, out_path=None, save_json=None):
    """Peform validation using the AmodalSynthDrive validation"""
    time_start = datetime.now()

    IMAGE_SIZE = [1080, 1920]

    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    model.eval()

    val_dataset = datasets.AmSynthDrive(
        camera=["front", "back"], amodal=True, split="val", show_extra_info=True
    )

    stats = Stats(device="cuda:0")

    n = len(val_dataset)
    for val_id in tqdm(range(n)):
        image1, image2, flow_amgt, _, _, info = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        info = {k: [v] for k, v in info.items()}

        # extract full flow ground truth
        flow_fg_gt = flow_amgt[0]
        flow_fg_gt = flow_fg_gt[None].cuda()

        flows = 0
        flow_count = 0

        t_start = time.time()

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h : h + TRAIN_SIZE[0], w : w + TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h : h + TRAIN_SIZE[0], w : w + TRAIN_SIZE[1]]

            (flow_fg_p, *_), _ = model(image1_tile, image2_tile)

            padding = (
                w,
                IMAGE_SIZE[1] - w - TRAIN_SIZE[1],
                h,
                IMAGE_SIZE[0] - h - TRAIN_SIZE[0],
                0,
                0,
            )
            flows += F.pad(flow_fg_p * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count

        t_delta = time.time() - t_start

        epe = torch.sum((flow_pre - flow_fg_gt) ** 2, dim=1).sqrt()
        err_fg = stats.push(**info, layer="full", epe=epe)

        if out_path is not None:
            path = Path(out_path) / f"frame_{str(val_id).zfill(4)}_flow.png"
            write_flow_png(path, flow_pre[0].permute(1, 2, 0))

            path = Path(out_path) / f"frame_{str(val_id).zfill(4)}_flow_gt.png"
            write_flow_png(path, flow_fg_gt[0].permute(1, 2, 0))

        tqdm.write(
            f"validating {val_id+1}/{n}, time: {t_delta:0.2}s, epe: {err_fg[0][0]:.4}"
        )

    stats = stats.collect()

    if save_json is not None:
        with open(save_json, "w") as fd:
            json.dump(stats, fd)

    log.info(f"Eval AmSynthDrive:")
    for k, v in stats["total"]["layers"]["full"].items():
        log.info(f"  {k}: {v}")

    return stats["total"]


@torch.no_grad()
def validate_amsynthdrive(model, sigma=0.5, save=None, save_json=None, batch_size=1):
    """Peform validation using the AmodalSynthDrive flow dataset"""
    IMAGE_SIZE = [1080, 1920]
    hws = compute_grid_indices(IMAGE_SIZE)
    weights = compute_weight(hws, IMAGE_SIZE, TRAIN_SIZE, sigma)

    model.eval()

    val_dataset = datasets.AmSynthDrive(
        camera=["front", "back"], amodal=True, split="val", show_extra_info=True
    )

    val_dataset = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False,
    )

    n_fullcls = labels.N_CLASSES_FULL
    n_amcls = labels.N_CLASSES_AMODAL

    epe_stats = Stats(device="cuda:0")

    for batch_id, data in enumerate(tqdm(val_dataset)):
        image1, image2, flow_gt, _valid, sseg_gt, info = data
        image1 = image1.cuda()
        image2 = image2.cuda()

        B, _, H, W = image1.shape

        flow_fg_gt, flow_bg_gt, *flow_am_gt = flow_gt
        flow_fg_gt, flow_bg_gt = flow_fg_gt.cuda(), flow_bg_gt.cuda()
        flow_ams_gt = [(f.cuda(), m.cuda()) for f, m in flow_am_gt]

        flow_count = torch.zeros((B, 1, H, W), device=image1.device)
        flows_fg = torch.zeros((B, 2, H, W), device=image1.device)
        flows_bg = torch.zeros((B, 2, H, W), device=image1.device)
        flows_am = [
            [
                torch.zeros((B, 2, H, W), device=image1.device),
                torch.zeros((B, 1, H, W), device=image1.device),
                torch.zeros((B, n_amcls, H, W), device=image1.device),
                torch.zeros((B, 1, H, W), device=image1.device),
                torch.zeros((B, 1, H, W), device=image1.device),
            ]
            for _ in range(len(flow_am_gt))
        ]
        ssegs_fg = torch.zeros((B, n_fullcls, H, W), device=image1.device)

        t_start = time.time()

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[:, :, h : h + TRAIN_SIZE[0], w : w + TRAIN_SIZE[1]]
            image2_tile = image2[:, :, h : h + TRAIN_SIZE[0], w : w + TRAIN_SIZE[1]]

            (flow_fg_p, flow_bg_p, sseg_fg_p, *flow_am_p), _ = model(
                image1_tile, image2_tile
            )

            padding = (
                w,
                IMAGE_SIZE[1] - w - TRAIN_SIZE[1],
                h,
                IMAGE_SIZE[0] - h - TRAIN_SIZE[0],
                0,
                0,
            )

            flows_fg += F.pad(flow_fg_p * weights[idx], padding)
            flows_bg += F.pad(flow_bg_p * weights[idx], padding)

            sseg_fg_p = F.softmax(sseg_fg_p, dim=1)
            ssegs_fg += F.pad(sseg_fg_p * weights[idx], padding)

            for k in range(len(flow_am_gt)):
                flow, mask, amcls, mask_vis, mask_occ = flow_am_p[k]

                mask = torch.sigmoid(mask)
                amcls = F.softmax(amcls, dim=1)
                mask_vis = torch.sigmoid(mask_vis)
                mask_occ = torch.sigmoid(mask_occ)

                flows_am[k][0] += F.pad(flow * weights[idx], padding)
                flows_am[k][1] += F.pad(mask * weights[idx], padding)
                flows_am[k][2] += F.pad(amcls * weights[idx], padding)
                flows_am[k][3] += F.pad(mask_vis * weights[idx], padding)
                flows_am[k][4] += F.pad(mask_occ * weights[idx], padding)

            flow_count += F.pad(weights[idx], padding)

        flow_fg_p = flows_fg / flow_count
        flow_bg_p = flows_bg / flow_count
        sseg_fg_p = ssegs_fg / flow_count
        flow_ams_p = [
            (
                f / flow_count,
                m / flow_count,
                s / flow_count,
                v / flow_count,
                o / flow_count,
            )
            for f, m, s, v, o in flows_am
        ]

        t_delta = time.time() - t_start

        epe = torch.sum((flow_fg_p - flow_fg_gt) ** 2, dim=1).sqrt()
        err_fg = epe_stats.push(**info, layer="full", epe=epe)

        epe = torch.sum((flow_bg_p - flow_bg_gt) ** 2, dim=1).sqrt()
        epe_stats.push(**info, layer="empty", epe=epe)

        # save full and empty flow results
        if save is not None:
            for b in range(B):
                camera = info["camera"][b]
                sequence = info["sequence"][b]
                frame = info["frame"][b]

                path = Path(save) / camera / sequence / f"frame_{frame}_full_flow.png"
                write_flow_png(path, flow_fg_p[b].permute(1, 2, 0).cpu())

                path = (
                    Path(save) / camera / sequence / f"frame_{frame}_full_flow_gt.png"
                )
                write_flow_png(path, flow_fg_gt[b].permute(1, 2, 0).cpu())

                path = Path(save) / camera / sequence / f"frame_{frame}_empty_flow.png"
                write_flow_png(path, flow_bg_p[b].permute(1, 2, 0).cpu())

                path = (
                    Path(save) / camera / sequence / f"frame_{frame}_empty_flow_gt.png"
                )
                write_flow_png(path, flow_bg_gt[b].permute(1, 2, 0).cpu())

        for j in range(len(flow_ams_gt)):
            flow_am_gt, mask_am_gt = flow_ams_gt[j]
            flow_am_p, mask_am_p, sseg_am_p, mask_amvis_p, mask_amocc_p = flow_ams_p[j]

            flow_am_p = flow_am_p * (mask_am_p > 0.5)
            flow_am_gt = flow_am_gt * (mask_am_gt > 0.5)[:, None, ...]

            epe = torch.sum((flow_am_p - flow_am_gt) ** 2, dim=1).sqrt()
            epe_stats.push(
                **info,
                layer=f"amodal{j}",
                epe=epe,
                gt_mask=mask_am_gt,
                pred_mask=mask_am_p,
            )

            if save is not None:
                for b in range(B):
                    camera = info["camera"][b]
                    sequence = info["sequence"][b]
                    frame = info["frame"][b]

                    path = (
                        Path(save)
                        / camera
                        / sequence
                        / f"frame_{frame}_amodal{j}_flow.png"
                    )
                    flow = flow_am_p[b].permute(1, 2, 0).cpu()
                    mask = mask_am_p[b].squeeze(0).cpu()
                    write_flow_png(path, flow, mask)

                    path = (
                        Path(save)
                        / camera
                        / sequence
                        / f"frame_{frame}_amodal{j}_flow_gt.png"
                    )
                    flow = flow_am_gt[b].permute(1, 2, 0).cpu()
                    mask = mask_am_gt[b].cpu()
                    write_flow_png(path, flow, mask)

        for b in range(B):
            val_id = batch_id * B + b
            tqdm.write(f"step: {val_id}, time: {t_delta}s, epe-full: {err_fg[b][0]:.4}")

    evaldata = {
        "frames": epe_stats.data,
        "total": epe_stats.total(),
    }

    if save_json is not None:
        with open(save_json, "w") as fd:
            json.dump(evaldata, fd)

    log.info(f"Eval AmSynthDrive:")
    log.info(f"  layers:")
    for ty, data in evaldata["total"]["layers"].items():
        for k, v in data.items():
            log.info(f"    {ty}::{k}: {v}")

    log.info(f"  total:")
    for k, v in evaldata["total"].items():
        if k == "layers":
            continue

        log.info(f"    {k}: {v}")

    return evaldata["total"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="load model")
    parser.add_argument("--eval", help="eval benchmark")
    parser.add_argument("--small", action="store_true", help="use small model")
    args = parser.parse_args()

    exp_func = None
    cfg = None
    if args.eval == "sintel_submission":
        exp_func = create_sintel_submission
        cfg = get_submission_cfg()
    elif args.eval == "kitti_submission":
        exp_func = create_kitti_submission
        cfg = get_submission_cfg()
    elif args.eval == "sintel_validation":
        exp_func = validate_sintel
        cfg = get_submission_cfg()
    elif args.eval == "kitti_validation":
        exp_func = validate_kitti
        cfg = get_submission_cfg()
    elif args.eval == "amsynthdrive_modal_validation":
        exp_func = validate_amsynthdrive_modal
        cfg = get_amsynthdrive_cfg()
    elif args.eval == "amsynthdrive_validation":
        exp_func = validate_amsynthdrive
        cfg = get_amsynthdrive_cfg()
    else:
        print(f"ERROR: {args.eval} is not valid")
    cfg.update(vars(args))

    print(cfg)
    model = torch.nn.DataParallel(build_flowformer(cfg))

    log.info(f"Loading ckpt from {cfg.model}")
    try:
        model.load_state_dict(torch.load(cfg.model), strict=True)
    except RuntimeError as e:
        log.warning(f"Failed to load state dict in strict mode: {e}")
        log.warning(f"Falling back to strict=False")
        model.load_state_dict(torch.load(cfg.model), strict=False)

    model.cuda()
    model.eval()

    exp_func(model.module)
