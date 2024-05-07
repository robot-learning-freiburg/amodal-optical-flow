import torch

MAX_FLOW = 400


def sequence_loss(flow_preds, flow_gt, valid, cfg):
    """Loss function defined over sequence of flow predictions"""

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    flow_gt_thresholds = [5, 10, 20]

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }

    flow_gt_length = torch.sum(flow_gt**2, dim=1).sqrt()
    flow_gt_length = flow_gt_length.view(-1)[valid.view(-1)]
    for t in flow_gt_thresholds:
        e = epe[flow_gt_length < t]
        metrics.update({f"{t}-th-5px": (e < 5).float().mean().item()})

    return flow_loss, metrics


def smooth_l1_loss(diff):
    cond = diff.abs() < 1
    loss = torch.where(cond, 0.5 * diff**2, diff.abs() - 0.5)
    return loss


def sequence_loss_smooth(flow_preds, flow_gt, valid, cfg):
    """Loss function defined over sequence of flow predictions"""

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    flow_gt_thresholds = [5, 10, 20]

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = smooth_l1_loss((flow_preds[i] - flow_gt))
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        "epe": epe.mean().item(),
        "1px": (epe < 1).float().mean().item(),
        "3px": (epe < 3).float().mean().item(),
        "5px": (epe < 5).float().mean().item(),
    }

    flow_gt_length = torch.sum(flow_gt**2, dim=1).sqrt()
    flow_gt_length = flow_gt_length.view(-1)[valid.view(-1)]
    for t in flow_gt_thresholds:
        e = epe[flow_gt_length < t]
        metrics.update({f"{t}-th-5px": (e < 5).float().mean().item()})

    return flow_loss, metrics


def amodal_sequence_loss(flow_preds, flow_gt, valid, sseg_gt, cfg):
    """Loss function defined over sequence of amodal flow predictions"""
    mask_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    sseg_loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=255)

    gamma = cfg.gamma
    max_flow = cfg.max_flow
    n_predictions = len(flow_preds)

    flow_full_gt, flow_empty_gt, *flow_amodal_gt = flow_gt
    sseg_full_gt, sseg_empty_gt, *sseg_amodal_gt = sseg_gt

    # exlude invalid pixels and extremely large diplacements
    mag_full = torch.sum(flow_full_gt**2, dim=1).sqrt()
    mag_empty = torch.sum(flow_empty_gt**2, dim=1).sqrt()
    mag_amodal = [torch.sum(f**2, dim=1).sqrt() for f, _ in flow_amodal_gt]

    valid_full = (valid[0] >= 0.5) & (mag_full < max_flow)
    valid_empty = (valid[1] >= 0.5) & (mag_empty < max_flow)
    valid_amodal = [(v >= 0.5) & (m < max_flow) for v, m in zip(valid[2:], mag_amodal)]

    loss_total = 0.0
    loss_total_flow_fg = 0.0
    loss_total_flow_bg = 0.0
    loss_total_sseg_fg = 0.0
    loss_total_flow_am = 0.0
    loss_total_mask_am = 0.0
    loss_total_sseg_am = 0.0
    loss_total_mask_amvis = 0.0
    loss_total_mask_amocc = 0.0

    for i in range(n_predictions):
        flow_fg_p, flow_bg_p, sseg_fg_p, *flow_ams_p = flow_preds[i]

        i_weight = gamma ** (n_predictions - i - 1)

        flow_loss_fg = (flow_fg_p - flow_full_gt).abs()
        flow_loss_fg = flow_loss_fg * valid_full[:, None, ...]
        flow_loss_fg = flow_loss_fg.mean()
        loss_total_flow_fg += i_weight * flow_loss_fg

        flow_loss_bg = (flow_bg_p - flow_empty_gt).abs()
        flow_loss_bg = flow_loss_bg * valid_empty[:, None, ...]
        flow_loss_bg = flow_loss_bg.mean()
        loss_total_flow_bg += i_weight * flow_loss_bg

        sseg_loss_fg = sseg_loss_fn(sseg_fg_p, sseg_full_gt)
        sseg_loss_fg = sseg_loss_fg.mean()
        loss_total_sseg_fg += i_weight * sseg_loss_fg

        i_loss = flow_loss_fg + flow_loss_bg + sseg_loss_fg

        mask_occluder = torch.zeros_like(
            flow_amodal_gt[0][1], device=flow_amodal_gt[0][1].device
        )

        for j in range(len(flow_amodal_gt)):
            flow_am_gt, mask_am_gt = flow_amodal_gt[j]
            sseg_am_gt = sseg_amodal_gt[j]

            flow_am_p, mask_am_p, sseg_am_p, mask_amvis_p, mask_amocc_p = flow_ams_p[j]

            mask_am_p = mask_am_p.squeeze(1)
            mask_amvis_p = mask_amvis_p.squeeze(1)
            mask_amocc_p = mask_amocc_p.squeeze(1)

            mask_amvis_gt = ((mask_am_gt - mask_occluder) > 0.5).float()
            mask_amocc_gt = ((mask_am_gt - mask_amvis_gt) > 0.5).float()

            mask_occluder = ((mask_occluder + mask_am_gt) > 0.5).float()

            valid_am = valid_amodal[j]

            # Note: We deliberately do not mask the predicted flow. Instead we
            # force it to predict zero flow to add more constraints and
            # guidance. Without this, the network tends to produce bad/random
            # results in masked-out areas, which might affect results in the
            # areas that we care for.
            flow_am_gt = flow_am_gt * (mask_am_gt > 0.5)[:, None, ...]

            flow_loss_amodal = (flow_am_p - flow_am_gt).abs()
            flow_loss_amodal = flow_loss_amodal * valid_am[:, None, ...]
            flow_loss_amodal = flow_loss_amodal.mean()
            i_loss += flow_loss_amodal

            mask_loss = mask_loss_fn(mask_am_p, mask_am_gt)
            mask_loss = mask_loss * valid_am
            mask_loss = mask_loss.mean()
            i_loss += mask_loss

            sseg_loss_amodal = sseg_loss_fn(sseg_am_p, sseg_am_gt)
            sseg_loss_amodal = sseg_loss_amodal.mean()
            i_loss += sseg_loss_amodal

            mask_loss_vis = mask_loss_fn(mask_amvis_p, mask_amvis_gt)
            mask_loss_vis = mask_loss_vis * valid_am
            mask_loss_vis = mask_loss_vis.mean()
            i_loss += mask_loss_vis

            mask_loss_occ = mask_loss_fn(mask_amocc_p, mask_amocc_gt)
            mask_loss_occ = mask_loss_occ * valid_am
            mask_loss_occ = mask_loss_occ.mean()
            i_loss += mask_loss_occ

            loss_total_flow_am += i_weight * flow_loss_amodal
            loss_total_mask_am += i_weight * mask_loss
            loss_total_sseg_am += i_weight * sseg_loss_amodal
            loss_total_mask_amvis += i_weight * mask_loss_vis
            loss_total_mask_amocc += i_weight * mask_loss_occ

        loss_total += i_weight * i_loss

    # compute metrics on last result
    metrics = {
        "loss-total": loss_total.cpu().item(),
        "loss-flow-fg": loss_total_flow_fg.cpu().item(),
        "loss-flow-bg": loss_total_flow_bg.cpu().item(),
        "loss-sseg-fg": loss_total_sseg_fg.cpu().item(),
        "loss-flow-am": loss_total_flow_am.cpu().item(),
        "loss-mask-am": loss_total_mask_am.cpu().item(),
        "loss-sseg-am": loss_total_sseg_am.cpu().item(),
        "loss-mask-amvis": loss_total_mask_amvis.cpu().item(),
        "loss-mask-amocc": loss_total_mask_amocc.cpu().item(),
    }

    flow_fg_p, flow_bg_p, sseg_fg_p, *flow_ams_p = flow_preds[-1]

    flow_fg_p = flow_fg_p.detach()
    flow_bg_p = flow_bg_p.detach()
    sseg_fg_p = sseg_fg_p.detach()
    flow_ams_p = [
        (f.detach(), m.detach(), s.detach(), v.detach(), o.detach())
        for f, m, s, v, o in flow_ams_p
    ]

    epe = torch.sum((flow_fg_p - flow_full_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid_full.view(-1)]
    metrics |= {
        "full-epe": epe.mean().cpu().item(),
        "full-1px": (epe < 1).float().mean().cpu().item(),
        "full-3px": (epe < 3).float().mean().cpu().item(),
        "full-5px": (epe < 5).float().mean().cpu().item(),
    }

    epe = torch.sum((flow_bg_p - flow_empty_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid_empty.view(-1)]
    metrics |= {
        "empty-epe": epe.mean().cpu().item(),
        "empty-1px": (epe < 1).float().mean().cpu().item(),
        "empty-3px": (epe < 3).float().mean().cpu().item(),
        "empty-5px": (epe < 5).float().mean().cpu().item(),
    }

    mask_occluder = torch.zeros_like(
        flow_amodal_gt[0][1], device=flow_amodal_gt[0][1].device
    )

    for j in range(len(flow_amodal_gt)):
        flow_am_gt, mask_am_gt = flow_amodal_gt[j]
        flow_am_p, mask_am_p, sseg_am_p, mask_amvis_p, mask_amocc_p = flow_ams_p[j]
        valid_am = valid_amodal[j]

        mask_am_p = mask_am_p.squeeze(1)
        mask_am_p = torch.sigmoid(mask_am_p)

        mask_am_union = (mask_am_p > 0.5) | (mask_am_gt > 0.5)
        mask_am_union = mask_am_union.float().sum()
        mask_am_inter = (mask_am_p > 0.5) & (mask_am_gt > 0.5)
        mask_am_inter = mask_am_inter.float().sum()

        if mask_am_union >= 1e-10:
            metrics |= {
                f"ammask-{j}-iou": (mask_am_inter / mask_am_union).item(),
            }

            flow_am_p = flow_am_p * (mask_am_p > 0.5)[:, None, ...]
            flow_am_gt = flow_am_gt * (mask_am_gt > 0.5)[:, None, ...]

            epe = torch.sum((flow_am_p - flow_am_gt) ** 2, dim=1).sqrt()
            epe = epe.view(-1)[valid_am.view(-1)]

            metrics |= {
                f"amflow-{j}-epe": epe.mean().cpu().item(),
                f"amflow-{j}-1px": (epe < 1).float().mean().cpu().item(),
                f"amflow-{j}-3px": (epe < 3).float().mean().cpu().item(),
                f"amflow-{j}-5px": (epe < 5).float().mean().cpu().item(),
            }

        mask_amvis_p = mask_amvis_p.squeeze(1)
        mask_amvis_p = torch.sigmoid(mask_amvis_p)

        mask_amvis_union = (mask_amvis_p > 0.5) | (mask_amvis_gt > 0.5)
        mask_amvis_union = mask_amvis_union.float().sum()
        mask_amvis_inter = (mask_amvis_p > 0.5) & (mask_amvis_gt > 0.5)
        mask_amvis_inter = mask_amvis_inter.float().sum()

        if mask_amvis_union >= 1e-10:
            metrics |= {
                f"ammask-vis-{j}-iou": (mask_amvis_inter / mask_amvis_union).item(),
            }

        mask_amocc_p = mask_amocc_p.squeeze(1)
        mask_amocc_p = torch.sigmoid(mask_amocc_p)

        mask_amocc_union = (mask_amocc_p > 0.5) | (mask_amocc_gt > 0.5)
        mask_amocc_union = mask_amocc_union.float().sum()
        mask_amocc_inter = (mask_amocc_p > 0.5) & (mask_amocc_gt > 0.5)
        mask_amocc_inter = mask_amocc_inter.float().sum()

        if mask_amocc_union >= 1e-10:
            metrics |= {
                f"ammask-occ-{j}-iou": (mask_amocc_inter / mask_amocc_union).item(),
            }

    return loss_total, metrics
