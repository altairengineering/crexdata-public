from typing import Dict, Iterable, Optional
import time
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lpips
from pytorch_msssim import ssim
from imageio import imwrite

from utils import collate_rays, MetricLogger
from common.utils import get_optimizer
from data.ram_rays_dataset import RamRaysDataset
from nerfs.color_space import linear_to_srgb, color_space_transformer
from nerfs.ray_rendering import render_image
from nerfs.losses import compute_mse_loss


def runtime_evaluate(
    P,
    model,
    test_loader,  # ImageMetaDataset DataLoader
    steps,
    logger,
    scene_box,
) -> dict:
    """
    Runtime task adaptation and full-image evaluation.

    Phase A: for each batch of image metadata, build a RamRaysDataset of support
    rays and adapt the model in-place (runtime_adapt).

    Phase B: render all test images with the adapted model and compute PSNR,
    SSIM, and LPIPS over the image set.
    """
    device = next(model.parameters()).device
    use_amp = bool(getattr(P, "use_amp", False))
    num_support = int(getattr(P, "support_rays", 4096))
    ray_samples = int(getattr(P, "ray_samples", 64))
    chunk_points = int(getattr(P, "chunk_points", 1 << 16))
    metrics_space = getattr(P, "color_space", "linear")

    logger.log(
        f"======================================= TTO: {getattr(P, 'tto', P.inner_iter)} ======================================="
    )

    # Phase A: adaptation on support rays
    model.train(True)
    total_support = 0
    logger.log("[Phase 1] Task adaptation on validation images")
    total_batches = 0

    optimizer = get_optimizer(P, model)
    t_adapt_start = time.time()

    for batch in test_loader:
        total_batches += 1
        metas = batch["metas"]

        rays_ds = RamRaysDataset(
            metas,
            center_pixels=True,
            ray_gen_kwargs={
                "scene_box": scene_box,
                "near_far_override": (P.near, P.far),
            },
        )

        total_support += int(len(rays_ds))

        data_loader = DataLoader(
            rays_ds,
            batch_size=num_support,
            shuffle=True,
            drop_last=False,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=4,
            pin_memory=True,
            collate_fn=collate_rays,
        )

        out = runtime_adapt(
            P=P, model=model, data_loader=data_loader, optimizer=optimizer, steps=steps
        )

        logger.log(
            f"[Batch {total_batches}] Rays={len(rays_ds)} last_loss={out['loss']:.6f}"
        )
        del rays_ds

    if device.type == "cuda":
        torch.cuda.synchronize()
    adapt_time_sec = time.time() - t_adapt_start

    logger.log(
        f"[ADAPTATION END] [{adapt_time_sec:.2f}s] Total support rays seen: "
        f"{total_support} in {total_batches} batches."
    )
    torch.cuda.empty_cache()

    # Phase B: rendering and metrics
    logger.log("[Phase 2] Rendering images")

    model.eval()
    scorer_lpips = lpips.LPIPS(net="alex").to(device)
    meter = MetricLogger(delimiter="  ")

    out_root_rendered = (Path("..") / "logs" / P.fname / "rendered").resolve()
    out_pred = out_root_rendered / ("pred" + str(getattr(P, "tto", steps)))
    out_gt = out_root_rendered / "gt"
    out_gt.mkdir(parents=True, exist_ok=True)
    out_pred.mkdir(parents=True, exist_ok=True)

    idx = 0
    t0 = time.time()
    for batch in test_loader:
        metas = batch["metas"]
        rgbs = batch["rgbs_raw"]

        for b, md in enumerate(metas):
            H, W = int(md.H), int(md.W)
            fx, fy, cx, cy = md.intrinsics.flatten().tolist()[:4]
            c2w = md.c2w
            gt_img_srgb = (rgbs[b].float() / 255.0).clamp_(0, 1).to(device)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                pred_lin, _, _ = render_image(
                    model,
                    H=H,
                    W=W,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    c2w=c2w,
                    scene_box=scene_box,
                    ray_samples=ray_samples,
                    bg_color_default=getattr(P, "bg_color_default", "white"),
                    chunk_points=chunk_points,
                    center_pixels=True,
                    use_amp=use_amp,
                )

            pred_bchw_lin = pred_lin.permute(2, 0, 1).unsqueeze(0)
            gt_bchw_srgb = gt_img_srgb.permute(2, 0, 1).unsqueeze(0)
            pred_bchw, gt_bchw = color_space_transformer(
                pred_bchw_lin, gt_bchw_srgb, metrics_space
            )

            mse = F.mse_loss(pred_bchw, gt_bchw, reduction="mean")
            psnr_val = (-10.0 * torch.log10(mse.clamp_min(1e-8))).item()
            ssim_val = ssim(pred_bchw, gt_bchw, data_range=1.0).mean().item()

            if metrics_space == "srgb":
                pred_srgb_img = pred_bchw
            else:
                pred_srgb_img = linear_to_srgb(pred_lin).permute(2, 0, 1).unsqueeze(0)

            pred_lp = pred_srgb_img * 2 - 1
            gt_lp = gt_bchw_srgb * 2 - 1
            lpips_val = scorer_lpips(pred_lp, gt_lp).mean().item()

            meter.meters["psnr"].update(psnr_val, n=1)
            meter.meters["ssim"].update(ssim_val, n=1)
            meter.meters["lpips"].update(lpips_val, n=1)
            logger.scalar_summary("eval_image/psnr", psnr_val, idx)
            logger.scalar_summary("eval_image/ssim", ssim_val, idx)
            logger.scalar_summary("eval_image/lpips", lpips_val, idx)

            logger.log(
                " * [IMG %d]  PSNR %.3f | SSIM %.3f | LPIPS %.3f"
                % (idx + 1, psnr_val, ssim_val, lpips_val)
            )

            pred_srgb = (
                (pred_srgb_img.squeeze(0).permute(1, 2, 0).clamp(0, 1) * 255.0 + 0.5)
                .byte()
                .cpu()
                .numpy()
            )
            gt_srgb = (gt_img_srgb.clamp(0, 1) * 255.0 + 0.5).byte().cpu().numpy()
            imwrite(out_pred / f"{idx:06d}.png", pred_srgb)
            imwrite(out_gt / f"{idx:06d}.png", gt_srgb)

            meter.meters["batch_time"].update(time.time() - t0, n=1)
            t0 = time.time()
            idx += 1

    meter.synchronize_between_processes()
    logger.log(
        " * [RENDERING END] imgs %d | PSNR %.3f | SSIM %.3f | LPIPS %.3f"
        % (idx, meter.psnr.global_avg, meter.ssim.global_avg, meter.lpips.global_avg)
    )
    logger.scalar_summary("eval/psnr", meter.psnr.global_avg, num_support)
    logger.scalar_summary("eval/ssim", meter.ssim.global_avg, num_support)
    logger.scalar_summary("eval/lpips", meter.lpips.global_avg, num_support)

    metrics = {
        "psnr": meter.psnr.global_avg,
        "ssim": meter.ssim.global_avg,
        "lpips": meter.lpips.global_avg,
        "duration": float(adapt_time_sec),
    }
    return metrics


def runtime_adapt(
    *,
    P,
    model,
    data_loader: Iterable,  # yields (rays, rgbs)
    optimizer: torch.optim.Optimizer,
    steps: Optional[
        int
    ] = None,  # number of optimizer steps (None => run all batches once)
    active_module: Optional[int] = None,
    grad_clip: Optional[float] = 1.0,
) -> Dict[str, float]:
    """
    Run in-place adaptation: optimizer updates the model parameters directly.
    If `steps` is None: iterate over data_loader once (one epoch).
    If `steps` is an int: perform exactly `steps` optimizer updates, looping over
    data_loader as many times as needed (infinite stream semantics).
    """
    device = next(model.parameters()).device
    use_amp = bool(getattr(P, "use_amp", True))

    model.train()
    base = model.submodules[active_module] if active_module is not None else model

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())
    last_loss = None
    step_count = 0

    # Case 1: no explicit step budget -> old behavior (one pass over data_loader)
    if steps is None:
        for rays, rgbs in data_loader:
            rays, rgbs = rays.to(device, non_blocking=True), rgbs.to(
                device, non_blocking=True
            )

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(
                enabled=use_amp and torch.cuda.is_available(), dtype=torch.float16
            ):
                loss = compute_mse_loss(
                    P,
                    model=base,
                    data={"rays": rays, "rgbs": rgbs},
                    params=None,
                    active_module=active_module,
                    reduction="mean",
                )

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(base.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            last_loss = float(loss.detach())
            step_count += 1

    # TODO: needs to be cleaned up. maintain until viewer side is reorganized!
    # Case 2: explicit step budget -> loop over loader indefinitely
    else:
        steps = int(steps)
        data_iter = iter(data_loader)
        while step_count < steps:
            try:
                rays, rgbs = next(data_iter)
            except StopIteration:
                # restart a new epoch over the same support set
                data_iter = iter(data_loader)
                rays, rgbs = next(data_iter)

            rays, rgbs = rays.to(device, non_blocking=True), rgbs.to(
                device, non_blocking=True
            )

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(
                enabled=use_amp and torch.cuda.is_available(), dtype=torch.float16
            ):
                loss = compute_mse_loss(
                    P,
                    model=base,
                    data={"rays": rays, "rgbs": rgbs},
                    params=None,
                    active_module=active_module,
                    reduction="mean",
                )

            scaler.scale(loss).backward()

            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(base.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            last_loss = float(loss.detach())
            step_count += 1

    return {"loss": (0.0 if last_loss is None else last_loss), "steps": step_count}
