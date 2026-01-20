import random
import time

import torch
from torch.amp import autocast

from utils import psnr
from common.utils import to_device_tree
from nerfs.losses import compute_loss
from .meta_core import (
    task_adapt,
    meta_update,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_step(
    P,
    step,
    model,
    optimizer,
    task_data,
    metric_logger,
    logger,
    scheduler=None,
    grad_scaler=None,
):
    """
    Offline Stage: meta-training step over a sequence of tasks grouped by region (cid).

    This function runs inner-loop adaptation on support rays, evaluates the adapted
    parameters on query rays, aggregates losses at region level, and applies the
    outer meta-update. It also records timing and performance metrics to assist with
    debugging, monitoring training dynamics, and identifying bottlenecks.
    """
    t_step_start = time.perf_counter()

    model.train()
    device = next(model.parameters()).device

    time_setup = time_data = time_inner = time_outer = 0.0

    # -------- setup  --------
    t0 = time.perf_counter()
    total_tasks = sum(len(v) for v in task_data.values())
    cids = list(task_data.keys())  # assumed 0..K-1, dense
    rnd = random.Random(getattr(P, "seed", 0) + step)
    rnd.shuffle(cids)
    num_regions = len(cids)

    region_inner_sum = {cid: torch.tensor(0.0, device=device) for cid in cids}
    region_inner_count = {cid: 0 for cid in cids}
    region_query_sum = {cid: torch.tensor(0.0, device=device) for cid in cids}
    region_query_count = {cid: 0 for cid in cids}

    time_setup += time.perf_counter() - t0

    # -------- iterate regions and tasks --------
    for cid in cids:
        region_tasks = task_data[cid]
        expert_nerf = model.submodules[cid]

        for task in region_tasks:
            t1 = time.perf_counter()
            # normalize structure
            if hasattr(task, "support") and hasattr(task, "query"):
                sup_i, qry_i = task.support, task.query
            else:
                sup_i, qry_i = task["support"], task["query"]

            sup_i = to_device_tree(sup_i, device)
            qry_i = to_device_tree(qry_i, device)
            time_data += time.perf_counter() - t1

            if getattr(P, "fim", False) and hasattr(expert_nerf, "fim_reset"):
                expert_nerf.fim_reset()

            assert sup_i["rays"].device == device and qry_i["rays"].device == device
            n_sup = int(sup_i["rays"].shape[0])
            n_q = int(qry_i["rays"].shape[0])

            if n_sup == 0 or n_q == 0:
                logger.log(f"[WARN] Empty task in region {cid}; skipping.")
                continue

            # ----- INNER: adapt on support -----
            t2 = time.perf_counter()
            fast_i, inner_losses = task_adapt(
                P,
                model,
                sup_i,
                P.inner_lr,
                P.inner_iter,
                active_module=cid,
            )
            time_inner += time.perf_counter() - t2

            last_inner_loss = (
                inner_losses[-1] if inner_losses else torch.tensor(0.0, device=device)
            )  # mean over support rays

            region_inner_sum[cid] += last_inner_loss.detach() * n_sup
            region_inner_count[cid] += n_sup

            # ----- QUERY: evaluate with adapted params -----
            t3 = time.perf_counter()
            with autocast(
                device_type="cuda",
                enabled=getattr(P, "mixed_precision", False),
                dtype=torch.float16,
            ):
                loss_q = compute_loss(
                    P,
                    model,
                    qry_i,
                    params=fast_i,
                    active_module=cid,
                )  # mean over query rays
            time_outer += time.perf_counter() - t3

            region_query_sum[cid] += loss_q * n_q
            region_query_count[cid] += n_q

    # -------- Sample-weighted reduction --------
    total_sup = sum(region_inner_count[cid] for cid in cids)
    total_q = sum(region_query_count[cid] for cid in cids)

    if total_q == 0:
        logger.log(
            "[WARN] train_step_ray: no query samples in any region; skipping step"
        )
        return

    region_loss_in = {
        cid: (
            region_inner_sum[cid] / region_inner_count[cid]
            if region_inner_count[cid] > 0
            else torch.tensor(0.0, device=device)
        )
        for cid in cids
    }
    region_loss_out = {
        cid: (
            region_query_sum[cid] / region_query_count[cid]
            if region_query_count[cid] > 0
            else torch.tensor(0.0, device=device)
        )
        for cid in cids
    }

    loss_in = (
        sum(region_inner_sum[cid] for cid in cids) / total_sup
        if total_sup > 0
        else torch.tensor(0.0, device=device)
    )
    loss_out = sum(region_query_sum[cid] for cid in cids) / total_q  # mean for logging

    # loss_out_meta = loss_out # fed avg in this case
    loss_out_meta = (
        num_regions * loss_out
    )  # fed-avg scaled by regions so that K does not affect result

    # -------- OUTER: meta update --------
    t4 = time.perf_counter()
    meta_update(
        P,
        model,
        optimizer,
        loss_out_meta,
        scheduler=scheduler,
        grad_scaler=grad_scaler,
    )
    time_outer += time.perf_counter() - t4

    if getattr(model, "use_occ", False):
        model.maybe_update_expert_occupancies(step, params=None)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_step_total = time.perf_counter() - t_step_start
    t_misc = max(0.0, t_step_total - (time_setup + time_data + time_inner + time_outer))
    # -------- global metrics --------
    metric_logger.meters["batch_time"].update(t_step_total, n=1)
    metric_logger.meters["tasks"].update(total_tasks, n=1)
    metric_logger.meters["loss_in"].update(float(loss_in), n=total_sup)
    metric_logger.meters["psnr_in"].update(float(psnr(loss_in)), n=total_sup)
    metric_logger.meters["loss_out"].update(float(loss_out), n=total_q)
    metric_logger.meters["psnr_out"].update(float(psnr(loss_out)), n=total_q)
    metric_logger.synchronize_between_processes()

    # -------- periodic logging (global + per-region + timing) --------
    if step % P.print_step == 0:
        # global scalars
        logger.log_dirname(f"Step {step}")
        logger.scalar_summary("train/loss_in", float(loss_in), step)
        logger.scalar_summary("train/loss_out", float(loss_out), step)
        logger.scalar_summary("train/psnr_in", float(psnr(loss_in)), step)
        logger.scalar_summary("train/psnr_out", float(psnr(loss_out)), step)
        logger.scalar_summary("train/batch_time", metric_logger.batch_time.value, step)

        # per-region diagnostics
        for cid in cids:
            r_rays_in = region_inner_count[cid]
            r_rays_out = region_query_count[cid]

            r_loss_in = float(region_loss_in[cid]) if r_rays_in > 0 else 0.0
            r_loss_out = float(region_loss_out[cid]) if r_rays_out > 0 else 0.0
            r_psnr_in = float(psnr(region_loss_in[cid])) if r_rays_in > 0 else 0.0
            r_psnr_out = float(psnr(region_loss_out[cid])) if r_rays_out > 0 else 0.0

            logger.scalar_summary(f"train/region_{cid}/rays_in", r_rays_in, step)
            logger.scalar_summary(f"train/region_{cid}/rays_out", r_rays_out, step)
            logger.scalar_summary(f"train/region_{cid}/loss_in", r_loss_in, step)
            logger.scalar_summary(f"train/region_{cid}/loss_out", r_loss_out, step)
            logger.scalar_summary(f"train/region_{cid}/psnr_in", r_psnr_in, step)
            logger.scalar_summary(f"train/region_{cid}/psnr_out", r_psnr_out, step)
        # timing
        logger.scalar_summary("train/time_setup", time_setup, step)
        logger.scalar_summary("train/time_data", time_data, step)
        logger.scalar_summary("train/time_inner", time_inner, step)
        logger.scalar_summary("train/time_outer", time_outer, step)
        logger.scalar_summary("train/time_misc", t_misc, step)
        logger.scalar_summary("train/time_total_step", t_step_total, step)

        logger.log(
            "[TRAIN] [Step %d] [LossIn %.6f] [LossOut %.6f] "
            "[PSNRIn %.2f] [PSNROut %.2f] [InnerLR %.6f]"
            % (
                step,
                float(loss_in),
                float(loss_out),
                float(psnr(loss_in)),
                float(psnr(loss_out)),
                float(P.inner_lr),
            )
        )
        logger.log(
            "[TIME] [Step %d] setup=%.4fs data=%.4fs inner=%.4fs "
            "outer=%.4fs misc=%.4fs total=%.4fs"
            % (
                step,
                time_setup,
                time_data,
                time_inner,
                time_outer,
                t_misc,
                t_step_total,
            )
        )

        metric_logger.reset()
