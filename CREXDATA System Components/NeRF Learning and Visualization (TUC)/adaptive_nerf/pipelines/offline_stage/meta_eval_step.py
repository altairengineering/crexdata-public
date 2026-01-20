import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

import time

import torch
from torch.amp import autocast

from utils import psnr, MetricLogger
from common.utils import to_device_tree
from nerfs.losses import compute_mse_loss
from .meta_core import task_adapt


def eval_step(P, model, test_loader, steps, logger) -> float:
    """
    Evaluate meta-learned NeRF generalization to unseen views.

    Builds support/query tasks from test_loader, performs inner-loop adaptation
    per task, and returns the sample-weighted PSNR over query rays.
    """
    model.eval()
    P.fim = False  # disable FIM during eval if present

    device = next(model.parameters()).device
    inner_lr = P.inner_lr
    iterations = int(getattr(P, "tto", getattr(P, "inner_iter", 1)))
    mixed_precision = bool(getattr(P, "mixed_precision", False))
    metric_logger = MetricLogger(delimiter="  ")
    tasks_cap = int(getattr(P, "max_test_tasks", 5))
    tasks_seen = 0

    support_loss_sum_global = torch.tensor(0.0, device=device)
    support_ray_count_global = 0
    query_loss_sum_global = torch.tensor(0.0, device=device)
    query_ray_count_global = 0

    for task_data in test_loader:
        batch_start_t = time.time()
        task_data = to_device_tree(task_data, device)
        cids = sorted(task_data.keys())

        for cid in cids:
            for task in task_data[cid]:
                sup_i, qry_i = task["support"], task["query"]

                n_sup = int(sup_i["rays"].shape[0])
                n_q = int(qry_i["rays"].shape[0])

                if n_sup == 0 or n_q == 0:
                    logger.log(f"[EVAL][WARN] Empty task in region {cid}; skipping.")
                    continue

                # Inner adaptation on support rays
                fast, inner_losses = task_adapt(
                    P,
                    model,
                    sup_i,
                    inner_lr,
                    iterations,
                    active_module=cid,
                )
                inner_last = (
                    inner_losses[-1]
                    if inner_losses
                    else torch.tensor(0.0, device=device)
                )

                # Query loss after adaptation
                with torch.no_grad(), autocast(
                    device_type="cuda",
                    enabled=mixed_precision,
                    dtype=torch.float16,
                ):
                    qa = compute_mse_loss(
                        P,
                        model=model,
                        data=qry_i,
                        params=fast,
                        active_module=cid,
                        reduction="mean",
                    )

                support_loss_sum_global += inner_last.detach() * n_sup
                support_ray_count_global += n_sup
                query_loss_sum_global += qa.detach() * n_q
                query_ray_count_global += n_q

        batch_time = time.time() - batch_start_t
        metric_logger.meters["batch_time"].update(batch_time, n=1)
        metric_logger.meters["eval_context"].update(len(cids), n=1)

        tasks_seen += 1
        if tasks_seen >= tasks_cap:
            break

    if query_ray_count_global == 0:
        logger.log(
            "[EVAL] No valid query rays found in test_loader; returning PSNR=0.0"
        )
        model.train(True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return 0.0

    loss_in_global = (
        support_loss_sum_global / support_ray_count_global
        if support_ray_count_global > 0
        else torch.tensor(0.0, device=device)
    )
    loss_out_global = query_loss_sum_global / query_ray_count_global
    psnr_in_global = psnr(loss_in_global)
    psnr_out_global = psnr(loss_out_global)

    metric_logger.meters["loss_in"].update(
        float(loss_in_global), n=support_ray_count_global
    )
    metric_logger.meters["psnr_in"].update(
        float(psnr_in_global), n=support_ray_count_global
    )
    metric_logger.meters["loss_out"].update(
        float(loss_out_global), n=query_ray_count_global
    )
    metric_logger.meters["psnr_out"].update(
        float(psnr_out_global), n=query_ray_count_global
    )
    metric_logger.synchronize_between_processes()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.log(
        " * [EVAL] [LossIn %.6f] [LossOut %.6f] [PSNRIn %.3f] [PSNROut %.3f]"
        % (
            metric_logger.loss_in.global_avg,
            metric_logger.loss_out.global_avg,
            metric_logger.psnr_in.global_avg,
            metric_logger.psnr_out.global_avg,
        )
    )
    logger.scalar_summary("eval/tto", float(iterations), steps)
    logger.scalar_summary("eval/loss_in", metric_logger.loss_in.global_avg, steps)
    logger.scalar_summary("eval/loss_out", metric_logger.loss_out.global_avg, steps)
    logger.scalar_summary("eval/psnr_in", metric_logger.psnr_in.global_avg, steps)
    logger.scalar_summary("eval/psnr_out", metric_logger.psnr_out.global_avg, steps)

    return float(metric_logger.psnr_out.global_avg)
