import torch

from common.utils import get_scheduler, is_resume
from utils import MetricLogger, save_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def meta_trainer(
    P,
    train_func,
    test_func,
    model,
    optimizer,
    train_loader,
    test_loader,
    logger,
):
    """
    Offline Stage Trainr. Organizes main meta-training loop for gradient-based meta-learning.
    """
    metric_logger = MetricLogger(delimiter="  ")
    scheduler = get_scheduler(P, optimizer)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=True) if P.use_amp else None

    # ------------------------ resume (may include occ_grid) ------------------------
    start_step, best, psnr = is_resume(
        P, model, optimizer, scheduler, grad_scaler, prefix=P.prefix
    )

    # ------------------------------ training -------------------------------
    logger.log_dirname("Start training")
    logger.log_custom_dict(P)

    for it, train_batch in enumerate(train_loader):
        step = start_step + it + 1
        if step > P.outer_steps:
            break
        # ------------------------------ one training step --------------------------
        train_func(
            P,
            step,
            model,
            optimizer,
            train_batch,
            metric_logger=metric_logger,
            logger=logger,
            scheduler=scheduler,
            grad_scaler=grad_scaler,
        )

        # -------------------------- evaluation & checkpoint ------------------------
        """ evaluation & save the best model """
        if step % P.eval_step == 0:
            psnr = test_func(P, model, test_loader, step, logger=logger)
            if best < psnr:
                best = psnr
                save_checkpoint(
                    P,
                    step,
                    model,
                    optimizer,
                    logger.logdir,
                    is_best=True,
                    best=best,
                    scheduler=scheduler,
                    scaler=grad_scaler,
                    keep_occ_grids=True,
                )

            logger.scalar_summary("eval/best", best, step)
            logger.log(
                "[EVAL] [Step %3d] [PSNR %5.2f] [Best %5.2f]" % (step, psnr, best)
            )

        """ save model per save_step steps"""
        if step % P.save_step == 0:
            save_checkpoint(
                P,
                step,
                model,
                optimizer,
                logger.logdir,
                is_best=False,
                best=best,
                scheduler=scheduler,
                scaler=grad_scaler,
                keep_occ_grids=True,
            )

    """ save last model"""
    save_checkpoint(
        P,
        step,
        model,
        optimizer,
        logger.logdir,
        is_best=False,
        best=best,
        scheduler=scheduler,
        scaler=grad_scaler,
        keep_occ_grids=True,
    )
