from typing import Dict, Optional, OrderedDict

import torch
import torch.nn.functional as F

from .ray_rendering import render_rays
from .color_space import color_space_transformer


def compute_mse_loss(
    P,
    model,
    data,
    params=None,
    active_module=None,
    reduction="mean",
):
    """Standard MSE loss (optionally per-sample with reduction='none')."""
    gt_rgb = data["rgbs"]
    rays = data["rays"]
    pred_rgb, *_ = render_rays(
        model,
        rays,
        ray_samples=P.ray_samples,
        params=params,
        active_module=active_module,
        chunk=P.chunk_points,
    )
    pred_rgb, gt_rgb = color_space_transformer(
        pred_rgb, gt_rgb, color_space=P.color_space
    )
    return F.mse_loss(pred_rgb, gt_rgb, reduction=reduction)


def compute_fim_loss(
    P,
    model,
    data: Dict[str, torch.Tensor],
    params: Optional[OrderedDict] = None,
    active_module=None,
    *,
    grad_buffer: Optional[Dict[str, torch.Tensor]] = None,  # filled on SUPPORT only
    update_fisher: bool = False,  # update F (SUPPORT only)
    clamp_factor=5,
) -> torch.Tensor:
    """
    Fisher-weighted loss.
    - SUPPORT path (need_grads=True): returns scalar loss and fills `grad_buffer` with
      *weighted* grads for the tracked params; optionally updates the Fisher from UNWEIGHTED grads.
    - QUERY path (need_grads=False): returns scalar loss only; does NOT update the Fisher.

    This function does exactly ONE forward pass. In per-sample mode it uses TWO autograd.grad
    calls on the same graph (retain first) to get exact Σ w_i ∇ℓ_i without per-sample grad tensors.
    """
    need_grads = bool(grad_buffer) or bool(update_fisher)

    # ---- forward once ----
    gt_rgb = data["rgbs"]
    rays = data["rays"]
    pred, *_ = render_rays(
        model,
        rays,
        ray_samples=P.ray_samples,
        params=params,
        active_module=active_module,
        chunk=P.chunk_points,
    )
    pred, gt_rgb = color_space_transformer(pred, gt_rgb, color_space=P.color_space)

    # per-sample and batch losses
    mse_i = F.mse_loss(pred, gt_rgb, reduction="none").mean(dim=-1)  # (B,)
    base_loss = mse_i.mean()  # scalar

    expert_module = model.submodules[active_module]
    # resolve submodule; if no FIM head available, fall back to base loss
    if not hasattr(model, "fisher_store") or not hasattr(model, "fim_loss"):
        return base_loss
    fs = expert_module.fisher_store
    if not fs.tracked:
        return base_loss

    tracked_names, tracked_tensors = zip(*fs.tracked)
    tracked_tensors = list(tracked_tensors)

    algo = str(getattr(P, "algo", "")).lower()
    first_order = algo in ("fomaml", "reptile")
    per_sample = bool(getattr(P, "fim_per_sample", False))

    # Need to keep graph alive if:
    #  - QUERY path (outer backward still to run), or
    #  - SUPPORT+per-sample (a second grad call follows).
    retain_first = (not need_grads) or per_sample

    grad_base = torch.autograd.grad(
        base_loss,
        tracked_tensors,
        create_graph=not first_order,
        allow_unused=True,
        retain_graph=retain_first,
    )
    grad_dict = {
        name: g.detach() for (name, g) in zip(tracked_names, grad_base) if g is not None
    }

    # ---- compute weights and loss ----
    if per_sample:
        w_i = expert_module.fim_loss.fim_weight(
            grad_dict,
            mse_i=mse_i,
            per_sample=True,
            clamp=(1 / clamp_factor, clamp_factor),
        )  # (B,)
        fim_loss = (w_i.detach() * mse_i).mean()

        if not need_grads:
            # QUERY: return scalar; no Fisher update, no inner grads
            return fim_loss

        # SUPPORT: exact Σ_i w_i ∇ℓ_i via a second grad (same forward)
        grad_weighted = torch.autograd.grad(
            fim_loss,
            tracked_tensors,
            create_graph=not first_order,
            allow_unused=True,
            retain_graph=False,
        )

    else:
        # batch-scalar weight
        w = expert_module.fim_loss.fim_weight(
            grad_dict, per_sample=False, clamp=(1 / clamp_factor, clamp_factor)
        )
        fim_loss = w.detach() * base_loss

        if not need_grads:
            # QUERY: return scalar; no Fisher update, no inner grads
            return fim_loss

        wd = w.detach()
        grad_weighted = [None if g is None else (wd * g) for g in grad_base]

    if update_fisher and grad_dict:
        with torch.no_grad():
            fs.update_from_grads({k: v.pow(2) for k, v in grad_dict.items()})

    if grad_buffer is not None:
        for name, g in zip(tracked_names, grad_weighted):
            if g is not None:
                grad_buffer[name] = g

    return fim_loss


def compute_loss(
    P,
    model,
    data,
    params=None,
    active_module=None,
    **kwargs,
):
    """Dispatcher for loss computation."""
    if P.fim:
        return compute_fim_loss(P, model, data, params, active_module, **kwargs)
    else:
        return compute_mse_loss(P, model, data, params, active_module)
