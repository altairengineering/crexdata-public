from collections import OrderedDict
from typing import List
import numpy as np

import torch
from torch.amp import autocast

from nerfs.losses import compute_loss


# =============================================================================
# Task adaptation
# =============================================================================
def task_adapt(
    P,
    model,
    support,
    inner_lr,
    iterations,
    active_module=None,
):
    algo = str(getattr(P, "algo", "")).lower()
    first_order = algo in ("fomaml", "reptile")

    # init fast ONCE
    base = model.submodules[active_module] if active_module is not None else model
    fast = extract_module_params(base, copy=(algo == "reptile"))
    inner_losses = []

    amp_enabled = (
        bool(getattr(P, "use_amp", True)) and torch.cuda.is_available() and first_order
    )

    for _ in range(int(iterations)):
        grad_buf = {}

        # Forward (builds graph) under autocast(fp16) if enabled
        with autocast(device_type="cuda", enabled=amp_enabled, dtype=torch.float16):
            loss = compute_loss(
                P,
                model,
                support,
                params=fast,
                active_module=active_module,
                grad_buffer=grad_buf,
                update_fisher=True,
            )

        if grad_buf:
            # FIM provided ready-weighted grads (dict keyed by param name)
            grads = (grad_buf.get(n, None) for n in fast.keys())
        else:
            # Plain MSE: grads for fast params
            grads = torch.autograd.grad(
                loss,
                tuple(fast.values()),
                create_graph=not first_order,  # False for FoMAML
                allow_unused=True,
            )

        fast = OrderedDict(
            (n, w if g is None else (w - inner_lr * g.to(w.dtype)))
            for (n, w), g in zip(fast.items(), grads)
        )

        inner_losses.append(loss.detach())

    return fast, inner_losses


# =============================================================================
# Meta update
# =============================================================================
def meta_update(
    P,
    model,
    optimizer,
    loss_out: torch.Tensor,  # maml
    scheduler=None,
    grad_scaler=None,  # maml
    fast_list: List[OrderedDict] = None,  # reptile
):
    """
    Unified outer/meta update (single or batched).

    Args:
        fast_list: list of fast params (Reptile)
        loss_out:  scalar query loss (MAML/FOMAML)
    """
    algo = P.algo.lower()
    if algo in ("maml", "fomaml"):
        maml_meta_update(
            optimizer,
            loss_out,
            scaler=grad_scaler,
            grad_clip=getattr(P, "grad_clip", 1.0),
        )
    elif algo == "reptile":
        reptile_meta_update(
            P,
            model,
            fast_list=fast_list,
        )
    else:
        raise ValueError(f"Unsupported algo {algo!r}")

    with torch.no_grad():
        for cid, expert in enumerate(model.submodules):
            total_norm = 0.0
            for p in expert.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
            total_norm = total_norm**0.5
            print(f"debug/outer_grad_norm_region_{cid}: ", total_norm)

    if scheduler is not None:
        scheduler.step()

    lrs = [g["lr"] for g in optimizer.param_groups]
    print(f"group LRs = {lrs}")


def maml_meta_update(optimizer, loss_out, scaler=None, grad_clip=1.0):
    if not torch.isfinite(loss_out):
        print(f"[WARN] Skipping meta-update: non-finite loss_out={loss_out.item()}")
        return

    optimizer.zero_grad(set_to_none=True)
    use_amp = scaler is not None and getattr(scaler, "is_enabled", lambda: False)()
    if use_amp:
        # AMP branch: scale loss, backward, unscale, clip, step via scaler
        scaler.scale(loss_out).backward()
        scaler.unscale_(optimizer)  # make grads real for clipping
        clip_all_grads(optimizer, grad_clip)
        scaler.step(optimizer)  # IMPORTANT: use scaler.step
        scaler.update()
    else:
        # FP32 branch
        loss_out.backward()
        clip_all_grads(optimizer, grad_clip)
        optimizer.step()


@torch.no_grad()
def reptile_meta_update(P, model, fast_list):
    """
    Batched Reptile update (Algorithm 2):
        Δ̄ = (1/n) Σ_i (W_i - θ)
        θ ← θ + lr * Δ̄
    """
    if len(fast_list) == 0:
        raise ValueError("Reptile update called with empty fast_list")

    # snapshot θ at batch start (the current shared weights)
    theta = snapshot_params(model)

    # accumulate mean delta
    sum_delta = {k: torch.zeros_like(v) for k, v in theta.items()}
    for fast in fast_list:
        for k, v in fast.items():
            if k in sum_delta:
                sum_delta[k].add_(v.detach() - theta[k])

    n = float(len(fast_list))
    updated_names = []
    for name, p in model.meta_named_parameters():
        if name in sum_delta:
            delta = sum_delta[name] / n
            if torch.isfinite(delta).all() and delta.abs().sum() > 0:
                p.add_(P.lr * delta)
                updated_names.append(name)

    # Debug message: which parameters were updated
    print(
        "Reptile meta-update: updated %d parameter tensors: %s",
        len(updated_names),
        ", ".join(updated_names) if updated_names else "<none>",
    )


def clip_all_grads(optimizer, grad_clip=1.0):
    if grad_clip is None:
        return
    params = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p.grad is not None:
                params.append(p)
    if params:
        torch.nn.utils.clip_grad_norm_(params, grad_clip)


# =============================================================================
# Model helpers
# =============================================================================
def extract_module_params(submodule, copy=True) -> OrderedDict:
    """Snapshot submodule[cid] leaf params → OrderedDict(name->Tensor, requires_grad=True)."""
    # reptile case
    if copy:
        return OrderedDict(
            (n, p.detach().clone().requires_grad_(True))
            for n, p in submodule.meta_named_parameters()
        )
    # Maml based parameter injection
    return OrderedDict((n, p) for n, p in submodule.meta_named_parameters())


def snapshot_params(model):
    """Return a {name: tensor} snapshot of meta-parameters θ (detached)."""
    return {n: p.detach().clone() for n, p in model.meta_named_parameters()}


def snapshot_model_dict(model):
    """Return a state_dict snapshot of meta-parameters θ (detached)."""
    return {n: p.detach().clone() for n, p in model.state_dict().items()}


# =============================================================================
# Debug Helpers
# =============================================================================


def compare_params(p_before, p_after):
    """
    Compute mean absolute difference between two param dicts.
    Returns:
      mean_abs_diff (float),
      per_param_abs_mean (dict[name]->float)
    """
    diffs = {n: (p_after[n] - p_before[n]).abs().mean().item() for n in p_before.keys()}
    mean_diff = sum(diffs.values()) / max(1, len(diffs))
    return mean_diff, diffs


def analyze_grads(grads, fast=None, topk=100, name="inner_loop"):
    """
    Analyze gradient magnitudes for debugging.

    Args:
        grads (Iterable[Tensor]): Gradients from autograd.grad.
        fast (OrderedDict, optional): Corresponding parameters (for naming + relative scales).
        topk (int): Number of layers to show in sorted summaries.
        name (str): Optional label for logging context.

    Returns:
        global_norm (float): Overall L2 norm of all grads.
    """
    grad_info = []
    total_norm_sq = 0.0
    eps = 1e-12

    for i, g in enumerate(grads):
        if g is None:
            continue

        gn = g.norm().item()
        total_norm_sq += gn**2

        layer_name = list(fast.keys())[i] if fast is not None else f"param_{i}"
        param_norm = fast[layer_name].norm().item() if fast is not None else None
        rel_scale = gn / (param_norm + eps) if param_norm is not None else None

        grad_info.append(
            {
                "name": layer_name,
                "grad_norm": gn,
                "param_norm": param_norm,
                "rel_scale": rel_scale,
                "mean": g.mean().item(),
                "std": g.std().item(),
                "max": g.abs().max().item(),
            }
        )

    if not grad_info:
        print(f"[{name}] No valid gradients found.")
        return 0.0

    global_norm = np.sqrt(total_norm_sq)

    # Print global summary
    print(f"\n[{name}] Gradient Summary:")
    print(f"  Global grad norm: {global_norm:.3e}")
    print(f"  Mean grad norm:   {np.mean([x['grad_norm'] for x in grad_info]):.3e}")
    print(f"  Max grad norm:    {np.max([x['grad_norm'] for x in grad_info]):.3e}")

    # Show top-k layers
    sorted_info = sorted(grad_info, key=lambda x: x["grad_norm"], reverse=True)
    print(f"\n  Top-{topk} layers by grad norm:")
    for x in sorted_info[:topk]:
        rel = f"(rel={x['rel_scale']:.2e})" if x["rel_scale"] is not None else ""
        print(f"   {x['name']:<40} | grad={x['grad_norm']:.3e} {rel}")

    return global_norm
