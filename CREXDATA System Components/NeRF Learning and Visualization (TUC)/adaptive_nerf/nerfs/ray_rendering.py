from typing import Optional, Tuple
import warnings

warnings.simplefilter("once", category=UserWarning)

import torch
from torch import Tensor
import torch.nn.functional as F
import nerfacc

from nerfs.ray_sampling import (
    get_ray_directions,
    get_rays,
    clamp_rays_near_far,
)
from models.trunc_exp import trunc_exp
from nerfs.scene_box import SceneBox


# ============================== BG helpers ===============================


def _get_bg_rgb(
    model, dirs: torch.Tensor, params, rgb_sigma_or_map, N: int, bg_color_default: str
) -> Optional[torch.Tensor]:
    """
    Return background RGB for compositing.

    If the model exposes a background head (`use_bg_nerf == True`), call it with the
    (N,3) ray directions. Otherwise, fall back to a deterministic default color.

    Args:
        model: Container or expert. May expose `use_bg_nerf` and `.background_color`.
        dirs: (N,3) normalized ray directions.
        params: Optional fast-weights/params.
        rgb_sigma_or_map: Used only to copy dtype/device when falling back (can be None).
        N: number of rays.
        bg_color_default: "white"|"black"|"random"|"last_sample"|"none".

    Returns:
        (N,3) tensor in [0,1] or None when bg_color_default="none".
    """
    if getattr(model, "use_bg_nerf", False):
        return model.background_color(dirs)  # (N,3) in [0,1]
    return get_bg_default_color(rgb_sigma_or_map, N, bg_color_default)


def get_bg_default_color(
    rgb_sigma, N: int, bg_color: str = "white"
) -> Optional[torch.Tensor]:
    """
    Deterministic fallback background color.

    Args:
        rgb_sigma: used to carry device/dtype; can be (N,S,4), (N,3), or None.
        N: number of rays.
        bg_color: "white"|"black"|"random"|"last_sample"|"none".

    Returns:
        (N,3) RGB in [0,1], or None if bg_color == "none".
    """
    device = None if rgb_sigma is None else rgb_sigma.device
    dtype = None if rgb_sigma is None else rgb_sigma.dtype

    if bg_color == "none":
        return None
    if bg_color == "white":
        return torch.ones(N, 3, device=device, dtype=dtype)
    if bg_color == "black":
        return torch.zeros(N, 3, device=device, dtype=dtype)
    if bg_color == "random":
        return torch.rand(N, 3, device=device, dtype=dtype)
    if bg_color == "last_sample":
        if rgb_sigma is None or rgb_sigma.dim() != 3 or rgb_sigma.size(-1) < 3:
            raise ValueError(
                "bg_color='last_sample' requires rgb_sigma of shape (N,S,4) or (N,S,>=3)."
            )
        return rgb_sigma[:, -1, :3]
    raise ValueError(f"Unknown background policy: {bg_color}")


def apply_bg_mask(
    rgb_lin: torch.Tensor, mask_invalid: torch.Tensor, policy: str
) -> None:
    """
    In-place BG fill for invalid rays (alpha≈0), typically for visualizers after compositing.

    Args:
        rgb_lin: (N,3), will be modified in-place.
        mask_invalid: (N,) bool mask.
        policy: "white"|"black"|"random"|"none"|"last_sample" (treated as "none" here).
    """
    if not mask_invalid.any():
        return
    policy = str(policy).lower()
    if policy == "white":
        rgb_lin[mask_invalid] = 1.0
    elif policy == "black":
        rgb_lin[mask_invalid] = 0.0
    elif policy == "random":
        n = int(mask_invalid.sum().item())
        rgb_lin[mask_invalid] = torch.rand(
            n, 3, device=rgb_lin.device, dtype=rgb_lin.dtype
        )
    elif policy in ("none", "last_sample"):
        pass  # leave as-is
    else:
        rgb_lin[mask_invalid] = 1.0  # fallback


# ============================== Core volume rendering ===============================


def volume_render(
    rgb_sigma: torch.Tensor,  # (N,S,4) with [rgb(0..1), sigma>=0] unless raw_* True
    t_vals: torch.Tensor,  # (N,S)
    bg_rgb: Optional[torch.Tensor] = None,  # (N,3) or None
    *,
    raw_rgb: bool = False,  # True if rgb are logits; apply sigmoid here
    raw_sigma: bool = False,  # True if sigma are logits; apply trunc_exp here
    sigma_scale: float = 1.0,  # Optional σ scale (e.g., >1.0 if under-dense)
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Standard NeRF compositing using discrete transmittance integration.

    - Computes per-interval distances Δ, alphas α = 1 - exp(-σ⋅Δ), transmittance T, and weights w = α⋅T.
    - Composite color, depth (expected t), and accumulated opacity.
    - Optionally composites a background color for the remaining transmittance.

    Returns:
        rgb_map: (N,3)
        depth_map: (N,)
        weights: (N,S)
        acc_map: (N,)
    """
    rgb_raw = rgb_sigma[..., :3]
    sigma_in = rgb_sigma[..., 3]

    rgb = torch.sigmoid(rgb_raw) if raw_rgb else rgb_raw.clamp(0.0, 1.0)
    sigma = trunc_exp(sigma_in) if raw_sigma else sigma_in.clamp_min(0.0)
    if sigma_scale != 1.0:
        sigma = sigma * float(sigma_scale)

    # Δ distances
    dists = (t_vals[:, 1:] - t_vals[:, :-1]).clamp_min(1e-4)
    dists = torch.cat([dists, dists[:, -1:]], dim=1)  # finite last Δ

    # α, T, weights
    alpha = (1.0 - torch.exp(-sigma * dists)).clamp_(0.0, 1.0 - 1e-7)
    T = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=1), dim=1
    )[:, :-1]
    weights = alpha * T

    # composites
    rgb_map = (weights.unsqueeze(-1) * rgb).sum(dim=1)
    depth_map = (weights * t_vals).sum(dim=1)
    acc_map = weights.sum(dim=1)

    if bg_rgb is not None:
        bg_rgb = bg_rgb.to(rgb_map.device, dtype=rgb_map.dtype, non_blocking=True)
        rgb_map = rgb_map + (1.0 - acc_map.unsqueeze(-1)) * bg_rgb

    return rgb_map, depth_map, weights, acc_map


# ============================== Helpers ===============================


@torch.no_grad()
def _intersect_rays_aabb(rays: Tensor, scene_box: SceneBox) -> Tensor:
    """
    Slab test to prefilter rays that can hit an AABB.

    Args:
        rays: (N,8) [o(3), d(3), near, far]
        xyz_min, xyz_max: (3,) expert AABB corners.

    Returns:
        (N,) bool mask where intersection with [near,far] is non-empty.
    """
    o, d = rays[:, :3], rays[:, 3:6]
    near, far = rays[:, 6:7], rays[:, 7:8]
    eps = 1e-9
    invd = torch.where(torch.abs(d) > eps, 1.0 / d, torch.full_like(d, 1.0 / eps))
    t0 = (scene_box.min[None, :] - o) * invd
    t1 = (scene_box.max[None, :] - o) * invd
    tmin = torch.minimum(t0, t1).amax(dim=-1, keepdim=True)
    tmax = torch.maximum(t0, t1).amin(dim=-1, keepdim=True)
    t_enter = torch.maximum(tmin, near)
    t_exit = torch.minimum(tmax, far)
    return (t_exit > t_enter).squeeze(-1)


@torch.no_grad()
def _merge_segments_union(ray_indices_list, t0_list, t1_list):
    """
    Merge per-expert segments per ray into a single partition (boundary union).

    Why: each expert proposes [t0,t1) segments independently; to avoid double-counting
    opacity we must integrate once over a **unified** set of intervals per ray.

    Inputs:
        ray_indices_list: List[Tensor(M_k,)] per expert, global ray ids.
        t0_list, t1_list: List[Tensor(M_k,)] per expert.
        n_rays: total number of rays N (for shape metadata only).

    Returns:
        merged_ri: (M,) global ray ids for merged segments
        merged_t0: (M,)
        merged_t1: (M,)
    """
    if len(t0_list) == 0:
        device = torch.device("cpu")
        return (
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
        )

    ri = torch.cat(ray_indices_list, 0)
    t0 = torch.cat(t0_list, 0)
    t1 = torch.cat(t1_list, 0)

    # sort by (ray, start)
    order = torch.argsort(ri * 1e9 + t0)  # lexsort fallback
    ri, t0, t1 = ri[order], t0[order], t1[order]

    # find ray boundaries
    changes = torch.ones_like(ri, dtype=torch.bool)
    changes[1:] = ri[1:] != ri[:-1]
    starts = torch.nonzero(changes, as_tuple=False).squeeze(1)
    starts = torch.cat([starts, torch.tensor([ri.numel()], device=ri.device)])

    out_ri, out_t0, out_t1 = [], [], []
    for a, b in zip(starts[:-1].tolist(), starts[1:].tolist()):
        rid = int(ri[a].item())
        bounds = torch.cat([t0[a:b], t1[a:b]], dim=0)
        bnd = torch.unique(bounds, sorted=True)
        if bnd.numel() < 2:
            continue
        out_ri.append(
            torch.full((bnd.numel() - 1,), rid, dtype=torch.long, device=ri.device)
        )
        out_t0.append(bnd[:-1])
        out_t1.append(bnd[1:])

    if len(out_t0) == 0:
        device = ri.device
        return (
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
            torch.zeros(0, dtype=torch.float32, device=device),
        )
    return torch.cat(out_ri, 0), torch.cat(out_t0, 0), torch.cat(out_t1, 0)


# ============================== Stratified rendering (baseline) ===============================


@torch.no_grad()
def stratified_t_vals(
    near: Tensor, far: Tensor, ray_samples: int, randomized: bool = True
) -> Tensor:
    """
    Uniformly sample S points per ray in [near,far], with optional stratified jitter for training.

    Args:
        near: (N,) per-ray near plane.
        far : (N,) per-ray far plane.
        ray_samples: S samples per ray.
        randomized: if True, jitter each interval (training stratification).

    Returns:
        t_vals: (N,S) sampled distances along each ray.
    """
    t_lin = torch.linspace(0.0, 1.0, ray_samples, device=near.device).unsqueeze(
        0
    )  # (1,S)
    t_vals = near.unsqueeze(1) * (1.0 - t_lin) + far.unsqueeze(1) * t_lin
    if randomized:
        mids = 0.5 * (t_vals[:, :-1] + t_vals[:, 1:])
        low = torch.cat([t_vals[:, :1], mids], dim=1)
        high = torch.cat([mids, t_vals[:, -1:]], dim=1)
        t_vals = low + (high - low) * torch.rand_like(low)
    return t_vals


def render_rays_stratified(
    model,
    rays: Tensor,
    ray_samples: int,
    params=None,
    active_module: Optional[int] = None,
    bg_color_default: str = "white",
    chunk: int = 1_000_000,
    sigma_scale=1.0,
    **kwargs,
):
    """
    Stratified renderer (no occupancy). Good for warmup.

    - Samples S uniform depths per ray.
    - Builds (N*S,6) [xyz,dir] and queries either a single expert (active_module) or the container (soft blend).
    - Standard volume rendering with optional background compositing.

    Returns:
        rgb_map: (N,3), depth_map: (N,), weights: (N,S), acc_map: (N,)
    """
    o, d = rays[:, :3], rays[:, 3:6]
    near, far = rays[:, 6], rays[:, 7]

    t_vals = stratified_t_vals(
        near, far, ray_samples, randomized=model.training
    )  # (N,S)
    pts = o.unsqueeze(1) + d.unsqueeze(1) * t_vals.unsqueeze(-1)  # (N,S,3)
    dirs = d.unsqueeze(1).expand_as(pts)  # (N,S,3)
    id6 = torch.cat([pts, dirs], dim=-1).reshape(-1, 6)  # (N*S,6)

    outs = []

    model_eff = model.submodules[active_module] if active_module is not None else model
    for start in range(0, id6.shape[0], chunk):
        outs.append(model_eff(id6[start : start + chunk], params=params))  # (m,4)
    rgb_sigma = torch.cat(outs, dim=0).view(pts.shape[0], pts.shape[1], 4)  # (N,S,4)

    bg_rgb = _get_bg_rgb(
        model,
        dirs[:, 0],
        params,
        rgb_sigma,
        N=rgb_sigma.size(0),
        bg_color_default=bg_color_default,
    )
    rgb_map, depth_map, weights, acc_map = volume_render(
        rgb_sigma,
        t_vals,
        bg_rgb=bg_rgb,
        raw_rgb=False,
        raw_sigma=False,
        sigma_scale=sigma_scale,
    )

    return rgb_map, depth_map, weights, acc_map


# ============================== Occupancy rendering (Soft MoE) ===============================
def render_rays_occ(
    model,
    rays: Tensor,  # (N,8)
    *,
    params=None,
    bg_color_default: str = "white",
    chunk: int = 1_000_000,
    render_step_size=None,
    alpha_thre=None,
    cone_angle=None,
    active_module: Optional[int] = None,
    **kwargs,
):
    """
    Occupancy-guided **soft Mixture-of-Experts** renderer.

    - If `model` is a container and `active_module is not None`: render only that expert
      (for per-expert training).
    - Else (container full render): for each expert:
        (a) AABB prefilter rays, (b) per-expert occupancy marching,
        (c) per-ray **union of segments** across experts, (d) evaluate experts **only at
        midpoints where their soft routing weight > 0**, (e) **blend σ and rgb BEFORE**
        computing weights, (f) **single** nerfacc integration. This preserves soft-boundary
        blending without opacity double counting.

    Returns:
        rgb_map: (N,3)
        depth:   (N,)
        weights: (M,1) packed sample weights
        acc:     (N,)
    """
    N = rays.shape[0]
    o, d = rays[:, :3], rays[:, 3:6]

    # ---------- Container with optional active expert ----------
    if active_module is not None:
        sub = model.submodules[active_module]
        return render_expert_occ(
            sub,
            rays,
            params=params,
            bg_color_default=bg_color_default,
            chunk=chunk,
            render_step_size=render_step_size,
            alpha_thre=alpha_thre,
            cone_angle=cone_angle,
        )

    # ---------- Full container: per-expert marching + union + soft blend ----------
    per_ray_idx, per_t0, per_t1 = [], [], []
    for k, expert in enumerate(model.submodules):
        hit = _intersect_rays_aabb(rays, scene_box=expert.scene_box)
        if not hit.any():
            continue
        rays_k = rays[hit]
        ri_k, t0_k, t1_k = expert.occupancy_marching(
            rays_k,
            params=(
                model.get_subdict(params, f"submodules.{k}")
                if params is not None
                else None
            ),
            render_step_size=render_step_size,
            alpha_thre=alpha_thre,
            cone_angle=cone_angle,
        )
        if t0_k.numel() == 0:
            continue
        hit_idx = hit.nonzero(as_tuple=False).squeeze(1)
        per_ray_idx.append(hit_idx[ri_k])
        per_t0.append(t0_k)
        per_t1.append(t1_k)

    if len(per_t0) == 0:
        acc = rays.new_zeros(N)
        bg_rgb = _get_bg_rgb(model, d, params, None, N, bg_color_default)
        depth = acc.clone()
        weights = torch.zeros(1, 1, device=rays.device, dtype=rays.dtype)
        return bg_rgb, depth, weights, acc

    merged_ri, merged_t0, merged_t1 = _merge_segments_union(per_ray_idx, per_t0, per_t1)
    M = merged_t0.numel()
    t_mid = 0.5 * (merged_t0 + merged_t1)
    x_mid = o[merged_ri] + d[merged_ri] * t_mid[:, None]
    d_mid = d[merged_ri]

    # soft routing weights at midpoints
    with torch.no_grad():
        W, _ = model._routing(x_mid.view(1, -1, 3))
        if W is None:
            _, hard = model._routing(x_mid.view(1, -1, 3))
            K = len(model.submodules)
            W = x_mid.new_zeros(1, M, K)
            W[0, torch.arange(M), hard.view(-1)] = 1.0
        W = W.squeeze(0)  # (M,K)

    # evaluate only where W[:,k] > 0 and blend BEFORE weights
    K = len(model.submodules)
    SIG = x_mid.new_zeros(M, K)  # σ_k(x_mid)
    RGB = x_mid.new_zeros(M, K, 3)  # c_k(x_mid)

    eps = 1e-8
    params_k = [
        model.get_subdict(params, f"submodules.{k}") if params is not None else None
        for k in range(K)
    ]
    for k, expert in enumerate(model.submodules):
        mask = W[:, k] > eps
        if not mask.any():
            continue
        idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        xb, db = x_mid[idx], d_mid[idx]
        p_k = params_k[k]

        sig_l, rgb_l = [], []
        for s in range(0, idx.numel(), chunk):
            xd = torch.cat([xb[s : s + chunk], db[s : s + chunk]], dim=-1)  # (m,6)
            out = expert(xd, params=p_k)  # (m,4)
            rgb_l.append(out[..., :3])
            sig_l.append(out[..., 3])

        SIG[idx, k] = torch.cat(sig_l, 0)
        RGB[idx, k] = torch.cat(rgb_l, 0)

    s_num = (W * SIG).sum(dim=1, keepdim=True).clamp_min(1e-12)  # (M,1)
    sigma_mix = s_num.squeeze(1)  # (M,)
    rgb_mix = (W[..., None] * SIG[..., None] * RGB).sum(dim=1) / s_num  # (M,3)

    packed = nerfacc.pack_info(merged_ri, N)
    weights = nerfacc.render_weight_from_density(
        t_starts=merged_t0, t_ends=merged_t1, sigmas=sigma_mix, packed_info=packed
    )[0][..., None]

    weights_1d = weights.squeeze(-1)  # (M,)
    rgb_map = nerfacc.accumulate_along_rays(weights_1d, rgb_mix, merged_ri, N)
    depth = nerfacc.accumulate_along_rays(
        weights_1d, t_mid[:, None], merged_ri, N
    ).squeeze(-1)
    acc = nerfacc.accumulate_along_rays(weights_1d, None, merged_ri, N).squeeze(-1)

    bg_rgb = _get_bg_rgb(model, d, params, rgb_map, N, bg_color_default)
    rgb_map = rgb_map + (1.0 - acc)[..., None] * bg_rgb
    return rgb_map, depth, weights, acc


def render_expert_occ(
    model,
    rays: Tensor,  # (N,8)
    *,
    params=None,
    bg_color_default: str = "white",
    chunk: int = 1_000_000,
    render_step_size=None,
    alpha_thre=None,
    cone_angle=None,
):
    N = rays.shape[0]
    o, d = rays[:, :3], rays[:, 3:6]

    # ---------- Single expert path ----------
    if getattr(model, "occ_grid", None) is not None:
        ray_indices, t_starts, t_ends = model.occupancy_marching(
            rays,
            params=params,
            render_step_size=render_step_size,
            alpha_thre=alpha_thre,
            cone_angle=cone_angle,
        )
        if t_starts.numel() == 0:
            acc = rays.new_zeros(N)
            bg_rgb = _get_bg_rgb(model, d, params, None, N, bg_color_default)
            depth = acc.clone()
            weights = torch.zeros(1, 1, device=rays.device, dtype=rays.dtype)
            return bg_rgb, depth, weights, acc

        # ensure 1D (robust to accidental 0-D tensors)
        ray_indices = torch.atleast_1d(ray_indices)
        t_starts = torch.atleast_1d(t_starts)
        t_ends = torch.atleast_1d(t_ends)

        t_mid = 0.5 * (t_starts + t_ends)
        x = o[ray_indices] + d[ray_indices] * t_mid[:, None]
        ds = d[ray_indices]

        sigmas, rgbs = [], []
        for s in range(0, x.shape[0], chunk):
            xd = torch.cat([x[s : s + chunk], ds[s : s + chunk]], dim=-1)  # (m, 6)
            out = model(xd, params=params)  # (m, 4)
            rgbs.append(out[..., :3])
            sigmas.append(out[..., 3])

        rgb = torch.cat(rgbs, 0)
        sigma = torch.cat(sigmas, 0)

        packed = nerfacc.pack_info(ray_indices, N)
        weights = nerfacc.render_weight_from_density(
            t_starts=t_starts, t_ends=t_ends, sigmas=sigma, packed_info=packed
        )[0][..., None]
        weights_1d = weights.squeeze(-1)  # (M,)
        rgb_map = nerfacc.accumulate_along_rays(weights_1d, rgb, ray_indices, N)
        depth = nerfacc.accumulate_along_rays(
            weights_1d, t_mid[:, None], ray_indices, N
        ).squeeze(-1)
        acc = nerfacc.accumulate_along_rays(weights_1d, None, ray_indices, N).squeeze(
            -1
        )

        bg_rgb = _get_bg_rgb(model, d, params, rgb_map, N, bg_color_default)
        rgb_map = rgb_map + (1.0 - acc)[..., None] * bg_rgb
        return rgb_map, depth, weights, acc


# ============================== Rendering Entrypoints ===============================


def render_rays(model, rays, *args, **kwargs):
    if model.use_occ:
        if not model.occ_ready:
            return render_rays_stratified(model, rays, *args, **kwargs)
        # Cheap boolean to silence per-call warnings in hot path
        if getattr(model, "warned_occ_ready", False) is False:
            warnings.warn("[OCC] Using nerfacc occupancy renderer (warmup concluded).")
            model.warned_occ_ready = True
        return render_rays_occ(model, rays, *args, **kwargs)
    else:
        return render_rays_stratified(model, rays, *args, **kwargs)


@torch.no_grad()
def render_image(
    model,
    *,
    H: int,
    W: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    c2w: Tensor,  # (3,4) or (4,4)
    scene_box,  # passed to get_rays
    params=None,  # fast weights (OrderedDict) or None
    active_module: Optional[int] = None,
    ray_samples: int = 64,
    chunk_points: int = 1 << 16,
    bg_color_default: str = "white",
    center_pixels: bool = True,
    use_amp: bool = False,
) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """
    Convenience utility to render a full image.

    - Builds per-pixel rays from intrinsics + pose.
    - Uses stratified renderer (or occupancy renderer automatically via `render_rays`).
    - Returns linear RGB in [0,1], and flattened depth/acc maps if available.
    """
    device = next(model.parameters()).device
    dirs = get_ray_directions(
        H, W, fx, fy, cx, cy, center_pixels=center_pixels, device=device
    )
    rays = get_rays(dirs, c2w.to(device), scene_box=scene_box).view(-1, 8)
    rays, _ = clamp_rays_near_far(rays, near_far_override=(None, None))

    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
        rgb_lin, depth, _, acc = render_rays(
            model,
            rays,
            ray_samples=ray_samples,
            params=params,
            active_module=active_module,
            bg_color_default=bg_color_default,
            chunk=chunk_points,
        )

    rgb_lin = rgb_lin.view(H, W, 3).float().clamp_(0, 1)
    return (
        rgb_lin,
        (None if depth is None else depth.view(-1)),
        (None if acc is None else acc.view(-1)),
    )


# ============================== Debug helper ===============================


def _tstats(x: torch.Tensor, name: str):
    """
    Quick tensor stats/health check for debugging numerical issues.
    """
    with torch.no_grad():
        x = x.detach()
        finite = torch.isfinite(x)
        n, n_finite = x.numel(), finite.sum().item()
        n_nan, n_inf = torch.isnan(x).sum().item(), torch.isinf(x).sum().item()
        x_f = x[finite]
        if x_f.numel() > 0:
            print(
                f"[{name}] shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
                f"min={float(x_f.min()):.4g} max={float(x_f.max()):.4g} "
                f"mean={float(x_f.mean()):.4g} std={float(x_f.std()):.4g} "
                f"finite={n_finite}/{n} nan={n_nan} inf={n_inf}"
            )
        else:
            print(
                f"[{name}] shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
                f"ALL NON-FINITE. n={n} nan={n_nan} inf={n_inf}"
            )
