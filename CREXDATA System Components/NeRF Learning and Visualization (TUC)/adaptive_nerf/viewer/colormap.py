from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass(frozen=True)
class ColormapOptions:
    colormap: str = "default"
    normalize: bool = False
    colormap_min: float = 0.0
    colormap_max: float = 1.0
    invert: bool = False


@torch.no_grad()
def apply_float_colormap(image: Tensor, colormap: str = "default") -> Tensor:
    import matplotlib

    if colormap == "default":
        colormap = "turbo"
    image = torch.nan_to_num(image, 0.0).clamp(0, 1)
    if colormap == "gray":
        return image.repeat_interleave(3, dim=-1)
    table = torch.tensor(
        matplotlib.colormaps[colormap].colors, device=image.device, dtype=torch.float32
    )  # (256,3)
    idx = (image * 255.0).long().clamp_(0, 255)[..., 0]
    return table[idx]


@torch.no_grad()
def apply_colormap(
    image: Tensor,
    colormap_options: ColormapOptions = ColormapOptions(),
    eps: float = 1e-9,
) -> Tensor:
    if image.shape[-1] == 3 and torch.is_floating_point(image):
        return image
    if image.dtype == torch.bool:
        out = torch.zeros(
            *image.shape[:-1], 3, device=image.device, dtype=torch.float32
        )
        out[image[..., 0]] = 1.0
        return out
    if image.shape[-1] > 3:
        return apply_pca_colormap(image)

    x = image
    if colormap_options.normalize:
        x = x - torch.min(x)
        x = x / torch.max(x).clamp_min_(eps)
    rng = float(colormap_options.colormap_max - colormap_options.colormap_min)
    x = x * rng + float(colormap_options.colormap_min)
    x = x.clamp_(0.0, 1.0)
    if colormap_options.invert:
        x = 1.0 - x
    return apply_float_colormap(x, colormap=colormap_options.colormap)


@torch.no_grad()
def apply_depth_colormap(
    depth: Tensor,  # (...,1)
    accumulation: Optional[Tensor] = None,  # (...,1) or (...,)
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    colormap_options: ColormapOptions = ColormapOptions(),
    *,
    use_acc_for_range: bool = True,  # mask near/far by acc>0 (no info loss)
    acc_eps: float = 1e-6,
) -> Tensor:
    """Depth -> RGB without percentile clipping. Range = finite (and optionally acc>0)."""

    d = depth[..., 0]  # (...,)

    valid = torch.isfinite(d)
    if use_acc_for_range and (accumulation is not None):
        acc = accumulation
        if acc.ndim == d.ndim:
            pass
        else:
            acc = acc.squeeze(-1)
        valid = valid & (acc > acc_eps)

    if valid.any():
        near = float(d[valid].min().item()) if near_plane is None else float(near_plane)
        far = float(d[valid].max().item()) if far_plane is None else float(far_plane)
        if far <= near:
            far = near + 1e-6
    else:
        # fallback; keep behavior defined
        near, far = (
            0.0 if near_plane is None else float(near_plane),
            1.0 if far_plane is None else float(far_plane),
        )
        if far <= near:
            far = near + 1e-6

    x = ((d - near) / (far - near)).clamp_(0.0, 1.0).unsqueeze(-1)  # (...,1)
    if colormap_options.invert:
        x = 1.0 - x

    colored = apply_colormap(
        x,
        colormap_options=ColormapOptions(
            colormap=colormap_options.colormap,
            normalize=False,  # already normalized
            invert=False,
        ),
    )  # (...,3)

    if accumulation is not None:
        acc = accumulation.to(colored.device, colored.dtype)
        if acc.ndim == colored.ndim - 1:
            acc = acc.unsqueeze(-1)  # (...,1)
        elif acc.shape[-1] != 1:
            acc = acc[..., :1]
        acc = torch.nan_to_num(acc, 0.0).clamp_(0.0, 1.0)
        colored = colored * acc + (1.0 - acc)

    return colored


@torch.no_grad()
def apply_pca_colormap(
    image: Tensor,  # (..., C) with C > 3
    pca_mat: Tensor | None = None,  # optional (C, 3) to keep colors stable
    ignore_zeros: bool = True,
) -> Tensor:
    """
    Project high-D features to RGB via PCA (first 3 components) with robust scaling.
    Returns (..., 3) in [0,1].

    - If pca_mat is None, compute it on-the-fly (unstable colors across frames).
    - If you pass a cached pca_mat (C,3), colors remain consistent across frames/views.
    """
    orig = image.shape
    C = orig[-1]
    assert C > 3, f"apply_pca_colormap expects channels>3, got {C}"

    x = image.reshape(-1, C)

    if ignore_zeros:
        valids = x.abs().amax(dim=-1) > 0
    else:
        valids = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)

    if pca_mat is None:
        # Compute PCA basis on valid rows only
        # pca_lowrank returns V (C, q)
        _, _, V = torch.pca_lowrank(x[valids], q=3, niter=20)
        pca_mat = V[:, :3]  # (C,3)

    # Project to 3D
    y = x @ pca_mat  # (N,3)

    # Robust per-channel scaling using MAD (median absolute deviation)
    yv = y[valids]
    med = yv.median(dim=0).values
    d = (yv - med).abs()
    mad = d.median(dim=0).values.clamp_min(1e-12)
    s = d / mad
    thr = 2.0  # outlier threshold in MAD units
    r_in = yv[s[:, 0] < thr, 0]
    g_in = yv[s[:, 1] < thr, 1]
    b_in = yv[s[:, 2] < thr, 2]

    # Shift/scale each channel to [0,1] based on inliers
    if r_in.numel() > 0:
        yv[:, 0] = (yv[:, 0] - r_in.min()) / (r_in.max() - r_in.min() + 1e-12)
    if g_in.numel() > 0:
        yv[:, 1] = (yv[:, 1] - g_in.min()) / (g_in.max() - g_in.min() + 1e-12)
    if b_in.numel() > 0:
        yv[:, 2] = (yv[:, 2] - b_in.min()) / (b_in.max() - b_in.min() + 1e-12)

    # Clamp and reshape back
    y[valids] = yv.clamp(0, 1)
    y[~valids] = 0
    return y.reshape(*orig[:-1], 3)
