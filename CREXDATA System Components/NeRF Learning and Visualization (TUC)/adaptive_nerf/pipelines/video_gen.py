"""
DRB-native NeRF video generator with RUB c2w poses.

Assumes:
- World translations are in DRB.
- c2w stores RUB columns ([right, up, back]) in DRB coords.
- get_ray_directions returns RUB camera rays with z_cam = -1.
"""

import math
from contextlib import nullcontext
from typing import Optional

import torch
import imageio.v3 as iio
from tqdm import tqdm

from nerfs.ray_sampling import get_ray_directions, get_rays
from nerfs.ray_rendering import render_rays
from nerfs.scene_box import SceneBox


def _rub_pose_from_pos(
    center: torch.Tensor, cam: torch.Tensor, up_world: torch.Tensor
) -> torch.Tensor:
    """Build c2w with RUB columns given camera and scene center in DRB."""
    fwd = center - cam
    fwd = fwd / fwd.norm().clamp_min_(1e-12)
    right = torch.linalg.cross(fwd, up_world)
    right = right / right.norm().clamp_min_(1e-12)
    up = torch.linalg.cross(right, fwd)
    R = torch.stack([right, up, -fwd], dim=1)
    c2w = torch.eye(4, device=cam.device, dtype=cam.dtype)
    c2w[:3, :3] = R
    c2w[:3, 3] = cam
    return c2w


def _rub_pose_look_same_D(
    center: torch.Tensor, cam: torch.Tensor, up_world: torch.Tensor
) -> torch.Tensor:
    """Build c2w that looks at center projected to camera height (D axis)."""
    look = torch.stack([cam[0], center[1], center[2]])
    fwd = look - cam
    fwd = fwd / fwd.norm().clamp_min_(1e-12)
    right = torch.cross(fwd, up_world)
    right = right / right.norm().clamp_min_(1e-12)
    up = torch.cross(right, fwd)
    R = torch.stack([right, up, -fwd], dim=1)
    c2w = torch.eye(4, device=cam.device, dtype=cam.dtype)
    c2w[:3, :3] = R
    c2w[:3, 3] = cam
    return c2w


def _rub_pose_from_fwd(
    cam: torch.Tensor, fwd: torch.Tensor, up_world: torch.Tensor
) -> torch.Tensor:
    """Build c2w with given forward direction in DRB."""
    fwd = fwd / fwd.norm().clamp_min_(1e-12)
    right = torch.cross(fwd, up_world)
    right = right / right.norm().clamp_min_(1e-12)
    up = torch.cross(right, fwd)
    R = torch.stack([right, up, -fwd], dim=1)
    c2w = torch.eye(4, device=cam.device, dtype=cam.dtype)
    c2w[:3, :3] = R
    c2w[:3, 3] = cam
    return c2w


# Pose generators --------------------------------------------------------------
def poses_turntable_drb(
    *,
    center_drb: torch.Tensor,
    radius: float,
    phi_deg: float = 20.0,
    n_poses: int = 120,
    device: str | torch.device = "cuda",
    tilt_deg: float = 0.0,
) -> torch.Tensor:
    """
    Generate a simple turntable orbit around center in DRB.

    Returns (T, 4, 4) c2w matrices with RUB columns.
    """
    device = torch.device(device)
    dtype = torch.float32
    center = center_drb.to(device=device, dtype=dtype)
    up_world = torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=dtype)

    phi = math.radians(max(phi_deg, 12.0))
    s_phi, c_phi = math.sin(phi), math.cos(phi)
    thetas = torch.linspace(0, 2 * math.pi, n_poses + 1, device=device, dtype=dtype)[
        :-1
    ]

    poses = []
    s_tilt, c_tilt = math.sin(math.radians(tilt_deg)), math.cos(math.radians(tilt_deg))

    for th in thetas:
        d = -radius * s_phi
        r = radius * c_phi * torch.cos(th)
        b = radius * c_phi * torch.sin(th)
        cam = center + torch.tensor([d, r, b], device=device, dtype=dtype)

        fwd = center - cam
        fwd = fwd / fwd.norm().clamp_min_(1e-12)
        right = torch.linalg.cross(fwd, up_world)
        right = right / right.norm().clamp_min_(1e-12)
        up = torch.linalg.cross(right, fwd)
        back = -fwd
        R = torch.stack([right, up, back], dim=1)

        if abs(tilt_deg) > 1e-6:
            k = back / back.norm().clamp_min_(1e-12)
            K = torch.tensor(
                [[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]],
                device=device,
                dtype=dtype,
            )
            R = R @ (
                torch.eye(3, device=device, dtype=dtype) * c_tilt
                + (1 - c_tilt) * (k[:, None] @ k[None, :])
                + s_tilt * K
            )

        c2w = torch.eye(4, device=device, dtype=dtype)
        c2w[:3, :3] = R
        c2w[:3, 3] = cam
        poses.append(c2w)

    return torch.stack(poses, 0)


def gen_path_east_west(
    center: torch.Tensor,
    extent: torch.Tensor,
    *,
    n_poses: int,
    height_frac: float = -0.12,
    margin: float = 0.90,
    device="cuda",
) -> torch.Tensor:
    """Sweep along R (west→east) at constant height, looking at center."""
    dev = torch.device(device)
    dtype = torch.float32
    up_world = torch.tensor([-1.0, 0.0, 0.0], device=dev, dtype=dtype)
    halfD, halfR, halfB = (
        0.5 * extent[0].item(),
        0.5 * extent[1].item(),
        0.5 * extent[2].item(),
    )

    D = torch.tensor(height_frac * halfD, device=dev, dtype=dtype)
    R0, R1 = -margin * halfR, +margin * halfR
    Boff = 0.15 * halfB
    ts = torch.linspace(0, 1, n_poses, device=dev)
    u = 0.5 * (1 - torch.cos(torch.pi * ts))

    poses = []
    for t, w in zip(ts, u):
        R = (1 - w) * R0 + w * R1
        B = Boff * torch.sin(2 * torch.pi * t)
        cam = center + torch.tensor([D, R, B], device=dev, dtype=dtype)
        poses.append(_rub_pose_look_same_D(center, cam, up_world))
    return torch.stack(poses, 0)


def gen_path_north_south(
    center: torch.Tensor,
    extent: torch.Tensor,
    *,
    n_poses: int,
    height_frac: float = -0.12,
    margin: float = 0.90,
    device="cuda",
) -> torch.Tensor:
    """Sweep along B (north→south) at constant height, looking at center."""
    dev = torch.device(device)
    dtype = torch.float32
    up_world = torch.tensor([-1.0, 0.0, 0.0], device=dev, dtype=dtype)
    halfD, halfR, halfB = (
        0.5 * extent[0].item(),
        0.5 * extent[1].item(),
        0.5 * extent[2].item(),
    )

    D = torch.tensor(height_frac * halfD, device=dev, dtype=dtype)
    B0, B1 = -margin * halfB, +margin * halfB
    Roff = 0.15 * halfR
    ts = torch.linspace(0, 1, n_poses, device=dev)
    u = 0.5 * (1 - torch.cos(torch.pi * ts))

    poses = []
    for t, w in zip(ts, u):
        B = (1 - w) * B0 + w * B1
        R = Roff * torch.sin(2 * torch.pi * t)
        cam = center + torch.tensor([D, R, B], device=dev, dtype=dtype)
        poses.append(_rub_pose_look_same_D(center, cam, up_world))
    return torch.stack(poses, 0)


def gen_path_spiral_inside(
    center: torch.Tensor,
    extent: torch.Tensor,
    *,
    n_poses: int,
    turns: float = 2.0,
    radial_frac: float = 0.6,
    height_center_frac: float = -0.15,
    height_amp_frac: float = 0.10,
    phi_deg: float = 18.0,
    device="cuda",
) -> torch.Tensor:
    """
    Spiral camera path entirely inside the scene AABB.

    Radius is limited to radial_frac * (min_extent/2) with mild height variation.
    """
    dev = torch.device(device)
    dtype = torch.float32
    up_world = torch.tensor([-1.0, 0.0, 0.0], device=dev, dtype=dtype)
    halfR, halfB, halfD = (
        0.5 * extent[1].item(),
        0.5 * extent[2].item(),
        0.5 * extent[0].item(),
    )
    min_half = 0.5 * float(extent.min().item())

    r_base = radial_frac * min_half

    def radius(theta):
        return r_base * (0.85 + 0.15 * torch.cos(theta * 0.5))

    d_center = height_center_frac * halfD
    d_amp = height_amp_frac * halfD

    thetas = torch.linspace(0, 2 * math.pi * turns, n_poses, device=dev)
    poses = []
    for th in thetas:
        rad = radius(th)
        r = rad * torch.cos(th)
        b = rad * torch.sin(th)
        d = d_center + d_amp * torch.sin(0.5 * th)

        r = torch.clamp(r, -halfR * 0.95, +halfR * 0.95)
        b = torch.clamp(b, -halfB * 0.95, +halfB * 0.95)
        d = torch.clamp(d, -halfD * 0.90, +halfD * 0.90)

        cam = center + torch.tensor(
            [d.item(), r.item(), b.item()], device=dev, dtype=dtype
        )
        poses.append(_rub_pose_from_pos(center, cam, up_world))
    return torch.stack(poses, 0)


def gen_path_full_coverage(
    center: torch.Tensor,
    extent: torch.Tensor,
    *,
    n_poses: int,
    rows: int = 6,
    cols: int = 9,
    height_start_frac: float = -0.18,
    height_end_frac: float = +0.18,
    device="cuda",
) -> torch.Tensor:
    """
    Grid-like path that sweeps a DRB-aligned grid over the scene with smooth motion.

    Returns (T, 4, 4) c2w matrices approximating n_poses frames.
    """
    dev = torch.device(device)
    dtype = torch.float32
    up_world = torch.tensor([-1.0, 0.0, 0.0], device=dev, dtype=dtype)
    halfR, halfB, halfD = (
        0.5 * extent[1].item(),
        0.5 * extent[2].item(),
        0.5 * extent[0].item(),
    )

    Rvals = torch.linspace(-0.85 * halfR, +0.85 * halfR, cols, device=dev)
    Bvals = torch.linspace(-0.85 * halfB, +0.85 * halfB, rows, device=dev)

    waypoints = []
    for i, b in enumerate(Bvals):
        Rs = Rvals if (i % 2 == 0) else torch.flip(Rvals, dims=[0])
        waypoints.extend((r.item(), b.item()) for r in Rs)
    M = len(waypoints)

    legs = max(1, M - 1)
    f_per_leg = max(2, int(math.ceil(n_poses / legs)))
    total = legs * f_per_leg
    t_heights = torch.linspace(0.0, 1.0, total, device=dev)
    d_start = height_start_frac * halfD
    d_end = height_end_frac * halfD

    poses = []

    def ease(u):
        return 0.5 * (1 - torch.cos(torch.tensor(math.pi, device=dev) * u))

    idx = 0
    for k in range(legs):
        r0, b0 = waypoints[k]
        r1, b1 = waypoints[k + 1]
        for j in range(f_per_leg):
            u = j / f_per_leg
            w = ease(u)
            r = (1 - w) * r0 + w * r1
            b = (1 - w) * b0 + w * b1
            tH = t_heights[idx]
            d = (1 - tH) * d_start + tH * d_end
            idx += 1
            cam = center + torch.tensor([d, r, b], device=dev, dtype=dtype)
            poses.append(_rub_pose_from_pos(center, cam, up_world))

    poses = torch.stack(poses, 0)
    if poses.shape[0] > n_poses:
        poses = poses[:n_poses]
    elif poses.shape[0] < n_poses:
        poses = torch.cat(
            [poses, poses[-1:].expand(n_poses - poses.shape[0], -1, -1)], dim=0
        )
    return poses


# Fog suppression --------------------------------------------------------------
def suppress_fog_inplace(
    rgb: torch.Tensor,
    weights: torch.Tensor,
    acc: torch.Tensor,
    *,
    bg_val: float,
    acc_thr: float = 0.05,
    wmax_thr: float = 0.08,
    entropy_thr: float = 1.5,
) -> dict:
    """
    Suppress low-opacity / hazy pixels in-place based on opacity and weight shape.

    Returns a small stats dict (fraction of low-opacity and fogged pixels).
    """
    acc1 = acc.squeeze(-1)
    low = acc1 < acc_thr
    if low.any():
        rgb[low] = bg_val

    wsum = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    p = weights / wsum
    entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=1)
    wmax = weights.max(dim=1).values

    fog = (acc1 < max(acc_thr * 1.3, 0.10)) & (
        (wmax < wmax_thr) | (entropy > entropy_thr)
    )
    if fog.any():
        rgb[fog] = bg_val

    return {
        "pct_low": float(low.float().mean().item()),
        "pct_fog": float(fog.float().mean().item()),
    }


# Main entrypoint --------------------------------------------------------------
@torch.inference_mode()
def render_video(
    model: torch.nn.Module,
    *,
    # image & intrinsics
    H: int,
    W: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    # scene / marching
    scene_box: Optional[SceneBox] = None,
    near: Optional[float] = None,
    far: Optional[float] = None,
    # path selection
    camera_path: str = "turntable",
    n_poses: int = 120,
    phi_deg: float = 20.0,
    tilt_deg: float = 0.0,
    radius: Optional[float] = None,
    center_drb: Optional[torch.Tensor] = None,
    # render
    ray_samples: int = 96,
    chunk: int = 262_144,
    bg_color_default: str = "white",
    fps: int = 30,
    out_path: str = "video.mp4",
    device: str | torch.device = "cuda",
    center_pixels: bool = True,
) -> str:
    """
    Render a NeRF video along a DRB camera path.

    Supports camera_path in {"turntable", "north_south", "east_west",
    "spiral_in", "full_coverage"} and writes an MP4 to out_path.
    """
    dev = torch.device(device)
    dtype = torch.float32
    model = model.to(dev).eval()
    if hasattr(model, "use_bg_nerf"):
        model.use_bg_nerf = False

    def bg_color_val(bg_color_default: str) -> float:
        return 0.0 if bg_color_default == "black" else 1.0

    dirs_hw = get_ray_directions(
        H=H, W=W, fx=fx, fy=fy, cx=cx, cy=cy, center_pixels=center_pixels, device=dev
    )

    if scene_box is not None:
        aabb = scene_box.aabb.to(device=dev, dtype=dtype)
        center = 0.5 * (aabb[0] + aabb[1]) if center_drb is None else center_drb.to(dev)
        extent = (aabb[1] - aabb[0]).abs()
    else:
        center = (
            torch.zeros(3, device=dev, dtype=dtype)
            if center_drb is None
            else center_drb.to(dev)
        )
        extent = torch.tensor([8.0, 8.0, 8.0], device=dev, dtype=dtype)

    if camera_path == "turntable":
        if radius is None:
            radius = (0.5 * float(extent.norm().item())) * 1.5
        poses = poses_turntable_drb(
            center_drb=center,
            radius=float(radius),
            phi_deg=phi_deg,
            n_poses=n_poses,
            device=dev,
            tilt_deg=tilt_deg,
        )
    elif camera_path == "north_south":
        poses = gen_path_north_south(center, extent, n_poses=n_poses, device=dev)
    elif camera_path == "east_west":
        poses = gen_path_east_west(center, extent, n_poses=n_poses, device=dev)
    elif camera_path == "spiral_in":
        poses = gen_path_spiral_inside(center, extent, n_poses=n_poses, device=dev)
    elif camera_path == "full_coverage":
        poses = gen_path_full_coverage(center, extent, n_poses=n_poses, device=dev)
    else:
        raise ValueError(f"Unknown camera_path: {camera_path}")

    camera_drop = 0.15 * extent[0]
    poses[:, 0, 3] += camera_drop

    if scene_box is not None:
        near_bias = 0.15 * (0.5 * float(extent.norm().item()))
    else:
        if near is None or far is None:
            raise ValueError("Provide near/far when no scene_box is given.")
        near_bias = 0.02 * float(far)

    frames_u8 = []
    amp_ctx = (
        torch.cuda.amp.autocast(dtype=torch.float16)
        if dev.type == "cuda"
        else nullcontext()
    )

    fog_params = {"acc_thr": 0.05, "wmax_thr": 0.08, "entropy_thr": 1.4}

    for c2w in tqdm(
        poses, total=poses.shape[0], desc=f"[render:{camera_path}]", ncols=90
    ):
        if scene_box is not None:
            rays_hw = get_rays(dirs_hw, c2w, scene_box=scene_box)
        else:
            rays_hw = get_rays(dirs_hw, c2w, near=float(near), far=float(far))
        rays = rays_hw.view(-1, 8).contiguous()

        t_near, t_far = rays[:, 6], rays[:, 7]
        t_near.clamp_(min=0.0)
        t_near.add_(near_bias).clamp_max_(t_far - 1e-4)

        valid = rays[:, 7] > rays[:, 6]
        idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
        N = rays.shape[0]
        rgb_out = torch.full(
            (N, 3), bg_color_val(bg_color_default), device=dev, dtype=torch.float32
        )

        with amp_ctx:
            for s in range(0, idx.numel(), chunk):
                sel = idx[s : s + chunk]
                if sel.numel() == 0:
                    break
                r = rays.index_select(0, sel)
                rgb, depth, weights, acc = render_rays(
                    model=model,
                    rays=r,
                    ray_samples=ray_samples,
                    params=None,
                    active_module=None,
                    bg_color_default=bg_color_default,
                    chunk=chunk,
                )

                suppress_fog_inplace(
                    rgb,
                    weights,
                    acc,
                    bg_val=bg_color_val(bg_color_default),
                    acc_thr=fog_params["acc_thr"],
                    wmax_thr=fog_params["wmax_thr"],
                    entropy_thr=fog_params["entropy_thr"],
                )

                rgb_out.index_copy_(0, sel, rgb)

        frame = rgb_out.view(H, W, 3).clamp_(0, 1)
        frames_u8.append((frame * 255).byte().cpu().numpy())

    iio.imwrite(out_path, frames_u8, fps=fps, codec="libx264", quality=8)
    return out_path
