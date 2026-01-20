#!/usr/bin/env python3
"""
This is an extenstion of MegaNeRF's:
https://github.com/cmusatyalab/mega-nerf/blob/main/scripts/create_cluster_masks.py

Copyright (c) 2021 cmusatyalab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Create cluster masks using Mega-NeRF-style Voronoi routing (2D YZ or full 3D XYZ).

Overview
--------
For every image, assign each ray to one or more routing centroids using the
classic Mega-NeRF Voronoi rule with optional boundary overlap:

1) For each ray r, sample S points uniformly along its effective [near, far] segment.
   - If --near/--far are not provided, near/far are auto-derived from a
     Nerfstudio-style SceneBox AABB intersection (recommended).
2) For each sampled point x and centroid c in the chosen routing subspace
   (YZ if --cluster_2d else XYZ), compute:
       ratio(x, c) = dist(x, c) / min_c' dist(x, c')
3) A ray belongs to centroid c if:    min_x ratio(x, c) <= boundary_margin
   - boundary_margin = 1.0 → strict Voronoi (no overlap)
   - boundary_margin > 1.0 → controlled overlap near boundaries

Centroids:
- Grid   : uniform tiles (2D YZ) or cubes (3D XYZ)
- K-Means: in 2D (YZ) or 3D (XYZ), with optional camera-pixel weighting

I/O & Features
--------------
- Resumable: skips images whose per-centroid zipped masks already exist and load OK
- Distributed: torch.distributed (NCCL) rank-strided split across images
- Robust I/O: masks are stored per-centroid as zipped tensors to limit inode usage
- SceneBox logging (rank 0 only): AABB, scale, and per-split ray intersection stats

Memory tuning
-----------------
 --ray_chunk_size       (rays per block)
 --sample_chunk_size    (sampled points per cdist block; prefer multiples of S)

Example Usage
-----------------
 ./scripts/create_clusters.py --data_path data/drz/out/prepared  --centroid_mode grid --grid_dim 2 2 --cluster_2d --boundary_margin 1.05 --ray_samples 256 --center_pixels --scene_scale 1.3 --output g22_grid_bm105_ss13 --resume

"""

import sys
from pathlib import Path
import warnings
import argparse
import datetime
import logging
import os
import zipfile
from typing import Iterable, Optional, Tuple, List
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.distributed as dist

from nerfs.ray_sampling import (
    get_ray_directions,
    get_rays,
    clamp_rays_near_far,
    unpack_rays,
)
from nerfs.scene_box import SceneBox

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        "Create cluster masks (Voronoi routing, 2D/3D grid or kmeans) + per-expert AABBs"
    )

    # I/O
    p.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Dataset root with coordinates.pt and train/val/metadata/*.pt",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output name for masks and params.pt folder",
    )
    p.add_argument(
        "--segmentation_path",
        type=Path,
        default=None,
        help="Optional directory with zipped per-image masks (.pt) to AND with clusters.",
    )
    p.add_argument("--resume", action="store_true")

    # Centroids
    p.add_argument("--centroid_mode", choices=["grid", "kmeans"], default="grid")
    p.add_argument(
        "--grid_dim",
        nargs="+",
        type=int,
        metavar="D",
        required=True,
        help="Grid dims. 2D: GY GZ (YZ tiles). 3D: GX GY GZ (XYZ cubes).",
    )
    p.add_argument(
        "--cluster_2d",
        action="store_true",
        help="Route in YZ only (ignore X). If unset, route in XYZ.",
    )

    p.add_argument("--kmeans_iters", type=int, default=50)
    p.add_argument("--kmeans_init", choices=["kmeans++", "random"], default="kmeans++")
    p.add_argument("--kmeans_seed", type=int, default=0)
    p.add_argument("--kmeans_weight_by_pixels", action="store_true")

    # Routing core
    p.add_argument(
        "--boundary_margin",
        type=float,
        default=1.0,
        help="Distance-ratio threshold; 1.0=Voronoi; >1.0 allows overlap.",
    )
    p.add_argument("--ray_samples", type=int, default=256, help="Samples per ray.")
    p.add_argument("--center_pixels", action="store_true")
    p.add_argument("--orig", action="store_true")

    # Blocking / memory
    p.add_argument(
        "--ray_chunk_size", type=int, default=256 * 1024, help="Rays per chunk."
    )
    p.add_argument(
        "--sample_chunk_size",
        type=int,
        default=512 * 1024 * 1024,
        help="[opt] Max sampled points per block; prefer multiples of ray_samples.",
    )
    p.add_argument(
        "--fp16",
        action="store_true",
        help="[opt] Use FP16/BF16 GEMMs for speed (routing math only).",
    )

    # Scene / bounds
    p.add_argument(
        "--scene_scale",
        type=float,
        default=1.0,
        help="SceneBox half-extent for Y/Z; X is limited by altitude band if provided.",
    )
    p.add_argument(
        "--altitude_range",
        nargs=2,
        type=float,
        default=None,
        help="Altitude [min,max] meters (ENU Up). If omitted, uses coordinates.pt if present.",
    )
    p.add_argument(
        "--near", type=float, default=None, help="Global near override (meters)."
    )
    p.add_argument(
        "--far", type=float, default=None, help="Global far override (meters)."
    )
    p.add_argument(
        "--altitude_pad",
        type=float,
        default=10.0,
        help="Global altitude padding (meters) for Scenebox.",
    )
    p.add_argument(
        "--box_margin",
        type=float,
        default=0.0,
        help="Per-axis dilation (in scene units) applied to each expert AABB before saving.",
    )
    return p.parse_args()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _setup_logger(rank: int) -> None:
    logging.basicConfig(
        level=(logging.INFO if rank == 0 else logging.ERROR), format="%(message)s"
    )


def _log(rank: int, *msg) -> None:
    if rank == 0:
        logging.info(" ".join(str(m) for m in msg))


def _init_distributed(out_dir: Path, resume: bool) -> Tuple[int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=24))
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local)
        if rank == 0:
            out_dir.mkdir(parents=True, exist_ok=resume)
        dist.barrier()
        return rank, world, torch.device("cuda", local)
    else:
        out_dir.mkdir(parents=True, exist_ok=resume)
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, dev


def _meta_list(ds_root: Path, split: str) -> List[Path]:
    return sorted((ds_root / split / "metadata").glob("*.pt"))


def _save_zip_tensor(path: Path, tensor: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        with zf.open(path.name, "w") as f:
            torch.save(tensor, f)


def _zip_load_ok(path: Path, inner: Optional[str] = None) -> bool:
    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            name = inner or path.name
            if name not in zf.namelist():
                name = zf.namelist()[0]
            with zf.open(name, "r") as f:
                _ = torch.load(f, map_location="cpu")
        return True
    except Exception:
        return False


def all_ok_for_image(K: int, out_dir: Path, filename: str) -> bool:
    return all(
        _zip_load_ok(out_dir / str(cid) / filename, filename) for cid in range(K)
    )


def _read_zip_mask(zip_path: Path) -> Optional[torch.Tensor]:
    if not zip_path.exists():
        return None
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            inner = zf.namelist()[0]
            with zf.open(inner, "r") as f:
                return torch.load(f, map_location="cpu")
    except Exception:
        return None


def _cam_weights(meta_paths: Iterable[Path]) -> Optional[torch.Tensor]:
    ws = []
    for p in meta_paths:
        md = torch.load(p, map_location="cpu")
        ws.append(int(md["H"]) * int(md["W"]))
    return torch.tensor(ws, dtype=torch.float32) if ws else None


# --------------------------------------------------------------------------------------
# Centroid generators
# --------------------------------------------------------------------------------------


def _grid_centroids(
    cam_pos: torch.Tensor, gx: int, gy: int, gz: int, cluster_2d: bool
) -> torch.Tensor:
    if cam_pos.numel() == 0:
        C = (gy * gz) if cluster_2d else (gx * gy * gz)
        return torch.zeros((C, 3), dtype=torch.float32)

    minp, maxp = cam_pos.min(0).values, cam_pos.max(0).values

    if cluster_2d:
        x_c = (minp[0] + maxp[0]) * 0.5
        dY, dZ = (maxp[1] - minp[1]) / gy, (maxp[2] - minp[2]) / gz
        Y = minp[1] + (torch.arange(gy) + 0.5) * dY
        Z = minp[2] + (torch.arange(gz) + 0.5) * dZ
        YY, ZZ = torch.meshgrid(Y, Z, indexing="ij")
        X = torch.full_like(YY, x_c)
        return torch.stack((X, YY, ZZ), -1).reshape(-1, 3)

    dX = (maxp[0] - minp[0]) / max(gx, 1)
    dY = (maxp[1] - minp[1]) / max(gy, 1)
    dZ = (maxp[2] - minp[2]) / max(gz, 1)
    X = minp[0] + (torch.arange(gx) + 0.5) * dX
    Y = minp[1] + (torch.arange(gy) + 0.5) * dY
    Z = minp[2] + (torch.arange(gz) + 0.5) * dZ
    XX, YY, ZZ = torch.meshgrid(X, Y, Z, indexing="ij")
    return torch.stack((XX, YY, ZZ), -1).reshape(-1, 3)


def _kmeans_init(
    points: torch.Tensor,
    K: int,
    seed: int,
    method: str,
    weights: Optional[torch.Tensor],
) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    if method == "random":
        idx = torch.randperm(points.size(0), generator=g)[:K]
        return points[idx].clone()

    centers = torch.empty(K, points.size(1), dtype=points.dtype)
    if weights is None:
        centers[0] = points[torch.randint(points.size(0), (1,), generator=g)]
    else:
        p = (weights / weights.sum()).cpu()
        centers[0] = points[torch.multinomial(p, 1, generator=g)]
    for k in range(1, K):
        D = torch.cdist(points, centers[:k])
        m2 = D.min(1).values ** 2
        w = weights if weights is not None else 1.0
        probs = (m2 * w).clamp_min_(1e-12)
        probs = probs / probs.sum()
        centers[k] = points[torch.multinomial(probs, 1, generator=g)]
    return centers


def _run_kmeans(
    points: torch.Tensor,
    K: int,
    iters: int,
    init: str,
    seed: int,
    weights: Optional[torch.Tensor],
) -> torch.Tensor:
    centers = _kmeans_init(points, K, seed, init, weights)
    w = (
        weights
        if weights is not None
        else torch.ones(points.size(0), dtype=points.dtype)
    )
    for _ in range(max(1, iters)):
        D = torch.cdist(points, centers)
        a = D.argmin(1)
        for k in range(K):
            m = a == k
            if not m.any():
                centers[k] = points[D[:, k].argmax()]
            else:
                wk, pk = w[m], points[m]
                centers[k] = (wk[:, None] * pk).sum(0) / wk.sum()
    return centers


# --------------------------------------------------------------------------------------
# Routing cores — returns boolean masks (N, C) on CPU; streams per-expert AABBs on-device
# --------------------------------------------------------------------------------------


@torch.inference_mode()
def compute_voronoi_opt(
    rays: torch.Tensor,
    *,
    ray_samples: int,
    ray_chunk_size: int,
    sample_chunk_size: int,
    centroids: torch.Tensor,
    cluster_2d: bool,
    device: torch.device,
    boundary_margin: float,
    fp16: bool = False,  # prefers bf16 if available
    # NEW (optional) for streaming per-expert AABBs + counts
    update_aabbs: bool = False,
    mins_out: Optional[torch.Tensor] = None,  # (C,3) on device
    maxs_out: Optional[torch.Tensor] = None,  # (C,3) on device
    counts_out: Optional[torch.Tensor] = None,  # (C,)  int64 on device
) -> torch.Tensor:
    """
    Optimized Voronoi routing. Strict (margin==1): argmin_c d^2(x,c).
    Overlap (margin>1): d^2(x,c) <= m^2 * min_{c'} d^2(x,c').
    If update_aabbs=True, updates mins/maxs/counts for experts using only samples assigned
    to them. Updates happen on the same device (NCCL-safe).
    """
    if not rays.is_cuda:
        warnings.warn(
            "Rays not on CUDA; falling back to compute_voronoi_orig (no AABB updates)."
        )
        return compute_voronoi_orig(
            rays,
            ray_samples=ray_samples,
            ray_chunk_size=ray_chunk_size,
            sample_chunk_size=sample_chunk_size,
            centroids=centroids,
            cluster_2d=cluster_2d,
            device=device,
            boundary_margin=boundary_margin,
        )

    rays_o, rays_d, near_b, far_b = unpack_rays(rays)
    N = int(rays_o.shape[0])

    start = 1 if cluster_2d else 0
    k = 2 if cluster_2d else 3

    if fp16 and device.type == "cuda":
        work_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        work_dtype = torch.float32

    Csub_w = (
        centroids[:, start:].to(device=device, dtype=work_dtype).contiguous()
    )  # (C,k) for matmuls
    Csub32 = Csub_w.to(torch.float32)  # (C,k) for stable d^2
    c_norm2_32 = (Csub32 * Csub32).sum(dim=1).view(1, -1)  # (1,C)

    C = int(Csub32.shape[0])
    S = int(ray_samples)
    m2 = float(boundary_margin * boundary_margin)

    out_mask = torch.zeros((N, C), dtype=torch.bool, device=device)
    z_w = torch.linspace(0.0, 1.0, S, device=device, dtype=work_dtype)

    for r0 in range(0, N, ray_chunk_size):
        r1 = min(N, r0 + ray_chunk_size)
        R = r1 - r0

        origins_w = rays_o[r0:r1].to(device=device, dtype=work_dtype)  # (R,3)
        directions_w = rays_d[r0:r1].to(device=device, dtype=work_dtype)  # (R,3)
        near_w = near_b[r0:r1, 0].to(device=device, dtype=work_dtype)  # (R,)
        far_w = far_b[r0:r1, 0].to(device=device, dtype=work_dtype)  # (R,)

        z_nearfar_w = torch.lerp(near_w[:, None], far_w[:, None], z_w[None, :])  # (R,S)
        Z_full32 = z_nearfar_w.to(torch.float32)  # (R,S)
        O_full32 = origins_w.to(torch.float32)  # (R,3)
        D_full32 = directions_w.to(torch.float32)  # (R,3)

        Osub_w = origins_w[:, start : start + k]  # (R,k)
        Dsub_w = directions_w[:, start : start + k]  # (R,k)
        Xsub_w = (
            Osub_w[:, None, :] + Dsub_w[:, None, :] * z_nearfar_w[..., None]
        )  # (R,S,k)

        ray_has = torch.zeros((R, C), dtype=torch.bool, device=device)

        Qtotal = R * S
        Qblock = max(S, (sample_chunk_size // max(S, 1)) * S)

        for q0 in range(0, Qtotal, Qblock):
            q1 = min(Qtotal, q0 + Qblock)
            qR = (q1 - q0) // S
            if qR == 0:
                q1 = min(q0 + S, Qtotal)
                qR = 1
            r_off = q0 // S

            Xb_sub_w = Xsub_w.reshape(-1, k)[q0:q1].reshape(qR, S, k)  # (qR,S,k)
            Xbf_sub_w = Xb_sub_w.reshape(-1, k)  # (qR*S,k)

            x2 = (Xbf_sub_w.to(torch.float32) ** 2).sum(dim=1, keepdim=True)  # (qR*S,1)
            ip = Xbf_sub_w.to(torch.float32) @ Csub32.t()  # (qR*S,C)
            d2 = x2 + c_norm2_32 - 2.0 * ip  # (qR*S,C)
            d2.clamp_min_(0.0)

            if boundary_margin == 1.0:
                nn = d2.argmin(dim=1).view(qR, S)  # (qR,S)
                rr = (
                    torch.arange(r_off, r_off + qR, device=device)
                    .view(-1, 1)
                    .expand(qR, S)
                )
                ray_has[rr.reshape(-1), nn.reshape(-1)] = True

                if update_aabbs:
                    assert mins_out is not None and maxs_out is not None
                    experts = torch.unique(nn)
                    if experts.numel() > 0:
                        for cid in experts.tolist():
                            m = nn == cid
                            if not m.any():
                                continue
                            ri, si = m.nonzero(as_tuple=True)
                            t_sel = Z_full32[ri + r_off, si]  # (Nsel,)
                            x_sel = O_full32[ri + r_off] + D_full32[
                                ri + r_off
                            ] * t_sel.unsqueeze(
                                1
                            )  # (Nsel,3)
                            # on-device min/max
                            mins_out[cid] = torch.minimum(
                                mins_out[cid], x_sel.min(dim=0).values
                            )
                            maxs_out[cid] = torch.maximum(
                                maxs_out[cid], x_sel.max(dim=0).values
                            )
                            if counts_out is not None:
                                counts_out[cid] += x_sel.shape[0]

            else:
                d2_min = d2.min(dim=1, keepdim=True).values  # (qR*S,1)
                thr = m2 * d2_min
                ok = (d2 <= thr).view(qR, S, C)  # (qR,S,C) bool
                ok_any = ok.any(dim=1)  # (qR,C)
                ray_has[r_off : r_off + qR, :] |= ok_any

                if update_aabbs:
                    assert mins_out is not None and maxs_out is not None
                    experts = torch.nonzero(ok_any.any(dim=0), as_tuple=False).squeeze(
                        1
                    )
                    for cid in experts.tolist():
                        m = ok[..., cid]  # (qR,S)
                        if not m.any():
                            continue
                        ri, si = m.nonzero(as_tuple=True)
                        t_sel = Z_full32[ri + r_off, si]
                        x_sel = O_full32[ri + r_off] + D_full32[
                            ri + r_off
                        ] * t_sel.unsqueeze(1)
                        mins_out[cid] = torch.minimum(
                            mins_out[cid], x_sel.min(dim=0).values
                        )
                        maxs_out[cid] = torch.maximum(
                            maxs_out[cid], x_sel.max(dim=0).values
                        )
                        if counts_out is not None:
                            counts_out[cid] += x_sel.shape[0]

        out_mask[r0:r1] = ray_has

    return out_mask.to(torch.bool).cpu()


@torch.inference_mode()
def compute_voronoi_orig(
    rays: torch.Tensor,
    *,
    ray_samples: int,
    ray_chunk_size: int,
    sample_chunk_size: int,
    centroids: torch.Tensor,
    cluster_2d: bool,
    device: torch.device,
    boundary_margin: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Reference Voronoi routing using Euclidean distances (torch.cdist) and min ratio.
    (No AABB streaming here; used as CPU fallback.)
    """
    rays_o, rays_d, near_b, far_b = unpack_rays(rays)
    N = int(rays_o.shape[0])
    C = int(centroids.shape[0])
    S = int(ray_samples)

    start = 1 if cluster_2d else 0
    cents_sub = centroids[:, start:].to(
        device=device, dtype=torch.float32
    )  # (C, 2 or 3)

    out_mask = torch.zeros((N, C), dtype=torch.bool, device=device)

    for r0 in range(0, N, ray_chunk_size):
        r1 = min(N, r0 + ray_chunk_size)
        R = r1 - r0

        ro = rays_o[r0:r1].to(device=device, dtype=torch.float32)
        rd = rays_d[r0:r1].to(device=device, dtype=torch.float32)
        near = near_b[r0:r1, 0].to(device=device, dtype=torch.float32)
        far = far_b[r0:r1, 0].to(device=device, dtype=torch.float32)

        z_vals = torch.linspace(0.0, 1.0, S, device=device, dtype=torch.float32)  # (S,)
        t = torch.lerp(near[:, None], far[:, None], z_vals[None, :])  # (R,S)
        xyz = ro[:, None, :] + rd[:, None, :] * t[..., None]  # (R,S,3)
        sub = xyz.view(-1, 3)[:, start:]  # (Q,2 or 3)

        min_ratio_chunk = torch.full((R, C), float("inf"), device=device)
        Q = sub.shape[0]
        q_step = max(S, (sample_chunk_size // max(S, 1)) * S)

        k = 0
        while k < Q:
            q = min(q_step, Q - k)
            block = sub[k : k + q]  # (q,k)
            D = torch.cdist(block, cents_sub)  # (q,C)
            m = D.min(dim=1, keepdim=True).values  # (q,1)
            ratio = D / (m + eps)  # (q,C)

            full = q // S
            if full > 0:
                r_off = k // S
                rmin = ratio[: full * S].view(full, S, C).amin(dim=1)  # (full,C)
                min_ratio_chunk[r_off : r_off + full] = torch.minimum(
                    min_ratio_chunk[r_off : r_off + full], rmin
                )

            rem = q % S
            if rem > 0:
                r_partial = (k // S) + full
                rmin = ratio[-rem:].amin(dim=0)  # (C,)
                min_ratio_chunk[r_partial] = torch.minimum(
                    min_ratio_chunk[r_partial], rmin
                )

            k += q

        out_mask[r0:r1] = min_ratio_chunk <= boundary_margin

    return out_mask.to(torch.bool).cpu()


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


@torch.inference_mode()
def main(h: argparse.Namespace) -> None:
    out = h.data_path / "masks" / h.output
    rank, world, device = _init_distributed(out, h.resume)
    _setup_logger(rank)

    ds = h.data_path
    coord = torch.load(ds / "coordinates.pt", map_location="cpu")

    pose_scale = float(coord.get("pose_scale_factor", 1.0))
    origin_drb_x = float(coord.get("origin_drb", [0.0, 0.0, 0.0])[0])

    # Altitude band (optional)
    if h.altitude_range is not None:
        min_enu_m, max_enu_m = map(float, h.altitude_range)
    elif "altitude_range_enu" in coord:
        min_enu_m, max_enu_m = map(float, coord["altitude_range_enu"])
    else:
        min_enu_m, max_enu_m = (
            0.0,
            0.0,
        )  # no band; X extents will be from scene scale only

    if min_enu_m > max_enu_m:
        min_enu_m, max_enu_m = max_enu_m, min_enu_m

    # ENU Up -> DRB Down (meters) → normalized (match c2w translation units)
    x_min_drb_m = -max_enu_m
    x_max_drb_m = -min_enu_m
    if x_min_drb_m > x_max_drb_m:
        x_min_drb_m, x_max_drb_m = x_max_drb_m, x_min_drb_m

    x_min_norm = (x_min_drb_m - origin_drb_x) / pose_scale
    x_max_norm = (x_max_drb_m - origin_drb_x) / pose_scale

    alt_bounds_norm = (
        (x_min_norm, x_max_norm)
        if (h.altitude_range is not None or "altitude_range_enu" in coord)
        else None
    )

    # Global SceneBox (normalized DRB)
    aabb = torch.tensor(
        [
            [alt_bounds_norm[0], -h.scene_scale, -h.scene_scale],
            [alt_bounds_norm[1], h.scene_scale, h.scene_scale],
        ],
        dtype=torch.float32,
    ).to(device)

    global_box = SceneBox.from_bound(aabb=aabb)
    expand_tensor = torch.tensor(
        [
            [h.altitude_pad / pose_scale, 0, 0],
        ],
        dtype=torch.float32,
    )
    global_box = global_box.expand(expand_tensor)
    aabb_global = global_box.aabb.detach().to(torch.float32)  # (2,3) on device
    aabb_min_g, aabb_max_g = aabb_global[0], aabb_global[1]
    _log(rank, f"Global SceneBox: {global_box}")

    # Metadata
    train_meta, val_meta = _meta_list(ds, "train"), _meta_list(ds, "val")
    all_meta = train_meta + val_meta
    if len(all_meta) == 0:
        raise RuntimeError(f"No metadata found in {ds}/{{train,val}}/metadata")

    # Grid dims
    dims = list(map(int, h.grid_dim))
    if h.cluster_2d:
        if len(dims) != 2:
            raise ValueError("For cluster_2d=True use --grid_dim GY GZ.")
        gx, gy, gz = 1, dims[0], dims[1]
    else:
        if len(dims) == 2:
            gx, gy, gz = 1, dims[0], dims[1]
        elif len(dims) == 3:
            gx, gy, gz = dims
        else:
            raise ValueError("For 3D grid use --grid_dim GX GY GZ.")
    K = gx * gy * gz

    # Camera positions
    poses = torch.stack(
        [torch.load(p, map_location="cpu")["c2w"] for p in all_meta], dim=0
    ).to(torch.float32)
    cams = poses[..., :3, 3]  # (N,3)

    # Centroids
    wts = _cam_weights(all_meta) if h.kmeans_weight_by_pixels else None
    if h.centroid_mode == "grid":
        cents = _grid_centroids(cams, gx, gy, gz, h.cluster_2d)
    else:
        if h.cluster_2d:
            pts_yz = cams[:, 1:].cpu()
            cents_yz = _run_kmeans(
                pts_yz,
                K,
                h.kmeans_iters,
                h.kmeans_init,
                h.kmeans_seed,
                (wts.cpu() if wts is not None else None),
            )
            x_mid = (cams[:, 0].min() + cams[:, 0].max()) * 0.5
            cents = torch.cat([torch.full((K, 1), float(x_mid)), cents_yz], dim=1)
        else:
            cents = _run_kmeans(
                cams.cpu(),
                K,
                h.kmeans_iters,
                h.kmeans_init,
                h.kmeans_seed,
                (wts.cpu() if wts is not None else None),
            )

    # Save global params early
    if rank == 0:
        torch.save(
            {
                "format_version": 3,
                "centroid_mode": h.centroid_mode,
                "centroids": cents.detach().cpu(),
                "grid_dim": (gx, gy, gz),
                "cluster_2d": bool(h.cluster_2d),
                "boundary_margin": float(h.boundary_margin),
                "ray_samples": int(h.ray_samples),
                "aabb_global": global_box.aabb.detach().cpu().contiguous(),
                "scene_scale": float(h.scene_scale),
                "near_far_override_m": (
                    (float(h.near) if h.near is not None else None),
                    (float(h.far) if h.far is not None else None),
                ),
            },
            out / "params.pt",
        )

    if dist.is_initialized():
        dist.barrier()

    cents = cents.to(device=device, dtype=torch.float32)
    C = cents.size(0)

    # Near/Far override in normalized units
    near_far_override = (
        (float(h.near) / pose_scale) if h.near is not None else None,
        (float(h.far) / pose_scale) if h.far is not None else None,
    )

    # Per-expert AABBs & counts on device (NCCL-safe)
    mins = torch.full((C, 3), float("inf"), dtype=torch.float32, device=device)
    maxs = torch.full((C, 3), float("-inf"), dtype=torch.float32, device=device)
    cnts = torch.zeros(C, dtype=torch.long, device=device)

    # Process splits
    for split in ("train", "val"):
        meta = _meta_list(ds, split)
        idxs = np.arange(rank, len(meta), world)

        _log(rank, f"[{split}] images: {len(meta)} | rank {rank}/{world}")
        _log(rank, f"[{split}] boundary_margin={h.boundary_margin}")

        # Counters
        ctr_device = device if device.type == "cuda" else torch.device("cpu")
        tot_pix_rank = torch.zeros((), dtype=torch.long, device=ctr_device)
        pix_per_cell = torch.zeros(C, dtype=torch.long, device=ctr_device)
        imgs_with_pix = torch.zeros(C, dtype=torch.long, device=ctr_device)
        rays_total_rank = torch.zeros((), dtype=torch.long, device=ctr_device)
        rays_intersect_rank = torch.zeros((), dtype=torch.long, device=ctr_device)

        for i in tqdm(idxs, disable=(rank != 0), desc=f"masks:{split}"):
            mp = meta[i]
            stem = mp.stem
            fname = stem + ".pt"

            if h.resume and all_ok_for_image(C, out, fname):
                continue

            md = torch.load(mp, map_location="cpu")
            H, W = int(md["H"]), int(md["W"])
            fx, fy, cx, cy = md["intrinsics"]
            c2w = md["c2w"].to(device)

            # Camera-frame directions (RUB) -> world DRB rays via c2w; normalized space
            dirs = get_ray_directions(H, W, fx, fy, cx, cy, h.center_pixels, device)
            img_rays = get_rays(
                dirs,
                c2w,
                scene_box=global_box,
                aabb_max_bound=1e10,
                aabb_invalid_value=float("inf"),
            ).view(
                -1, 8
            )  # [ox,oy,oz, dx,dy,dz, near, far] normalized DRB

            # Apply optional global near/far override
            img_rays, valid = clamp_rays_near_far(img_rays, near_far_override)

            Npix = img_rays.shape[0]
            rays_total_rank += Npix
            rays_intersect_rank += valid.sum()

            valid_img = valid.view(H, W).cpu()

            # Voronoi assignment + AABB streaming
            if h.orig:
                voronoi_mask = compute_voronoi_orig(
                    img_rays,
                    ray_samples=h.ray_samples,
                    ray_chunk_size=h.ray_chunk_size,
                    sample_chunk_size=h.sample_chunk_size,
                    centroids=cents,
                    cluster_2d=h.cluster_2d,
                    device=device,
                    boundary_margin=h.boundary_margin,
                )  # (N, C) CPU
            else:
                voronoi_mask = compute_voronoi_opt(
                    img_rays,
                    ray_samples=h.ray_samples,
                    ray_chunk_size=h.ray_chunk_size,
                    sample_chunk_size=h.sample_chunk_size,
                    centroids=cents,
                    cluster_2d=h.cluster_2d,
                    device=device,
                    boundary_margin=h.boundary_margin,
                    fp16=h.fp16,
                    update_aabbs=True,
                    mins_out=mins,
                    maxs_out=maxs,
                    counts_out=cnts,
                )

            voronoi_mask = voronoi_mask.view(H, W, C)

            # Optional per-pixel segmentation
            seg = None
            if h.segmentation_path:
                seg = _read_zip_mask(Path(h.segmentation_path) / (stem + ".pt"))
                if seg is not None:
                    seg = seg.view(H, W).bool()

            # Save per-centroid masks
            for cid in range(C):
                m = voronoi_mask[..., cid] & valid_img
                if seg is not None:
                    m = m & seg
                s = int(m.sum())
                pix_per_cell[cid] += s
                if s > 0:
                    imgs_with_pix[cid] += 1
                _save_zip_tensor(out / f"{cid}" / (fname), m.contiguous())

            tot_pix_rank += H * W

        # Reduce split stats
        if dist.is_initialized():
            dist.barrier()
            dist.all_reduce(tot_pix_rank, op=dist.ReduceOp.SUM)
            dist.all_reduce(pix_per_cell, op=dist.ReduceOp.SUM)
            dist.all_reduce(imgs_with_pix, op=dist.ReduceOp.SUM)
            dist.all_reduce(rays_total_rank, op=dist.ReduceOp.SUM)
            dist.all_reduce(rays_intersect_rank, op=dist.ReduceOp.SUM)

        if rank == 0:
            total = int(tot_pix_rank.item())
            pct = (pix_per_cell.double() / max(1, total) * 100.0).tolist()
            _log(
                rank,
                f"[{split}] SceneBox ray coverage: {int(rays_intersect_rank.item()):,} / {int(rays_total_rank.item()):,} "
                f"({(float(rays_intersect_rank.item())/max(1,int(rays_total_rank.item()))*100.0):.3f}%)",
            )
            _log(rank, f"[{split}] total_pixels={total:,}")
            _log(
                rank,
                f"[{split}] pixels_per_centroid={[int(x) for x in pix_per_cell.cpu().tolist()]}",
            )
            _log(
                rank, f"[{split}] coverage_pct_per_centroid={[round(x,4) for x in pct]}"
            )
            _log(
                rank,
                f"[{split}] images_with_pixels_per_centroid={[int(x) for x in imgs_with_pix.cpu().tolist()]}",
            )

    # --- Distributed reduction for AABBs & counts ---
    if dist.is_initialized():
        dist.barrier()
        dist.all_reduce(mins, op=dist.ReduceOp.MIN)
        dist.all_reduce(maxs, op=dist.ReduceOp.MAX)
        dist.all_reduce(cnts, op=dist.ReduceOp.SUM)

    # --- Clamp to global AABB & fix empties ---
    # mins/maxs in normalized coords; clamp to global box
    mins = torch.maximum(mins, aabb_min_g)
    maxs = torch.minimum(maxs, aabb_max_g)

    # For experts with zero samples, create a tiny epsilon box around centroid
    empties = cnts == 0
    if empties.any():
        # epsilon ~ 1e-6 of global extent
        extent = (aabb_max_g - aabb_min_g).abs()
        eps = torch.clamp(extent * 1e-6, min=1e-7)
        cen_full = cents  # (C,3), already on device
        # Clamp centroid to global then pad by eps
        cclamped = torch.minimum(torch.maximum(cen_full, aabb_min_g), aabb_max_g)
        mins[empties] = torch.maximum(cclamped[empties] - eps, aabb_min_g)
        maxs[empties] = torch.minimum(cclamped[empties] + eps, aabb_max_g)

    # --- Optional dilation of per-expert AABBs (normalized coords) ---
    if getattr(h, "box_margin", 0.0) and h.box_margin > 0.0:
        margin = float(h.box_margin) / pose_scale
        mins = torch.maximum(mins - margin, aabb_min_g)
        maxs = torch.minimum(maxs + margin, aabb_max_g)
    # Altitude same as the boc for everyone!
    mins[:, 0] = aabb_min_g[0]
    maxs[:, 0] = aabb_max_g[0]

    # Move to CPU for saving/logging
    mins_cpu = mins.detach().cpu()
    maxs_cpu = maxs.detach().cpu()
    cnts_cpu = cnts.detach().cpu()
    cents_cpu = cents.detach().cpu()
    aabb_global_cpu = aabb_global.detach().cpu()

    # --- Save per-expert SceneBoxes ---
    if rank == 0:
        torch.save(
            {
                "format_version": 3,
                "aabb_global": aabb_global_cpu,  # (2,3)
                "mins": mins_cpu,  # (C,3)
                "maxs": maxs_cpu,  # (C,3)
                "counts": cnts_cpu,  # (C,)
                "centroids": cents_cpu,  # (C,3)
                "grid_dim": (gx, gy, gz),
                "cluster_2d": bool(h.cluster_2d),
                "boundary_margin": float(h.boundary_margin),
                "ray_samples": int(h.ray_samples),
                "scene_scale": float(h.scene_scale),
            },
            out / "scene_boxes.pt",
        )

        # Human-readable log
        lines = []
        gmin = aabb_global_cpu[0].tolist()
        gmax = aabb_global_cpu[1].tolist()
        lines.append("==== GLOBAL ====")
        lines.append(f"global.min = {np.round(gmin,6).tolist()}")
        lines.append(f"global.max = {np.round(gmax,6).tolist()}")
        lines.append("")
        lines.append("==== PER-EXPERT LOCAL BOXES (normalized DRB) ====")

        for cid in range(C):
            mn = np.round(mins_cpu[cid].tolist(), 6).tolist()
            mx = np.round(maxs_cpu[cid].tolist(), 6).tolist()
            _log(rank, f"[AABB] expert={cid:03d} mins={mn} maxs={mx}")
            ct = int(cnts_cpu[cid].item())
            cen = np.round(cents_cpu[cid].tolist(), 6).tolist()
            lines.append(
                f"[{cid:03d}] count={ct:9d}  centroid={cen}  min={mn}  max={mx}"
            )
        (out / "scene_boxes.txt").write_text("\n".join(lines))

        # Short, organized logging to console
        _log(rank, "==== LOCAL SCENEBOX SUMMARY ====")
        _log(rank, f"Global AABB min={gmin}, max={gmax}")
        nonempty = int((cnts_cpu > 0).sum().item())
        _log(rank, f"Experts with samples: {nonempty}/{C}")
        _log(rank, f"Saved per-expert boxes to: {out/'scene_boxes.pt'}")
        _log(rank, f"Readable dump:         {out/'scene_boxes.txt'}")

    _log(rank, f"Done. Masks saved to: {out}")


if __name__ == "__main__":
    main(parse_args())
