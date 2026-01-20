#!/usr/bin/env python3
"""
Logs detailed dataset statistics and reports all poses, rotations, and translations in the canonical
**DRB world frame** (Down, Right, Back).

Storage convention (as produced by prepare_dataset.py)
------------------------------------------------------
- Rotation (3×3) in metadata is an operator that maps **RUB camera vectors → DRB world vectors**.
  In other words, for a camera-space RUB vector v_cam, the world vector is:
      v_world_drb = R_saved @ v_cam_rub
  This matrix already includes any ECEF→ENU world-frame change and the ENU→DRB basis mapping,
  as well as the camera-basis change RUB→RDF applied on the right during preparation.

- Translation (3×1) is stored in **DRB coordinates**, normalized to [-1, 1] by:
      t_norm_drb = (t_world_drb - origin_drb) / pose_scale_factor

This logger uses the rotation directly in **DRB** and denormalizes translation as needed.

Outputs
-------
Creates `info.txt` under the dataset root with:
  - Image counts, resolutions, and ray totals
  - Pose normalization and origin/scale
  - Camera positions in normalized and world DRB coordinates
  - Scene extent and baseline distances
  - Rotation determinant and mean forward vector (in DRB)
  - Intrinsics and FOV ranges

Example
-------
    ./scripts/log_dataset_info.py --dataset_path data/drz/out/data
"""

import sys
from pathlib import Path
from typing import List, Tuple
import math
from collections import Counter
from argparse import ArgumentParser

import torch

# ---------------------------------------------------------------
# CLI
# ---------------------------------------------------------------


def parse_args():
    p = ArgumentParser(description="Log dataset statistics in DRB coords.")
    p.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Root with coordinates.pt and train/val/metadata/*.pt",
    )
    return p.parse_args()


class Logger:
    def __init__(self, path: str):
        self.f = open(path, "w", encoding="utf-8")

    def write(self, s: str):
        line = str(s).rstrip()
        print(line)
        self.f.write(line + "\n")
        self.f.flush()

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------
def _quantiles(x: torch.Tensor, pct: List[float]) -> List[float]:
    if x.numel() == 0:
        return [float("nan")] * len(pct)
    q = torch.tensor(pct, dtype=torch.float32, device=x.device) / 100.0
    return torch.quantile(x, q).tolist()


def _range_mean_std(x: torch.Tensor) -> str:
    if x.numel() == 0:
        return "n/a"
    return (
        f"min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}, std={x.std():.6f}"
    )


def _fov_deg(W: float, H: float, fx: float, fy: float) -> Tuple[float, float]:
    return (
        2.0 * math.degrees(math.atan2(0.5 * W, fx)),
        2.0 * math.degrees(math.atan2(0.5 * H, fy)),
    )


def _hdr(s: str, logger: Logger) -> None:
    logger.write("\n" + s)
    logger.write("-" * len(s))


# ---------------------------------------------------------------
# Core
# ---------------------------------------------------------------


def load_coords(ds_root: Path):
    coords = torch.load(ds_root / "coordinates.pt", map_location="cpu")
    origin_drb = torch.as_tensor(coords["origin_drb"], dtype=torch.float32)
    pose_scale = float(coords["pose_scale_factor"])
    alt_band = coords.get("altitude_range_enu", None)
    if alt_band is not None:
        alt_band = torch.as_tensor(alt_band, dtype=torch.float32)
    return origin_drb, pose_scale, alt_band


def list_metadata(ds_root: Path) -> Tuple[List[Path], List[Path]]:
    train = sorted((ds_root / "train" / "metadata").glob("*.pt"))
    val = sorted((ds_root / "val" / "metadata").glob("*.pt"))
    return train, val


def _signed_roll_deg(R_stack: torch.Tensor) -> torch.Tensor:
    """
    Signed roll (deg) around camera forward axis.
    R_stack: (N,3,3), RUB->DRB operator.
      right = R[:, :, 0], up = R[:, :, 1], back = R[:, :, 2]
      forward = -back
    """
    fwd = -R_stack[:, :, 2]  # (N,3) DRB forward
    up = R_stack[:, :, 1]  # (N,3) DRB up (camera up)
    world_up = torch.tensor(
        [-1.0, 0.0, 0.0], dtype=R_stack.dtype, device=R_stack.device
    )  # DRB Up = -Down(+X)

    def proj(v, n):
        return v - (torch.sum(v * n, dim=1, keepdim=True)) * n

    u_proj = torch.nn.functional.normalize(proj(up, fwd), dim=1)
    w_proj = torch.nn.functional.normalize(proj(world_up.expand_as(up), fwd), dim=1)

    cross_uw = torch.cross(w_proj, u_proj, dim=1)
    sin_th = torch.sum(cross_uw * fwd, dim=1)  # oriented sine about fwd
    cos_th = torch.sum(w_proj * u_proj, dim=1).clamp(-1, 1)  # cosine
    return torch.rad2deg(torch.atan2(sin_th, cos_th))  # (N,)


def log_dataset_info(ds_root: Path) -> None:
    log_path = ds_root / "info.txt"
    logger = Logger(str(log_path))

    origin_drb, pose_scale, alt_band = load_coords(ds_root)
    t_meta, v_meta = list_metadata(ds_root)
    all_meta = t_meta + v_meta

    # ---------------- Dataset Overview ----------------
    _hdr("Dataset", logger)
    logger.write(f"path                : {ds_root}")
    logger.write(
        f"images (train/val)  : {len(t_meta)} / {len(v_meta)} (total={len(all_meta)})"
    )

    # ---------------- Pose Normalization ----------------
    _hdr("Pose normalization", logger)
    logger.write(f"origin_drb (m)      : {origin_drb.tolist()}")
    logger.write(f"pose_scale_factor   : {pose_scale:.6f}")
    if alt_band is not None:
        logger.write(
            f"altitude_range_enu (m): [{float(alt_band[0]):.3f}, {float(alt_band[1]):.3f}]"
        )

    # Accumulators
    cams_drb, R_list_drb = [], []
    intr_fx, intr_fy, intr_cx, intr_cy = [], [], [], []
    fovx, fovy = [], []
    res_ct = Counter()
    rays_train, rays_val = 0, 0

    # ---------------- Collect Metadata ----------------
    def collect(paths, ray_count):
        for mp in paths:
            md = torch.load(mp, map_location="cpu")
            H, W = int(md["H"]), int(md["W"])
            res_ct[(W, H)] += 1
            ray_count += W * H

            # Translation (normalized DRB)
            t_norm_drb = md["c2w"][:3, 3].to(torch.float32)
            cams_drb.append(t_norm_drb.unsqueeze(0))

            # Rotation: RUB->DRB operator; treat as DRB directly
            R_saved = md["c2w"][:3, :3].to(torch.float32)
            R_list_drb.append(R_saved)

            fx, fy, cx, cy = md["intrinsics"].tolist()
            intr_fx.append(fx)
            intr_fy.append(fy)
            intr_cx.append(cx)
            intr_cy.append(cy)
            xdeg, ydeg = _fov_deg(W, H, fx, fy)
            fovx.append(xdeg)
            fovy.append(ydeg)

        return ray_count

    rays_train = collect(t_meta, rays_train)
    rays_val = collect(v_meta, rays_val)

    cams_drb = torch.cat(cams_drb, dim=0) if cams_drb else torch.empty(0, 3)
    cams_world = cams_drb * pose_scale + origin_drb

    # ---------------- Ray Counts ----------------
    _hdr("Ray counts", logger)
    total_rays = rays_train + rays_val
    logger.write(f"train rays          : {rays_train:,} (~{rays_train/1e6:.3f} M)")
    logger.write(f"val rays            : {rays_val:,}   (~{rays_val/1e6:.3f} M)")
    logger.write(f"total rays          : {total_rays:,} (~{total_rays/1e6:.3f} M)")

    # ---------------- Image Resolutions ----------------
    _hdr("Image resolutions (W×H : count)", logger)
    for (W, H), c in sorted(res_ct.items()):
        logger.write(f"{W}×{H} : {c}")

    # ---------------- Camera Positions (Normalized DRB) ----------------
    _hdr("Camera positions — normalized DRB", logger)
    if cams_drb.numel() == 0:
        logger.write("no cameras found")
    else:
        pct = [0, 1, 5, 50, 95, 99, 100]
        for label, v in zip(
            ("Down(+X)", "Right(+Y)", "Back(+Z)"),
            (cams_drb[:, 0], cams_drb[:, 1], cams_drb[:, 2]),
        ):
            logger.write(f"{label}: {_range_mean_std(v)}")
            logger.write(
                f"{label} pct {pct}: {[round(x,6) for x in _quantiles(v,pct)]}"
            )
        out_of_range = (cams_drb.abs() > 1.0001).any(dim=1).sum().item()
        if out_of_range:
            logger.write(
                f"⚠ WARNING: {out_of_range} camera translations fall outside [-1,1] after normalization."
            )

    # ---------------- Camera Positions (World Meters) ----------------
    _hdr("Camera positions — world meters", logger)
    if cams_world.numel() == 0:
        logger.write("no cameras found")
    else:
        pct = [0, 1, 5, 50, 95, 99, 100]
        Xw, Yw, Zw = cams_world[:, 0], cams_world[:, 1], cams_world[:, 2]
        for label, v in zip(("Down(+X)", "Right(+Y)", "Back(+Z)"), (Xw, Yw, Zw)):
            logger.write(f"{label}(m): {_range_mean_std(v)}")
            logger.write(
                f"{label} pct(m) {pct}: {[round(x,3) for x in _quantiles(v,pct)]}"
            )

        bbox_min, bbox_max = cams_world.min(0).values, cams_world.max(0).values
        diag = torch.norm(bbox_max - bbox_min).item()
        logger.write(f"\nScene extent (world m): bbox diag={diag:.3f}")
        if cams_world.shape[0] > 1:
            idx = torch.randperm(len(cams_world))[: min(500, len(cams_world))]
            d = torch.cdist(cams_world[idx], cams_world[idx])
            d = d[d > 0]
            if d.numel() > 0:
                qs = torch.quantile(d, torch.tensor([0.05, 0.5, 0.95]))
                logger.write(
                    f"Baseline distance among cameras (m): min={d.min():.3f}, "
                    f"p05={qs[0]:.3f}, med={qs[1]:.3f}, p95={qs[2]:.3f}, max={d.max():.3f}"
                )

    # ---------------- Rotation Consistency (DRB) ----------------
    _hdr("Rotation consistency — DRB", logger)
    if R_list_drb:
        R_stack = torch.stack(R_list_drb, dim=0)  # (N,3,3)
        dets = torch.det(R_stack)
        ortho_err = torch.norm(
            torch.transpose(R_stack, 1, 2) @ R_stack - torch.eye(3), dim=(1, 2)
        )
        logger.write(
            f"det(R_drb): mean={dets.mean().item():.6f}, min={dets.min().item():.6f}, max={dets.max().item():.6f}"
        )
        logger.write(
            f"orthogonality ||R^T R - I||_F: mean={ortho_err.mean().item():.2e}, "
            f"p95={torch.quantile(ortho_err, torch.tensor(0.95)).item():.2e}, max={ortho_err.max().item():.2e}"
        )

        forwards = -R_stack[:, :, 2]  # DRB forward
        ups = R_stack[:, :, 1]  # DRB up

        mean_forward = forwards.mean(0)
        logger.write(f"Mean forward vector (DRB frame): {mean_forward.tolist()}")

        ex = torch.tensor([1.0, 0.0, 0.0], dtype=forwards.dtype)  # Down(+X)
        ey = torch.tensor([0.0, 1.0, 0.0], dtype=forwards.dtype)  # Right(+Y)
        ez = torch.tensor([0.0, 0.0, 1.0], dtype=forwards.dtype)  # Back(+Z)

        # Pitch: angle to Down(+X). 0° = nadir, 90° = horizontal, 180° = up.
        dot_down = torch.clamp(forwards @ ex, -1, 1)
        pitch_deg = torch.rad2deg(torch.acos(dot_down))

        # Yaw: heading on YZ plane; atan2(Back, Right)
        yaw_deg = torch.rad2deg(torch.atan2(forwards @ ez, forwards @ ey))

        # Signed roll about forward
        roll_deg = _signed_roll_deg(R_stack)

        def pct(x):
            return f"{(100.0*x):.1f}%"

        bands = [5, 15, 30, 45, 60, 180]
        counts = [
            (pitch_deg < bands[0]).float().mean().item(),
            ((pitch_deg >= bands[0]) & (pitch_deg < bands[1])).float().mean().item(),
            ((pitch_deg >= bands[1]) & (pitch_deg < bands[2])).float().mean().item(),
            ((pitch_deg >= bands[2]) & (pitch_deg < bands[3])).float().mean().item(),
            ((pitch_deg >= bands[3]) & (pitch_deg < bands[4])).float().mean().item(),
            (pitch_deg >= bands[4]).float().mean().item(),
        ]
        logger.write(
            "Pitch from Down (DRB): "
            f"<5° {pct(counts[0])}, 5–15° {pct(counts[1])}, 15–30° {pct(counts[2])}, "
            f"30-45° {pct(counts[3])}, 45–60° {pct(counts[4])}, ≥60° {pct(counts[5])}"
        )

        logger.write(
            f"Pitch stats (deg): mean={pitch_deg.mean().item():.2f}, "
            f"median={pitch_deg.median().item():.2f}, p95={pitch_deg.quantile(0.95).item():.2f}"
        )
        logger.write(
            f"Roll  stats (deg): mean={roll_deg.mean().item():.2f}, "
            f"median={roll_deg.median().item():.2f}, p95={roll_deg.quantile(0.95).item():.2f}"
        )
        logger.write(
            f"Yaw   stats (deg): mean={yaw_deg.mean().item():.2f}, std={yaw_deg.std().item():.2f}"
        )

    # ---------------- Intrinsics & FOV ----------------
    _hdr("Intrinsics & FOV", logger)
    if intr_fx:
        fx, fy = torch.tensor(intr_fx), torch.tensor(intr_fy)
        cx, cy = torch.tensor(intr_cx), torch.tensor(intr_cy)
        logger.write(f"fx: {_range_mean_std(fx)}")
        logger.write(f"fy: {_range_mean_std(fy)}")
        logger.write(f"cx: {_range_mean_std(cx)}")
        logger.write(f"cy: {_range_mean_std(cy)}")

        fovx_t = torch.tensor(fovx)
        fovy_t = torch.tensor(fovy)
        logger.write(f"FOVx(deg): {_range_mean_std(fovx_t)}")
        logger.write(f"FOVy(deg): {_range_mean_std(fovy_t)}")

        try:
            sample = torch.load((t_meta or v_meta)[0], map_location="cpu")
            H0, W0 = int(sample["H"]), int(sample["W"])
            cx_rel = (cx / max(W0, 1e-8) - 0.5).abs().mean().item()
            cy_rel = (cy / max(H0, 1e-8) - 0.5).abs().mean().item()
            logger.write(
                f"Principal point offset (mean abs): "
                f"|cx/W-0.5|={cx_rel:.4f}, |cy/H-0.5|={cy_rel:.4f}"
            )
        except Exception:
            pass

        if abs(fx.mean().item() - fy.mean().item()) / max(1e-8, fy.mean().item()) > 0.1:
            logger.write("⚠ WARNING: fx and fy differ by >10% (non-square pixels).")

    logger.write("\nDataset statistics complete.")


# ---------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------


def main():
    args = parse_args()
    ds_root = Path(args.dataset_path)
    if not (ds_root / "coordinates.pt").exists():
        print("ERROR: coordinates.pt not found", file=sys.stderr)
        sys.exit(1)
    log_dataset_info(ds_root)


if __name__ == "__main__":
    main()
