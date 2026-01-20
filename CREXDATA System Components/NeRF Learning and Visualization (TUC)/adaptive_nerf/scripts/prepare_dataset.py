#!/usr/bin/env python3
"""
This is an extenstion of MegaNeRF's:
https://github.com/cmusatyalab/mega-nerf/blob/main/scripts/colmap_to_mega_nerf.py

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

COLMAP → (optional ECEF→ENU) → DRB → RUB (OpenGL-style storage)
================================================================

What this script does (in plain terms)
--------------------------------------
It takes a COLMAP reconstruction and writes a dataset that NeRF-style frameworks
(Mega-NeRF, etc.) can read without any frame-gotchas.

- We undistort and save RGB images.
- We write per-image camera metadata (intrinsics + camera-to-world).
- We also save a global origin and scale so everything is centered and sized
  consistently for training.

Outputs
-------
- train/rgbs/, val/rgbs/
- train/metadata/, val/metadata/
- coordinates.pt  (contains: origin_drb, pose_scale_factor)

Conventions (the one thing you must remember)
---------------------------------------------
- **Translations** are in **DRB** world coordinates:
  - D (Down), R (Right), B (Back).
- **Rotations** are stored in **RUB** (OpenGL-style):
  - R (Right), U (Up), B (Back), with the camera looking along **−Z**.
- This pairing is intentional: directions generated in camera RUB map
  directly to world DRB via the saved `c2w` matrix.

Why this pairing works
----------------------
Rendering code usually:
1) Builds camera-ray directions in **RUB**: `[ (i−cx)/fx, −(j−cy)/fy, −1 ]`.
2) Applies the saved rotation (RUB→DRB) to get **world** directions.
3) Uses the saved translation (in DRB) as the **world** ray origin.

Because both pieces are consistent (R maps to DRB, t lives in DRB), rays land
in the right place. No extra flips or swaps needed.

End-to-end transform flow (high level)
--------------------------------------
1) **COLMAP camera frame (RDF)**
   - Right, Down, Forward. COLMAP gives poses that map world points into this
     camera-local frame.

2) **(Optional) ECEF → ENU**
   - If enabled, we convert camera centers from global Earth (ECEF) to a local
     East–North–Up frame (ENU) using a chosen reference (first/mean/median/manual).

3) **World → DRB (internal world basis)**
   - We remap world axes to DRB:
     Down = −Up, Right = East, Back = North.
     This keeps a right-handed system where the camera looks along −Z.

4) **Write metadata in Mega-NeRF’s on-disk convention**
   - Intrinsics: `[fx, fy, cx, cy]`, image size (H, W).
   - `c2w` rotation: stored in **RUB** layout (OpenGL-style).
   - `c2w` translation: stored in **DRB**, **normalized** by `pose_scale_factor`
     around `origin_drb`.

Normalization (origin & scale)
------------------------------
- We store `origin_drb` and `pose_scale_factor` in `coordinates.pt`.
- Translations in each per-image metadata are centered/scaled using these values.
- If your renderer needs metric space, restore with:
  `t_metric = origin_drb + pose_scale_factor * t_drb`.

Compatibility note (Mega-NeRF & friends)
----------------------------------------
- Many pipelines generate camera rays in **RUB** and expect world coords in **DRB**.
- Our metadata is written to align with that expectation exactly, so you can plug
  it in without extra frame conversions.

Expected input layout
---------------------
data_path/
  ├── model/    # COLMAP sparse model (cameras.bin, images.bin, points3D.bin)
  └── images/   # All registered images used by the COLMAP model

Example usage
-------------
./scripts/prepare_dataset.py --data_path data/drz --output_path data/drz/out/prepared --val_split 0.3 --scale_strategy camera_max --ecef_to_enu --enu_ref median
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import argparse
from argparse import Namespace
import shutil
import numpy as np
import torch
import cv2
from tqdm import tqdm
import pymap3d as pm
from datetime import datetime

from data.colmap_utils import qvec2rotmat, read_model, get_cam_intrinsics
from data.transformations import (
    choose_enu_origin,
    ecef_to_enu_rot,
    ellipsoid_wgs84,
    enu_span_meters,
    is_likely_ecef,
    ENU_TO_DRB,
    RDF_TO_RUB,
)


# -------------------- CLI --------------------
def parse_args() -> Namespace:
    p = argparse.ArgumentParser(
        description="Convert COLMAP model to Mega-NeRF style dataset with optional ECEF→ENU conversion."
    )
    p.add_argument(
        "--data_path",
        type=str,
        default="data/drz2",
        help="COLMAP project dir with 'model' and 'images'.",
    )
    p.add_argument(
        "--output_path", type=str, default="data/drz2/out/debug", help="Output dir."
    )
    p.add_argument("--val_split", type=float, default=0.0, help="Validation fraction.")
    p.add_argument(
        "--seed", type=int, default=17, help="Random seed for deterministic split."
    )
    p.add_argument(
        "--scale_strategy",
        type=str,
        default="camera_max",
        choices=["camera_max", "camera_p99", "bbox_diag"],
        help="How to determine scene scale.",
    )
    p.add_argument(
        "--ecef_to_enu",
        action="store_true",
        help="Convert world coordinates from ECEF to local ENU.",
    )
    p.add_argument(
        "--enu_ref",
        type=str,
        default="median",
        choices=["first", "mean", "median", "custom"],
        help="ENU origin selection.",
    )
    p.add_argument(
        "--enu_ref_lat", type=float, default=None, help="Custom ENU lat (deg)."
    )
    p.add_argument(
        "--enu_ref_lon", type=float, default=None, help="Custom ENU lon (deg)."
    )
    p.add_argument(
        "--enu_ref_alt",
        type=float,
        default=None,
        help="Custom ENU alt (ellipsoidal meters).",
    )
    p.add_argument(
        "--points_low_alt_percentile",
        type=float,
        default=0.01,
        help="Percentile for lowest points3D altitude.",
    )
    p.add_argument(
        "--maximum_allowed_height",
        type=float,
        default=200.0,
        help="Max allowed height below lowest camera (m).",
    )
    p.add_argument(
        "--verify_enu_geodetic",
        action="store_true",
        help="At the end (and only if --ecef_to_enu is set), print ENU→geodetic reconversion for manual comparison.",
    )
    return p.parse_args()


# -------------------- Main --------------------
def main(hparams: Namespace) -> None:
    # Load COLMAP model
    cameras, images, points3D = read_model(str(Path(hparams.data_path) / "model"))
    images_ordered = sorted(
        images.values(), key=lambda x: x.id
    )  # canonical order by ID
    ordered_indices = np.arange(len(images_ordered), dtype=int)
    print(
        f"Loaded {len(images_ordered)} images; {len(points3D)} 3D points; {len(cameras)} cameras."
    )

    # Collect camera centers and COLMAP rotations (RDF, world→camera)
    camera_centers_world = []
    R_w2c_rdf_list = []

    for img in images_ordered:
        R_w2c_rdf = np.asarray(qvec2rotmat(img.qvec), dtype=np.float64)
        t_world = np.asarray(img.tvec, dtype=np.float64).reshape(3, 1)
        C_world = (-R_w2c_rdf.T @ t_world).reshape(3)  # camera center in world frame

        # Consistency check: R_w2c * C_world + t ≈ 0
        residual = R_w2c_rdf @ C_world + t_world.ravel()
        if np.linalg.norm(residual) > 1e-6:
            print(
                f"[WARN] COLMAP consistency residual for {img.id} ({img.name}): {np.linalg.norm(residual):.3e}"
            )

        camera_centers_world.append(C_world)
        R_w2c_rdf_list.append(R_w2c_rdf)

    camera_centers_world = np.stack(camera_centers_world, axis=0)  # (N,3)
    R_w2c_rdf = np.stack(R_w2c_rdf_list, axis=0)  # (N,3,3)

    # Heuristic for ECEF
    mean_radius = np.linalg.norm(camera_centers_world, axis=1).mean()
    print(
        f"Mean radius (ecef heuristic): {mean_radius:.1f} m  (ECEF? {'YES' if is_likely_ecef(camera_centers_world) else 'NO'})"
    )
    if hparams.ecef_to_enu and not is_likely_ecef(camera_centers_world):
        print(
            "[WARN] --ecef_to_enu requested but centers don't look like ECEF. Proceeding anyway."
        )

    # Points3D (COLMAP world coords)
    pts_xyz_world = None
    if points3D and len(points3D) > 0:
        pts_xyz_world = np.stack(
            [np.asarray(p.xyz, dtype=np.float64) for p in points3D.values()], axis=0
        )

    # c2w in RDF from COLMAP
    R_c2w_rdf = np.transpose(R_w2c_rdf, (0, 2, 1))  # (N,3,3)

    # Optional ECEF → ENU (world-frame change) for translations AND rotations
    if hparams.ecef_to_enu:
        ell = ellipsoid_wgs84()
        lats, lons, alts = np.array(
            [pm.ecef2geodetic(X, Y, Z, ell=ell) for X, Y, Z in camera_centers_world]
        ).T

        lat_min, lat_max = lats.min(), lats.max()
        lon_min, lon_max = lons.min(), lons.max()
        alt_min, alt_max = alts.min(), alts.max()
        dlat_m, dlon_m = enu_span_meters(
            lat_min,
            lat_max,
            lon_min,
            lon_max,
            lats.mean(),
            lons.mean(),
            alts.mean(),
            ell,
        )

        print("Geodetic spans:")
        print(f"  Latitude Δ={lat_max - lat_min:.8f}° (~{dlat_m:.1f} m)")
        print(f"  Longitude Δ={lon_max - lon_min:.8f}° (~{dlon_m:.1f} m)")
        print(f"  Altitude Δ={alt_max - alt_min:.3f} m")

        lat0, lon0, h0, desc = choose_enu_origin(
            hparams.enu_ref, lats, lons, alts, ordered_indices, hparams
        )
        print(
            f"\nENU origin: {desc} → lat={lat0:.8f}°, lon={lon0:.8f}°, ellipsoidal height={h0:.3f} m"
        )

        # Camera centers: ECEF→ENU (meters)
        cam_enu = np.array(
            [
                pm.ecef2enu(X, Y, Z, lat0, lon0, h0, ell=ell)
                for X, Y, Z in camera_centers_world
            ]
        )

        # World rotation change: ECEF → ENU (left-multiply)
        Q_ecef2enu = ecef_to_enu_rot(lat0, lon0)  # (3,3)
        R_c2w_rdf = Q_ecef2enu @ R_c2w_rdf  # broadcasting over (N,3,3)

        # Points3D → ENU for altitude (Up) distribution
        if pts_xyz_world is not None:
            pts_enu = np.array(
                [
                    pm.ecef2enu(x, y, z, lat0, lon0, h0, ell=ell)
                    for x, y, z in pts_xyz_world
                ]
            )
            pts_up_enu = pts_enu[:, 2]
        else:
            pts_up_enu = None
        enu_ref_coords = (float(lat0), float(lon0), float(h0))
    else:
        # Assume world is already ENU in meters
        cam_enu = camera_centers_world.copy()
        lat0 = lon0 = h0 = None
        print("\n[INFO] Assuming input world is already ENU-aligned (East, North, Up).")
        pts_up_enu = pts_xyz_world[:, 2] if pts_xyz_world is not None else None
        enu_ref_coords = None

    # Altitude (ENU Up) range
    cam_up_enu = cam_enu[:, 2]
    lowest_camera_alt = float(cam_up_enu.min())
    highest_camera_alt = float(cam_up_enu.max())
    lowest_acceptable_alt = lowest_camera_alt - hparams.maximum_allowed_height

    if pts_up_enu is not None and pts_up_enu.size > 0:
        p_low = float(np.quantile(pts_up_enu, hparams.points_low_alt_percentile))
        lowest_point_alt = max(p_low, lowest_acceptable_alt)
        if p_low != lowest_point_alt:
            print(
                f"[WARN] {hparams.points_low_alt_percentile*100:.0f}% percentile of points3D ({p_low:.3f} m) "
                f"is lower than camera lowest-{hparams.maximum_allowed_height:.0f}m ({lowest_acceptable_alt:.3f} m); using latter."
            )
    else:
        lowest_point_alt = float(lowest_acceptable_alt)
        print(
            f"[WARN] points3D is empty; using lowest camera altitude-{hparams.maximum_allowed_height:.0f}m as 'lowest point'."
        )

    altitude_range_enu_m = torch.FloatTensor([lowest_point_alt, highest_camera_alt])
    print(
        f"[ALTITUDE] range ENU (m): lowest point = {lowest_point_alt:.3f}, highest camera = {highest_camera_alt:.3f}"
    )

    # ENU → DRB translation (meters) via single matrix
    # [D, R, B]^T = ENU_TO_DRB @ [E, N, U]^T  ==  [-U, E, -N]^T
    # ENU → DRB translation (meters), all images
    T_drb = cam_enu @ ENU_TO_DRB.T  # (N,3)

    # Precompute RUB→DRB rotation for all images (saves time & avoids drift)
    R_saved_rub_to_drb = (ENU_TO_DRB @ R_c2w_rdf @ RDF_TO_RUB).astype(
        np.float32
    )  # (N,3,3)

    # Sanity: rotation/translation live in the SAME world basis (sample check)
    v_enu0 = cam_enu[0]
    v_drb_from_rot0 = ENU_TO_DRB @ v_enu0
    if not np.allclose(v_drb_from_rot0, T_drb[0], atol=1e-6):
        print("[ERR] R/T world-basis mismatch. ENU_TO_DRB@ENU != t_drb for i=0.")
        print("   ENU_TO_DRB@ENU:", v_drb_from_rot0, " t_drb:", T_drb[0])
        raise SystemExit(3)

    # Report DRB translation stats (meters, pre-normalization)
    print(f"\n{T_drb.shape[0]} images (pre-normalization)")
    print(f"[DRB] Down  range (m): {T_drb[:, 0].min():.3f} .. {T_drb[:, 0].max():.3f}")
    print(f"[DRB] Right range (m): {T_drb[:, 1].min():.3f} .. {T_drb[:, 1].max():.3f}")
    print(f"[DRB] Back  range (m): {T_drb[:, 2].min():.3f} .. {T_drb[:, 2].max():.3f}")

    # Origin and scale in DRB (meters), computed from translations
    max_vals = T_drb.max(axis=0)
    min_vals = T_drb.min(axis=0)
    origin_drb_m = (max_vals + min_vals) * 0.5
    dists_from_origin = np.linalg.norm(T_drb - origin_drb_m[None, :], axis=1)

    if hparams.scale_strategy == "camera_max":
        pose_scale_factor = float(dists_from_origin.max())
    elif hparams.scale_strategy == "camera_p99":
        pose_scale_factor = float(np.quantile(dists_from_origin, 0.99))
    else:  # bbox_diag
        diag = np.linalg.norm(max_vals - min_vals)
        pose_scale_factor = float(max(diag * 0.5, 1e-8))

    print(f"Origin (DRB, m): {origin_drb_m.tolist()}")
    print(f"Pose scale factor (m): {pose_scale_factor:.6f}")

    coordinates = {
        "origin_drb": torch.from_numpy(origin_drb_m.astype(np.float32)),  # meters, DRB
        "pose_scale_factor": pose_scale_factor,  # meters
        "altitude_range_enu": altitude_range_enu_m,  # meters, ENU
        "enu_ref_coords": enu_ref_coords,  # (lat, lon, h) or None
    }
    origin_drb_np = coordinates["origin_drb"].numpy().astype(np.float32)

    # Prepare output directories
    out_dir = Path(hparams.output_path)
    if out_dir.exists():
        resp = input(f"[WARNING] {out_dir} exists. Overwrite? [y/N]: ").strip().lower()
        if resp not in ("y", "yes"):
            print("Aborting.")
            return
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / split / "metadata").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "rgbs").mkdir(parents=True, exist_ok=True)

    # Train/Val split (by lexicographic name for deterministic mapping)
    all_imgs_by_name = sorted(images.values(), key=lambda x: x.name)
    N = len(all_imgs_by_name)
    num_val = max(0, int(round(hparams.val_split * N)))

    # Evenly spaced validation indices for full spatial/temporal coverage (drone data)
    if num_val > 0:
        val_positions = np.linspace(0, N - 1, num=num_val, endpoint=True)
        val_ids = set(np.round(val_positions).astype(int).tolist())
    else:
        val_ids = set()

    print(f"{num_val} images reserved for validation (evenly spaced).")

    # Build index map from name → index in images_ordered (ID order)
    index_by_name = {img.name: i for i, img in enumerate(images_ordered)}
    # Guard: ensure mapping is consistent
    for k, v in index_by_name.items():
        assert (
            images_ordered[v].name == k
        ), "Name/ID order mismatch — indexing bug risk."

    # Save loop
    mappings_f = (out_dir / "mappings.txt").open("w")
    try:
        for i, img in enumerate(
            tqdm(all_imgs_by_name, desc="Saving images and metadata")
        ):
            split = "val" if i in val_ids else "train"
            split_dir = out_dir / split

            # Intrinsics and undistortion
            cam = cameras[img.camera_id]
            K, distortion, dist_how = get_cam_intrinsics(cam)

            src_path = Path(hparams.data_path) / "images" / img.name
            distorted = cv2.imread(str(src_path))
            if distorted is None:
                raise FileNotFoundError(f"Cannot read image: {src_path}")

            if dist_how == "fisheye":
                undistorted = cv2.fisheye.undistortImage(distorted, K, distortion)
            elif dist_how == "opencv":
                undistorted = cv2.undistort(distorted, K, distortion)
            else:
                undistorted = distorted  # camera model with no/unknown distortion

            cv2.imwrite(str(split_dir / "rgbs" / f"{i:06d}.jpg"), undistorted)

            # Pose
            j = index_by_name[img.name]  # index in ID-ordered arrays

            R_save = R_saved_rub_to_drb[j]  # (3,3) float32
            t_save_m = T_drb[j].astype(np.float32)  # (3,)

            # Normalize translation
            t_norm = (t_save_m - origin_drb_np) / pose_scale_factor
            # Pack: rotation (RUB→DRB), translation DRB-normalized
            T3x4 = torch.zeros(3, 4, dtype=torch.float32)
            T3x4[:, :3] = torch.from_numpy(R_save)
            T3x4[:, 3] = torch.from_numpy(t_norm)

            meta = {
                "H": undistorted.shape[0],
                "W": undistorted.shape[1],
                "c2w": T3x4,  # rotation is RUB→DRB; translation is DRB-normalized
                "intrinsics": torch.tensor(
                    [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32
                ),
                "distortion": torch.tensor(distortion, dtype=torch.float32),
            }
            torch.save(meta, split_dir / "metadata" / f"{i:06d}.pt")
            mappings_f.write(f"{img.name},{i:06d}.pt\n")
    finally:
        mappings_f.close()

    torch.save(coordinates, out_dir / "coordinates.pt")
    print("\nDataset preparation complete.")

    # --- Optional: Verify ENU <-> Geodetic consistency (for manual comparison) ---
    if hparams.verify_enu_geodetic and hparams.ecef_to_enu:
        ell = ellipsoid_wgs84()
        print("\n[VERIFY] Recomputing ENU→Geodetic for verification...")
        enu_to_geo = np.array(
            [pm.enu2geodetic(e, n, u, lat0, lon0, h0, ell=ell) for e, n, u in cam_enu]
        )
        print(
            f"[VERIFY] Sample ENU→Geodetic[0]: lat={enu_to_geo[0,0]:.8f}, lon={enu_to_geo[0,1]:.8f}, alt={enu_to_geo[0,2]:.3f}"
        )
        print(
            f"[VERIFY] ENU→Geodetic lat range: {enu_to_geo[:,0].min():.8f} .. {enu_to_geo[:,0].max():.8f}"
        )
        print(
            f"[VERIFY] ENU→Geodetic lon range: {enu_to_geo[:,1].min():.8f} .. {enu_to_geo[:,1].max():.8f}"
        )
        print(
            f"[VERIFY] ENU→Geodetic alt range: {enu_to_geo[:,2].min():.3f} .. {enu_to_geo[:,2].max():.3f}"
        )


if __name__ == "__main__":
    main(parse_args())
