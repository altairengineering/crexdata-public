#!/usr/bin/env python3
"""
Update an existing prepared dataset with *only* newly registered COLMAP images.

What this script does:
  • Loads the canonical scene frame from <prepared_dir>/coordinates.pt
      - Required: origin_drb, pose_scale_factor
      - Optional: enu_ref_coords=(lat,lon,h) — if present, we reuse it to map
        ECEF→ENU for *new* images only. If absent, we assume the COLMAP world
        is already ENU-like and skip the conversion.
  • Reads the latest COLMAP sparse model from update_model_path/
  • Detects images that are *not* already in <prepared_dir>/mappings.txt
  • For each new image only:
      - Resolve the actual file path robustly (handles ../, absolute paths, etc.)
      - Undistort with the same intrinsics/distortion convention
      - Compute RUB→DRB rotation + DRB translation, normalize translation
      - Save to <prepared_dir>/continual/<batch_tag>/{rgbs,metadata}
      - Append to <prepared_dir>/mappings.txt
  • Writes a <prepared_dir>/continual/<batch_tag>/manifest.json with per-image records.

It does NOT:
  • Touch or recompute train/val, origin/scale, or old metadata
  • Rebuild COLMAP or run BA; it assumes your COLMAP model is already updated

Example:
  ./scripts/update_dataset.py --update_model_path data/drz/model  --image_path data/drz/images --prepared_dir data/drz/out/prepared --batch_tag batch_0001
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import argparse
from argparse import Namespace
import numpy as np
import torch
import cv2
from tqdm import tqdm
import pymap3d as pm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from data.colmap_utils import qvec2rotmat, read_model, get_cam_intrinsics
from data.transformations import (
    ellipsoid_wgs84,
    ecef_to_enu_rot,
    ENU_TO_DRB,
    RDF_TO_RUB,
)


# -------------------- CLI --------------------
def parse_args() -> Namespace:
    p = argparse.ArgumentParser(
        description="Append only NEW images to an existing prepared dataset (consistent with main preparer)."
    )
    p.add_argument(
        "--update_model_path",
        required=True,
        help="COLMAP project dir (expects sparse model in model_path.",
    )
    p.add_argument(
        "--prepared_dir",
        required=True,
        help="Existing prepared dataset directory (with coordinates.pt, mappings.txt).",
    )
    p.add_argument(
        "--image_path", default=None, help="Where the actual image files live."
    )
    p.add_argument(
        "--batch_tag",
        default=None,
        help="Name for the new batch under prepared_dir/continual/<batch_tag>. Defaults to an increasing index.",
    )
    # Optional ECEF→ENU forcing when coordinates.pt lacks a ref but you still need it
    p.add_argument(
        "--force_ecef_to_enu",
        action="store_true",
        help="Force ECEF→ENU using --enu_ref.* when coordinates.pt lacks a ref.",
    )
    p.add_argument(
        "--enu_ref",
        default="custom",
        choices=["custom"],
        help="ENU origin policy when forcing (only 'custom' supported here).",
    )
    p.add_argument("--enu_ref_lat", type=float, default=None)
    p.add_argument("--enu_ref_lon", type=float, default=None)
    p.add_argument("--enu_ref_alt", type=float, default=None)
    # Behavior when new translations exceed the normalized range
    p.add_argument(
        "--on_overflow",
        default="abort",
        choices=["abort", "clip"],
        help="If |t_norm|>1: abort (safe, default) or clip to [-1,1] (dangerous).",
    )
    # Optional verification for newly added cameras only
    p.add_argument(
        "--verify_enu_geodetic",
        action="store_true",
        help="At the end, print ENU→geodetic reconversion for the newly added images (requires ENU ref).",
    )
    return p.parse_args()


def main(hp: Namespace):
    update_model_path = Path(hp.update_model_path)
    prep_dir = Path(hp.prepared_dir)

    # 1) Load coordinates & mappings
    coords_path = prep_dir / "coordinates.pt"
    if not coords_path.exists():
        raise FileNotFoundError(f"coordinates.pt not found at {coords_path}")
    coordinates = torch.load(coords_path)

    origin_drb = (
        coordinates["origin_drb"].numpy()
        if isinstance(coordinates.get("origin_drb"), torch.Tensor)
        else np.asarray(coordinates["origin_drb"], dtype=np.float32)
    )
    pose_scale = float(coordinates["pose_scale_factor"])
    enu_ref = coordinates.get("enu_ref_coords", None)  # may be absent/None

    mappings_path = prep_dir / "mappings.txt"
    existing_names: set[str] = set()
    existing_ids: list[int] = []
    if mappings_path.exists():
        with mappings_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                name, idpt = line.split(",")
                existing_names.add(name)
                try:
                    existing_ids.append(int(Path(idpt).stem))
                except Exception:
                    pass
    next_id = (max(existing_ids) + 1) if existing_ids else 0

    # 2) Load updated COLMAP model
    cameras, images, points3D = read_model(update_model_path)
    images_ordered = sorted(images.values(), key=lambda x: x.id)
    name_to_idx = {im.name: i for i, im in enumerate(images_ordered)}

    # Precompute COLMAP world → camera and camera centers (world)
    R_w2c_rdf_list = []
    C_world_list = []
    for im in images_ordered:
        R_w2c = np.asarray(qvec2rotmat(im.qvec), dtype=np.float64)
        t = np.asarray(im.tvec, dtype=np.float64).reshape(3, 1)
        Cw = (-R_w2c.T @ t).reshape(3)
        R_w2c_rdf_list.append(R_w2c)
        C_world_list.append(Cw)
    R_w2c_rdf = np.stack(R_w2c_rdf_list, axis=0)
    R_c2w_rdf = np.transpose(R_w2c_rdf, (0, 2, 1))
    C_world = np.stack(C_world_list, axis=0)

    # 3) Decide ENU conversion rules (consistent with preparer)
    ell = ellipsoid_wgs84()
    if enu_ref is None:
        print(
            "[UPDATE] No ENU reference in coordinates.pt → assuming COLMAP world is already ENU-like."
        )
        Q_ecef2enu = None
    else:
        lat0, lon0, h0 = map(float, enu_ref)
        print(
            f"[UPDATE] Reusing ENU reference from coordinates.pt: lat={lat0:.8f}, lon={lon0:.8f}, h={h0:.3f}"
        )
        Q_ecef2enu = ecef_to_enu_rot(lat0, lon0)

    if enu_ref is None and hp.force_ecef_to_enu:
        if None in (hp.enu_ref_lat, hp.enu_ref_lon, hp.enu_ref_alt):
            raise ValueError(
                "--force_ecef_to_enu requires --enu_ref_lat, --enu_ref_lon, --enu_ref_alt"
            )
        lat0, lon0, h0 = (
            float(hp.enu_ref_lat),
            float(hp.enu_ref_lon),
            float(hp.enu_ref_alt),
        )
        print(
            f"[UPDATE] Forcing ECEF→ENU with user reference: lat={lat0:.8f}, lon={lon0:.8f}, h={h0:.3f}"
        )
        enu_ref = (lat0, lon0, h0)
        Q_ecef2enu = ecef_to_enu_rot(lat0, lon0)

    # 4) Prepare batch dirs
    new_root = prep_dir / "continual"
    new_root.mkdir(parents=True, exist_ok=True)

    if hp.batch_tag is None:
        i = 1
        while True:
            tentative = f"batch_{i:04d}"
            if not (new_root / tentative).exists():
                hp.batch_tag = tentative
                break
            i += 1
    batch_dir = new_root / hp.batch_tag
    (batch_dir / "rgbs").mkdir(parents=True, exist_ok=True)
    (batch_dir / "metadata").mkdir(parents=True, exist_ok=True)

    image_path = Path(hp.image_path) if hp.image_path else None

    # 5) Process only NEW images
    manifest_items = []
    added = 0
    # collect ENU for verification (new images only)
    added_cam_enu = []
    added_names = []

    with mappings_path.open("a") as map_f:
        for im in tqdm(images_ordered, desc="Saving new images & metadata"):
            if im.name in existing_names:
                continue

            j = name_to_idx[im.name]

            # ENU camera center and rotation in ENU (if applicable)
            if Q_ecef2enu is not None:
                X, Y, Z = C_world[j]
                E, N, U = pm.ecef2enu(X, Y, Z, lat0, lon0, h0, ell=ell)
                cam_enu = np.array([E, N, U], dtype=np.float64)
                R_c2w_rdf_in_ENU = Q_ecef2enu @ R_c2w_rdf[j]
            else:
                # assume COLMAP already ENU-like
                E, N, U = C_world[j]
                cam_enu = np.array([E, N, U], dtype=np.float64)
                R_c2w_rdf_in_ENU = R_c2w_rdf[j]

            # ENU→DRB translation (meters) and saved rotation (RUB→DRB)
            t_drb_m = (cam_enu @ ENU_TO_DRB.T).astype(np.float32)
            R_saved = (ENU_TO_DRB @ R_c2w_rdf_in_ENU @ RDF_TO_RUB).astype(np.float32)

            # Normalize translation (guard overflow)
            origin_drb_f32 = (
                origin_drb.astype(np.float32)
                if isinstance(origin_drb, np.ndarray)
                else np.asarray(origin_drb, dtype=np.float32)
            )
            t_norm = (t_drb_m - origin_drb_f32) / pose_scale
            max_abs = float(np.abs(t_norm).max())
            if max_abs > 1.0 + 1e-6 and hp.on_overflow == "abort":
                raise RuntimeError(
                    f"New camera '{im.name}' exceeds normalized range |t_norm|={max_abs:.3f} > 1. "
                    "Create a new scene version (recompute origin/scale with old+new) or rerun with --on_overflow clip."
                )
            if max_abs > 1.0 + 1e-6 and hp.on_overflow == "clip":
                print(
                    f"[WARN] Clipping translation for '{im.name}' (|t_norm|={max_abs:.3f} > 1)"
                )
                t_norm = np.clip(t_norm, -1.0, 1.0)

            # Intrinsics / undistort (same convention as preparer)
            cam = cameras[im.camera_id]
            K, distortion, dist_how = get_cam_intrinsics(cam)

            print(im.name)

            input
            img_bgr = cv2.imread(str(image_path) + "/" + im.name)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot read image: {image_path}")

            if dist_how == "fisheye":
                undist = cv2.fisheye.undistortImage(img_bgr, K, distortion)
            elif dist_how == "opencv":
                undist = cv2.undistort(img_bgr, K, distortion)
            else:
                undist = img_bgr  # camera model with no/unknown distortion

            new_id = next_id
            next_id += 1

            cv2.imwrite(str(batch_dir / "rgbs" / f"{new_id:06d}.jpg"), undist)

            T3x4 = torch.zeros(3, 4, dtype=torch.float32)
            T3x4[:, :3] = torch.from_numpy(R_saved)
            T3x4[:, 3] = torch.from_numpy(t_norm.astype(np.float32))

            meta = {
                "H": undist.shape[0],
                "W": undist.shape[1],
                "c2w": T3x4,  # rotation is RUB→DRB; translation is DRB-normalized
                "intrinsics": torch.tensor(
                    [K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32
                ),
                "distortion": torch.tensor(distortion, dtype=torch.float32),
            }
            torch.save(meta, batch_dir / "metadata" / f"{new_id:06d}.pt")

            map_f.write(f"{im.name},{new_id:06d}.pt\n")

            manifest_items.append(
                {
                    "image_name": im.name,
                    "id": f"{new_id:06d}",
                    "camera_id": int(im.camera_id),
                    "undistort": dist_how,
                    "pose_rev": 0,
                }
            )
            added += 1

            # collect for verification (new images only)
            added_cam_enu.append(cam_enu)
            added_names.append(im.name)

    # 6) Write manifest
    manifest = {
        "batch_tag": hp.batch_tag,
        "count": added,
        "items": manifest_items,
    }
    (prep_dir / "continual").mkdir(parents=True, exist_ok=True)
    with (batch_dir / "manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[UPDATE] Added {added} new images to {batch_dir}")

    # 7) Optional: verify ENU→Geodetic for newly added images only
    if hp.verify_enu_geodetic:
        if enu_ref is None:
            print(
                "[VERIFY] Skipped ENU→Geodetic: no ENU reference available (not in coordinates.pt and not forced)."
            )
        else:
            if len(added_cam_enu) == 0:
                print("[VERIFY] No new images added; nothing to verify.")
            else:
                ell = ellipsoid_wgs84()
                added_cam_enu = np.asarray(added_cam_enu, dtype=np.float64)
                latlonh = np.array(
                    [
                        pm.enu2geodetic(e, n, u, lat0, lon0, h0, ell=ell)
                        for (e, n, u) in added_cam_enu
                    ]
                )
                lat_min, lat_max = float(latlonh[:, 0].min()), float(
                    latlonh[:, 0].max()
                )
                lon_min, lon_max = float(latlonh[:, 1].min()), float(
                    latlonh[:, 1].max()
                )
                h_min, h_max = float(latlonh[:, 2].min()), float(latlonh[:, 2].max())
                print("\n[VERIFY] ENU→Geodetic for newly added images:")
                print(f"[VERIFY] lat range: {lat_min:.8f} .. {lat_max:.8f}")
                print(f"[VERIFY] lon range: {lon_min:.8f} .. {lon_max:.8f}")
                print(f"[VERIFY] alt range: {h_min:.3f} .. {h_max:.3f}")
                # sample first
                print(
                    f"[VERIFY] sample[0]: name='{added_names[0]}', lat={latlonh[0,0]:.8f}, lon={latlonh[0,1]:.8f}, alt={latlonh[0,2]:.3f}"
                )


if __name__ == "__main__":
    main(parse_args())
