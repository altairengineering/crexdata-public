import math
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import torch

from nerfs.color_space import linear_to_srgb


def uint8_from_linear01(img_lin: torch.Tensor) -> np.ndarray:
    img_srgb = linear_to_srgb(img_lin).clamp(0, 1)
    return (img_srgb * 255.0 + 0.5).to(torch.uint8).cpu().numpy()


def rub_to_drb_3x3(device, dtype):
    return torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype
    )


def drb_to_rub_3x3(device, dtype):
    """Inverse of rub_to_drb_3x3 (here it’s just transpose)."""
    M = rub_to_drb_3x3(device, dtype)
    return M.transpose(0, 1)


def safe_active_module(selection: str, model: torch.nn.Module) -> Optional[int]:
    try:
        if selection == "All":
            return None
        idx = int(selection)
        K = getattr(model, "num_submodules", None)
        if K is None:
            K = (
                len(getattr(model, "submodules", []))
                if hasattr(model, "submodules")
                else 0
            )
        return idx if 0 <= idx < K else None
    except Exception:
        return None


def safe_bg(selection: str) -> str:
    allowed = {"white", "black", "random", "none", "last_sample", "bg_nerf"}
    return selection if selection in allowed else "white"


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def rub_pose_look(
    center: np.ndarray, cam: np.ndarray, up_world: np.ndarray, *, same_D: bool = False
) -> np.ndarray:
    tgt = center.copy()
    if same_D:
        tgt[0] = cam[0]  # keep horizon level (same D)
    fwd = _normalize(tgt - cam)
    right = _normalize(np.cross(fwd, up_world))
    up = _normalize(np.cross(right, fwd))
    R = np.stack([right, up, -fwd], axis=1)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3] = cam
    return c2w


def yaw_rotate_RB(vec_RB: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Rotate a vector (ordered [D,R,B]) in the R–B plane around the Up axis.
    Works for either DRB or RUB, as long as the vector uses [D,R,B] ordering."""
    R, B = vec_RB[1], vec_RB[2]
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    R2 = c * R - s * B
    B2 = s * R + c * B
    out = vec_RB.copy()
    out[1], out[2] = R2, B2
    return out


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# * -------------------------------- Verify data correct! -----------------------------


_VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
_ID_RE = re.compile(r"^\d{6}$")  # 6-digit ids like 000123


def _list_images(img_dir: Path) -> List[Path]:
    return sorted([p for p in img_dir.iterdir() if p.suffix in _VALID_IMG_EXTS])


def _list_metadata(meta_dir: Path) -> List[Path]:
    return sorted([p for p in meta_dir.iterdir() if p.suffix == ".pt"])


def _filestem_ids(paths: List[Path]) -> List[str]:
    return [p.stem for p in paths]


def _looks_like_id(s: str) -> bool:
    return _ID_RE.match(s) is not None


def _find_prepared_root_from_batch(batch_dir: Path) -> Optional[Path]:
    """
    update_dataset.py writes batches to: <prepared_dir>/continual/<batch_tag>
    So prepared_root is batch_dir.parents[1].
    """
    # e.g., .../prepared/continual/batch_xxxx  -> parents = [batch_xxxx, continual, prepared, ...]
    if len(batch_dir.parents) < 2:
        return None
    candidate = batch_dir.parents[1]
    if (candidate / "coordinates.pt").exists():
        return candidate
    return None


def verify_continual_batch_dir(
    batch_dir_str: str, sample_meta_check: int = 3
) -> Dict[str, Any]:
    """
    Verifies structure and basic consistency of a continual batch folder:
      batch_dir/
        rgbs/*.jpg|png
        metadata/*.pt
        manifest.json   (optional but recommended)
    And checks the prepared root above it for coordinates.pt and mappings.txt.
    """
    report: Dict[str, Any] = {"ok": False, "errors": [], "warnings": [], "summary": {}}

    batch_dir = Path(batch_dir_str).expanduser().resolve()
    if not batch_dir.exists():
        report["errors"].append(f"Path does not exist: {batch_dir}")
        return report
    if not batch_dir.is_dir():
        report["errors"].append(f"Not a directory: {batch_dir}")
        return report

    # Required subdirs
    rgbs_dir = batch_dir / "rgbs"
    meta_dir = batch_dir / "metadata"
    if not rgbs_dir.is_dir():
        report["errors"].append(f"Missing subdir: {rgbs_dir}")
    if not meta_dir.is_dir():
        report["errors"].append(f"Missing subdir: {meta_dir}")
    if report["errors"]:
        return report

    imgs = _list_images(rgbs_dir)
    metas = _list_metadata(meta_dir)
    if len(imgs) == 0:
        report["errors"].append(f"No images found under {rgbs_dir} (expected JPG/PNG).")
    if len(metas) == 0:
        report["errors"].append(f"No metadata .pt files found under {meta_dir}.")
    if report["errors"]:
        return report

    img_ids = _filestem_ids(imgs)
    meta_ids = _filestem_ids(metas)
    # Must look like 6-digit ids and match as sets
    bad_img_ids = [s for s in img_ids if not _looks_like_id(s)]
    bad_meta_ids = [s for s in meta_ids if not _looks_like_id(s)]
    if bad_img_ids:
        report["errors"].append(
            f"Non-id image filenames: {bad_img_ids[:5]}{' ...' if len(bad_img_ids)>5 else ''}"
        )
    if bad_meta_ids:
        report["errors"].append(
            f"Non-id metadata filenames: {bad_meta_ids[:5]}{' ...' if len(bad_meta_ids)>5 else ''}"
        )
    if report["errors"]:
        return report

    img_set, meta_set = set(img_ids), set(meta_ids)
    missing_meta = sorted(list(img_set - meta_set))[:10]
    missing_imgs = sorted(list(meta_set - img_set))[:10]
    if missing_meta:
        report["errors"].append(
            f"{len(img_set - meta_set)} ids have an image but no metadata, e.g. {missing_meta}"
        )
    if missing_imgs:
        report["errors"].append(
            f"{len(meta_set - img_set)} ids have metadata but no image, e.g. {missing_imgs}"
        )
    if report["errors"]:
        return report

    # Locate prepared root and verify coordinates/mappings
    prepared_root = _find_prepared_root_from_batch(batch_dir)
    if prepared_root is None:
        report["errors"].append(
            "Could not locate prepared root (expected <prepared_dir>/continual/<batch>); "
            "coordinates.pt not found two levels up."
        )
        return report

    coords = prepared_root / "coordinates.pt"
    mappings = prepared_root / "mappings.txt"
    if not coords.exists():
        report["errors"].append(f"Missing coordinates.pt at {coords}")
        return report
    if not mappings.exists():
        report["warnings"].append(
            f"mappings.txt not found at {mappings} (will still proceed)."
        )

    # Optional: spot-check a few metadata tensors for required schema
    sample = metas[:sample_meta_check]
    meta_problems = []
    for mp in sample:
        try:
            md = torch.load(mp)
            H = int(md.get("H", -1))
            W = int(md.get("W", -1))
            c2w = md.get("c2w", None)
            intr = md.get("intrinsics", None)
            if H <= 0 or W <= 0:
                meta_problems.append(f"{mp.name}: bad H/W ({H},{W})")
            if c2w is None or tuple(c2w.shape) != (3, 4):
                meta_problems.append(f"{mp.name}: c2w must be (3,4)")
            if intr is None or (len(intr) != 4):
                meta_problems.append(f"{mp.name}: intrinsics must be [fx,fy,cx,cy]")
        except Exception as e:
            meta_problems.append(f"{mp.name}: load error: {e}")

    if meta_problems:
        report["errors"].append("Metadata schema problems: " + "; ".join(meta_problems))
        return report

    report["ok"] = True
    report["summary"] = {
        "batch_dir": str(batch_dir),
        "prepared_root": str(prepared_root),
        "counts": {"images": len(imgs), "metadata": len(metas)},
        "example_id": img_ids[0],
    }
    if not (batch_dir / "manifest.json").exists():
        report["warnings"].append("manifest.json not found (optional but recommended).")
    return report
