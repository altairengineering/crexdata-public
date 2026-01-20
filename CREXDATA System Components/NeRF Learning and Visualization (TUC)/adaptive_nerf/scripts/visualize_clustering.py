#!/usr/bin/env python3
"""
Clustering Visualization (Mega-NeRF-style, concise)

Visualizes clustering using saved centroids + masks:

- viz/regions.png                          : 2D Voronoi region map (+contours if boundary_margin>1)
- viz/scatter_cameras.png                  : cameras colored by nearest centroid
- viz/clustering/<split>/<stem>_assign.png : RGB overlay colored by module (+magenta for overlaps)

Inputs
- Dataset: <dataset_path>/<split>/{metadata,rgbs}
  • metadata/*.pt with keys: H, W, c2w(3x4), intrinsics=[fx,fy,cx,cy]
  • rgbs/<stem>.jpg
- Masks: <mask_path>/{0..K-1}/<split>/<stem>.pt (zipped .pt preferred; raw .pt allowed)
- Params: <mask_path>/params.pt with: centroids (K×3), cluster_2d (bool), boundary_margin (float)

Example:
  ./scripts/visualize_clustering.py --dataset_path data/drz/out/prepared --mask_path data/drz/out/prepared/masks/g22_grid_bm15_ss14
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import zipfile
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2  # for reading RGBs


# -----------------------------
# Logging (always verbose)
# -----------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


log = logging.getLogger("viz")


# -----------------------------
# Loaders & helpers
# -----------------------------
@dataclass
class Params:
    centroids: torch.Tensor
    cluster_2d: bool
    boundary_margin: float


def _load_params(mask_path: Path) -> Params:
    p = mask_path / "params.pt"
    if not p.exists():
        log.warning("params.pt not found at %s; using defaults.", p)
        return Params(torch.empty(0, 3), True, 1.0)
    dd = torch.load(p, map_location="cpu")
    cents = torch.as_tensor(dd.get("centroids", []), dtype=torch.float32)
    return Params(
        centroids=cents,
        cluster_2d=bool(dd.get("cluster_2d", True)),
        boundary_margin=float(dd.get("boundary_margin", 1.0)),
    )


def _gather_meta_paths(dataset_path: Path, split: str) -> List[Path]:
    mdir = dataset_path / split / "metadata"
    if not mdir.exists():
        log.warning("No metadata dir for split '%s': %s", split, mdir)
        return []
    return sorted([p for p in mdir.glob("*.pt") if p.is_file()])


def _load_cam_centers(meta_paths: List[Path]) -> torch.Tensor:
    cams = []
    for mp in meta_paths:
        md = torch.load(mp, map_location="cpu")
        c2w = md.get("c2w", None)
        if isinstance(c2w, torch.Tensor) and c2w.shape == (3, 4):
            cams.append(c2w[:, 3].to(torch.float32).unsqueeze(0))
        else:
            log.info("Skipping %s (missing/invalid c2w)", mp.name)
    return torch.cat(cams, dim=0) if cams else torch.empty(0, 3)


def _meta_to_rgb_path(dataset_path: Path, split: str, meta_path: Path) -> Path:
    return dataset_path / split / "rgbs" / f"{meta_path.stem}.jpg"


# -----------------------------
# Mask reading
# -----------------------------
def _load_mask_from_disk(cell_dir: Path, stem: str) -> Optional[torch.Tensor]:
    pt_path = cell_dir / f"{stem}.pt"
    if not pt_path.exists():
        return None
    try:
        if zipfile.is_zipfile(pt_path):
            with zipfile.ZipFile(pt_path, "r") as zf:
                inner = stem + ".pt"
                if inner not in zf.namelist():
                    inner = zf.namelist()[0]
                with zf.open(inner, "r") as f:
                    obj = torch.load(f, map_location="cpu")
        else:
            obj = torch.load(pt_path, map_location="cpu")

        if isinstance(obj, torch.Tensor):
            m = obj
        elif isinstance(obj, dict):
            for k in ("mask", "m", "assign", "binary"):
                if k in obj:
                    m = obj[k]
                    break
            else:
                return None
        else:
            return None
        if m.ndim != 2:
            return None
        return m.to(torch.uint8) != 0
    except Exception as e:
        log.debug("Failed to read mask %s: %s", pt_path, e)
        return None


def _compose_assignment(masks_path: Path, stem: str, K: int, H: int, W: int):
    """Return (assign, overlap) or None:
    assign: HxW uint16 in [0..K-1] (K = unassigned)
    overlap: HxW bool (True where >=2 modules cover the pixel)
    """
    have_any = False
    assign = np.full((H, W), fill_value=K, dtype=np.uint16)  # K = unassigned
    cover_count = np.zeros((H, W), dtype=np.uint8)

    for mid in range(K):
        cell_dir = masks_path / str(mid)
        m = _load_mask_from_disk(cell_dir, stem)
        if m is None:
            continue
        m = m.numpy().astype(bool)
        have_any = True
        cover_count[m] += 1
        sel = m & (assign == K)  # first-come wins
        assign[sel] = mid

    if not have_any:
        return None
    overlap = cover_count >= 2
    return assign, overlap


# -----------------------------
# Split-level plotting (camera/centroid space only)
# -----------------------------
def save_overview_scatter(
    path: Path, centroids: torch.Tensor, cams: torch.Tensor, *, cluster_2d: bool
) -> None:
    if centroids.numel() == 0:
        log.warning("No centroids; skipping camera scatter.")
        return
    if cluster_2d:
        cents2 = centroids[:, 1:].cpu().numpy()
        cams2 = cams[:, 1:].cpu().numpy() if cams.numel() > 0 else np.empty((0, 2))
        xlabel, ylabel = "Y", "Z"
    else:
        cents2 = centroids[:, [0, 2]].cpu().numpy()
        cams2 = cams[:, [0, 2]].cpu().numpy() if cams.numel() > 0 else np.empty((0, 2))
        xlabel, ylabel = "X", "Z"

    assign = None
    if cams2.size and cents2.size:
        dists = np.linalg.norm(cams2[:, None, :] - cents2[None, :, :], axis=2)
        assign = dists.argmin(axis=1)

    base_cmap = plt.get_cmap("tab10")
    colors = [base_cmap(i % 10) for i in range(cents2.shape[0])]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=140)

    if assign is not None:
        for k in range(cents2.shape[0]):
            sel = assign == k
            if not np.any(sel):
                continue
            ax.scatter(
                cams2[sel, 0],
                cams2[sel, 1],
                s=14,
                alpha=0.9,
                color=colors[k],
                label=f"mod {k}",
            )

    ax.scatter(
        cents2[:, 0],
        cents2[:, 1],
        s=40,
        marker="x",
        linewidths=1.8,
        color="black",
        label="centroids",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Cameras + centroids (colored by nearest centroid)")
    ax.legend(loc="best", fontsize=14, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log.info("Saved %s", path)


def save_region_viz2d(
    path: Path,
    centroids: torch.Tensor,
    cams: Optional[torch.Tensor],
    *,
    cluster_2d: bool,
    boundary_margin: float,
    res: int = 600,
) -> None:
    if centroids.numel() == 0:
        log.warning("No centroids; skipping regions map.")
        return

    if cluster_2d:
        cents2 = centroids[:, 1:].cpu().numpy()
        cams2 = (
            cams[:, 1:].cpu().numpy()
            if (cams is not None and cams.numel() > 0)
            else None
        )
        xlabel, ylabel = "Y", "Z"
    else:
        cents2 = centroids[:, [0, 2]].cpu().numpy()
        cams2 = (
            cams[:, [0, 2]].cpu().numpy()
            if (cams is not None and cams.numel() > 0)
            else None
        )
        xlabel, ylabel = "X", "Z"

    # Determine plotting bounds directly from cameras (no AABB/SceneBox)
    if cams2 is not None and cams2.size:
        lo = cams2.min(axis=0)
        hi = cams2.max(axis=0)
        # Small padding (2%) for nicer framing
        pad = 0.02 * (hi - lo + 1e-9)
        lo -= pad
        hi += pad
    else:
        # Fallback square around centroids
        lo = cents2.min(axis=0) - 1.0
        hi = cents2.max(axis=0) + 1.0

    xs = np.linspace(lo[0], hi[0], res)
    ys = np.linspace(lo[1], hi[1], res)
    X, Y = np.meshgrid(xs, ys)
    G = np.stack([X.ravel(), Y.ravel()], axis=1)

    dists = np.linalg.norm(G[:, None, :] - cents2[None, :, :], axis=2)
    dmin = dists.min(axis=1, keepdims=True)
    assign = dists.argmin(axis=1)
    Z = assign.reshape(res, res)

    base_cmap = plt.get_cmap("tab10")
    colors = [base_cmap(i % 10) for i in range(cents2.shape[0])]
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(colors)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
    ax.imshow(
        Z,
        origin="lower",
        extent=[lo[0], hi[0], lo[1], hi[1]],
        interpolation="nearest",
        alpha=0.25,
        cmap=cmap,
    )

    if boundary_margin > 1.0:
        ratio = dists / (dmin + 1e-8)
        for k in range(cents2.shape[0]):
            Mk = (ratio[:, k] <= boundary_margin).reshape(res, res)
            ax.contour(
                xs, ys, Mk.astype(np.uint8), levels=[0.5], linewidths=1.0, alpha=0.8
            )

    if cams2 is not None and cams2.size:
        ax.scatter(cams2[:, 0], cams2[:, 1], s=8, alpha=0.6, c="black", label="cameras")
    ax.scatter(
        cents2[:, 0],
        cents2[:, 1],
        s=40,
        marker="x",
        linewidths=1.8,
        color="black",
        label="centroids",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("Routing regions (Voronoi / overlap)")
    ax.legend(loc="best", fontsize=14, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    log.info("Saved %s", path)


# -----------------------------
# Overlay & color helpers
# -----------------------------
def _build_palette(K: int):
    cmap = plt.get_cmap("tab10")
    cols = []
    for i in range(K):
        r, g, b, _ = cmap(i % 10)
        cols.append((int(b * 255), int(g * 255), int(r * 255)))  # BGR for OpenCV
    return cols


def _color_overlay(rgb: np.ndarray, assign: np.ndarray, palette, alpha=0.35):
    out = rgb.copy()
    H, W = assign.shape
    for mid, col in enumerate(palette):
        mask = assign == mid
        if not mask.any():
            continue
        color = np.array(col, dtype=np.float32)
        out[mask] = (alpha * color + (1 - alpha) * out[mask]).astype(np.uint8)
    return out


def _overlay_split(
    dataset_path: Path,
    mask_path: Path,
    split: str,
    out_root: Path,
    params: Params,
    limit: int = 0,
) -> None:
    meta_paths = _gather_meta_paths(dataset_path, split)
    if limit > 0:
        meta_paths = meta_paths[:limit]
    out_dir = out_root / "clustering" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    K = int(params.centroids.shape[0])
    palette = _build_palette(max(K, 1))

    log.info("[%s] overlaying %d images ...", split, len(meta_paths))
    for mp in tqdm(meta_paths, desc=f"[{split}] overlay"):
        md = torch.load(mp, map_location="cpu")
        H, W = int(md["H"]), int(md["W"])
        stem = mp.stem

        res = _compose_assignment(mask_path, stem, K, H, W)
        if res is None:
            log.warning("No masks for %s (%s). Skipping.", stem, split)
            continue

        assign, overlap = res
        rgb_path = _meta_to_rgb_path(dataset_path, split, mp)
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            log.warning("Missing RGB: %s", rgb_path)
            continue

        over = _color_overlay(rgb, assign, palette, alpha=0.35)

        if overlap.any():
            ov_tint = np.array([255, 0, 255], dtype=np.uint8)  # magenta
            over[overlap] = (0.25 * ov_tint + 0.75 * over[overlap]).astype(np.uint8)

        fig, ax = plt.subplots(figsize=(12, 5), dpi=140)
        ax.imshow(cv2.cvtColor(over, cv2.COLOR_BGR2RGB))
        ax.axis("off")

        from matplotlib.patches import Patch

        handles = [
            Patch(facecolor=np.array(p[::-1]) / 255.0, edgecolor="none")
            for p in palette
        ]
        labels = [f"module {i}" for i in range(K)]
        handles.append(Patch(facecolor=(1.0, 0.0, 1.0), edgecolor="none", alpha=0.7))
        labels.append("overlap")

        fig.subplots_adjust(right=0.80)
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(0.82, 0.5),
            fontsize=12,
            framealpha=0.9,
        )
        fig.tight_layout()
        out_path = out_dir / f"{stem}_assign.png"
        fig.savefig(out_path)
        plt.close(fig)
        log.info("Saved %s", out_path)

    log.info("[%s] overlays done.", split)


# -----------------------------
# Orchestrator
# -----------------------------
def _split_level_figures(viz_root: Path, dataset_path: Path, params: Params) -> None:
    # Use all cameras from both splits for nicer extents (no AABB/SceneBox)
    meta_train = _gather_meta_paths(dataset_path, "train")
    meta_val = _gather_meta_paths(dataset_path, "val")
    cams = _load_cam_centers(meta_train + meta_val)

    save_overview_scatter(
        viz_root / "scatter_cameras.png",
        params.centroids,
        cams,
        cluster_2d=params.cluster_2d,
    )
    save_region_viz2d(
        viz_root / "regions.png",
        params.centroids,
        cams,
        cluster_2d=params.cluster_2d,
        boundary_margin=params.boundary_margin,
        res=700,
    )


def run_visualize(h):
    viz_root = h.mask_path / "viz"
    viz_root.mkdir(parents=True, exist_ok=True)
    params = _load_params(h.mask_path)

    log.info(
        "Centroids: %s | cluster_2d=%s | boundary_margin=%.3f",
        list(params.centroids.shape),
        params.cluster_2d,
        params.boundary_margin,
    )
    _split_level_figures(viz_root, h.dataset_path, params)
    if not h.only_clusters:
        _overlay_split(
            h.dataset_path, h.mask_path, "train", viz_root, params, h.max_images
        )
        _overlay_split(
            h.dataset_path, h.mask_path, "val", viz_root, params, h.max_images
        )


# -----------------------------
# CLI
# -----------------------------
def _get_opts() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize clustering (scatter, regions, overlays)."
    )
    p.add_argument("--dataset_path", type=Path, required=True)
    p.add_argument("--mask_path", type=Path, required=True)
    p.add_argument(
        "--only_clusters",
        action="store_true",
        help="Only clustrs. No mask per image overlay",
    )
    p.add_argument(
        "--max_images", type=int, default=0, help="Cap images per split (0 = all)"
    )
    return p.parse_args()


def main():
    setup_logging()
    h = _get_opts()
    log.info(
        "dataset=%s | masks=%s | output=%s",
        h.dataset_path,
        h.mask_path,
        h.mask_path / "viz",
    )
    run_visualize(h)
    log.info("Done.")


if __name__ == "__main__":
    main()
