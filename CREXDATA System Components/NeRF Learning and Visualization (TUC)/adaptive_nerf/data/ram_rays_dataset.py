"""
This module is an extension of the implementation of MegaNeRF's memory_dataset.py tailored for our system.
https://github.com/cmusatyalab/mega-nerf/blob/main/mega_nerf/datasets/memory_dataset.py

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

"""

import multiprocessing
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import os

import torch
from torch.utils.data import Dataset

from .image_metadata import ImageMetadata
from nerfs.ray_sampling import get_rays, get_ray_directions, clamp_rays_near_far

torch.multiprocessing.set_sharing_strategy("file_system")


# -----------------------------------------------------------------------------
# Helper: per-image preprocessing (used in multiprocessing)
# -----------------------------------------------------------------------------
def _process_single_image(
    md: ImageMetadata,
    center_pixels: bool,
    val_balancing: bool,
    ray_gen_kwargs: dict,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Processes a single ImageMetadata entry and returns (rgbs, rays, indices)."""
    torch.set_num_threads(1)  # prevent thread explosion inside each process

    scene_box = ray_gen_kwargs["scene_box"]
    near_far_override = ray_gen_kwargs.get("near_far_override", None)
    if md is None:
        return None
    # Load image
    img = md.load_image()
    if img is None:
        return None

    # Ensure (H, W, 3)
    if img.ndim == 2 and img.shape[-1] == 3:
        img = img.view(md.H, md.W, 3)
    elif img.ndim == 3 and img.shape[0] == 3:
        img = img.permute(1, 2, 0).contiguous()
    elif img.ndim == 3 and img.shape[-1] == 3:
        pass
    else:
        return None

    # Load mask
    keep_mask = md.load_mask()
    if keep_mask is not None:
        if keep_mask.ndim == 1:
            keep_mask = keep_mask.view(md.H, md.W)

    # Apply Mega-NeRF validation balancing
    if md.is_val and val_balancing:
        if keep_mask is None:
            keep_mask = torch.ones(md.H, md.W, dtype=torch.bool)
        keep_mask = RamRaysDataset._apply_meganerf_val_balancing_static(
            keep_mask, md.H, md.W
        )

    # Skip empty
    if keep_mask is not None and keep_mask.sum().item() == 0:
        return None

    # Directions and rays
    fx, fy, cx, cy = md.intrinsics
    directions = get_ray_directions(
        md.H, md.W, fx, fy, cx, cy, center_pixels, device=torch.device("cpu")
    )
    c2w = md.c2w.cpu()
    image_rays = get_rays(directions, c2w, scene_box=scene_box)

    # Apply mask (on CPU)
    if keep_mask is not None:
        flat_mask = keep_mask.view(-1)
        image_rays = image_rays.view(-1, 8)[flat_mask]
        img = img.view(-1, 3)[flat_mask]
    else:
        image_rays = image_rays.view(-1, 8)
        img = img.view(-1, 3)

    # Clamp near/far
    image_rays, valid = clamp_rays_near_far(
        image_rays, near_far_override=near_far_override
    )
    if not valid.any():
        return None

    # Filter valid
    image_rays = image_rays[valid]
    img = img[valid].to(torch.float32).div_(255.0)

    indices = torch.full((img.shape[0],), md.image_index, dtype=torch.int32)
    return img, image_rays, indices


# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------
class RamRaysDataset(Dataset):
    def __init__(
        self,
        metadata_items: List[ImageMetadata],
        center_pixels: bool,
        val_balancing: bool = False,
        ray_gen_kwargs: Optional[dict] = None,
        num_workers: Optional[int] = None,
    ):
        """
        Multi-processing RamRaysDataset prodcessing. Inspired from MefaNeRF MemoryDataset but utilizes more cpus for increased efficiency
        """
        super().__init__()

        # Safe unpack of ray_gen_kwargs
        if ray_gen_kwargs is None:
            raise ValueError(
                "ray_gen_kwargs must contain keys: 'scene_box' and 'near_far_override'"
            )
        if "scene_box" not in ray_gen_kwargs:
            raise ValueError(
                "ray_gen_kwargs must contain keys: 'scene_box' and 'near_far_override'"
            )

        # Multiprocessing configuration
        cpu_count = os.cpu_count() or 1
        if num_workers is None:
            num_workers = min(8, max(1, cpu_count // 2))  # reasonable default

        rgbs, rays, indices = [], [], []

        scene_box = ray_gen_kwargs.get("scene_box", None)
        if scene_box is not None and scene_box.aabb.device != torch.device("cpu"):
            ray_gen_kwargs["scene_box"] = scene_box.to(torch.device("cpu"))

        with torch.no_grad():
            worker_fn = partial(
                _process_single_image,
                center_pixels=center_pixels,
                val_balancing=val_balancing,
                ray_gen_kwargs=ray_gen_kwargs,
            )

            if len(metadata_items) > 8 and num_workers > 1:
                print(
                    f"[RamRaysDataset] Using multiprocessing with {num_workers} workers..."
                )

                ctx = multiprocessing.get_context("spawn")
                with ProcessPoolExecutor(
                    max_workers=num_workers,
                    mp_context=ctx,
                    max_tasks_per_child=50,
                ) as executor:
                    for res in tqdm(
                        executor.map(worker_fn, metadata_items, chunksize=8),
                        total=len(metadata_items),
                    ):
                        if res is None:
                            continue
                        img_sel, image_rays, indices_tensor = res
                        rgbs.append(img_sel.contiguous())
                        rays.append(image_rays.contiguous())
                        indices.append(indices_tensor.contiguous())
                        del res, img_sel, image_rays, indices_tensor
            else:
                print("[RamRaysDataset] Using single-process mode.")
                for md in tqdm(metadata_items, desc="Processing Images"):
                    res = _process_single_image(
                        md, center_pixels, val_balancing, ray_gen_kwargs
                    )
                    if res is None:
                        continue
                    img_sel, image_rays, indices_tensor = res
                    rgbs.append(img_sel.contiguous())
                    rays.append(image_rays.contiguous())
                    indices.append(indices_tensor.contiguous())
                    del res, img_sel, image_rays, indices_tensor

        # Final concatenation
        if not rgbs:
            print("Warning: MemoryDataset ended up empty. Check masks/val logic.")
            self._rgbs = torch.zeros((0, 3), dtype=torch.float32)
            self._rays = torch.zeros((0, 8), dtype=torch.float32)
            self._img_indices = torch.zeros((0,), dtype=torch.int32)
            self._num_images = 0
        else:
            self._rgbs = torch.cat(rgbs, dim=0).contiguous()
            self._rays = torch.cat(rays, dim=0).contiguous()
            self._img_indices = torch.cat(indices, dim=0).contiguous()
            self._num_images = len(rgbs)
            self._img_unique_ids = torch.unique(self._img_indices).cpu().tolist()
            del rgbs, rays, indices

    # -------------------------------------------------------------------------
    # Dataset interface
    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        return self._rgbs.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            "rgbs": self._rgbs[idx],
            "rays": self._rays[idx],
            "img_indices": self._img_indices[idx],
        }

    # -------------------------------------------------------------------------
    # Static utility for multiprocessing compatibility
    # -------------------------------------------------------------------------
    @staticmethod
    def _apply_meganerf_val_balancing_static(
        keep_mask: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """Static variant for use in multiprocessing workers."""
        keep_mask = keep_mask.view(H, W)
        left = keep_mask[:, : W // 2]
        right = keep_mask[:, W // 2 :]
        discard_pos = int(right.sum().item())
        if discard_pos > 0:
            candidates = torch.arange(H * W, device=keep_mask.device).view(H, W)[
                :, : W // 2
            ]
            not_kept_left = candidates[~left]
            if not_kept_left.numel() > 0:
                perm = torch.randperm(not_kept_left.numel(), device=keep_mask.device)
                to_add = not_kept_left[perm[:discard_pos]]
                flat = keep_mask.view(-1)
                flat.scatter_(0, to_add, torch.ones_like(to_add, dtype=torch.bool))
                keep_mask = flat.view(H, W)
        keep_mask[:, W // 2 :] = False
        return keep_mask.view(-1).bool()
