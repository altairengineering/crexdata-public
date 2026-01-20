"""
This module is an extension of the implementation of MegaNeRF's image_metadata.py tailored for our system.
https://github.com/cmusatyalab/mega-nerf/blob/main/mega_nerf/image_metadata.py

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

from pathlib import Path
from zipfile import ZipFile
from PIL import Image
import warnings

warnings.filterwarnings(
    "ignore", message="The given NumPy array is not writable", category=UserWarning
)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class ImageMetadata:
    def __init__(
        self,
        image_path: Path,
        c2w: torch.Tensor,
        W: int,
        H: int,
        intrinsics: torch.Tensor,
        image_index: int,
        is_val=False,
        mask_dir: Path | None = None,
    ):
        self.image_path = image_path
        self.c2w = c2w
        self.W = W
        self.H = H
        self.intrinsics = intrinsics
        self.image_index = image_index
        self.is_val = is_val
        self.mask_path = (
            (mask_dir / f"{self.image_path.stem}.pt") if mask_dir is not None else None
        )

    def __repr__(self):
        return (
            f"ImageMetadata(\n"
            f"  path={self.image_path},\n"
            f"  image_index={self.image_index},\n"
            f"  W={self.W}, H={self.H},\n"
            f"  intrinsics=\n{self.intrinsics},\n"
            f"  c2w=\n{self.c2w}\n"
            f")"
        )

    def load_image(self) -> torch.Tensor:
        img = Image.open(self.image_path).convert("RGB")
        if img.size != (self.W, self.H):
            img = img.resize((self.W, self.H), Image.LANCZOS)
        arr = np.array(img, dtype=np.uint8)  # H x W x 3, RGB
        return torch.from_numpy(arr)

    def load_mask(
        self,
    ) -> torch.Tensor:
        """
        Load mask from: masks_root/<cell_id>/<stem>.pt  (plain or zipped).
        Resize to this image's (H, W) with nearest if needed.
        Return flat bool tensor of shape (H*W,), or None if file missing.
        """
        if self.mask_path is None or not self.mask_path.exists():
            return None

        # load: try plain .pt first, else zipped .pt
        try:
            m = torch.load(self.mask_path, map_location="cpu")
        except Exception:
            with ZipFile(self.mask_path, "r") as zf:
                inner = zf.namelist()[0]
                with zf.open(inner) as f:
                    m = torch.load(f, map_location="cpu")

        # ensure 2D (H_mask, W_mask)
        if m.ndim == 1:
            if m.numel() == self.H * self.W:
                m = m.view(self.H, self.W)
            else:
                return None
        if m.ndim != 2:
            return None

        # resize to current (H, W) if needed (nearest keeps mask discrete)
        if (m.shape[0], m.shape[1]) != (self.H, self.W):
            m = (
                F.interpolate(
                    m.unsqueeze(0).unsqueeze(0).float(),
                    size=(self.H, self.W),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
            )

        return m.bool()  # (H,W)


class ImageMetaDataset(Dataset):
    def __init__(self, meta_list: list[ImageMetadata]):
        self.items = meta_list

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        md = self.items[i]
        rgbs = md.load_image()
        return {
            "meta": md,
            "rgbs_raw": rgbs,
        }
