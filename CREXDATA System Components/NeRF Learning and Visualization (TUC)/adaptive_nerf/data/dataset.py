from pathlib import Path
from typing import Optional, Tuple, List

import torch

from .ram_rays_dataset import RamRaysDataset
from .image_metadata import ImageMetadata
from utils import discover_cluster_cells


def get_dataset(P, dataset, only_test=False, ray_gen_kwargs: dict = None):
    """Build NeRF train/val ray datasets for 'drz' in flat or masked (cell) layout."""

    train_set = None
    test_set = None
    P.data_size = None
    if dataset == "drz":
        data_path = "." / Path(P.data_path) / "out" / P.data_dirname

        coordinate_info = torch.load(data_path / "coordinates.pt", map_location="cpu")
        origin_drb, pose_scale_factor = (
            coordinate_info["origin_drb"],
            coordinate_info["pose_scale_factor"],
        )
        print(f"Origin: {origin_drb}, scale factor: {pose_scale_factor}")
        print(
            "(near,far) Derived from intersections with Scenebox and override clips. Determined by clustering step!"
        )

        # If no mask_dir: original single full-scene datasets (backward-compatible)
        if not hasattr(P, "mask_dirname") or P.mask_dirname is None:
            train_metadata, val_metadata = get_image_metadata(
                data_path, P.downscale, mask_dir=None
            )

            camera_positions = torch.cat(
                [x.c2w[:3, 3].unsqueeze(0) for x in train_metadata + val_metadata]
            )
            print(
                "Camera range in metric space: {} {}".format(
                    camera_positions.min(dim=0)[0] * pose_scale_factor + origin_drb,
                    camera_positions.max(dim=0)[0] * pose_scale_factor + origin_drb,
                )
            )
            print(
                "Camera range in [-1, 1] space: {} {}".format(
                    camera_positions.min(dim=0)[0], camera_positions.max(dim=0)[0]
                )
            )
            print(
                f"Using {len(train_metadata)} training and {len(val_metadata)} validation images."
            )

            kwargs = {"center_pixels": True, "ray_gen_kwargs": ray_gen_kwargs}

            print(f"Processing Training Images")
            train_set = RamRaysDataset(
                metadata_items=train_metadata,
                num_workers=P.num_workers,
                **kwargs,
            )
            print(f"Processing Validation Images")
            test_set = RamRaysDataset(
                metadata_items=val_metadata,
                num_workers=P.num_workers,
                **kwargs,
            )
            if only_test:
                return test_set

            return train_set, test_set

        else:
            # ====== Masked, per-cell subdatasets (Mega-NeRF style) ======
            mask_dir = data_path / "masks" / P.mask_dirname
            mask_root = Path(mask_dir)
            mask_cells = discover_cluster_cells(mask_root)
            assert (
                mask_cells == P.num_submodules
            ), f"Mismatch. Mask directory contains {mask_cells} regions but the experiment is configured for {P.num_submodules}."
            print(f"Discovered {mask_cells} cells in masks: {mask_root}")

            train_sets, val_sets = [], []
            expert_box_list = ray_gen_kwargs.pop("expert_box_list")

            for cell_id in range(P.num_submodules):
                cell_mask_dir = mask_root / f"{cell_id}"

                train_md, val_md = get_image_metadata(
                    data_path, P.downscale, cell_mask_dir, only_test
                )

                if len(train_md) == 0 and len(val_md) == 0:
                    continue

                if P.cap_images is not None:
                    train_md = cap_metadata(train_md, P.cap_images)
                    val_md = cap_metadata(val_md, P.cap_images)

                ray_gen_kwargs["scene_box"] = expert_box_list[cell_id].to("cpu")
                kwargs = {"center_pixels": True, "ray_gen_kwargs": ray_gen_kwargs}

                print(f"Context generation for subgrid {cell_id}...")

                train_ds = (
                    RamRaysDataset(
                        metadata_items=train_md,
                        num_workers=P.num_workers,
                        **kwargs,
                    )
                    if not only_test
                    else None
                )
                val_ds = None
                if val_md is not None and len(val_md) > 0:
                    val_ds = RamRaysDataset(
                        metadata_items=val_md,
                        num_workers=P.num_workers,
                        **kwargs,
                    )

                train_print = (
                    ""
                    if only_test
                    else f"  • Train: {len(train_ds):,} rays from {train_ds._num_images} images\n"
                )
                val_print = (
                    f"  • Val:   {len(val_ds):,} rays from {val_ds._num_images} images"
                    if val_ds is not None
                    else f"  • Val:   0 rays from 0 images"
                )
                print(
                    f"[Subgrid {cell_id}] Context complete!\n" + train_print + val_print
                )
                # If a cell ends up empty (e.g., all masks empty), skip it
                if train_ds is not None and len(train_ds) > 0:
                    train_sets.append(train_ds)
                if val_ds is not None and len(val_ds) > 0:
                    val_sets.append(val_ds)

            P.dim_in, P.dim_out = 6, 4
            P.data_type = "ray"
            return train_sets, val_sets
    else:
        raise NotImplementedError()


def cap_metadata(md_list, cap_images):
    """Restrict ImageMetadata list if cap_images is set."""
    if cap_images is None or cap_images <= 0:
        return md_list  # take all
    if len(md_list) <= cap_images:
        return md_list  # nothing to cut

    idx = torch.randperm(len(md_list))[:cap_images].tolist()
    return [md_list[i] for i in idx]


def get_meta_lookups(train_md, val_md):
    """Build H/W lookup dicts by image_index for train and val metadata lists."""

    train_meta_lookup = (
        {md.image_index: {"H": md.H, "W": md.W} for md in train_md}
        if train_md is not None and len(train_md) > 0
        else None
    )

    val_meta_lookup = (
        {md.image_index: {"H": md.H, "W": md.W} for md in val_md}
        if val_md is not None and len(val_md) > 0
        else None
    )
    return train_meta_lookup, val_meta_lookup


def _list_metadata_files(d: Path) -> List[Path]:
    """Return sorted .pt files under directory d, or [] if missing."""
    if not d.exists() or not d.is_dir():
        return []
    files = [p for p in d.iterdir() if p.is_file() and p.suffix == ".pt"]
    files.sort(key=lambda x: x.name)
    return files


def get_image_metadata(
    data_path: str,
    scale_factor: float,
    mask_dir: Optional[str] = None,
    only_test: bool = False,
) -> Tuple[List[ImageMetadata], List[ImageMetadata]]:
    """
    Load ImageMetadata from a COLMAP-converted dataset.

    Supports:
      - Flat layout: <data_path>/{metadata,rgbs} → all as val, train=[].
      - Split layout: <data_path>/train/metadata and val/ or test/ metadata.

    Always returns (train_items, val_items); falls back to ([], []) if nothing is found.
    """

    root = Path(data_path)

    # ---- 1) Flat layout: <root>/metadata + <root>/rgbs ----
    flat_meta_dir = root / "metadata"
    flat_rgbs_dir = root / "rgbs"
    flat_meta = _list_metadata_files(flat_meta_dir)

    if flat_meta and flat_rgbs_dir.exists():
        all_paths = flat_meta[:]
        image_indices = {
            p.name: i for i, p in enumerate(sorted(all_paths, key=lambda x: x.name))
        }

        val_items = [
            get_metadata_item(p, image_indices[p.name], scale_factor, True, mask_dir)
            for p in flat_meta
        ]
        train_items = [] if only_test else []
        return train_items, val_items

    # ---- 2) Split layout: train + (val|test) ----
    train_meta_dir = root / "train" / "metadata"
    val_meta_dir = root / "val" / "metadata"
    test_meta_dir = root / "test" / "metadata"

    train_paths = _list_metadata_files(train_meta_dir)
    eval_paths = _list_metadata_files(val_meta_dir) or _list_metadata_files(
        test_meta_dir
    )

    # If we found any split files, build indices over the union (sorted by filename)
    if train_paths or eval_paths:
        all_paths = train_paths + eval_paths
        all_paths.sort(key=lambda x: x.name)
        image_indices = {p.name: i for i, p in enumerate(all_paths)}

        train_items = (
            [
                get_metadata_item(
                    p, image_indices[p.name], scale_factor, False, mask_dir
                )
                for p in train_paths
            ]
            if not only_test
            else []
        )

        val_items = [
            get_metadata_item(p, image_indices[p.name], scale_factor, True, mask_dir)
            for p in eval_paths
        ]
        return train_items, val_items

    return [], []


def get_metadata_item(
    metadata_path: Path,
    image_index: int,
    scale_factor: float,
    is_val: bool = False,
    mask_dir: Path | None = None,
) -> ImageMetadata:
    """Load a single ImageMetadata entry from its .pt file and corresponding RGB image."""

    image_path = None
    for extension in [".jpg", ".JPG", ".png", ".PNG"]:
        candidate = (
            metadata_path.parent.parent
            / "rgbs"
            / "{}{}".format(metadata_path.stem, extension)
        )
        if candidate.exists():
            image_path = candidate
            break

    if image_path is None:
        return

    metadata = torch.load(metadata_path, map_location="cpu")

    return ImageMetadata(
        image_path,
        metadata["c2w"],
        int(round(metadata["W"] * scale_factor)),
        int(round(metadata["H"] * scale_factor)),
        metadata["intrinsics"] * scale_factor,
        image_index,
        is_val,
        mask_dir,
    )
