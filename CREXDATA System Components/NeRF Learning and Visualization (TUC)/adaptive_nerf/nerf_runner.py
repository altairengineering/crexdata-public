"""
runner.py — unified entrypoint with a single argument space and an --op switch.

Usage examples:
  python runner.py --op train <all your usual args...>
  python runner.py --op eval  --checkpoint_path runs/exp123 --fname best --tto 0,1,5
  python runner.py --op view  --checkpoint_path runs/exp123 --fname best \
                   --view_support episodic --viewer_port 7070 --ray_samples 64

Notes:
- Single argparse source: args.parse_args() contains EVERYTHING, including --op.
- One build_context(P, op) function builds a plain dict tailored to the operation.
- Three ops now: train(ctx), eval(ctx), view(ctx). No helper classes, no extra pre-parsing.
"""

import copy
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.task_dataset import TaskDataset
from common.args import parse_args
from utils import (
    Logger,
    set_random_seed,
    seed_worker as worker_init_fn,
    collate_episodes as collate_episodes_fn,
    collate_image_meta_data,
    load_clustering_meta,
    load_scene_boxes,
    resolve_checkpoint_dir,
    load_model_checkpoint,
)

from common.utils import get_optimizer
from data.dataset import get_dataset, get_image_metadata
from data.image_metadata import ImageMetaDataset
from data.multi_loader import MultiLoader
from models.inr.meta_container import MetaContainer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def build_context(P, op: str) -> dict:
    """Prepare a minimal, op-aware context dictionary used by train/eval/view."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Runtime knobs common to all ops
    P.world_size = torch.cuda.device_count()
    P.data_parallel = False
    P.rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(P.rank)

    set_random_seed(P.seed)

    ctx = {
        "P": P,
        "device": device,
        "rank": 0,
        "logger": None,
        "train_set": None,
        "test_set": None,
        "train_loader": None,
        "test_loader": None,
        "model": None,
        "optimizer": None,
        "global_box": None,
        "expert_box_list": None,
        "support_iter": None,
    }

    # resolve checkpoint path from job ID if needed
    if getattr(P, "checkpoint_path", None):
        P.checkpoint_path = resolve_checkpoint_dir(P.checkpoint_path, logs_root="..")
    print("Using checkpoint path:", P.checkpoint_path)

    # ---- clustering meta / SceneBox ----
    mask_dir = Path(P.data_path) / "out" / P.data_dirname / "masks" / P.mask_dirname
    print(mask_dir)
    clustering_params = load_clustering_meta(mask_dir)
    _data_path = "." / Path(P.data_path) / "out" / P.data_dirname
    coordinate_info = torch.load(_data_path / "coordinates.pt", map_location="cpu")

    global_box, expert_box_list = load_scene_boxes(mask_dir=mask_dir, device=device)

    ctx["global_box"] = global_box
    ctx["expert_box_list"] = expert_box_list
    P.near = (
        P.near / coordinate_info["pose_scale_factor"] if P.near is not None else None
    )
    P.far = P.far / coordinate_info["pose_scale_factor"] if P.far is not None else None
    print("Global_Scenebox: ", global_box)
    for i in range(len(expert_box_list)):
        print(f"Expert {i}: {expert_box_list[i]}")

    # ---- per-expert / base kwargs for Instant-NGP ----

    hash_enc_conf = {
        "levels": getattr(P, "high_levels", 16),
        "features_per_level": getattr(P, "high_features_per_level", 2),
        "log2_hashmap_size": getattr(P, "high_log2_hashmap_size", 20),
        "max_res": getattr(P, "high_max_res", 4096),
        "min_res": getattr(P, "high_min_res", 16),
        "interpolation": getattr(P, "interpolation", "Linear"),
    }

    # ---- per-expert / base kwargs for MetaNeRF* (instant/standard) ----
    nerf_kwargs = {
        "expert_box_list": expert_box_list,
        "hidden": P.dim_hidden,
        "sigma_depth": P.sigma_depth,
        "color_depth": P.color_depth,
        "dir_encoding": P.dir_encoding,
        "color_hidden": P.color_hidden,
        "use_sigmoid_rgb": True,
        "hash_enc_conf": hash_enc_conf,
    }

    # ---- occupancy config for the container ----
    occ_conf = {
        "use_occ": P.use_occ,
        "resolution": 128,  # per-level grid resolution
        "levels": 4,  # number of levels (multi-scale grid)
        "render_step_size": None,  # if None, defaults to scene_diag / 1000.0
        "occ_thre": 1e-2,  # final opacity threshold (after warmup)
        "alpha_thre": 1e-2,  # final opacity threshold (after warmup)
        "alpha_thre_start": 0.0,  # starting alpha threshold
        "alpha_thre_end": 1e-2,  # end alpha threshold (can equal alpha_thre)
        "cosine_anneal": True,  # smooth cosine ramp-up of alpha_thre
        "warmup_steps": 256,  # number of steps with alpha_thre=0 # !
        "update_interval": 16,  # update every N steps after warmup# !
        "ema_decay": 0.95,  # smoothing factor for occ density updates
        "cone_angle": 0.004,  # for step size scaling with ray depth
        "near_plane": (
            P.near
            if P.near is not None
            else 0.05 / coordinate_info["pose_scale_factor"]
        ),  # frustum start
        "far_plane": (
            P.far if P.far is not None else 1e3 / coordinate_info["pose_scale_factor"]
        ),  # frustum end
        "occ_frozen": False,  # set True to stop updating grid
        "occ_ready": False,  # internal flag; leave False at init
    }

    boundary_margin = min(max(1, P.bm), clustering_params["boundary_margin"])

    # ---- model ----
    model = MetaContainer(
        num_submodules=P.num_submodules,
        centroids=clustering_params["centroids"],
        aabb=global_box.aabb,
        nerf_variant=P.nerf_variant,  # "instant" or "standard"
        # routing
        boundary_margin=boundary_margin,
        cluster_2d=clustering_params["cluster_2d"],
        joint_training=P.data_parallel,
        # background
        use_bg_nerf=not P.no_bg_nerf,
        bg_hidden=P.bg_hidden,
        bg_encoding=getattr(P, "bg_encoding", "spherical"),
        # occupancy
        occ_conf=occ_conf,
        **nerf_kwargs,
    ).to(device)

    ctx["model"] = model
    g = torch.Generator()
    g.manual_seed(P.seed)

    logger = Logger(fn=P.fname, today=False, rank=P.rank, logs_path="../logs/")
    ctx["logger"] = logger

    # ---- per-op context ----
    if op == "train":

        # episodic datasets via TaskDataset
        ray_gen_kwargs = {
            "expert_box_list": expert_box_list,
            "near_far_override": (P.near, P.far),
        }
        train_set, test_set = get_dataset(
            P, dataset=P.dataset, ray_gen_kwargs=ray_gen_kwargs
        )

        if model.use_occ:
            train_md, val_md = get_image_metadata(
                data_path=_data_path, scale_factor=P.downscale
            )
            all_md = [md for md in (train_md or []) + (val_md or []) if md is not None]
            print("Premarking invisible cells...")
            model.premark_invisible_expert_cells(
                all_md, near_plane=P.near if P.near is not None else 1e-3
            )
            print("Cell premark complete!")

        wrapper_kwargs = {
            "S_target": P.support_rays,
            "Q_target": P.query_rays,
            "image_cap": 0.4,
            "min_rays_cell": int((P.support_rays + P.query_rays) * 0.5),
            "assignment_checkpoint": 0.7,
            "routing_policy": "dda",
            "cells": (1, P.cell_dim, P.cell_dim),
        }
        print("Creating tasks...")
        train_wrapped = [
            TaskDataset(ram_ds=ds, cell_id=i, **wrapper_kwargs)
            for i, ds in enumerate(train_set)
        ]
        print(
            f"Task cells per expert: {[len(train_wrapped[i].eligible_cells) for i in range(len(train_wrapped))]}"
        )
        test_wrapped = [
            TaskDataset(ram_ds=ds, cell_id=i, **wrapper_kwargs)
            for i, ds in enumerate(test_set)
        ]
        print("Task generation complete!")

        train_loader_kwargs = {
            "batch_size": P.batch_size,
            "collate_fn": collate_episodes_fn,
            "num_workers": 8,
            "prefetch_factor": 2,
            "pin_memory": True,
            "shuffle": False,
            "drop_last": False,
            "worker_init_fn": worker_init_fn,
            "generator": g,
        }
        test_loader_kwargs = {
            "batch_size": P.test_batch_size,
            "shuffle": False,
            "pin_memory": True,
            "num_workers": 0,
            "collate_fn": collate_episodes_fn,
            "worker_init_fn": worker_init_fn,
            "generator": g,
        }

        ctx["train_loader"] = MultiLoader(
            [DataLoader(ds, **train_loader_kwargs) for ds in train_wrapped]
        )
        ctx["test_loader"] = MultiLoader(
            [DataLoader(ds, **test_loader_kwargs) for ds in test_wrapped]
        )
        ctx["optimizer"] = get_optimizer(P, ctx["model"])

    elif op in ["eval", "video"]:

        if not getattr(P, "checkpoint_path", None):
            raise ValueError("--checkpoint_path is required when --op eval")

        # metadata-only eval loader (kept as you had it)
        data_path = Path(P.data_path) / "out" / P.data_dirname
        _, test_meta_list = get_image_metadata(data_path, P.downscale, only_test=False)

        test_meta = ImageMetaDataset(test_meta_list)
        ctx["test_loader"] = DataLoader(
            test_meta,
            shuffle=False,
            batch_size=P.test_batch_size,
            pin_memory=True,
            num_workers=0,
            collate_fn=collate_image_meta_data,
        )

        load_model_checkpoint(P, model, ctx["logger"])

    elif op == "view":
        # For viewing, we optionally load a checkpoint and build a small support iterator (episodic or rays)
        if P.checkpoint_path is None:
            raise ValueError("--checkpoint_path needs to be added")

        load_model_checkpoint(P, model, ctx["logger"])

        data_path = "." / Path(P.data_path) / "out" / P.data_dirname
        # scene info
        ctx["coordinate_info"] = torch.load(
            data_path / "coordinates.pt", map_location="cpu"
        )

    else:
        raise ValueError("Only train, eval, and view ops are supported")

    return ctx


# -----------------------------
# Ops
# -----------------------------


def train(ctx: dict):
    from pipelines.offline_stage import setup
    from pipelines.offline_stage.trainer import meta_trainer

    P = ctx["P"]

    train_func, test_func, _, _ = setup(P.algo, P)

    ctx["logger"].log(P)
    ctx["logger"].log(ctx["model"])

    meta_trainer(
        P,
        train_func,
        test_func,
        ctx["model"],
        ctx["optimizer"],
        ctx["train_loader"],
        ctx["test_loader"],
        ctx["logger"],
    )
    ctx["logger"].close_writer()


def eval(ctx: dict):
    from pipelines.online_stage.runtime_adapt import runtime_evaluate as test_func

    P = ctx["P"]
    model = ctx["model"]
    base_state = copy.deepcopy(model.state_dict())

    tto_list = (
        [int(P.tto)]
        if isinstance(P.tto, int)
        else [int(x) for x in str(P.tto).split(",")]
    )

    all_results = []

    print(f"Args: {P}\n")
    for step in tto_list:
        P.tto = step
        set_random_seed(P.seed)

        # reset model to meta-learned initialization
        model.load_state_dict(base_state)

        metrics = test_func(
            P,
            model,
            ctx["test_loader"],
            step,
            logger=ctx["logger"],
            scene_box=ctx["global_box"],
        )

        all_results.append(
            {
                "tto_steps": step,
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
                "duration": metrics["duration"],
            }
        )

    df = pd.DataFrame(all_results).sort_values("tto_steps").reset_index(drop=True)
    latex_table = df.to_latex(index=False, float_format="%.3f", escape=True)
    print(df)
    print(latex_table)


def video(ctx: dict):
    from pipelines.video_gen import render_video

    P = ctx["P"]
    print(ctx["device"])
    first_batch = next(iter(ctx["test_loader"]))
    metas = first_batch["metas"]  # list[ImageMetadata]
    md0 = metas[0]  # ImageMetadata with attrs: H, W, intrinsics, c2w, ...

    H = int(md0.H)
    W = int(md0.W)
    intr = torch.as_tensor(md0.intrinsics, dtype=torch.float32).flatten()
    fx, fy, cx, cy = [float(x) for x in intr[:4]]

    # robust numeric radius from SceneBox (avoid None!)
    aabb = ctx["global_box"].aabb  # (2,3)
    extent = (aabb[1] - aabb[0]).abs()  # (3,)

    def inside_radius_from_box(extent: torch.Tensor, frac: float = 0.6) -> float:
        """
        Radius that stays inside the AABB. frac∈(0,1]; 1.0 touches faces.
        """
        half_min = 0.5 * float(extent.min().item())
        return frac * half_min

    radius_inside = inside_radius_from_box(extent, frac=0.6)

    # Output location
    out_vid = (
        Path("logs")
        / "eval"
        / str(getattr(P, "fname", P.fname))
        / "videos"
        / "orbit.mp4"
    )
    out_vid.parent.mkdir(parents=True, exist_ok=True)

    render_video(
        model=ctx["model"],
        W=W,
        H=H,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        scene_box=ctx["global_box"],
        n_poses=210,
        out_path=str(out_vid),
        phi_deg=45.0,
        camera_path=P.camera_path,
        radius=radius_inside,
        ray_samples=int(getattr(P, "ray_samples", 64)),
        fps=30,
        device=ctx["device"],
        chunk=P.chunk_points,
    )
    print(f"[eval] orbit video written to: {out_vid}")


def view(ctx: dict):
    """Spin up the lightweight Viser viewer bound to the current model."""
    P = ctx["P"]
    from viewer.viewer import launch_viewer
    import time

    viewer = launch_viewer(
        P=P,
        model=ctx["model"],
        coordinate_info=ctx["coordinate_info"],
        scene_box=ctx["global_box"],  # (2,3) tensor
        device=ctx["device"],
        host=str(getattr(P, "viewer_host", "0.0.0.0")),
        port=int(getattr(P, "viewer_port", 7070)),
        open_browser=bool(int(getattr(P, "viewer_open_browser", 0))),
    )

    timeout = getattr(P, "viewer_timeout", -1)
    if timeout is None:
        timeout = -1

    if timeout > 0:
        end = time.time() + timeout
        while time.time() < end and viewer.viewer_state["running"]:
            time.sleep(1.0)

        if not viewer.viewer_state["running"]:
            ctx["logger"].log("[VIEW] Viewer stopped by use.")
        else:
            ctx["logger"].log("[VIEW] Viewer timeout reached, shutting down.")
        try:
            viewer.stop()
        except Exception:
            pass
    else:
        try:
            while True:
                time.sleep(1.0)
                if not viewer.viewer_state["running"]:
                    ctx["logger"].log("[VIEW] Viewer stopped by user (server.stop()).")
                    break
        except KeyboardInterrupt:
            ctx["logger"].log("[VIEW] Viewer interrupted, shutting down.")
            try:
                viewer.stop()
            except Exception:
                pass

# -----------------------------
# Entrypoint
# -----------------------------
def main():
    P = parse_args()  
    ctx = build_context(P, P.op)

    if P.op == "train":
        train(ctx)
    elif P.op == "eval":
        eval(ctx)
    elif P.op == "video":
        video(ctx)
    else:
        view(ctx)


if __name__ == "__main__":
    main()
