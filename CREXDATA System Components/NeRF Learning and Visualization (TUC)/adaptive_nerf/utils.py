import json
from pathlib import Path
import pickle
import random
import shutil
import sys
from datetime import datetime
import os
import time
from collections import defaultdict, deque
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch.nn as nn
import numpy as np
import torch
from torchvision.utils import make_grid
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from einops import rearrange

from nerfs.scene_box import SceneBox


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
    TensorBoard logger and training directory manager.

    Args:
        fn (str): Base name for logging directory.
        ask (bool): Whether to prompt before overwriting logs.
        today (bool): Add today's date as prefix to log directory.
        rank (int): Rank for distributed training (only rank 0 logs).
    """

    def __init__(self, fn, today=False, rank=0, logs_path="logs/"):
        self.rank = rank
        self.log_path = logs_path
        self.logdir = None

        if self.rank == 0:
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            self.today = today

            logdir = self._make_dir(fn)
            logdir = self._resolve_unique_dir(logdir)
            os.makedirs(logdir, exist_ok=True)
            self.set_dir(logdir)

    def _resolve_unique_dir(self, path: str) -> str:
        """
        If `path` exists and is not empty, return path_v1, path_v2, ...
        until a free name is found.
        """
        if not os.path.exists(path) or len(os.listdir(path)) == 0:
            return path  # free or empty

        base = path
        idx = 1
        while True:
            new_path = f"{base}_v{idx}"
            if not os.path.exists(new_path):
                return new_path
            idx += 1

    def _make_dir(self, fn):
        if self.today:
            today = datetime.today().strftime("%y%m%d")
            logdir = self.log_path + today + "_" + fn
        else:
            logdir = self.log_path + fn
        return logdir

    def set_dir(self, logdir, log_fn="log.txt"):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.writer = SummaryWriter(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), "a")

    def close_writer(self):
        if self.rank == 0:
            self.writer.close()

    def log(self, string):
        if self.rank == 0:
            self.log_file.write("[%s] %s" % (datetime.now(), string) + "\n")
            self.log_file.flush()

            print("[%s] %s" % (datetime.now(), string))
            sys.stdout.flush()

    def log_dirname(self, string):
        if self.rank == 0:
            self.log_file.write("%s (%s)" % (string, self.logdir) + "\n")
            self.log_file.flush()

            print("%s (%s)" % (string, self.logdir))
            sys.stdout.flush()

    def log_custom_dict(self, P):
        train_inner_dict, eval_inner_dict = {}, {}
        if P.log_method == "step":
            for i in range(P.num_submodules):
                train_inner_dict[f"train_loss_in_step{i:02}"] = [
                    "Multiline",
                    [
                        f"train_loss_in_step{i:02}/loss_patch{j:02}"
                        for j in range(P.num_submodules)
                    ],
                ]
                train_inner_dict[f"train_psnr_in_step{i:02}"] = [
                    "Multiline",
                    [
                        f"train_psnr_in_step{i:02}/psnr_patch{j:02}"
                        for j in range(P.num_submodules)
                    ],
                ]
                eval_inner_dict[f"eval_loss_in_step{i:02}"] = [
                    "Multiline",
                    [
                        f"eval_loss_in_step{i:02}/loss_patch{j:02}"
                        for j in range(P.num_submodules)
                    ],
                ]
                eval_inner_dict[f"eval_psnr_in_step{i:02}"] = [
                    "Multiline",
                    [
                        f"eval_psnr_in_step{i:02}/psnr_patch{j:02}"
                        for j in range(P.num_submodules)
                    ],
                ]
        elif P.log_method == "patch":
            for i in range(P.num_submodules):
                train_inner_dict[f"train_loss_in_patch{i:02}"] = [
                    "Multiline",
                    [
                        f"train_loss_in_patch{i:02}/loss_step{j:02}"
                        for j in range(P.num_submodules)
                    ],
                ]
                train_inner_dict[f"train_psnr_in_patch{i:02}"] = [
                    "Multiline",
                    [
                        f"train_psnr_in_patch{i:02}/psnr_step{j:02}"
                        for j in range(P.num_submodules)
                    ],
                ]
                eval_inner_dict[f"eval_loss_in_patch{i:02}"] = [
                    "Multiline",
                    [
                        f"eval_loss_in_patch{i:02}/loss_step{j:02}"
                        for j in range(P.num_submodules)
                    ],
                ]
                eval_inner_dict[f"eval_psnr_in_patch{i:02}"] = [
                    "Multiline",
                    [
                        f"eval_psnr_in_patch{i:02}/psnr_step{j:02}"
                        for j in range(P.num_submodules)
                    ],
                ]

        layout = {"train": train_inner_dict, "eval": eval_inner_dict}
        self.writer.add_custom_scalars(layout)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if self.rank == 0:
            self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, gts, step):
        """Log a list of images."""
        if not torch.is_tensor(images):
            images = torch.stack(
                [images[i][:4].cpu().clamp(0, 1) for i in range(len(images))]
            )
            images = rearrange(images, "t b c h w -> (t b) c h w")
        else:
            images = images[:4].data.cpu().clamp(0, 1)

        gts = gts.value[:4].data.cpu().clamp(0, 1)
        img_grid = make_grid(torch.cat([gts, images], dim=0), nrow=4)[None]
        if self.rank == 0:
            self.writer.add_images(tag, img_grid, step)

    def video_summary(self, tag, videos, gts, step):
        """Log a list of videos."""
        if not torch.is_tensor(videos):
            videos = torch.stack(
                [videos[i][0].squeeze().cpu().clamp(0, 1) for i in range(len(videos))]
            )
            videos = rearrange(videos, "t b c h w -> (t b) c h w")
        else:
            videos = videos[0].data.cpu().clamp(0, 1)
        gts = gts.value[0].data.cpu().clamp(0, 1)
        vid_grid = make_grid(torch.cat([gts, videos], dim=0), nrow=4)[None]
        if self.rank == 0:
            self.writer.add_images(tag, vid_grid, step)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint(logdir, mode="best"):
    model_path = os.path.join(logdir, f"{mode}.model")
    optim_path = os.path.join(logdir, f"{mode}.optim")
    config_path = os.path.join(logdir, f"{mode}.configs")
    scheduler_path = os.path.join(logdir, f"{mode}.scheduler")
    scaler_path = os.path.join(logdir, f"{mode}.scaler")

    print(f"=> Loading checkpoint from '{logdir}'")

    # If there's no model file, nothing to load
    if not os.path.exists(model_path):
        print(f"[WARN] No model checkpoint found at '{model_path}'")
        return None, None, {}, None, None

    # ---- mandatory: model ----
    model_state = torch.load(model_path, map_location="cpu")

    # ---- optional: optimizer ----
    optim_state = (
        torch.load(optim_path, map_location="cpu")
        if os.path.exists(optim_path)
        else None
    )

    # ---- optional: scheduler ----
    scheduler_state = (
        torch.load(scheduler_path, map_location="cpu")
        if os.path.exists(scheduler_path)
        else None
    )

    # ---- optional: scaler ----
    scaler_state = (
        torch.load(scaler_path, map_location="cpu")
        if os.path.exists(scaler_path)
        else None
    )

    # ---- optional: config ----
    if os.path.exists(config_path):
        with open(config_path, "rb") as handle:
            cfg = pickle.load(handle)
    else:
        cfg = {}

    return model_state, optim_state, cfg, scheduler_state, scaler_state


def save_checkpoint(
    P,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    logdir: str,
    is_best: bool = False,
    best: float = 0.0,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    keep_occ_grids: bool = True,  # <<<<< controls whether occ grids are saved
):
    """Save model, optimizer, and optional Meta-SGD state in the legacy flat layout."""
    if getattr(P, "rank", 0) != 0:
        return

    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    tag = "best" if is_best else f"step{step}"

    # --- main components ---
    md_sd = model.state_dict()
    m_sd = (
        md_sd
        if keep_occ_grids
        else {k: v for k, v in md_sd.items() if ".occ_grid." not in k}
    )
    torch.save(model.state_dict(), logdir / f"{tag}.model")
    torch.save(optimizer.state_dict(), logdir / f"{tag}.optim")
    if scheduler is not None:
        torch.save(scheduler.state_dict(), logdir / f"{tag}.scheduler")
    if scaler is not None:
        torch.save(scaler.state_dict(), logdir / f"{tag}.scaler")

    # --- save full config (P) ---
    torch.save(vars(P), logdir / f"{tag}.P")

    # --- lightweight metadata (for resuming) ---
    opts = {"step": step, "best": best, "is_best": is_best}
    with open(logdir / f"{tag}.configs", "wb") as handle:
        pickle.dump(opts, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model_checkpoint(P, model, logger=None):
    """
    Loads model weights from a specified file path. If a low-rank variant is specified, adjusts architecture.
    Also loads inner-loop learning rates for Meta-SGD.

    Args:
        P (argparse.Namespace): Parsed arguments including load_path and rank.
        model (nn.Module): Model into which weights will be loaded.
        logger (Logger, optional): Optional logger for logging output.
    """

    # consistent logger handle
    log_ = print if logger is None else logger.log

    ckpt_dir = Path(getattr(P, "checkpoint_path", ""))
    if not ckpt_dir.exists():
        log_(f"[WARN] Checkpoint path '{ckpt_dir}' does not exist.")
        return

    model_path = ckpt_dir / f"{P.prefix}.model"
    if not model_path.exists():
        log_(f"[WARN] No model checkpoint found at {model_path}.")
        return

    log_(f"Model architecture:\n{model}")
    log_(f"=> Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")

    # handle low-rank variant if needed
    if getattr(P, "rank", 0) != 0 and hasattr(model, "__init_low_rank__"):
        model.__init_low_rank__(rank=P.rank)

    # load weights
    result = model.load_state_dict(
        checkpoint, strict=not bool(getattr(P, "no_strict", False))
    )
    if result.missing_keys or result.unexpected_keys:
        log_(f"[WARN] Some keys not loaded: {result}")
    else:
        log_("[INFO] All keys matched successfully.")

    return model


def cycle(loader):
    while True:
        for x in loader:
            yield x


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    """
    Tracks and synchronizes training statistics over multiple steps or processes.
    Integrates `SmoothedValue` for stability.
    """

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def psnr(mse):
    return -10.0 * torch.log10(mse + 1e-24)


def collate_image_meta_data(batch):
    """
    returns {"metas": List[ImageMetadata], "rgbs_raw": List[HxWx3 uint8]}
    """
    metas = []
    rgbs = []
    for item in batch:
        m = item["meta"]
        x = item["rgbs_raw"]

        # Accept common shapes, coerce to HxWx3 uint8
        if isinstance(x, torch.Tensor) and x.ndim == 3:
            # HWC or CHW
            if x.shape[0] == m.H and x.shape[1] == m.W:  # HWC
                if x.shape[2] < 3:
                    raise ValueError("C<3")
                x = x[:, :, :3].contiguous().to(torch.uint8)
            elif x.shape[1] == m.H and x.shape[2] == m.W:  # CHW
                if x.shape[0] < 3:
                    raise ValueError("C<3")
                x = x[:3, :, :].permute(1, 2, 0).contiguous().to(torch.uint8)
            else:
                raise ValueError(
                    f"Unexpected 3D shape {tuple(x.shape)} for (H,W)=({m.H},{m.W})"
                )
        elif isinstance(x, torch.Tensor) and x.ndim == 2 and x.shape[0] == m.H * m.W:
            # (H*W, C)
            if x.shape[1] < 3:
                raise ValueError("C<3")
            x = x[:, :3].contiguous().view(m.H, m.W, 3).to(torch.uint8)
        else:
            raise ValueError(f"rgbs_raw must be HWC/CHW/(HW,C); got {tuple(x.shape)}")

        metas.append(m)
        rgbs.append(x)

    return {"metas": metas, "rgbs_raw": rgbs}


def collate_rays(batch):
    """
    Expect items as (rays, rgbs) or {"rays":..., "rgbs":...}, both torch.Tensors on CPU.
    Outputs pinned, contiguous CPU tensors so .to(device, non_blocking=True) overlaps.
    """
    if isinstance(batch[0], (tuple, list)):
        rays, rgbs = zip(*batch)  # tuples of tensors
        rays = torch.stack(rays, 0)
        rgbs = torch.stack(rgbs, 0)
    elif isinstance(batch[0], dict):
        rays = torch.stack([b["rays"] for b in batch], 0)
        rgbs = torch.stack([b["rgbs"] for b in batch], 0)
    else:
        raise TypeError("Unexpected RamRaysDataset item type")

    return rays.contiguous(), rgbs.contiguous()


def collate_episodes(batch: List[Dict]) -> List[Dict[str, Any]]:
    """
    For a single DataLoader (one region/cell), turn a list of episodes into a flat
    list of task dicts. We drop 'cell_id' here because MultiLoader already tags
    which region this batch belongs to.

    Input batch: [ Task(...) or {"support": {...}, "query": {...}, ...}, ... ]
    Output:      [ {"support": {...}, "query": {...}}, ... ]
    """
    # Import here to avoid circulars at module import time
    try:
        from data.task_dataset import Task
    except Exception:
        Task = None  # tolerate environments where this import path differs

    tasks: List[Dict[str, Any]] = []
    for ep in batch:
        if Task is not None and isinstance(ep, Task):
            # Convert Task dataclass to the dict format we train on
            if not hasattr(ep, "support") or not hasattr(ep, "query"):
                raise TypeError("Task missing 'support' or 'query'.")
            tasks.append(
                {
                    "support": ep.support,
                    "query": ep.query,
                }
            )
        elif isinstance(ep, dict):
            if "support" not in ep or "query" not in ep:
                raise TypeError("Episode dict missing 'support' or 'query'.")
            tasks.append(
                {
                    "support": ep["support"],
                    "query": ep["query"],
                }
            )
        else:
            raise TypeError(f"Unsupported episode type: {type(ep)}")
    return tasks


def seed_worker(worker_id):
    """Ensure reproducibility across workers and epochs."""
    worker_info = torch.utils.data.get_worker_info()
    base_seed = worker_info.seed % 2**32
    np.random.seed(base_seed)
    random.seed(base_seed)
    torch.manual_seed(base_seed)


def discover_cluster_cells(mask_dir: Path) -> int:
    # Prefer params.pt if present (Mega-NeRF style), else count subfolders
    params_pt = mask_dir / "params.pt"
    if params_pt.exists():
        params = torch.load(params_pt, map_location="cpu")
        return len(params.get("centroids", [])) or len(
            [p for p in mask_dir.iterdir() if p.is_dir()]
        )
    # fallback: subdirs named 0,1,2,...
    return len([p for p in mask_dir.iterdir() if p.is_dir()])


def load_clustering_meta(mask_dir_or_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and return the dict saved in params.pt, unchanged.

    Accepts either a directory containing params.pt or a direct path to params.pt.
    """
    p = Path(mask_dir_or_file)
    params_path = p if (p.is_file() and p.name == "params.pt") else (p / "params.pt")
    if not params_path.exists():
        raise FileNotFoundError(f"params.pt not found at: {params_path}")
    return torch.load(params_path, map_location="cpu")


def load_scene_boxes(
    mask_dir: Path, device: Optional[torch.device] = None
) -> Tuple[SceneBox, List[SceneBox]]:
    """
    Load global + per-expert boxes saved as format_version=3:

    """
    mask_dir = Path(mask_dir)

    boxes_path = mask_dir / "scene_boxes.pt"
    if not boxes_path.exists():
        raise FileNotFoundError(f"boxes.pt not found at {boxes_path}")

    meta = torch.load(boxes_path, map_location="cpu")

    aabb_global = meta["aabb_global"].to(dtype=torch.float32).contiguous()  # (2,3) CPU
    mins = meta["mins"].to(dtype=torch.float32).contiguous()  # (C,3) CPU
    maxs = meta["maxs"].to(dtype=torch.float32).contiguous()  # (C,3) CPU

    # Build SceneBoxes (note: these expect (2,3) AABBs in the same normalized DRB coords)
    aabb_global = aabb_global.to(device) if device is not None else aabb_global
    global_box = SceneBox(aabb=aabb_global)

    local_boxes: List[SceneBox] = []
    num_boxes = mins.shape[0]
    for c in range(num_boxes):
        aabb_expert = torch.stack([mins[c], maxs[c]], dim=0)
        if device is not None:
            aabb_expert = aabb_expert.to(device)
        local_boxes.append(SceneBox(aabb=aabb_expert))

    return global_box, local_boxes


def _contains_model_files(d: Path) -> bool:
    return d.is_dir() and any(f.is_file() and f.suffix == ".model" for f in d.iterdir())


def resolve_checkpoint_dir(path: str, logs_root: str = "../logs") -> str:
    """
    Resolve and return the directory that contains checkpoint files (*.model).

    Input may be:
      - a run directory
      - a job directory
      - the same paths with or without the 'logs/' prefix

    Resolution:
      - If the directory contains *.model files, return it.
      - Otherwise, repeatedly descend into the latest subdirectory (by name)
        until a directory containing *.model files is found.

    The returned path is the actual resolved filesystem path
    (including 'logs/' if that is where the checkpoints live).
    """
    p = Path(path)

    # logs/ is optional in input, but resolution must reflect reality
    if not p.exists():
        p = Path(logs_root) / path

    if not p.exists():
        p = Path(logs_root) / "logs" / path

    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(
            f"Checkpoint path not found: '{path}', '{Path(logs_root) / path}, or '{Path(logs_root) / 'logs' / path}'!"
        )

    cur = p
    for _ in range(8):  # safety bound
        if _contains_model_files(cur):
            return str(cur.resolve())

        subdirs = sorted(
            [d for d in cur.iterdir() if d.is_dir()],
            key=lambda x: x.name,
        )
        if not subdirs:
            break

        cur = subdirs[-1]  # descend into latest run

    raise FileNotFoundError(f"No '*.model' checkpoints found under: {p}")
