from copy import deepcopy
import math
import sys
from typing import Callable, Dict, Optional, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from common.utils import get_optimizer

from .base import BaseRunner
from utils import collate_rays, collate_image_meta_data
from data.image_metadata import ImageMetaDataset, ImageMetadata
from data.dataset import get_image_metadata
from data.infinite_loader import InfiniteDataLoader
from data.ram_rays_dataset import RamRaysDataset
from pipelines.online_stage import runtime_adapt


class RuntimeAdaptRunner(BaseRunner):
    """
    Viewer step = ONE inner update using runtime_adapt over a single support batch.

    Key behavior:
    - Infinite support stream (reshuffles each epoch, never ends)
    - Pause/resume keeps fast params and iterators
    - Reset Fast clears adaptation state
    - Live hyperparam updates handled via BaseRunner.update_*()
    """

    def __init__(
        self,
        model,
        device,
        log: Callable[[str], None],
        *,
        P,
        data_root: str,
        batch_dir: str,
        scene_box,
        only_test: bool = True,
        test_batch_size: int = 1,
        reset_on_start: bool = False,
    ):
        super().__init__(model, device, log, P=P)
        self.data_root = data_root
        self.batch_dir = batch_dir
        self.scene_box = scene_box
        self.only_test = bool(only_test)
        self.test_batch_size = int(test_batch_size)
        self.reset_on_start = bool(reset_on_start)
        self.downscale = float(getattr(self.P, "downscale", 1.0))

        # Adaptation state
        self.orig_state: Optional[Dict[str, torch.Tensor]] = None
        # optimizer
        self.orig_state = deepcopy(self.model.state_dict())
        self.optimizer = get_optimizer(self.P, self.model)
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=bool(getattr(self.P, "use_amp", False))
        )

        self.k_steps: int = 0

        # Iterators (persist across pause/resume)
        self._meta_loader: Optional[DataLoader] = None
        self._meta_iter = None
        self._current_metas: Optional[List[ImageMetadata]] = None
        self._support_loader: Optional[DataLoader] = None
        self._support_iter: Optional[InfiniteDataLoader] = None

        # Progress
        self._pbar: Optional[tqdm] = None
        self._total_steps: Optional[int] = None
        self._last_loss: Optional[float] = None
        self._psnr_hist: List[float] = []

    # -------------------- lifecycle --------------------

    def set_total_steps(self, total: int) -> None:
        """Set total expected steps for persistent tqdm."""
        self._total_steps = int(total)
        if self._pbar is not None:
            self._pbar.total = self._total_steps
            self._pbar.refresh()

    def _reset_progress(self) -> None:
        """Reset or close persistent tqdm so a new run starts cleanly."""
        if self._pbar is not None:
            try:
                self._pbar.close()
            except Exception:
                pass
        self._pbar = None
        self.k_steps = 0
        self._last_loss = None
        self._psnr_hist.clear()

    def start(self):
        """Initialize or rebuild loaders and progress bar."""
        self._build_meta_loader()
        if self.reset_on_start:
            self.reset_fast()

        # Lazy build of support loader on first step
        self._current_metas = None
        self._support_loader = None
        self._support_iter = None

        # Persistent progress bar
        if self._pbar is None:
            self._pbar = tqdm(
                total=self._total_steps,
                dynamic_ncols=True,
                leave=True,
                mininterval=0.2,
                smoothing=0.1,
                desc="Runtime-Adapt",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                file=sys.stdout,
            )

        self.log(
            f"Runtime-Adapt ready (resume={'yes' if self.fast is not None else 'no'})"
        )

    def stop(self):
        """Close resources."""
        if self._pbar is not None:
            try:
                self._pbar.close()
            except Exception:
                pass
            self._pbar = None

    # -------------------- step --------------------

    def step(self):
        """Perform one adaptation step on a single support batch."""
        if self._meta_iter is None:
            self.start()

        rays, rgbs = self._next_support_batch()

        metrics = runtime_adapt(
            P=self.P,
            model=self.model,
            data_loader=[(rays, rgbs)],
            optimizer=self.optimizer,
        )

        loss_val = float(metrics.get("loss", 0.0))
        psnr_val = 10.0 * math.log10(1.0 / max(loss_val, 1e-8))

        self.k_steps += 1
        self._last_loss = loss_val
        self._psnr_hist.append(psnr_val)
        mean_psnr = sum(self._psnr_hist[-50:]) / min(len(self._psnr_hist), 50)

        # tqdm live update
        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix(
                {"loss": f"{loss_val:.6f}", "psnr": f"{psnr_val:.2f}"}
            )

        # viewer log summary
        self.log(
            f"[step {self.k_steps:05d}] "
            f"loss={loss_val:.6f} | psnr={psnr_val:.2f} | avg_psnr={mean_psnr:.2f}"
        )

        del rays, rgbs
        torch.cuda.empty_cache()

    def get_render_params(self):
        return self.fast

    # -------------------- data handling --------------------

    def _build_meta_loader(self):
        """Build meta DataLoader from eval/test split."""
        data_dir = self.batch_dir if self.batch_dir else self.data_root
        _, eval_list = get_image_metadata(data_dir, self.downscale, only_test=True)

        if not eval_list:
            raise RuntimeError(
                f"No metadata found for eval set in: {data_dir}. "
                "Expected <data_dir>/metadata (flat) or <data_dir>/(val|test)/metadata."
            )

        print("batch_size", self.test_batch_size)

        meta_ds = ImageMetaDataset(eval_list)
        self._meta_loader = DataLoader(
            meta_ds,
            shuffle=True,
            batch_size=self.test_batch_size,
            pin_memory=True,
            num_workers=0,
            collate_fn=collate_image_meta_data,
        )
        self._meta_iter = iter(self._meta_loader)

    def _make_support_loader(self, metas: List[ImageMetadata]) -> DataLoader:
        rays_ds = RamRaysDataset(
            metas,
            center_pixels=True,
            ray_gen_kwargs={
                "scene_box": self.scene_box,
                "near_far_override": (
                    getattr(self.P, "near", None),
                    getattr(self.P, "far", None),
                ),
            },
        )
        num_support = int(
            getattr(self.P, "support_rays", getattr(self.P, "support_batch", 4096))
        )
        return DataLoader(
            rays_ds,
            batch_size=num_support,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_rays,
            persistent_workers=False,
        )

    def _ensure_support_stream(self):
        """Ensure infinite support iterator exists; rebuild if exhausted or invalidated."""
        if self._current_metas is None:
            try:
                batch = next(self._meta_iter)
            except StopIteration:
                self._meta_iter = iter(self._meta_loader)
                batch = next(self._meta_iter)
            self._current_metas = batch["metas"]
            self._support_loader = self._make_support_loader(self._current_metas)
            self._support_iter = InfiniteDataLoader(self._support_loader)

        if self._support_iter is None and self._support_loader is not None:
            self._support_iter = InfiniteDataLoader(self._support_loader)

    def _next_support_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_support_stream()
        return next(self._support_iter)
