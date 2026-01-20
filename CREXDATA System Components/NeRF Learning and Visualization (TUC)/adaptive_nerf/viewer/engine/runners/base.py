import os
import time, threading
from types import SimpleNamespace
from typing import Optional

import torch

model_lock = threading.Lock()


class BaseRunner:
    def __init__(self, model, device, log, *, P=None, sleep=0.02):
        self.model, self.device, self.log, self.sleep = model, device, log, sleep
        self.P = P or SimpleNamespace()
        self.fast = None
        self.k_steps = 0

    def start(self):
        pass

    def step(self):
        time.sleep(self.sleep)

    def stop(self):
        pass

    # ---------------- UI-facing helpers ----------------

    def reset_fast(self):
        """Revert adapted state; optionally zero internal step counters and progress."""
        assert self.orig_state is not None, "no original state to reset to"
        self.model.load_state_dict(self.orig_state)
        self.optimizer.state.clear()
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=bool(getattr(self.P, "use_amp", False))
        )
        self.k_steps = 0

        try:
            self._reset_progress()  # hook: implemented by subclasses that track progress bars
        except Exception:
            pass
        self.log("fast params reset")

    def save_checkpoint(self, path: str, include_base_model: bool = False):
        """
        Save the adapted state and minimal context. If include_base_model=True,
        also persist the base model weights (so the checkpoint is self-contained).
        """
        ckpt = {
            "type": "runtime_adapt",
            "fast": self.fast,
            "steps": self.k_steps,
            "P": {
                k: getattr(self.P, k)
                for k in ("inner_lr", "support_rays", "ray_samples", "chunk_points")
                if hasattr(self.P, k)
            },
            "downscale": self.downscale,
        }
        if include_base_model:
            ckpt["model_state_dict"] = {
                k: v.detach().cpu() for k, v in self.model.state_dict().items()
            }
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(ckpt, path)
        self.log(f"checkpoint saved → {path}")

    def load_checkpoint(self, path: str, restore_model: bool = False):
        """Load a checkpoint saved by save_checkpoint()."""
        ckpt = torch.load(path, map_location="cpu")
        self.fast = ckpt.get("fast", None)
        self.k_steps = int(ckpt.get("steps", 0))
        if "P" in ckpt:
            self.update_train_hparams(**ckpt["P"], log_update=False)
        if "downscale" in ckpt and float(ckpt["downscale"]) != self.downscale:
            self.update_data_hparams(downscale=float(ckpt["downscale"]))
        if restore_model and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.log(f"checkpoint loaded ← {path} (restore_model={restore_model})")

    def update_train_hparams(
        self,
        *,
        sigma_lr: Optional[float] = None,
        color_lr: Optional[float] = None,
        encoding_lr: Optional[float] = None,
        support_rays: Optional[int] = None,
        ray_samples: Optional[int] = None,
        chunk_points: Optional[int] = None,
        batch_size: Optional[int] = None,
        log_update: bool = False,
    ):
        """
        Update training knobs in-place; takes effect next step.
        Note:
        - Changing `support_rays` rebuilds the support loader on the next step.
        - Changing `batch_size` (meta/test) rebuilds the meta loader and also resets support stream.
        """
        if batch_size is not None:
            bs = int(batch_size)
            setattr(self.P, "test_batch_size", bs)
            setattr(self.P, "batch_size", bs)
            runner = getattr(self, "runner", None)
            if runner is not None and hasattr(runner, "set_test_batch_size"):
                runner.set_test_batch_size(bs)

        if sigma_lr is not None:
            setattr(self.P, "sigma_lr", float(sigma_lr))

        if color_lr is not None:
            setattr(self.P, "color_lr", float(color_lr))

        if encoding_lr is not None:
            setattr(self.P, "encoding_lr", float(encoding_lr))

        if not (color_lr is None and sigma_lr is None and encoding_lr is None):
            self.sync_optimizer_lrs()

        if support_rays is not None:
            sr = int(support_rays)
            setattr(self.P, "support_rays", sr)
            runner = getattr(self, "runner", None)
            if runner is not None and hasattr(runner, "invalidate_support_stream"):
                runner.invalidate_support_stream()

        if ray_samples is not None:
            setattr(self.P, "ray_samples", int(ray_samples))

        if chunk_points is not None:
            setattr(self.P, "chunk_points", int(chunk_points))

        if log_update:
            self.log(
                "updated hparams → "
                f"sigma_lr={getattr(self.P,'sigma_lr',None)} "
                f"color_lr={getattr(self.P,'color_lr',None)} "
                f"encoding_lr={getattr(self.P,'encoding_lr',None)} "
                f"support_rays={getattr(self.P,'support_rays',None)} "
                f"ray_samples={getattr(self.P,'ray_samples',None)} "
                f"chunk_points={getattr(self.P,'chunk_points',None)} "
                f"test_batch_size={getattr(self.P,'test_batch_size',None)}"
            )

    def update_data_hparams(self, *, downscale: Optional[float] = None):
        """
        Update data knobs. Changing downscale requires rebuilding meta loader,
        and forces rebuilding support stream on next step.
        """
        if downscale is not None and float(downscale) != self.downscale:
            self.downscale = float(downscale)
            self._build_meta_loader()  # rebuild metas with new scale
            # force rebuilding support stream on next step
            self._current_metas = None
            self._support_loader = None
            self._support_iter = None
            self.log(f"updated data → downscale={self.downscale} (meta loader rebuilt)")

    def sync_optimizer_lrs(self):
        """Update optimizer.param_groups['lr'] from self.P.* without recreating the optimizer."""
        opt = getattr(self, "optimizer", None)
        if opt is None:
            return

        base_lr = getattr(self.P, "lr", None)

        for group in opt.param_groups:
            name = group.get("name", None)

            if name == "sigma":
                lr = getattr(self.P, "sigma_lr", base_lr)
            elif name == "color":
                lr = getattr(self.P, "color_lr", base_lr)
            elif name == "encoding":
                lr = getattr(self.P, "encoding_lr", base_lr)
            elif name == "background":
                lr = getattr(self.P, "bg_lr", base_lr)
            else:
                lr = base_lr

            if lr is not None:
                group["lr"] = float(lr)


class ViewRunner(BaseRunner):
    pass
