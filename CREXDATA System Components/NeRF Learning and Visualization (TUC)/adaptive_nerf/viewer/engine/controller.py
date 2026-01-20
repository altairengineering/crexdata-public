import os
import threading
from typing import Optional, Callable, Dict, Any


class Controller:
    """
    Viewer-facing training controller.

    Responsibilities
    ----------------
    - Build the appropriate runner for the selected mode (viewer stays runner-agnostic).
    - Expose Start / Pause / Resume / Stop / Step-once with target-step semantics.
    - Propagate training / data hyperparameters to the runner.
    - Provide render parameters for the viewer via `get_render_params()`.
    """

    def __init__(
        self,
        model,
        device,
        server,
        set_status: Callable[[str, str], None],
        *,
        P: Any,
        scene_box: Any,
        data_root: str,
    ) -> None:
        # Core context
        self.model = model
        self.device = device
        self.server = server
        self._set_status_cb = set_status

        self.P = P
        self.scene_box = scene_box
        self.data_root = data_root

        # Execution state
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        self._pause_evt = threading.Event()
        self._stop_evt = threading.Event()
        self._pause_evt.clear()
        self._stop_evt.clear()

        self.mode: str = "View Model"
        self.current_step: int = 0
        self._target_steps: int = 0
        self.runner = None

        # Hparam caches (viewer → controller → runner)
        self._train_hparams: Dict[str, Any] = {}
        self._data_hparams: Dict[str, Any] = {}

        # Runtime state (set during “Scan & Verify”)
        self.runtime: Dict[str, Any] = {}

        self._on_running: Optional[Callable[[], None]] = None
        self._on_paused: Optional[Callable[[], None]] = None
        self._on_idle: Optional[Callable[[], None]] = None

    # --------------------------------------------------------------------- #
    # Lifecycle handlers (viewer hooks)
    # --------------------------------------------------------------------- #
    def set_lifecycle_handlers(
        self,
        *,
        on_running: Optional[Callable[[], None]] = None,
        on_paused: Optional[Callable[[], None]] = None,
        on_idle: Optional[Callable[[], None]] = None,
    ) -> None:
        """Register optional callbacks for running / paused / idle transitions."""
        self._on_running = on_running
        self._on_paused = on_paused
        self._on_idle = on_idle

    # --------------------------------------------------------------------- #
    # Public properties
    # --------------------------------------------------------------------- #
    @property
    def running(self) -> bool:
        return self._running

    @property
    def paused(self) -> bool:
        return self._paused

    # --------------------------------------------------------------------- #
    # Runner management
    # --------------------------------------------------------------------- #
    def ensure_runner(self, mode: str):
        """Ensure a runner exists for `mode`, creating it if needed."""
        self._ensure_runner(mode)
        return self.runner

    def _ensure_runner(self, mode: str) -> None:
        if self.mode == mode and self.runner is not None:
            return
        if self._running:
            self._status("Cannot change mode while running. Stop first.", "red")
            return

        if self.mode != mode:
            self.current_step = 0
            self._target_steps = 0

        self.mode = mode
        self.runner = self._build_runner(mode)

        if not self.runner:
            return

        # Apply cached hparams to the freshly built runner
        if self._train_hparams and hasattr(self.runner, "update_train_hparams"):
            try:
                self.runner.update_train_hparams(**self._train_hparams)
            except Exception as e:
                self._status(f"Applying train hparams failed: {e}", "red")

        if self._data_hparams and hasattr(self.runner, "update_data_hparams"):
            try:
                self.runner.update_data_hparams(**self._data_hparams)
            except Exception as e:
                self._status(f"Applying data hparams failed: {e}", "red")

    def _build_runner(self, mode: str):
        """Construct the concrete runner for the requested mode."""
        if mode == "Runtime-Adapt":
            from viewer.engine.runners.runtime_adapt import RuntimeAdaptRunner

            batch_dir = (self.runtime or {}).get("batch_dir")
            if not batch_dir:
                self._status(
                    "No batch directory set. Use 'Scan & Verify' first.", "red"
                )
                return None
            if not os.path.isdir(batch_dir):
                self._status(f"Batch directory not found: {batch_dir}", "red")
                return None

            # Keep P.downscale in sync with viewer, if provided
            if "downscale" in self._data_hparams:
                setattr(self.P, "downscale", float(self._data_hparams["downscale"]))

            try:
                return RuntimeAdaptRunner(
                    model=self.model,
                    device=self.device,
                    log=self._status,
                    P=self.P,
                    data_root=self.data_root,
                    batch_dir=batch_dir,
                    scene_box=self.scene_box,
                    only_test=True,
                    test_batch_size=self.P.test_batch_size,
                    reset_on_start=False,
                )
            except Exception as e:
                self._status(f"Failed to construct RuntimeAdaptRunner: {e}", "red")
                return None

        if mode == "Meta-Train":
            # Placeholder for future integration.
            # from viewer.engine.runners.meta_train import MetaTrainRunner
            # return MetaTrainRunner(...)
            self._status("Meta-Train runner not wired yet.", "yellow")
            return None

        # "View Model" or unknown mode → no runner
        return None

    # --------------------------------------------------------------------- #
    # Control surface
    # --------------------------------------------------------------------- #
    def start(self, mode: str, total_steps: int) -> None:
        """Start (or extend) training for a given `mode` up to target steps."""
        self._ensure_runner(mode)
        if self.runner is None:
            self._status(
                "Cannot start: runner not available (check batch dir?).", "red"
            )
            return

        total_steps = int(total_steps)
        if total_steps <= 0:
            self._status("Nothing to do (total_steps <= 0).", "yellow")
            return

        if not self._running:
            # New run
            self._target_steps = self.current_step + total_steps
            self.runner.set_total_steps(self._target_steps)
        else:
            # Extend existing run
            self._target_steps = max(
                self._target_steps, self.current_step + total_steps
            )
            self.runner.set_total_steps(self._target_steps)
            self._status(f"Extending target steps → {self._target_steps}", "yellow")
            return

        # Transition to running
        self._running = True
        self._paused = False
        self._pause_evt.clear()
        self._stop_evt.clear()
        self._status(f"Starting {self.mode}...", "yellow")
        self._emit_running()

        try:
            if hasattr(self.runner, "start"):
                self.runner.start()
        except Exception as e:
            self._running = False
            self._status(f"Runner start error: {e}", "red")
            return

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def step_once(self) -> None:
        """Perform a single runner step in the current mode."""
        self._ensure_runner(self.mode)
        if self.runner is None:
            self._status("No active runner. Use 'Scan & Verify' then Start.", "red")
            return

        self._status("Performing single step...", "yellow")
        try:
            self.runner.step()
            self.current_step += 1
            self._safe_redraw()
            self._status(
                f"Step completed ({self.current_step}/{self._target_steps})", "green"
            )
        except Exception as e:
            self._status(f"Step error: {e}", "red")

    def pause(self) -> None:
        if not self._running or self._paused:
            return
        self._paused = True
        self._pause_evt.set()
        self._status(f"Paused at step {self.current_step}/{self._target_steps}", "cyan")
        self._emit_paused()

    def resume(self) -> None:
        if not self._running or not self._paused:
            return
        self._paused = False
        self._pause_evt.clear()
        self._status(
            f"Resumed at step {self.current_step}/{self._target_steps}", "yellow"
        )
        self._emit_running()

    def stop(self) -> None:
        """Stop training and transition to idle."""
        self._running = False
        self._paused = False
        self._stop_evt.set()
        self._pause_evt.clear()
        self._status("Stopped by user.", "red")

        try:
            if self.runner and hasattr(self.runner, "stop"):
                self.runner.stop()
        except Exception:
            pass

        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=2.0)
            except Exception:
                pass

        self._thread = None
        self._emit_idle()

    def reset_fast(self) -> None:
        """Reset fast weights and local step counters (if supported by runner)."""
        if not self.runner or not hasattr(self.runner, "reset_fast"):
            self._status("Reset not available: no active runner.", "red")
            return

        try:
            self.runner.reset_fast()
            self.current_step = 0
            self._target_steps = 0
            self._safe_redraw()
        except Exception as e:
            self._status(f"Reset failed: {e}", "red")

    def save_checkpoint(self, path: str, include_base_model: bool = False) -> None:
        if not self.runner or not hasattr(self.runner, "save_checkpoint"):
            self._status("Save failed: no active runner.", "red")
            return
        try:
            self.runner.save_checkpoint(path, include_base_model)
            self._status(f"Checkpoint saved → {path}", "green")
        except Exception as e:
            self._status(f"Save failed: {e}", "red")

    def load_checkpoint(self, path: str, restore_model: bool = False) -> None:
        if not self.runner or not hasattr(self.runner, "load_checkpoint"):
            self._status("Load failed: no active runner.", "red")
            return
        try:
            self.runner.load_checkpoint(path, restore_model)
            self._safe_redraw()
            self._status(f"Checkpoint loaded ← {path}", "green")
        except Exception as e:
            self._status(f"Load failed: {e}", "red")

    def update_train_hparams(self, **kwargs: Any) -> None:
        """Update and forward training hyperparameters (inner_lr, support_rays, etc.)."""
        if not kwargs:
            return
        self._train_hparams.update(kwargs)
        if self.runner and hasattr(self.runner, "update_train_hparams"):
            try:
                self.runner.update_train_hparams(**kwargs)
            except Exception as e:
                self._status(f"Update train hparams failed: {e}", "red")

    def update_data_hparams(self, **kwargs: Any) -> None:
        """Update and forward data-related hyperparameters (e.g., downscale)."""
        if not kwargs:
            return
        self._data_hparams.update(kwargs)
        if self.runner and hasattr(self.runner, "update_data_hparams"):
            try:
                self.runner.update_data_hparams(**kwargs)
            except Exception as e:
                self._status(f"Update data hparams failed: {e}", "red")

    def get_render_params(self):
        """Return render-time params (e.g. fast weights) if provided by the runner."""
        try:
            if self.runner and hasattr(self.runner, "get_render_params"):
                return self.runner.get_render_params()
            if self.runner and hasattr(self.runner, "fast"):
                return self.runner.fast
        except Exception:
            pass
        return None

    # --------------------------------------------------------------------- #
    # Worker loop (background thread)
    # --------------------------------------------------------------------- #
    def _loop(self) -> None:
        try:
            while (
                self._running
                and not self._stop_evt.is_set()
                and self.current_step < self._target_steps
            ):
                if self._paused:
                    # Block without busy-spin; wake periodically to check stop
                    self._pause_evt.wait(timeout=0.2)
                    continue

                if self.runner is None:
                    self._status("Runner missing; aborting.", "red")
                    break

                try:
                    self.runner.step()
                except Exception as e:
                    self._status(f"Training step error: {e}", "red")
                    break

                self.current_step += 1
                self._safe_redraw()

            self._running = False
            self._status("Completed.", "green")
        except Exception as e:
            self._running = False
            self._status(f"Training error: {e}", "red")

        self._emit_idle()

    # --------------------------------------------------------------------- #
    # UI / status helpers
    # --------------------------------------------------------------------- #
    def _emit_running(self) -> None:
        try:
            if callable(self._on_running):
                self._on_running()
        except Exception:
            pass

    def _emit_paused(self) -> None:
        try:
            if callable(self._on_paused):
                self._on_paused()
        except Exception:
            pass

    def _emit_idle(self) -> None:
        try:
            if callable(self._on_idle):
                self._on_idle()
        except Exception:
            pass

    def _safe_redraw(self) -> None:
        """Ask the viewer server to redraw, ignoring failures."""
        try:
            self.server.request_redraw()
        except Exception:
            pass

    def _status(self, msg: str, color: str = "yellow") -> None:
        """Central, tolerant status reporter used by all controller paths."""
        try:
            self._set_status_cb(msg, color)
        except TypeError:
            # tolerate callbacks that only take msg
            try:
                self._set_status_cb(msg)
            except Exception:
                pass
        except Exception:
            pass
