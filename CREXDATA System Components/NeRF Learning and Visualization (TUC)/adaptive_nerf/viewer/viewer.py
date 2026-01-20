"""
Lightweight nerfview/Viser adapter for --op view.

Design:
- All viewer camera math happens in RUB (OpenGL-style).
- DRB world constants (center/origin) are converted to RUB once at init.
- At render time, incoming viewer pose (RUB) → DRB before get_rays.
"""

import os
import sys
import time
from functools import lru_cache
from dataclasses import dataclass
from typing import Optional, Tuple

import imageio.v3 as iio
import numpy as np
import torch
import viser
import nerfview
import viser.transforms as tf
from rich.console import Console

from nerfs.ray_rendering import render_rays
from nerfs.ray_sampling import clamp_rays_near_far, get_ray_directions, get_rays
from nerfs.scene_box import SceneBox
from viewer.colormap import ColormapOptions, apply_colormap, apply_depth_colormap
from viewer.engine.controller import Controller
from viewer.engine.runners.base import model_lock
from viewer.utils import (
    ensure_dir,
    rub_pose_look,  # expects RUB inputs
    rub_to_drb_3x3,  # RUB→DRB
    drb_to_rub_3x3,  # DRB→RUB
    safe_bg,
    safe_active_module,
    uint8_from_linear01,
    verify_continual_batch_dir,
)

CONSOLE = Console(width=120, file=sys.stderr)


# ========= Data classes & local caches =========
@dataclass
class Preset:
    ray_samples: int
    early_stop_eps: float

    @property
    def kwargs(self) -> dict:
        return dict(
            ray_samples=self.ray_samples,
            early_stop_eps=self.early_stop_eps,
            cone_angle=0.0,
        )


# Device-tensor cache: (H,W,fx,fy,cx,cy,device.type,device.index) -> torch.Tensor (on device)
@lru_cache(maxsize=8)
def _cached_ray_dirs_key(H, W, fx, fy, cx, cy):
    dirs = get_ray_directions(
        int(H),
        int(W),
        float(fx),
        float(fy),
        float(cx),
        float(cy),
        center_pixels=True,
        device=torch.device("cpu"),
    )
    return dirs.cpu().numpy()


# ========= Main entry =========
def launch_viewer(
    *,
    P,
    model: torch.nn.Module,
    coordinate_info: dict,
    scene_box: SceneBox,
    device: torch.device,
    host: str = "127.0.0.1",
    port: int = 7070,
    open_browser: bool = True,
    active_module: Optional[int] = None,
) -> Tuple[viser.ViserServer, "nerfview.Viewer"]:
    """Launch viewer with RUB-first camera ops and DRB-compatible rendering."""

    # ----- Init: device, types, transforms -----
    dev = torch.device(device)
    dtype = torch.float32
    model.to(dev).eval()

    origin_drb = coordinate_info["origin_drb"].to(dev)
    pose_scale = float(coordinate_info["pose_scale_factor"])

    aabb = scene_box.aabb.detach().cpu().numpy().astype(np.float32)  # [[min], [max]]
    center_drb_np = 0.5 * (aabb[0] + aabb[1])
    extent_drb_np = aabb[1] - aabb[0]  # normalized units
    scene_extent_norm = float(np.linalg.norm(extent_drb_np))
    scene_extent_world = scene_extent_norm * pose_scale  # meters (pre-normalization)

    # Center in RUB: valid to rotate centers; do NOT rotate extents for magnitude
    M_drb2rub = drb_to_rub_3x3(dev, dtype).cpu().numpy()
    center_rub_np = M_drb2rub @ center_drb_np
    up_world_rub_np = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # Render presets
    global FULL, PREVIEW
    FULL = Preset(ray_samples=P.ray_samples, early_stop_eps=1e-3)
    PREVIEW = Preset(ray_samples=max(32, P.ray_samples // 2), early_stop_eps=2e-3)

    # ----- Server -----
    try:
        server = viser.ViserServer(
            host=host, port=port, verbose=False, open_browser=open_browser
        )
    except TypeError:
        server = viser.ViserServer(host=host, port=port, verbose=False)
    server.scene.world_axes.visible = True

    # ----- Viewer/UI state -----
    state = {
        "zoom": 1.0,
        "exposure": 1.0,
        "gamma": 1.0,
        "display": "RGB",
        "last_c2w_np": None,
        "last_frame_u8": None,
    }
    training_state = {"target_steps": 0}  # Start/Continue semantics
    viewer_state = {"running": True}

    # ----- Visuals (background, samples/eps, occupancy, modules) -----
    with server.add_gui_folder("Visuals", expand_by_default=False):
        bg_default = str(getattr(P, "bg_color_default", "white"))
        bg_options = ["white", "black", "random", "none", "bg_nerf", "last_sample"]
        if bg_default not in {}:
            bg_default = "white"
        if getattr(model, "use_bg_nerf", False):
            bg_default = "bg_nerf"

        gui_bg = server.add_gui_dropdown(
            label="Background",
            options=bg_options,
            initial_value=bg_default,
        )

        K = getattr(model, "num_submodules", len(getattr(model, "submodules", []))) or 0
        sub_options = ["All"] + [str(i) for i in range(K)] if K else ["All"]
        gui_sub = server.add_gui_dropdown(
            "Active Module",
            options=sub_options,
            initial_value="All" if active_module is None else str(active_module),
        )

        gui_occ = server.add_gui_checkbox(
            "Use Occupancy", bool(getattr(model, "use_occ", False))
        )
        gui_full_samples = server.add_gui_slider(
            "Full Ray Samples", 24, 192, 8, int(FULL.ray_samples)
        )
        gui_prev_samples = server.add_gui_slider(
            "Preview Ray Samples", 16, 128, 8, int(PREVIEW.ray_samples)
        )

    # ----- Postprocessing -----
    with server.add_gui_folder("Postprocessing", expand_by_default=False):
        gui_display = server.add_gui_dropdown(
            "Display", ["RGB", "Depth", "Opacity"], initial_value="RGB"
        )
        gui_zoom = server.add_gui_slider("Zoom", 0.5, 2.0, 0.01, 1.0)

        depth_folder = server.add_gui_folder("Tonemap — Depth", expand_by_default=False)
        with depth_folder:
            gui_depth_cmap = server.add_gui_dropdown(
                "Colormap",
                ["turbo", "viridis", "inferno", "gray"],
                initial_value="turbo",
            )
            gui_depth_inv = server.add_gui_checkbox("Inverse (near bright)", True)

        opacity_folder = server.add_gui_folder(
            "Tonemap — Opacity", expand_by_default=False
        )
        with opacity_folder:
            gui_acc_cmap = server.add_gui_dropdown(
                "Colormap",
                ["inferno", "viridis", "turbo", "gray"],
                initial_value="inferno",
            )

        # Hide them initially
        depth_folder.visible = False
        opacity_folder.visible = False

        with server.add_gui_folder("Lighting", expand_by_default=True):
            gui_exposure = server.add_gui_slider("Exposure", 0.5, 2.0, 0.01, 1.0)
            gui_gamma = server.add_gui_slider("Gamma", 0.8, 2.2, 0.01, 1.0)

        with server.add_gui_folder("Clipping", expand_by_default=False):
            gui_nf_enable = server.add_gui_checkbox("Override Near/Far", True)
            # default_near_m = scene_extent_world / 400.0
            default_near_m = 70.0
            default_far_m = scene_extent_world
            gui_near_m = server.add_gui_slider(
                "Near (m)",
                min=0.0,
                max=scene_extent_world,
                step=0.001,
                initial_value=default_near_m,
            )
            gui_far_m = server.add_gui_slider(
                "Far (m)",
                min=0.1,
                max=scene_extent_world * 2.0,
                step=0.001,
                initial_value=default_far_m,
            )

        @gui_display.on_update
        def _(_):
            mode = gui_display.value
            depth_folder.visible = mode == "Depth"
            opacity_folder.visible = mode == "Opacity"

    # ----- Camera control helpers -----
    def _c2w_to_quat_pos(c2w_np: np.ndarray):
        R, t = c2w_np[:3, :3], c2w_np[:3, 3]
        wxyz = tf.SO3.from_matrix(R).wxyz
        return wxyz, t

    def _set_pose(client, c2w_np: np.ndarray):
        wxyz, pos = _c2w_to_quat_pos(c2w_np)
        with client.atomic():
            client.camera.wxyz, client.camera.position = wxyz, pos
        client.flush()

    def pose_look_center(c2w_np: np.ndarray, *, level: bool):
        cam = c2w_np[:3, 3]
        return rub_pose_look(center_rub_np, cam, up_world_rub_np, same_D=level).astype(
            np.float32
        )

    def pose_snap_dir(c2w_np: np.ndarray, fwd_rub: np.ndarray):
        cam = c2w_np[:3, 3]
        tgt = cam + fwd_rub
        return rub_pose_look(tgt, cam, up_world_rub_np, same_D=False).astype(np.float32)

    def pose_dolly(c2w_np: np.ndarray, *, forward: bool):
        """Use rotation-invariant magnitude from DRB extent (normalized units)."""
        c2w = c2w_np.copy()
        back = c2w[:3, 2]
        step = (
            0.05 * scene_extent_norm
        )  # or use scene_extent_world if you want (meters)
        delta = (-back if forward else back) * step
        c2w[:3, 3] += delta
        return c2w.astype(np.float32)

    @server.on_client_connect
    def _setup_camera_gui(client):
        with client.gui.add_folder("Controls", expand_by_default=False):
            btn_center = client.gui.add_button("Look At Center")
            btn_front = client.gui.add_button("Look Front")
            btn_right = client.gui.add_button("Look Right")
            btn_bottom = client.gui.add_button("Look Down")
            btn_in = client.gui.add_button("Dolly In")
            btn_out = client.gui.add_button("Dolly Out")
            btn_snap = client.gui.add_button("Save Screenshot")

            btn_terminate = client.gui.add_button(
                "Terminate Viewer",
                color="red",
            )

        def _bind(btn, fn):
            @btn.on_click
            def _(_evt):
                c2w = state.get("last_c2w_np")
                if c2w is not None:
                    _set_pose(client, fn(c2w.astype(np.float32)))

        _bind(btn_center, lambda c2w: pose_look_center(c2w, level=False))
        _bind(
            btn_front, lambda c2w: pose_snap_dir(c2w, np.array([0, 0, -1], np.float32))
        )
        _bind(
            btn_right, lambda c2w: pose_snap_dir(c2w, np.array([1, 0, 0], np.float32))
        )
        _bind(
            btn_bottom, lambda c2w: pose_snap_dir(c2w, np.array([0, -1, 0], np.float32))
        )
        _bind(btn_in, lambda c2w: pose_dolly(c2w, forward=True))
        _bind(btn_out, lambda c2w: pose_dolly(c2w, forward=False))

        @btn_snap.on_click
        def _(_):
            if state["last_frame_u8"] is not None:
                out_dir = os.path.join("logs", "viewer", "snapshots")
                ensure_dir(out_dir)
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(out_dir, f"frame_{ts}.png")
                iio.imwrite(path, state["last_frame_u8"])
                CONSOLE.print(
                    f"[bold green]📸 Snapshot saved →[/bold green] [cyan]{path}[/cyan]"
                )

        @btn_terminate.on_click
        def _(_evt):
            # mark viewer as not running
            viewer_state["running"] = False

            try:
                ctrl.stop()
            except Exception:
                pass

            CONSOLE.print("[bold red] Terminating viewer on user request...[/bold red]")
            # server.stop()

    # ----- Training: widgets & controller -----
    with server.add_gui_folder("Operation Mode", expand_by_default=False):
        gui_mode = server.add_gui_dropdown(
            "Mode",
            options=["View Model", "Runtime-Adapt"],
            initial_value="View Model",
        )
        gui_status = server.add_gui_text("Status", "idle")

        gui_start = server.add_gui_button("▶ Start", visible=False)
        gui_pause = server.add_gui_button("⏸ Pause", visible=False)
        gui_resume = server.add_gui_button("⏵ Resume", visible=False)
        gui_stop = server.add_gui_button("⏹ Stop", visible=False)
        gui_step = server.add_gui_button("Step Once", visible=False)

        # Batch dir + scan
        gui_batch_dir = server.add_gui_text("Batch directory", "", visible=False)
        gui_scan = server.add_gui_button("Scan & Verify", visible=False)

        runtime_folder = server.add_gui_folder(
            "Runtime Controls", expand_by_default=False
        )
        runtime_folder.visible = False  # hidden unless Runtime-Adapt is active

        with runtime_folder:

            # Runtime "settings"
            gui_adapt_steps = server.add_gui_slider(
                "Steps (Start)", 1, 1000, 1, 100, visible=False
            )
            gui_batch_size = server.add_gui_slider(
                "Batch Size", 1, 200, 1, P.test_batch_size, visible=False
            )

            min_lr = 1e-5
            max_lr = 1e-1
            step_lr = 1e-3
            gui_sigma_lr = server.add_gui_number(
                "SigmaMLP LR",
                initial_value=P.sigma_lr,
                min=min_lr,
                max=max_lr,
                step=step_lr,
                visible=False,
            )

            gui_color_lr = server.add_gui_number(
                "ColorMLP LR",
                initial_value=P.color_lr,
                min=min_lr,
                max=max_lr,
                step=step_lr,
                visible=False,
            )

            gui_encoding_lr = server.add_gui_number(
                "Encoding LR",
                initial_value=P.encoding_lr,
                min=min_lr,
                max=max_lr,
                step=step_lr,
                visible=False,
            )

            # Runtime low-level controls
            gui_support_rays = server.add_gui_slider(
                "support_rays",
                512,
                65536,
                256,
                int(getattr(P, "support_rays", 4096)),
                visible=False,
            )
            gui_ray_samples = server.add_gui_slider(
                "ray_samples",
                16,
                256,
                8,
                int(getattr(P, "ray_samples", 64)),
                visible=False,
            )
            gui_chunk_points = server.add_gui_slider(
                "chunk_points",
                50_000,
                10_000_000,
                50_000,
                int(getattr(P, "chunk_points", 5_000_000)),
                visible=False,
            )
            gui_downscale = server.add_gui_slider(
                "downscale",
                0.25,
                2.0,
                0.05,
                float(getattr(P, "downscale", 1.0)),
                visible=False,
            )
            btn_reset_fast = server.add_gui_button("Reset Fast", visible=False)

            # Checkpoint-related controls
            gui_ckpt_path = server.add_gui_text(
                "Path", "logs/viewer/ckpts/runtime_fast.pt", visible=False
            )
            gui_ckpt_inc_base = server.add_gui_checkbox(
                "Include Base Weights", False, visible=False
            )
            btn_save_ckpt = server.add_gui_button("💾 Save Checkpoint", visible=False)

    def _set_status(msg: str, color="yellow"):
        gui_status.value = msg
        CONSOLE.print(f"[bold {color}]⚙ {msg}[/bold {color}]")

    data_root = os.path.join(P.data_path, "out", P.data_dirname)
    ctrl = Controller(
        model, dev, server, _set_status, P=P, scene_box=scene_box, data_root=data_root
    )

    # Visibility helpers that reflect runner lifecycle
    def _show_controls_running():
        gui_start.visible = False
        gui_pause.visible = True
        gui_resume.visible = False
        gui_stop.visible = True
        gui_step.visible = gui_mode.value == "Runtime-Adapt"

    def _show_controls_paused():
        gui_start.visible = False
        gui_pause.visible = False
        gui_resume.visible = True
        gui_stop.visible = True
        gui_step.visible = gui_mode.value == "Runtime-Adapt"

    def _show_controls_idle():
        gui_start.visible = True
        gui_pause.visible = False
        gui_resume.visible = False
        gui_stop.visible = False
        gui_step.visible = gui_mode.value == "Runtime-Adapt"

    ctrl.set_lifecycle_handlers(
        on_running=_show_controls_running,
        on_paused=_show_controls_paused,
        on_idle=_show_controls_idle,
    )

    # ----- Training actions -----
    @gui_scan.on_click
    def _(_):
        path = gui_batch_dir.value.strip()
        if not path:
            _set_status("Please enter a batch directory path.", "red")
            return
        rep = verify_continual_batch_dir(path)
        if not rep["ok"]:
            msg = "Verification failed:\n- " + "\n- ".join(rep["errors"])
            if rep.get("warnings"):
                msg += "\nWarnings:\n- " + "\n- ".join(rep["warnings"])
            _set_status("Batch invalid. See console.", "red")
            CONSOLE.print(f"[bold red]❌ {msg}[/bold red]")
            return

        s = rep["summary"]
        warn = rep.get("warnings", [])
        _set_status(
            f"OK: {s['counts']['images']} images / {s['counts']['metadata']} metadata in {s['batch_dir']}",
            "green",
        )
        CONSOLE.print(
            "[bold green]✅ Batch verified[/bold green]\n"
            f"• prepared_root: {s['prepared_root']}\n"
            f"• batch_dir: {s['batch_dir']}\n"
            f"• counts: {s['counts']}\n"
            f"• example id: {s['example_id']}"
        )
        if warn:
            CONSOLE.print("[bold yellow]⚠ Warnings:[/bold yellow] " + "; ".join(warn))

        ctrl.runtime = getattr(ctrl, "runtime", {})
        ctrl.runtime["prepared_root"] = s["prepared_root"]
        ctrl.runtime["batch_dir"] = s["batch_dir"]

    @gui_mode.on_update
    def _(_):
        mode = gui_mode.value
        vis = lambda *xs: [setattr(x, "visible", True) for x in xs]
        hide = lambda *xs: [setattr(x, "visible", False) for x in xs]

        # Always hide everything first
        hide(
            gui_start,
            gui_pause,
            gui_resume,
            gui_stop,
            gui_step,
            gui_support_rays,
            gui_ray_samples,
            gui_chunk_points,
            gui_downscale,
            btn_reset_fast,
            gui_ckpt_path,
            gui_ckpt_inc_base,
            btn_save_ckpt,
            gui_batch_dir,
            gui_scan,
            gui_adapt_steps,
            gui_batch_size,
            gui_sigma_lr,
            gui_color_lr,
            gui_encoding_lr,
        )

        # Default: runtime folder hidden unless Runtime-Adapt
        if mode == "View Model":
            runtime_folder.visible = False
            _set_status("Viewer mode (no training controls)", "cyan")
            _show_controls_idle()

        elif mode == "Meta-Train":
            runtime_folder.visible = False
            vis(gui_start, gui_stop)
            _set_status("Meta-training ready.", "yellow")
            _show_controls_idle()

        elif mode == "Runtime-Adapt":
            runtime_folder.visible = True
            vis(
                gui_start,
                gui_stop,
                gui_step,
                gui_batch_dir,
                gui_scan,
                gui_adapt_steps,
                gui_batch_size,
                gui_sigma_lr,
                gui_color_lr,
                gui_encoding_lr,
                gui_support_rays,
                gui_ray_samples,
                gui_chunk_points,
                gui_downscale,
                btn_reset_fast,
                gui_ckpt_path,
                gui_ckpt_inc_base,
                btn_save_ckpt,
            )
            _set_status("Runtime adaptation ready.", "yellow")
            _show_controls_idle()

    @gui_start.on_click
    def _(_):
        mode = gui_mode.value
        if mode == "Runtime-Adapt":
            ctrl.ensure_runner("Runtime-Adapt")
            ctrl.update_train_hparams(
                sigma_lr=float(gui_sigma_lr.value),
                color_lr=float(gui_color_lr.value),
                encoding_lr=float(gui_encoding_lr.value),
                support_rays=int(gui_support_rays.value),
                ray_samples=int(gui_ray_samples.value),
                chunk_points=int(gui_chunk_points.value),
                batch_size=int(gui_batch_size.value),
            )
            ctrl.update_data_hparams(downscale=float(gui_downscale.value))
            if not getattr(ctrl, "_running", False):
                training_state["target_steps"] = ctrl.current_step + int(
                    gui_adapt_steps.value
                )
            steps_to_run = max(0, training_state["target_steps"] - ctrl.current_step)
            ctrl.start("Runtime-Adapt", total_steps=steps_to_run)
            _show_controls_running()
        elif mode == "Meta-Train":
            ctrl.start("Meta-Train", total_steps=100)
            _show_controls_running()
        else:
            ctrl.start("View Model", total_steps=1)

    @gui_pause.on_click
    def _(_):
        ctrl.pause()
        _show_controls_paused()

    @gui_resume.on_click
    def _(_):
        ctrl.resume()
        _show_controls_running()

    @gui_stop.on_click
    def _(_):
        ctrl.stop()
        _show_controls_idle()

    @gui_step.on_click
    def _(_):
        ctrl.ensure_runner("Runtime-Adapt")
        ctrl.step_once()

    # Live knob → controller
    def _apply(fn, *, require_runner: bool = False):
        if require_runner and getattr(ctrl, "runner", None) is None:
            _set_status(
                "No active runner. Switch to a training mode and press Start first.",
                "red",
            )
            return
        fn()

    @gui_color_lr.on_update
    def _(_):
        _apply(lambda: ctrl.update_train_hparams(color_lr=float(gui_color_lr.value)))

    @gui_sigma_lr.on_update
    def _(_):
        _apply(lambda: ctrl.update_train_hparams(sigma_lr=float(gui_sigma_lr.value)))

    @gui_encoding_lr.on_update
    def _(_):
        _apply(
            lambda: ctrl.update_train_hparams(encoding_lr=float(gui_encoding_lr.value))
        )

    @gui_support_rays.on_update
    def _(_):
        _apply(
            lambda: ctrl.update_train_hparams(support_rays=int(gui_support_rays.value))
        )

    @gui_ray_samples.on_update
    def _(_):
        _apply(
            lambda: ctrl.update_train_hparams(ray_samples=int(gui_ray_samples.value))
        )

    @gui_chunk_points.on_update
    def _(_):
        _apply(
            lambda: ctrl.update_train_hparams(chunk_points=int(gui_chunk_points.value))
        )

    @gui_downscale.on_update
    def _(_):
        _apply(lambda: ctrl.update_data_hparams(downscale=float(gui_downscale.value)))

    @btn_reset_fast.on_click
    def _(_):
        _apply(lambda: ctrl.reset_fast(), require_runner=True)

    @btn_save_ckpt.on_click
    def _(_):
        path = gui_ckpt_path.value.strip()
        if not path:
            _set_status("Please set a checkpoint path.", "red")
            return
        _apply(
            lambda: ctrl.save_checkpoint(
                path, include_base_model=bool(gui_ckpt_inc_base.value)
            ),
            require_runner=True,
        )

    training_handles = {
        "mode": gui_mode,
        "status": gui_status,
        "start": gui_start,
        "pause": gui_pause,
        "resume": gui_resume,
        "stop": gui_stop,
        "step": gui_step,
    }

    # ----- Render callback (no_grad) -----
    @torch.no_grad()
    def nerf_render_fn(camera_state, render_tab):
        # Old/new nerfview API compat
        if isinstance(render_tab, tuple) and len(render_tab) == 2:
            W, H = int(render_tab[0]), int(render_tab[1])
            rk = FULL.kwargs
        else:
            tab_state = render_tab
            if getattr(tab_state, "preview_render", False):
                W, H, rk = (
                    tab_state.render_width,
                    tab_state.render_height,
                    PREVIEW.kwargs,
                )
            else:
                W, H, rk = tab_state.viewer_width, tab_state.viewer_height, FULL.kwargs

        # Acquire model lock (non-blocking). If busy, return last frame.
        locked = model_lock.acquire(blocking=False)
        try:
            c2w_np = camera_state.c2w.astype(np.float32)
            state["last_c2w_np"] = c2w_np

            if not locked:
                if state["last_frame_u8"] is not None:
                    return state["last_frame_u8"]
                return np.zeros((H, W, 3), dtype=np.uint8)

            # GUI values snapshot
            bg_color = safe_bg(gui_bg.value)
            active_mod = safe_active_module(gui_sub.value, model)
            setattr(model, "use_occ", bool(gui_occ.value))
            FULL.ray_samples, PREVIEW.ray_samples = int(gui_full_samples.value), int(
                gui_prev_samples.value
            )

            state["display"] = str(gui_display.value)
            state["zoom"] = float(gui_zoom.value)
            state["exposure"] = float(gui_exposure.value)
            state["gamma"] = float(gui_gamma.value)
            depth_inv = bool(gui_depth_inv.value)
            depth_cmap = str(gui_depth_cmap.value)
            acc_cmap = str(gui_acc_cmap.value)

            # Intrinsics (+ zoom)
            K = camera_state.get_K([W, H]).copy()
            if abs(state["zoom"] - 1.0) > 1e-6:
                K[0, 0] *= state["zoom"]
                K[1, 1] *= state["zoom"]
            fx, fy, cx, cy = (
                float(K[0, 0]),
                float(K[1, 1]),
                float(K[0, 2]),
                float(K[1, 2]),
            )

            # Per-pixel directions (cached)
            dirs_np = _cached_ray_dirs_key(H, W, fx, fy, cx, cy)
            directions = torch.from_numpy(dirs_np).to(
                dev, dtype=torch.float32, non_blocking=False
            )

            # RUB→DRB normalization of pose
            c2w = torch.from_numpy(c2w_np).to(dev, dtype)
            if getattr(P, "viewer_rub_to_drb", True):
                M = rub_to_drb_3x3(dev, c2w.dtype)
                c2w[:3, :3] = M @ c2w[:3, :3]
                c2w[:3, 3] = M @ c2w[:3, 3]
            c2w[:3, 3] = (c2w[:3, 3] - origin_drb) / pose_scale

            # Rays + optional near/far override
            rays = (
                get_rays(directions, c2w, scene_box=scene_box).view(-1, 8).contiguous()
            )
            near_far_override = None
            if bool(gui_nf_enable.value):
                near_m, far_m = float(gui_near_m.value), float(gui_far_m.value)
                near_n = max(1e-5, float(np.nan_to_num(near_m / pose_scale, nan=0.0)))
                far_n = max(
                    near_n + 1e-5, float(np.nan_to_num(far_m / pose_scale, nan=1.0))
                )
                near_far_override = (near_n, far_n)
            rays, valid = clamp_rays_near_far(rays, near_far_override=near_far_override)

            # Sanitize rays once
            if not torch.isfinite(rays).all():
                rays = torch.nan_to_num(rays, nan=0.0, posinf=0.0, neginf=0.0)

            # ---- Render call (fixed chunk, as before) ----
            try:
                params_override = ctrl.get_render_params()  # adapted fast params if any
                rgb_lin, depth, weights, acc = render_rays(
                    model,
                    rays,
                    params=params_override,
                    active_module=active_mod,
                    bg_color_default=bg_color,
                    chunk=int(getattr(P, "chunk_points", 5_000_000)),
                    ray_samples=rk.get("ray_samples"),
                )
            except RuntimeError as e:
                # Optional safety net: return a blank frame on CUDA OOM instead of crashing
                if "out of memory" in str(e).lower() or "cuda error" in str(e).lower():
                    _set_status(
                        "CUDA OOM at fixed chunk_points — returning blank frame.", "red"
                    )
                    return np.zeros((H, W, 3), dtype=np.uint8)
                raise

            # ---- Invalid ray mask fill ----
            if not valid.all():
                m = ~valid
                if bg_color in ["white", "bg_nerf", "last_sample"]:
                    rgb_lin[m] = 1.0
                elif bg_color == "black":
                    rgb_lin[m] = 0.0
                elif bg_color == "random":
                    rgb_lin[m] = torch.rand(m.sum(), 3, device=dev)
                else:
                    rgb_lin[m] = 1.0

            # ---- Tonemapping ----
            disp = state["display"]
            if disp == "Depth" and depth is not None:
                d = torch.nan_to_num(depth.view(-1, 1), nan=float("inf"))
                near, far = d.min().item(), d.max().item()
                img_lin = apply_depth_colormap(
                    d,
                    accumulation=acc,
                    near_plane=near,
                    far_plane=far,
                    colormap_options=ColormapOptions(
                        colormap="default" if depth_cmap == "turbo" else depth_cmap,
                        invert=depth_inv,
                        normalize=False,
                    ),
                ).view(H, W, 3)
            elif disp == "Opacity" and acc is not None:
                a = acc.view(-1, 1).clamp(0, 1)
                img_lin = apply_colormap(
                    a, ColormapOptions(colormap=acc_cmap, normalize=True)
                ).view(H, W, 3)
            else:
                img_lin = rgb_lin.view(H, W, 3)
                img_lin = (
                    (img_lin * state["exposure"]).clamp(0, 1).pow(1.0 / state["gamma"])
                )

            out_u8 = uint8_from_linear01(img_lin)
            state["last_frame_u8"] = out_u8
            return out_u8

        finally:
            if locked:
                model_lock.release()

    # ----- Wire viewer -----
    viewer = nerfview.Viewer(server=server, render_fn=nerf_render_fn, mode="rendering")
    viewer._training_tab_handles = training_handles
    viewer.viewer_state = viewer_state

    return viewer
