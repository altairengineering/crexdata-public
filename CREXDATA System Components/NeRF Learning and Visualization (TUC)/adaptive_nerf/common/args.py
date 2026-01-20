import os
import sys
import argparse
import torch


def build_parser():
    parser = argparse.ArgumentParser(description="TUC's Adaptive NeRF Framework")
    parser.add_argument(
        "--op", type=str, default="train", choices=["train", "eval", "view", "video"]
    )

    # --- system
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true")

    # --- io/logging
    parser.add_argument("--eval_step", type=int, default=200)
    parser.add_argument("--save_step", type=int, default=1000)
    parser.add_argument("--print_step", type=int, default=1)
    parser.add_argument(
        "--log_method", type=str, default="step", choices=["step", "patch"]
    )

    # --- data
    parser.add_argument(
        "--dataset",
        type=str,
        default="drz",
        choices=["ffhq", "celeba", "imagenette", "voxceleb", "drz"],
    )
    parser.add_argument(
        "--data_type", type=str, default="ray", choices=["img", "video", "ray"]
    )
    parser.add_argument("--data_path", type=str, default="data/drz/")
    parser.add_argument("--data_dirname", type=str, default="balanced")
    parser.add_argument("--mask_dirname", type=str, default="g22_grid_bm110_ss11")
    parser.add_argument("--cap_images", type=int, default=None)
    parser.add_argument("--downscale", type=float, default=0.25)
    parser.add_argument("--near", type=float, default=None)
    parser.add_argument("--far", type=float, default=None)
    parser.add_argument("--bm", type=float, default=1.05)

    # --- episode gen
    parser.add_argument("--support_rays", type=int, default=4000)
    parser.add_argument("--query_rays", type=int, default=2000)
    parser.add_argument("--cell_dim", type=int, default=5)

    # --- dataloader
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--test_batch_size", type=int, default=1)

    # --- model
    parser.add_argument("--num_submodules", type=int, default=4)
    parser.add_argument(
        "--nerf_variant", type=str, default="instant", choices=["instant", "vanilla"]
    )
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--sigma_depth", type=int, default=2)
    parser.add_argument("--color_depth", type=int, default=2)
    parser.add_argument("--dim_hidden", type=int, default=64)
    parser.add_argument("--color_hidden", type=int, default=64)

    # --- hash encoding
    parser.add_argument("--max_res", type=int, default=4096)
    parser.add_argument("--log2_hashmap_size", type=int, default=20)
    parser.add_argument("--use_occ", action="store_true")
    parser.add_argument(
        "--xyz_encoding", type=str, default="hash", choices=["frequency", "hash"]
    )
    parser.add_argument(
        "--dir_encoding",
        type=str,
        default="spherical",
        choices=["frequency", "spherical"],
    )

    # --- background model
    parser.add_argument("--no_bg_nerf", action="store_true")
    parser.add_argument(
        "--bg_color_default",
        type=str,
        default="random",
        choices=["white", "black", "none", "last_sample", "random"],
    )
    parser.add_argument("--bg_hidden", type=int, default=32)
    parser.add_argument(
        "--bg_encoding",
        type=str,
        default="spherical",
        choices=["frequency", "spherical"],
    )

    # --- rendering
    parser.add_argument("--ray_samples", type=int, default=96)
    parser.add_argument("--chunk_points", type=int, default=262_144 * 17)
    parser.add_argument(
        "--color_space",
        type=str,
        default="linear",
        choices=["srgb", "linear", "identity"],
    )
    # --- FIM
    parser.add_argument("--fim", action="store_true")
    parser.add_argument("--fim_per_sample", action="store_true")
    parser.add_argument("--fim_lambda", type=float, default=0.1)
    parser.add_argument("--fim_beta", type=float, default=0.95)
    parser.add_argument("--fim_epsilon", type=float, default=1e-6)

    # --- optimizer
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adamw", "sgd", "adam"]
    )
    parser.add_argument("--encoding_lr", type=float, default=1e-2)
    parser.add_argument("--sigma_lr", type=float, default=2e-3)
    parser.add_argument("--color_lr", type=float, default=2e-3)
    parser.add_argument("--bg_lr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-4)

    # --- scheduler
    parser.add_argument("--no_scheduler", action="store_true")
    parser.add_argument(
        "--decay_factor", type=float, default=10
    )  # final LR = initial LR / decay_factor

    # --- training
    parser.add_argument("--inner_iter", type=int, default=8)
    parser.add_argument("--inner_lr", type=float, default=15e-3)
    parser.add_argument("--outer_steps", type=int, default=20_000)
    parser.add_argument(
        "--algo",
        type=str,
        default="fomaml",
        choices=["maml", "fomaml", "reptile"],
    )
    parser.add_argument("--max_test_tasks", type=int, default=4)

    # --- eval
    parser.add_argument("--tto", type=str, default="16")

    # ----- video & viewing
    parser.add_argument(
        "--camera_path",
        type=str,
        default="full_coverage",
        choices=["spiral_in", "turntable", "east_west", "north_south", "full_coverage"],
    )

    parser.add_argument(
        "--viewer_timeout",
        type=int,
        default=900,  # seconds, e.g. 1 hour=3600; set -1 for "run forever"
        help="Max lifetime for viewer in seconds (-1 = no limit).",
    )
    parser.add_argument("--viewer_public_host", type=str, default="192.168.1.17")

    # --- extras
    parser.add_argument("--configPath", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--log_date", action="store_true")
    parser.add_argument("--fname", default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--prefix", type=str, default="best")  # best/last/step{i}
    parser.add_argument("--no_strict", action="store_true")

    return parser


ARCH_KEYS = {
    # model
    "num_submodules",
    "nerf_variant",
    "num_layers",
    "sigma_depth",
    "color_depth",
    "dim_hidden",
    "color_hidden",
    # encodings / bg
    "max_res",
    "log2_hashmap_size",
    "xyz_encoding",
    "dir_encoding",
    "no_bg_nerf",
    "bg_hidden",
    "bg_encoding",
}


def _cli_provided_dests(parser: argparse.ArgumentParser, argv):
    """Return set of argparse dest names explicitly present on CLI."""
    opt_to_action = {}
    for action in parser._actions:
        for opt in action.option_strings:
            opt_to_action[opt] = action

    provided = set()
    i = 0
    while i < len(argv):
        act = opt_to_action.get(argv[i])
        if act is not None:
            provided.add(act.dest)
        i += 1
    return provided


def load_checkpoint_cfg(cfg_path: str):
    cfg_obj = torch.load(cfg_path, map_location="cpu", weights_only=False)
    return vars(cfg_obj) if hasattr(cfg_obj, "__dict__") else dict(cfg_obj)


def _enforce_arch_from_ckpt(args, ckpt_cfg: dict):
    """Force architecture-defining args from checkpoint config."""
    for k in ARCH_KEYS:
        if k in ckpt_cfg and hasattr(args, k):
            setattr(args, k, ckpt_cfg[k])
    return args


def parse_args():
    """
    Priority:
      defaults < checkpoint < json(if not on CLI) < CLI
    Exception:
      if checkpoint exists, ARCH_KEYS are always forced from checkpoint.
    """
    parser = build_parser()
    argv = sys.argv[1:]
    args = parser.parse_args(argv)

    if args.checkpoint_path == "":
        args.checkpoint_path = None

    cli_dests = _cli_provided_dests(parser, argv)

    ckpt_cfg = None
    if args.checkpoint_path:
        cfg_path = os.path.join(args.checkpoint_path, f"{args.prefix}.P")
        if os.path.exists(cfg_path):
            ckpt_cfg = load_checkpoint_cfg(cfg_path)

            # Force architecture from checkpoint (ignores CLI/JSON).
            _enforce_arch_from_ckpt(args, ckpt_cfg)

            # Non-arch: apply checkpoint only if not explicitly provided on CLI.
            for k, v in ckpt_cfg.items():
                if k in ARCH_KEYS:
                    continue
                if k in cli_dests:
                    continue
                if hasattr(args, k):
                    setattr(args, k, v)

    config_path = getattr(args, "configPath", None)
    if config_path is not None:
        import json

        with open(config_path, "r") as f:
            cfg = json.load(f)

        for k, v in cfg.items():
            if not hasattr(args, k):
                continue
            if k in cli_dests:
                continue
            setattr(args, k, v)

    # Re-force arch after JSON merge if checkpoint is present.
    if ckpt_cfg is not None:
        _enforce_arch_from_ckpt(args, ckpt_cfg)

    if args.fname is None:
        from datetime import datetime

        args.fname = f"{args.op}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return args
