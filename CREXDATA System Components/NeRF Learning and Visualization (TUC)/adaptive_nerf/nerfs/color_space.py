import torch


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    return torch.where(
        x <= 0.0031308,
        12.92 * x,
        1.055 * x.pow(1 / 2.4) - 0.055,
    )


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    # x in [0,1]
    return torch.where(
        x <= 0.04045,
        x / 12.92,
        ((x + 0.055) / 1.055).pow(2.4),
    )


def color_space_transformer(
    pred_linear: torch.Tensor, gt_tensor: torch.Tensor, color_space: str
):
    """
    Convert exactly one side so loss/metrics are in the *same* space.

    pred_linear: prediction in *linear* space.
    gt_tensor:   GT in *sRGB* [0,1] (as loaded from 8-bit images).
    """
    cs = str(color_space).lower()

    pred_lin32 = pred_linear.to(torch.float32)
    gt32 = gt_tensor.to(torch.float32).clamp(0, 1)

    if cs == "linear":
        # Compare in linear: convert GT sRGB -> linear
        pred = pred_lin32
        gt = srgb_to_linear(gt32)
        pred = pred.clamp(0, 1)
        gt = gt.clamp(0, 1)

    elif cs == "srgb":
        # Compare in sRGB: convert pred linear -> sRGB
        pred = linear_to_srgb(pred_lin32)
        gt = gt32
        pred = pred.clamp(0, 1)
        gt = gt.clamp(0, 1)

    elif cs == "identity":
        # Only valid if dataset already supplies linear GT
        if (gt32.max() > 1) or (gt32.min() < 0):
            raise ValueError(
                "GT out of [0,1]; identity mode assumes normalized linear GT."
            )
        pred = pred_lin32
        gt = gt32
    else:
        raise ValueError(
            f"Invalid color_space={color_space!r}; use 'linear'|'srgb'|'identity'"
        )

    # Match dtype/device to pred_linear for downstream code
    pred = pred.to(pred_linear.dtype).to(pred_linear.device)
    gt = gt.to(pred_linear.dtype).to(pred_linear.device)
    return pred, gt
