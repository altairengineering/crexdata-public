import math
import warnings
from typing import Literal, Optional

import torch
import torch.nn as nn

try:
    import tinycudann as tcnn

    _TCNN_AVAILABLE = True
except Exception:
    tcnn = None
    _TCNN_AVAILABLE = False

# =========================
# SHEncoder (TCNN or Torch)
# =========================
MAX_SH_DEGREE = 4


def num_sh_bases(degree: int) -> int:
    assert degree <= MAX_SH_DEGREE, f"We don't support degree > {MAX_SH_DEGREE}."
    return (degree + 1) ** 2


def components_from_spherical_harmonics(
    degree: int, directions: torch.Tensor
) -> torch.Tensor:
    """
    Compute real spherical harmonic components up to `degree`.
    directions: (...,3), assumed unit length.
    Returns (...,(degree+1)^2)
    """
    assert 0 <= degree <= MAX_SH_DEGREE
    assert directions.shape[-1] == 3

    x, y, z = directions[..., 0], directions[..., 1], directions[..., 2]
    xx, yy, zz = x * x, y * y, z * z

    comps = directions.new_zeros((*directions.shape[:-1], num_sh_bases(degree)))

    # l=0
    comps[..., 0] = 0.28209479177387814

    # l=1
    if degree > 0:
        comps[..., 1] = 0.4886025119029199 * y
        comps[..., 2] = 0.4886025119029199 * z
        comps[..., 3] = 0.4886025119029199 * x

    # l=2
    if degree > 1:
        comps[..., 4] = 1.0925484305920792 * x * y
        comps[..., 5] = 1.0925484305920792 * y * z
        comps[..., 6] = 0.9461746957575601 * zz - 0.31539156525251999
        comps[..., 7] = 1.0925484305920792 * x * z
        comps[..., 8] = 0.5462742152960396 * (xx - yy)

    # l=3
    if degree > 2:
        comps[..., 9] = 0.5900435899266435 * y * (3 * xx - yy)
        comps[..., 10] = 2.890611442640554 * x * y * z
        comps[..., 11] = 0.4570457994644658 * y * (5 * zz - 1)
        comps[..., 12] = 0.3731763325901154 * z * (5 * zz - 3)
        comps[..., 13] = 0.4570457994644658 * x * (5 * zz - 1)
        comps[..., 14] = 1.445305721320277 * z * (xx - yy)
        comps[..., 15] = 0.5900435899266435 * x * (xx - 3 * yy)

    # l=4
    if degree > 3:
        comps[..., 16] = 2.5033429417967046 * x * y * (xx - yy)
        comps[..., 17] = 1.7701307697799304 * y * z * (3 * xx - yy)
        comps[..., 18] = 0.9461746957575601 * x * y * (7 * zz - 1)
        comps[..., 19] = 0.6690465435572892 * y * z * (7 * zz - 3)
        comps[..., 20] = 0.10578554691520431 * (35 * zz * zz - 30 * zz + 3)
        comps[..., 21] = 0.6690465435572892 * x * z * (7 * zz - 3)
        comps[..., 22] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1)
        comps[..., 23] = 1.7701307697799304 * x * z * (xx - 3 * yy)
        comps[..., 24] = 0.6258357354491761 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
    return comps


class SHEncoder(nn.Module):
    """
    Real Spherical Harmonics encoder (like Nerfstudio's SHEncoding)
    - Uses TCNN 'SphericalHarmonics' if available, else a pure Torch fallback.

    Args:
        levels: number of SH levels (degree = levels - 1)
        implementation: "tcnn" | "torch"
    """

    def __init__(
        self, levels: int = 4, implementation: Literal["tcnn", "torch"] = "tcnn"
    ) -> None:
        super().__init__()
        if levels <= 0 or levels > MAX_SH_DEGREE + 1:
            raise ValueError(
                f"Supported levels ∈ [1, {MAX_SH_DEGREE + 1}], got {levels}"
            )

        self.levels = int(levels)
        self.degree = self.levels - 1
        self._out_dim = self.levels**2
        self._use_tcnn = False

        # --- TCNN fast path ---
        if implementation == "tcnn":
            if _TCNN_AVAILABLE:
                try:
                    self._tcnn_enc = tcnn.Encoding(
                        n_input_dims=3,
                        encoding_config={
                            "otype": "SphericalHarmonics",
                            "degree": self.levels,
                        },
                    )
                    self._use_tcnn = True
                except Exception as e:
                    warnings.warn(
                        f"[SHEncoder] tinycudann init failed ({e}); using Torch fallback."
                    )
            else:
                warnings.warn(
                    "[SHEncoder] tinycudann not available; using Torch fallback."
                )

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            d: (...,3) unit directions (normalize inside)
        Returns:
            (...,(levels)^2) encoding
        """
        assert d.shape[-1] == 3, f"Expected (...,3); got {tuple(d.shape)}"
        d = d / d.norm(dim=-1, keepdim=True).clamp_min_(1e-9)

        if self._use_tcnn:
            flat = d.reshape(-1, 3).contiguous().float()
            y = self._tcnn_enc(flat)
            return y.view(*d.shape[:-1], self._out_dim)

        # Torch fallback (identical math to Nerfstudio)
        with torch.cuda.amp.autocast(enabled=False):
            comps = components_from_spherical_harmonics(self.degree, d.float())
        return comps.to(dtype=d.dtype)


# ===========================================
# HashGridEncoder (TCNN or pure Torch fallback)
# ===========================================
INTERPOLATIONS = ["Nearest", "Linear", "Smoothstep"]


class HashGridEncoder(nn.Module):
    """
    Instant-NGP style HashGrid encoder (Nerfstudio-like), assuming inputs in [0,1].

    Constructor mirrors Nerfstudio:
      - levels:             number of levels (L)
      - min_res:          base resolution at level 0 (N_min)
      - max_res:          target resolution at last level (≈N_max)
      - log2_hashmap_size:log2(T) of parameters per level
      - features_per_level: channels per level (F)
      - hash_init_scale:  small init scale (stability)
      - implementation:   "tcnn" | "torch"
      - interpolation:    None | "Nearest" | "Linear" | "Smoothstep"
                           (None → default backend behavior; torch requires Linear/Nearest/Smoothstep)
    API:
      forward(x: (...,3) in [0,1]) -> (..., L*F)
      get_out_dim() -> int
      out_dim (property) -> int
    """

    def __init__(
        self,
        levels: int = 16,
        min_res: int = 16,
        max_res: int = 4096,
        log2_hashmap_size: int = 19,
        features_per_level: int = 2,
        hash_init_scale: float = 1e-3,
        implementation: Literal["tcnn", "torch"] = "tcnn",
        interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = None,
    ) -> None:
        super().__init__()
        # Hyperparameters
        self.levels = int(levels)
        self.min_res = int(min_res)
        self.max_res = int(max_res)
        self.features_per_level = int(features_per_level)
        self.log2_hashmap_size = int(log2_hashmap_size)
        self.hash_init_scale = float(hash_init_scale)
        self.hash_table_size = 2**self.log2_hashmap_size
        self.interpolation = interpolation  # may be None

        # Per-level resolutions (integer), growth factor like Nerfstudio
        L = self.levels
        self.growth_factor = (
            1.0
            if L <= 1
            else float(
                math.exp((math.log(self.max_res) - math.log(self.min_res)) / (L - 1))
            )
        )
        levels = torch.arange(L, dtype=torch.float32)
        scalings = torch.floor(self.min_res * (self.growth_factor**levels)).to(
            torch.int32
        )  # (L,)
        self.register_buffer("level_resolutions", scalings, persistent=False)

        # Per-level offset for hash tables
        self.register_buffer(
            "level_offsets",
            torch.arange(L, dtype=torch.int64) * self.hash_table_size,
            persistent=False,
        )

        self._out_dim = self.levels * self.features_per_level
        self._interp_warned = False
        self._use_tcnn = False

        # --- TCNN path ---
        if implementation == "tcnn":
            if _TCNN_AVAILABLE:
                enc_cfg = {
                    "otype": "HashGrid",
                    "n_levels": self.levels,
                    "n_features_per_level": self.features_per_level,
                    "log2_hashmap_size": self.log2_hashmap_size,
                    "base_resolution": self.min_res,
                    "per_level_scale": self.growth_factor,
                }
                if (
                    self.interpolation is not None
                    and self.interpolation in INTERPOLATIONS
                ):
                    enc_cfg["interpolation"] = self.interpolation
                try:
                    # Use fp16 in TCNN path (fast path like NGP)
                    self._tcnn_enc = tcnn.Encoding(
                        n_input_dims=3,
                        encoding_config=enc_cfg,
                        dtype=torch.float16,
                    )
                    self._use_tcnn = True
                except Exception as e:
                    warnings.warn(
                        f"[HashGridEncoder] tinycudann init failed ({e}); using Torch fallback.",
                        RuntimeWarning,
                    )
            else:
                warnings.warn(
                    "[HashGridEncoder] tinycudann not available; using Torch fallback.",
                    RuntimeWarning,
                )

        # --- Torch fallback ---
        if not self._use_tcnn:
            T = self.hash_table_size * self.levels
            F = self.features_per_level
            table = (torch.rand(T, F) * 2 - 1) * self.hash_init_scale
            self.hash_table = nn.Parameter(table)

            # 64-bit primes for hashing
            self.register_buffer(
                "hash_primes",
                torch.tensor([1, 2654435761, 805459861], dtype=torch.int64),
                persistent=False,
            )

            # Validate interpolation choices for torch
            if (
                self.interpolation is not None
                and self.interpolation not in INTERPOLATIONS
            ):
                warnings.warn(
                    f"[HashGridEncoder] interpolation '{self.interpolation}' not supported in torch backend; using 'Linear'.",
                    RuntimeWarning,
                )
                self.interpolation = "Linear"

    # ---------- Public API ----------
    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., 3) assumed to be in [0,1].
        Returns: (..., levels * features_per_level)
        """
        assert x.shape[-1] == 3, f"Expected (...,3), got {tuple(x.shape)}"

        if self._use_tcnn:
            flat = x.reshape(-1, 3).contiguous().half()  # TCNN expects fp16 here
            y = self._tcnn_enc(flat)
            return y.view(*x.shape[:-1], self._out_dim)

        return self._torch_forward(x)

    # ---------- Torch backend ----------
    def _hash(
        self, ix: torch.Tensor, iy: torch.Tensor, iz: torch.Tensor
    ) -> torch.Tensor:
        # Instant-NGP style 3D hash
        return (
            (ix.to(torch.int64) * self.hash_primes[0])
            ^ (iy.to(torch.int64) * self.hash_primes[1])
            ^ (iz.to(torch.int64) * self.hash_primes[2])
        ) % (2**self.log2_hashmap_size)

    def _gather(
        self,
        ix: torch.Tensor,
        iy: torch.Tensor,
        iz: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        idx = self._hash(ix, iy, iz)  # (..., L)
        feats32 = self.hash_table[
            idx + self.level_offsets
        ]  # (..., L, F) in parameter dtype (fp32)
        return feats32.to(dtype=out_dtype)

    def _torch_forward(self, x01: torch.Tensor) -> torch.Tensor:
        L = self.levels
        levels = self.level_resolutions.to(x01.device, dtype=x01.dtype)  # (L,)
        scaled = x01[..., None, :] * levels.view(
            *([1] * (x01.ndim - 1)), L, 1
        )  # (..., L, 3)

        mode = self.interpolation or "Linear"

        if mode == "Nearest":
            idx = torch.round(scaled).to(torch.int64)
            feats = self._gather(idx[..., 0], idx[..., 1], idx[..., 2], x01.dtype)
            return feats.view(*x01.shape[:-1], L * self.features_per_level)

        # Tri-linear (Linear or Smoothstep)
        floor = torch.floor(scaled)
        frac = scaled - floor
        floor = floor.to(torch.int64)
        ceil = floor + 1

        f000 = self._gather(floor[..., 0], floor[..., 1], floor[..., 2], x01.dtype)
        f001 = self._gather(floor[..., 0], floor[..., 1], ceil[..., 2], x01.dtype)
        f010 = self._gather(floor[..., 0], ceil[..., 1], floor[..., 2], x01.dtype)
        f011 = self._gather(floor[..., 0], ceil[..., 1], ceil[..., 2], x01.dtype)
        f100 = self._gather(ceil[..., 0], floor[..., 1], floor[..., 2], x01.dtype)
        f101 = self._gather(ceil[..., 0], floor[..., 1], ceil[..., 2], x01.dtype)
        f110 = self._gather(ceil[..., 0], ceil[..., 1], floor[..., 2], x01.dtype)
        f111 = self._gather(ceil[..., 0], ceil[..., 1], ceil[..., 2], x01.dtype)

        wx, wy, wz = frac[..., 0:1], frac[..., 1:2], frac[..., 2:3]
        if mode == "Smoothstep":
            wx = wx * wx * (3 - 2 * wx)
            wy = wy * wy * (3 - 2 * wy)
            wz = wz * wz * (3 - 2 * wz)
        elif mode != "Linear":
            if not self._interp_warned:
                warnings.warn(
                    f"[HashGridEncoder] unsupported interpolation '{mode}' in torch backend; using Linear.",
                    RuntimeWarning,
                )
                self._interp_warned = True

        c00 = f000 * (1 - wx) + f100 * wx
        c01 = f001 * (1 - wx) + f101 * wx
        c10 = f010 * (1 - wx) + f110 * wx
        c11 = f011 * (1 - wx) + f111 * wx
        c0 = c00 * (1 - wy) + c10 * wy
        c1 = c01 * (1 - wy) + c11 * wy
        feats = c0 * (1 - wz) + c1 * wz  # (..., L, F)

        return feats.flatten(start_dim=-2)


# ========================================
# FrequencyEncoder (TCNN or Torch version)
# ========================================
class FrequencyEncoder(nn.Module):
    """
    Fourier features (NeRF PE). TCNN 'Frequency' if available; fixed in_dim.
    API: forward(x:(...,D))->(...,D*(2*L+[1 if include_input])); out_dim()->int
    """

    def __init__(
        self, in_dim: int, pe_dim: int, include_input: bool = True, use_pi: bool = False
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.pe_dim = int(pe_dim)
        self.include_input = bool(include_input)
        self.use_pi = bool(use_pi)
        self._use_tcnn = False

        if _TCNN_AVAILABLE:
            try:
                self._tcnn = tcnn.Encoding(
                    n_input_dims=self.in_dim,
                    encoding_config={
                        "otype": "Frequency",
                        "n_frequencies": self.pe_dim,
                        "include_input": self.include_input,
                    },
                )
                self._use_tcnn = True
            except Exception as e:
                warnings.warn(
                    f"[FrequencyEncoder] tinycudann init failed ({e}); using Torch.",
                    RuntimeWarning,
                )

        bands = 2.0 ** torch.arange(self.pe_dim, dtype=torch.float32)
        self.register_buffer("bands", bands, persistent=False)

    @property
    def out_dim(self) -> int:
        return self.in_dim * (2 * self.pe_dim + (1 if self.include_input else 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[-1] == self.in_dim
        ), f"Expected (...,{self.in_dim}), got {tuple(x.shape)}"
        if self._use_tcnn:
            xin = x * (math.pi if self.use_pi else 1.0)
            y = self._tcnn(xin.reshape(-1, self.in_dim).contiguous().float())
            return y.view(*x.shape[:-1], self.out_dim)
        return self.torch_forward(x)

    def torch_forward(self, x: torch.Tensor) -> torch.Tensor:
        fb = self.bands.to(dtype=x.dtype, device=x.device)
        xin = x * (x.new_tensor(math.pi) if self.use_pi else 1)
        x_exp = xin[..., None] * fb  # (..., D, L)
        s = torch.sin(x_exp)  # stays in autocast dtype
        c = torch.cos(x_exp)
        pe = torch.cat([c, s], dim=-1).reshape(*x.shape[:-1], -1)
        return torch.cat([x, pe], dim=-1) if self.include_input else pe
