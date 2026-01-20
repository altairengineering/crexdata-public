from dataclasses import dataclass
from typing import Sequence, Tuple, Union
from jaxtyping import Float

import torch
from torch import Tensor
import viser.transforms as vtf


@dataclass
class SceneBox:
    """Minimal, efficient SceneBox with AABB stored as (2, 3): [min, max]."""

    aabb: Float[Tensor, "2 3"]

    # ---- properties for quick access ----
    @property
    def min(self) -> Tensor:
        return self.aabb[0]

    @property
    def max(self) -> Tensor:
        return self.aabb[1]

    @property
    def center(self) -> Tensor:
        return (self.aabb[0] + self.aabb[1]) * 0.5

    @property
    def extent(self) -> Tensor:
        return self.aabb[1] - self.aabb[0]

    def to(self, dev) -> "SceneBox":
        dev = torch.device(dev) if isinstance(dev, str) else dev
        return SceneBox(aabb=self.aabb.to(dev))

    # ---- pretty representation ----
    def __repr__(self) -> str:
        min_str = ", ".join(f"{x:.3f}" for x in self.min.cpu().tolist())
        max_str = ", ".join(f"{x:.3f}" for x in self.max.cpu().tolist())
        diag = self.get_diagonal_length().item()
        return f"SceneBox (min=[{min_str}], max=[{max_str}], diag={diag:.3f})"

    # Intersections
    def ray_aabb_intersect(
        self,
        origins: torch.Tensor,  # (N,3)
        directions: torch.Tensor,  # (N,3)
        eps: float = 1e-8,
        max_bound: float = 1e10,
        invalid_value: float = 1e10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized slab test with clamping & invalid tagging.

        - Clamp t to [0, max_bound]
        - Where no hit (tmax <= tmin), set both to `invalid_value`.
        """
        assert self.aabb.shape == (2, 3), "aabb must be (2,3)"

        # -------- device & dtype guard: move rays to AABB's device/dtype --------
        aabb_dev = self.aabb.device
        aabb_dtype = self.aabb.dtype

        out_dev = origins.device

        if origins.device != aabb_dev:
            origins = origins.to(aabb_dev)
        if directions.device != aabb_dev:
            directions = directions.to(aabb_dev)

        # unify dtype across all operands
        common_dtype = torch.promote_types(
            torch.promote_types(origins.dtype, directions.dtype), aabb_dtype
        )
        origins = origins.to(common_dtype)
        directions = directions.to(common_dtype)
        aabb_min = self.aabb[0].to(device=aabb_dev, dtype=common_dtype)
        aabb_max = self.aabb[1].to(device=aabb_dev, dtype=common_dtype)

        # -------- numeric guard on near-zero direction components --------
        eps_pos = torch.full_like(directions, eps)
        eps_neg = -eps_pos
        rd_safe = torch.where(
            directions.abs() < eps,
            torch.where(directions >= 0, eps_pos, eps_neg),
            directions,
        )
        inv = 1.0 / rd_safe

        # -------- slab intersections --------
        t0 = (aabb_min.unsqueeze(0) - origins) * inv  # (N,3)
        t1 = (aabb_max.unsqueeze(0) - origins) * inv  # (N,3)

        t_min = torch.minimum(t0, t1).amax(dim=-1)  # (N,)
        t_max = torch.maximum(t0, t1).amin(dim=-1)  # (N,)

        # clamp to [0, max_bound]
        t_min = t_min.clamp(min=0.0, max=max_bound)
        t_max = t_max.clamp(min=0.0, max=max_bound)

        # tag invalid (no forward hit)
        invalid = t_max <= t_min
        inv_val = torch.as_tensor(invalid_value, dtype=common_dtype, device=aabb_dev)
        t_min = torch.where(invalid, inv_val, t_min)
        t_max = torch.where(invalid, inv_val, t_max)
        return t_min.to(out_dev), t_max.to(out_dev)

    # ---- queries ----
    def within(self, pts: Float[Tensor, "n 3"], inclusive: bool = False) -> Tensor:
        if inclusive:
            return (pts >= self.aabb[0]).all(dim=-1) & (pts <= self.aabb[1]).all(dim=-1)
        return (pts > self.aabb[0]).all(dim=-1) & (pts < self.aabb[1]).all(dim=-1)

    def get_diagonal_length(self) -> Tensor:
        diff = self.aabb[1] - self.aabb[0]
        return torch.linalg.norm(diff)

    def get_centered_and_scaled_scene_box(
        self, scale_factor: Union[float, Tensor] = 1.0
    ) -> "SceneBox":
        c = self.center
        s = torch.as_tensor(
            scale_factor, device=self.aabb.device, dtype=self.aabb.dtype
        )
        return SceneBox(aabb=(self.aabb - c) * s)

    # ---- normalization ----
    @staticmethod
    def get_normalized_positions(
        positions: Float[Tensor, "*batch 3"], aabb: Float[Tensor, "2 3"]
    ) -> Tensor:
        lengths = aabb[1] - aabb[0]
        return (positions - aabb[0]) / lengths

    # ---- constructors ----
    @staticmethod
    def from_camera_poses(
        poses: Float[Tensor, "*batch 3 4"], scale_factor: float = 1.0
    ) -> "SceneBox":
        # just go with default
        xyzs = poses[..., :3, -1]
        mn = xyzs.min(dim=-2).values
        mx = xyzs.max(dim=-2).values
        aabb = torch.stack([mn, mx], dim=0) * scale_factor
        return SceneBox(aabb=aabb)

    @staticmethod
    def from_bound(aabb: torch.Tensor) -> "SceneBox":
        """
        Create a SceneBox directly from an explicit AABB tensor.

        Args:
            aabb: Tensor of shape (2, 3) with [min, max] in DRB convention.
                Example: tensor([[-0.1, -1.0, -1.0],
                                    [ 0.45,  1.0,  1.0]])
        """
        assert isinstance(aabb, torch.Tensor), "aabb must be a torch.Tensor"
        assert aabb.shape == (2, 3), f"Expected (2,3) AABB, got {tuple(aabb.shape)}"
        return SceneBox(aabb=aabb)

    def expand(self, pad: Union[float, torch.Tensor, Sequence[float]]) -> "SceneBox":
        """
        Expand the box by absolute padding (no halving).

        pad can be:
        - scalar: same pad on all axes for both min/max (ndim == 0)
        - (3,) or (1,3): per-axis symmetric pad (min -= p, max += p)
        - (2,3): explicit/asymmetric pad:
                pad[0] applies to the min side  (subtract)
                pad[1] applies to the max side  (add)

        Negative values shrink; positive expand.

        Returns:
            New SceneBox with expanded AABB.
        """
        pad_t = torch.as_tensor(pad, dtype=self.aabb.dtype, device=self.aabb.device)

        if pad_t.ndim == 0:
            # scalar → same pad everywhere, both sides
            pad_min = pad_t.expand(3)
            pad_max = pad_t.expand(3)
        elif pad_t.shape == (3,) or pad_t.shape == (1, 3):
            # symmetric per-axis
            p = pad_t.view(-1, 3)[-1]  # (3,)
            pad_min = p
            pad_max = p
        elif pad_t.shape == (2, 3):
            # asymmetric per-axis: [pad_min, pad_max]
            pad_min = pad_t[0]  # (3,)
            pad_max = pad_t[1]  # (3,)
        else:
            raise ValueError(
                f"pad must be scalar, (3,), (1,3), or (2,3); got shape {tuple(pad_t.shape)}"
            )

        mn = self.aabb[0] - pad_min
        mx = self.aabb[1] + pad_max

        # (optional) sanity check to avoid inverted boxes
        if not torch.all(mn < mx):
            raise ValueError(f"expand produced invalid AABB: min {mn} not < max {mx}")

        return SceneBox(aabb=torch.stack([mn, mx], dim=0))

    # ---- unification ----
    def union(self, other: "SceneBox") -> "SceneBox":
        mn = torch.minimum(self.aabb[0], other.aabb[0])
        mx = torch.maximum(self.aabb[1], other.aabb[1])
        return SceneBox(aabb=torch.stack([mn, mx], dim=0))

    @staticmethod
    def reduce_union(aabbs: Float[Tensor, "k 2 3"]) -> "SceneBox":
        mn = aabbs[:, 0, :].min(dim=0).values
        mx = aabbs[:, 1, :].max(dim=0).values
        return SceneBox(aabb=torch.stack([mn, mx], dim=0))


@dataclass
class OrientedBox:
    R: Float[Tensor, "3 3"]
    """R: rotation matrix."""
    T: Float[Tensor, "3"]
    """T: translation vector."""
    S: Float[Tensor, "3"]
    """S: scale vector."""

    def within(self, pts: Float[Tensor, "n 3"]):
        """Returns a boolean mask indicating whether each point is within the box."""
        R, T, S = self.R, self.T, self.S.to(pts)
        H = torch.eye(4, device=pts.device, dtype=pts.dtype)
        H[:3, :3] = R
        H[:3, 3] = T
        H_world2bbox = torch.inverse(H)
        pts = torch.cat((pts, torch.ones_like(pts[..., :1])), dim=-1)
        pts = torch.matmul(H_world2bbox, pts.T).T[..., :3]

        comp_l = torch.tensor(-S / 2)
        comp_m = torch.tensor(S / 2)
        mask = torch.all(torch.concat([pts > comp_l, pts < comp_m], dim=-1), dim=-1)
        return mask

    @staticmethod
    def from_params(
        pos: Tuple[float, float, float],
        rpy: Tuple[float, float, float],
        scale: Tuple[float, float, float],
    ):
        """Construct a box from position, rotation, and scale parameters."""
        R = torch.tensor(vtf.SO3.from_rpy_radians(rpy[0], rpy[1], rpy[2]).as_matrix())
        T = torch.tensor(pos)
        S = torch.tensor(scale)
        return OrientedBox(R=R, T=T, S=S)

    def to_aabb(self) -> Tuple[Tensor, Tensor]:
        # Tight world-aligned AABB of an OBB with center T, rotation R, full sizes S
        device, dtype = self.T.device, self.T.dtype
        half = (self.S * 0.5).to(device=device, dtype=dtype)  # (3,)
        Rabs = self.R.to(device=device, dtype=dtype).abs()  # (3,3)
        ext = Rabs @ half  # (3,)
        aabb_min = self.T - ext
        aabb_max = self.T + ext
        return aabb_min, aabb_max
