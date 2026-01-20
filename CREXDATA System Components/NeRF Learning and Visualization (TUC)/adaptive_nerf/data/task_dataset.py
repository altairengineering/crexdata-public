import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple

import torch
from torch.utils.data import IterableDataset


@dataclass
class Task:
    """One episode: support/query sampled from a single spatial cell."""

    support: Dict[str, torch.Tensor]
    query: Dict[str, torch.Tensor]
    cell_id: Optional[int] = None  # region (expert) id
    block_id: Optional[int] = None  # selected micro-cell id
    bounds: Optional[torch.Tensor] = None  # [2,3] cell AABB
    support_imgs: Optional[List[int]] = None  # unique, freq-sorted (debug)
    query_imgs: Optional[List[int]] = None  # unique, freq-sorted (debug)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


# -------------------------
# Micro-cell episodic dataset over RamRaysDataset
# -------------------------
class TaskDataset(IterableDataset):
    """
    Episodic NeRF dataset that samples support/query rays from a single voxel cell.

    Each episode selects a cell, then draws ray-disjoint (and when possible
    image-disjoint) support and query sets using a reproducible RNG.
    """

    def __init__(
        self,
        ram_ds,
        cell_id: int,
        S_target: int = 4000,
        Q_target: int = 2000,
        min_rays_cell: int = 6000,
        image_cap: Optional[
            float
        ] = None,  # e.g. 0.4 = max 40% per split from one image
        max_images_support: Optional[int] = 8,
        max_images_query: Optional[int] = 4,
        min_images_support: int = 2,
        min_images_query: int = 1,
        region_bounds: Optional[
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ] = None,
        cells: Tuple[int, int, int] = (1, 6, 6),  # (nx, ny, nz)
        cell_pick: str = "uniform",  # "uniform" or "sequential"
        assignment_checkpoint: float = 0.7,
        routing_policy: str = "alpha",  # "alpha" or "dda"
        image_disjoint_splits: bool = True,
        overlap_bias_exponent: float = 0.6,
        debug: bool = False,
        seed: int = 0,
    ):
        super().__init__()

        # Validate RamRaysDataset internals (must be fully in RAM)
        for attr in ("_rays", "_rgbs", "_img_indices"):
            if not hasattr(ram_ds, attr):
                raise ValueError(
                    f"RamRaysDataset missing attribute {attr}; got type {type(ram_ds)}"
                )

        self.rays = ram_ds._rays  # [N, 8]: [o(3), d(3), near, far]
        self.rgbs = ram_ds._rgbs  # [N, 3]
        self.imgix = ram_ds._img_indices  # [N]
        self.uv = None  # optional [N, 2] pixel coords

        self.max_images_support = max_images_support
        self.max_images_query = max_images_query
        self.min_images_support = int(min_images_support)
        self.min_images_query = int(min_images_query)

        self.seed = int(seed)
        self.rng = torch.Generator(device=torch.device("cpu"))
        self.rng.manual_seed(self.seed)

        self.cell_id = int(cell_id)
        self.S_target = int(S_target)
        self.Q_target = int(Q_target)
        self.min_rays_cell = int(min_rays_cell)
        self.image_cap = image_cap
        self.region_bounds_in = region_bounds
        self.cells = cells
        self.cell_pick = cell_pick
        self.assignment_checkpoint = float(max(0.0, min(1.0, assignment_checkpoint)))
        self.routing_policy = str(routing_policy).lower()
        assert self.routing_policy in ("alpha", "dda")
        self.image_disjoint_splits = bool(image_disjoint_splits)
        self.overlap_bias_exponent = float(overlap_bias_exponent)
        self.debug = bool(debug)

        self.N_total = int(self.rays.shape[0])
        self.device = self.rays.device

        # Region AABB and per-cell bounds
        self.aabb = self._init_region_aabb(self.rays, region_bounds)
        self.cell_bounds, self.cell_sizes = self._build_cell_bounds(
            self.aabb, self.cells, self.device
        )  # [C, 2, 3], [C, 3]

        # Route rays to cells and build caches
        cell_bins = self._route_and_bin(
            self.rays, self.aabb, self.cells, self.assignment_checkpoint
        )
        self._build_cell_cache(cell_bins)
        del cell_bins

        # Cells with enough rays for an episode
        self._cursor = 0
        self.eligible_cells = [
            i
            for i, total in enumerate(self._cell_total_counts)
            if total >= self.min_rays_cell
        ]
        if len(self.eligible_cells) == 0:
            warnings.warn(
                f"[Region {self.cell_id}] No eligible cells (min_rays_cell={self.min_rays_cell})."
            )

    # ---------- geometry ----------
    @staticmethod
    def _aabb_intersect(
        o: torch.Tensor, d: torch.Tensor, aabb: torch.Tensor, eps: float = 1e-12
    ):
        """Slab test supporting aabb [2,3] or [N,2,3]. Returns (hit, t_entry, t_exit)."""
        if aabb.dim() == 2:
            lo, hi = aabb[0], aabb[1]  # [3], [3]
        elif aabb.dim() == 3:
            lo, hi = aabb[:, 0, :], aabb[:, 1, :]  # [N,3], [N,3]
        else:
            raise ValueError(f"aabb must be [2,3] or [N,2,3], got {tuple(aabb.shape)}")

        parallel = d.abs() < eps
        inv_d = 1.0 / d
        t0 = (lo - o) * inv_d
        t1 = (hi - o) * inv_d
        tmin = torch.minimum(t0, t1)
        tmax = torch.maximum(t0, t1)
        outside_parallel = parallel & ~((o >= lo) & (o <= hi))
        miss_parallel = outside_parallel.any(dim=1)
        t_entry = tmin.max(dim=1).values
        t_exit = tmax.min(dim=1).values
        hit = (t_exit >= t_entry) & ~miss_parallel
        return hit, t_entry, t_exit

    @staticmethod
    def _region_segment(rays: torch.Tensor, aabb: torch.Tensor):
        """Clip ray to region and (near,far). Returns valid mask and [t0,t1], seg length."""
        o = rays[:, 0:3]
        d = rays[:, 3:6]
        have_bounds = rays.shape[1] >= 8
        near = rays[:, 6] if have_bounds else None
        far = rays[:, 7] if have_bounds else None
        hit, t_entry, t_exit = TaskDataset._aabb_intersect(o, d, aabb)
        zero = torch.zeros_like(t_entry)
        t0 = torch.maximum(t_entry, zero)
        t1 = t_exit.clone()
        if have_bounds:
            t0 = torch.maximum(t0, near)
            t1 = torch.minimum(t1, far)
        seg = t1 - t0
        valid = hit & (seg > 0)
        return valid, t0, t1, seg

    @staticmethod
    def _build_cell_bounds(
        aabb: torch.Tensor, cells: Tuple[int, int, int], device: torch.device
    ):
        """Make per-cell AABBs and sizes."""
        nx, ny, nz = cells
        lo, hi = aabb[0], aabb[1]
        size = (hi - lo).clamp(min=1e-9)
        xs = torch.linspace(0, 1, steps=nx + 1, device=device)
        ys = torch.linspace(0, 1, steps=ny + 1, device=device)
        zs = torch.linspace(0, 1, steps=nz + 1, device=device)
        x0, y0, z0 = xs[:-1], ys[:-1], zs[:-1]
        x1, y1, z1 = xs[1:], ys[1:], zs[1:]
        X0, Y0, Z0 = torch.meshgrid(x0, y0, z0, indexing="ij")
        X1, Y1, Z1 = torch.meshgrid(x1, y1, z1, indexing="ij")
        lo_norm = torch.stack([X0, Y0, Z0], dim=-1).reshape(-1, 3)
        hi_norm = torch.stack([X1, Y1, Z1], dim=-1).reshape(-1, 3)
        cell_bounds = torch.stack(
            [lo + size * lo_norm, lo + size * hi_norm], dim=1
        )  # [C,2,3]
        cell_sizes = (cell_bounds[:, 1] - cell_bounds[:, 0]).abs()  # [C,3]
        return cell_bounds, cell_sizes

    @staticmethod
    def _map_points_to_block_ids(
        p: torch.Tensor, aabb: torch.Tensor, cells: Tuple[int, int, int]
    ):
        """Map 3D points to grid cell indices (floor)."""
        nx, ny, nz = cells
        lo, hi = aabb[0], aabb[1]
        rel = ((p - lo) / (hi - lo).clamp(min=1e-9)).clamp(0.0, 1.0 - 1e-7)
        ix = torch.floor(rel[:, 0] * nx).to(torch.int64).clamp(0, nx - 1)
        iy = torch.floor(rel[:, 1] * ny).to(torch.int64).clamp(0, ny - 1)
        iz = torch.floor(rel[:, 2] * nz).to(torch.int64).clamp(0, nz - 1)
        cid = (ix * (ny * nz)) + (iy * nz) + iz
        return cid, ix, iy, iz

    @staticmethod
    def _overlap_len_with_cell(rays: torch.Tensor, cell_aabb: torch.Tensor):
        """Parametric length inside cell for each ray (>=0)."""
        o = rays[:, :3]
        d = rays[:, 3:6]
        have_bounds = rays.shape[1] >= 8
        near = rays[:, 6] if have_bounds else None
        far = rays[:, 7] if have_bounds else None
        hit, te, tx = TaskDataset._aabb_intersect(o, d, cell_aabb)
        zero = torch.zeros_like(te)
        t0 = torch.maximum(te, zero)
        t1 = tx.clone()
        if have_bounds:
            t0 = torch.maximum(t0, near)
            t1 = torch.minimum(t1, far)
        len_t = torch.where(hit, (t1 - t0).clamp_min(0.0), torch.zeros_like(te))
        return len_t

    @staticmethod
    def _init_region_aabb(rays: torch.Tensor, region_bounds):
        """Region bounds: given or inferred from near points."""
        if region_bounds is not None:
            return torch.tensor(region_bounds, dtype=torch.float32, device=rays.device)
        o = rays[:, 0:3]
        d = rays[:, 3:6]
        t = rays[:, 6:7]
        pts = o + d * t
        lo = pts.min(dim=0).values
        hi = pts.max(dim=0).values
        return torch.stack([lo, hi], dim=0)

    # ---------- DDA (max-overlap) ----------
    def _dda_transform(self, o: torch.Tensor, d: torch.Tensor):
        """World→grid transform: g = (p - lo)/cell_size (per-axis)."""
        lo = self.aabb[0]
        hi = self.aabb[1]
        cell = (hi - lo) / torch.tensor(self.cells, device=o.device, dtype=o.dtype)
        cell = torch.clamp(cell, min=1e-12)
        g_o = (o - lo) / cell
        g_d = d / cell
        return g_o, g_d

    @staticmethod
    def _dda_init(g_o: torch.Tensor, g_d: torch.Tensor, t0: torch.Tensor):
        """Init DDA at t0+: voxel indices, step dirs, tMax, tDelta in grid units."""
        eps = 1e-6
        p = g_o + g_d * (t0 + eps).unsqueeze(-1)
        ix = torch.floor(p[:, 0]).to(torch.int64)
        iy = torch.floor(p[:, 1]).to(torch.int64)
        iz = torch.floor(p[:, 2]).to(torch.int64)
        step_x = torch.sign(g_d[:, 0]).to(torch.int64).clamp(min=-1, max=1)
        step_y = torch.sign(g_d[:, 1]).to(torch.int64).clamp(min=-1, max=1)
        step_z = torch.sign(g_d[:, 2]).to(torch.int64).clamp(min=-1, max=1)

        def next_boundary(coord, step):
            return torch.where(
                step > 0, torch.floor(coord) + 1.0, torch.ceil(coord) - 1.0
            )

        nbx = next_boundary(p[:, 0], step_x)
        nby = next_boundary(p[:, 1], step_y)
        nbz = next_boundary(p[:, 2], step_z)
        inv_dx = 1.0 / g_d[:, 0]
        inv_dy = 1.0 / g_d[:, 1]
        inv_dz = 1.0 / g_d[:, 2]
        tMaxX = (nbx - p[:, 0]) * inv_dx
        tMaxY = (nby - p[:, 1]) * inv_dy
        tMaxZ = (nbz - p[:, 2]) * inv_dz
        tDeltaX = step_x.to(p.dtype) * inv_dx
        tDeltaY = step_y.to(p.dtype) * inv_dy
        tDeltaZ = step_z.to(p.dtype) * inv_dz
        big = 1e30
        for arr in (tMaxX, tMaxY, tMaxZ, tDeltaX, tDeltaY, tDeltaZ):
            arr.nan_to_num_(nan=big, posinf=big, neginf=big)
        return (
            ix,
            iy,
            iz,
            step_x,
            step_y,
            step_z,
            tMaxX,
            tMaxY,
            tMaxZ,
            tDeltaX,
            tDeltaY,
            tDeltaZ,
        )

    def _dda_maxoverlap(
        self,
        rays: torch.Tensor,
        t0: torch.Tensor,
        t1: torch.Tensor,
        max_steps: int = 64,
    ):
        """Traverse voxels in [t0,t1]; return (best_cid, best_len) by in-cell path.
        EXACT behavior: fixed max_steps (default 64), no dynamic caps.
        """
        o = rays[:, :3]
        d = rays[:, 3:6]
        g_o, g_d = self._dda_transform(o, d)

        (ix, iy, iz, sx, sy, sz, tMaxX, tMaxY, tMaxZ, tDeltaX, tDeltaY, tDeltaZ) = (
            self._dda_init(g_o, g_d, t0)
        )

        nx, ny, nz = self.cells
        nyz = ny * nz
        N = rays.shape[0]

        ix = ix.clamp(0, nx - 1)
        iy = iy.clamp(0, ny - 1)
        iz = iz.clamp(0, nz - 1)
        t = t0.clone()

        best_len = torch.zeros(N, device=rays.device, dtype=rays.dtype)
        best_cid = ((ix * nyz) + (iy * nz) + iz).to(torch.int64)

        for _ in range(max_steps):
            m = torch.min(torch.min(tMaxX, tMaxY), tMaxZ)
            t_next = torch.minimum(m, t1)
            dt = (t_next - t).clamp_min(0.0)
            cid = ((ix * nyz) + (iy * nz) + iz).to(torch.int64)
            improve = dt > best_len
            best_len = torch.where(improve, dt, best_len)
            best_cid = torch.where(improve, cid, best_cid)
            done = t_next >= t1
            if done.all():
                break
            adv_x = (tMaxX <= tMaxY) & (tMaxX <= tMaxZ)
            adv_y = (~(tMaxX <= tMaxY)) & (tMaxY <= tMaxZ)
            adv_z = ~(adv_x | adv_y)
            ix = torch.where(adv_x, (ix + sx).clamp(0, nx - 1), ix)
            iy = torch.where(adv_y, (iy + sy).clamp(0, ny - 1), iy)
            iz = torch.where(adv_z, (iz + sz).clamp(0, nz - 1), iz)
            tMaxX = torch.where(adv_x, tMaxX + tDeltaX, tMaxX)
            tMaxY = torch.where(adv_y, tMaxY + tDeltaY, tMaxY)
            tMaxZ = torch.where(adv_z, tMaxZ + tDeltaZ, tMaxZ)
            t = t_next

        return best_cid, best_len

    # ---------- alpha policy (with local max-overlap) ----------
    def _alpha_local_max_overlap(
        self,
        rays: torch.Tensor,
        aabb: torch.Tensor,
        cells: Tuple[int, int, int],
        alpha: float,
        t0: torch.Tensor,
        t1: torch.Tensor,
        seg: torch.Tensor,
    ):
        """α-point cell + 6-neighbor max-overlap; returns (cid_final, len_best) as before."""
        device = rays.device
        o = rays[:, :3]
        d = rays[:, 3:6]

        # α point (nudged inside segment)
        t_assign = t0 + alpha * (t1 - t0)
        t_assign = t_assign + 1e-6 * (t1 - t0)
        p_assign = o + d * t_assign.unsqueeze(-1)

        cid_primary, ix, iy, iz = self._map_points_to_block_ids(p_assign, aabb, cells)

        # Hoisted neighbor offsets (no behavior change)
        if not hasattr(self, "_nbr_dx"):
            self._nbr_dx = torch.tensor([-1, 1, 0, 0, 0, 0, 0], device=device)
            self._nbr_dy = torch.tensor([0, 0, -1, 1, 0, 0, 0], device=device)
            self._nbr_dz = torch.tensor([0, 0, 0, 0, -1, 1, 0], device=device)

        nx, ny, nz = cells
        nyz = ny * nz
        cand_ix = (ix.unsqueeze(1) + self._nbr_dx).clamp(0, nx - 1)
        cand_iy = (iy.unsqueeze(1) + self._nbr_dy).clamp(0, ny - 1)
        cand_iz = (iz.unsqueeze(1) + self._nbr_dz).clamp(0, nz - 1)

        # Ensure last candidate is the primary
        cand_ix[:, -1] = ix
        cand_iy[:, -1] = iy
        cand_iz[:, -1] = iz

        cand_cid = (cand_ix * nyz) + (cand_iy * nz) + cand_iz  # [N,7]
        cand_cid = cand_cid.to(torch.int64)

        # Per-candidate overlaps
        overlaps = []
        for k in range(cand_cid.shape[1]):
            cb = self.cell_bounds.index_select(0, cand_cid[:, k])  # [N,2,3]
            len_t = self._overlap_len_with_cell(rays, cb)
            overlaps.append(len_t.unsqueeze(1))
        overlaps = torch.cat(overlaps, dim=1)  # [N,7]

        best_k = overlaps.argmax(dim=1)
        cid_best = cand_cid[torch.arange(cand_cid.size(0), device=device), best_k]
        len_best = overlaps[torch.arange(cand_cid.size(0), device=device), best_k]

        # Same tolerance as before
        cell_diag = (self.cell_sizes.pow(2).sum(dim=1).sqrt()).median().item()
        tol_abs = max(1e-6 * cell_diag, 1e-9)
        tol_rel = 1e-6
        ok = len_best >= torch.maximum(
            torch.tensor(tol_abs, device=device), tol_rel * seg
        )

        cid_final = torch.where(ok, cid_best, cid_primary)
        return cid_final, len_best

    # ---------- routing & binning ----------
    def _choose_images_for_split(
        self,
        cid: int,
        min_imgs: int,
        max_imgs: Optional[int],
        forbid_imgs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Choose a random set of image ids for this split in cell `cid`.
        - Tries to avoid `forbid_imgs` (for S/Q image-disjointness).
        - Honors min/max constraints. If not enough remain, it will relax by
        borrowing from `forbid_imgs` only to meet `min_imgs`.
        """
        dev = self.device
        all_imgs = torch.unique(self._cell_flat_img[cid])

        if all_imgs.numel() == 0:
            return all_imgs  # empty

        # remove forbidden images first
        if forbid_imgs is not None and forbid_imgs.numel() > 0:
            keep_mask = ~torch.isin(all_imgs, forbid_imgs)
            pool = all_imgs[keep_mask]
        else:
            pool = all_imgs

        # how many to aim for
        Kmax = (
            all_imgs.numel()
            if (max_imgs is None or max_imgs <= 0)
            else min(max_imgs, all_imgs.numel())
        )
        Kmin = max(0, min(min_imgs, Kmax))

        # if we have enough non-forbidden images
        if pool.numel() >= Kmin:
            k = min(Kmax, pool.numel())
            order = torch.randperm(pool.numel(), generator=self.rng, device=dev)
            return pool.index_select(0, order[:k])

        # otherwise: take what we can from pool, then borrow from forbidden to reach Kmin
        chosen = pool
        if (
            forbid_imgs is not None
            and forbid_imgs.numel() > 0
            and chosen.numel() < Kmin
        ):
            # candidates = all_imgs ∩ forbid_imgs
            borrow = all_imgs[torch.isin(all_imgs, forbid_imgs)]
            need = min(Kmin, Kmax) - chosen.numel()
            if need > 0 and borrow.numel() > 0:
                order_b = torch.randperm(
                    borrow.numel(), generator=self.rng, device=dev
                )[: min(need, borrow.numel())]
                chosen = torch.cat([chosen, borrow.index_select(0, order_b)], dim=0)

        # final cap to Kmax
        if chosen.numel() > Kmax:
            order = torch.randperm(chosen.numel(), generator=self.rng, device=dev)[
                :Kmax
            ]
            chosen = chosen.index_select(0, order)

        return chosen

    def _sample_split_from_images(
        self,
        cid: int,
        target: int,
        images: torch.Tensor,
        forbid_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Random ray sampling restricted to `images` within cell `cid`.
        - Pure random from a flat shuffled pool, masked to those images.
        - Hard ray-level disjointness via `forbid_indices`.
        - Soft anti-domination using `image_cap` (if set), otherwise even-ish.
        """
        if target <= 0 or images is None or images.numel() == 0:
            return self._cell_flat_idx[cid][:0]

        flat_idx = self._cell_flat_idx[cid]
        flat_img = self._cell_flat_img[cid]
        dev = self.device

        # pool: only these images
        mask = torch.isin(flat_img, images)

        # forbid indices (S vs Q disjoint)
        if forbid_indices is not None and forbid_indices.numel() > 0:
            mask &= ~torch.isin(flat_idx, forbid_indices)

        pool_idx = flat_idx[mask]
        pool_img = flat_img[mask]

        if pool_idx.numel() == 0:
            return flat_idx[:0]

        need = min(int(target), int(pool_idx.numel()))

        # Fresh random order
        order = torch.randperm(pool_idx.numel(), generator=self.rng, device=dev)

        # No explicit per-image cap -> take first `need`
        if not (self.image_cap is not None and self.image_cap > 0):
            return pool_idx.index_select(0, order[:need])

        # With cap: greedy by image counts
        cap = max(1, int(math.ceil(float(self.image_cap) * need)))
        picked, counts = [], {}
        for pos in order.tolist():
            img_id = int(pool_img[pos].item())
            if counts.get(img_id, 0) >= cap:
                continue
            picked.append(pos)
            counts[img_id] = counts.get(img_id, 0) + 1
            if len(picked) >= need:
                break

        if not picked:
            return pool_idx[:0]
        pos_tensor = torch.tensor(picked, dtype=torch.long, device=dev)
        return pool_idx.index_select(0, pos_tensor)

    def _route_and_bin(
        self,
        rays: torch.Tensor,
        aabb: torch.Tensor,
        cells: Tuple[int, int, int],
        alpha: float,
    ):
        """Assign rays to cells; build bins; filter weak overlaps per cell.
        EXACT decisions; faster via batched recompute of overlap on selected cells.
        """
        device = rays.device
        nx, ny, nz = cells
        num_cells = nx * ny * nz

        valid, t0, t1, seg = self._region_segment(rays, aabb)
        if valid.sum().item() == 0:
            return [
                torch.empty(0, dtype=torch.long, device=device)
                for _ in range(num_cells)
            ]

        idx_valid = torch.nonzero(valid, as_tuple=False).reshape(-1)
        t0_v = t0.index_select(0, idx_valid)
        t1_v = t1.index_select(0, idx_valid)
        seg_v = seg.index_select(0, idx_valid)
        rays_v = rays.index_select(0, idx_valid)

        # --- routing (exact) ---
        if self.routing_policy == "dda":
            cid_final, _ = self._dda_maxoverlap(rays_v, t0_v, t1_v, max_steps=64)
        else:
            cid_final, _ = self._alpha_local_max_overlap(
                rays_v, aabb, cells, alpha, t0_v, t1_v, seg_v
            )

        # --- sort by cell (exactly like before) ---
        order = torch.argsort(cid_final)
        sorted_ids = cid_final.index_select(0, order)
        idx_ord = idx_valid.index_select(
            0, order
        )  # global indices in sorted-by-cell order
        rays_ord = self.rays.index_select(
            0, idx_ord
        )  # gather original rays in that order

        # --- batched recompute of overlap WITH THE SELECTED CELL (exact criterion) ---
        # gather each ray's selected cell AABB in the same order
        cell_aabbs_ord = self.cell_bounds.index_select(0, sorted_ids)  # [Nv, 2, 3]
        len_ord = self._overlap_len_with_cell(rays_ord, cell_aabbs_ord)  # [Nv]

        # per-cell tolerance, gathered in the same order
        size = (self.cell_bounds[:, 1] - self.cell_bounds[:, 0]).norm(dim=1)  # [C]
        tol_per_cell = torch.maximum(1e-6 * size, torch.tensor(1e-9, device=device))
        tol_ord = tol_per_cell.index_select(0, sorted_ids)  # [Nv]

        keep_ord = len_ord >= tol_ord

        # --- segment ranges for each cell (as before) ---
        starts = torch.cat(
            [
                torch.tensor([0], device=device),
                torch.nonzero(
                    sorted_ids[1:] != sorted_ids[:-1], as_tuple=False
                ).reshape(-1)
                + 1,
                torch.tensor([sorted_ids.numel()], device=device),
            ]
        )

        bins = [
            torch.empty(0, dtype=torch.long, device=device) for _ in range(num_cells)
        ]
        for i in range(starts.numel() - 1):
            s = int(starts[i].item())
            e = int(starts[i + 1].item())
            if e <= s:
                continue
            c = int(sorted_ids[s].item())
            # keep relative order within the cell, identical to original
            mask = keep_ord[s:e]
            if mask.any():
                bins[c] = idx_ord[s:e][mask]

        return bins

    # ---------- cached buckets (packed per cell) ----------
    def _build_cell_cache(self, cell_bins: List[torch.Tensor]):
        """Pack per-cell indices as a fully shuffled flat pool (pure randomness)."""
        C = len(cell_bins)
        dev = self.device

        self._cell_concat_idx = []
        self._cell_img_ids = []
        self._cell_img_starts = []
        self._cell_img_lengths = []
        self._cell_total_counts = []

        self._cell_flat_idx = []
        self._cell_flat_img = []

        need_uniq = bool(self.debug or self.image_disjoint_splits)

        for cid in range(C):
            idx_pool = cell_bins[cid]
            N = int(idx_pool.numel())

            if N == 0:
                empty_long = torch.empty(0, dtype=torch.long, device=dev)
                self._cell_flat_idx.append(empty_long)
                self._cell_flat_img.append(empty_long)
                self._cell_concat_idx.append(empty_long)
                self._cell_img_ids.append(empty_long)
                self._cell_img_starts.append(empty_long)
                self._cell_img_lengths.append(empty_long)
                self._cell_total_counts.append(0)
                continue

            # Fully random order (seeded, reproducible)
            perm = torch.randperm(N, generator=self.rng, device=dev)
            flat_idx = idx_pool.index_select(0, perm)
            flat_img = self.imgix.index_select(0, flat_idx)

            self._cell_flat_idx.append(flat_idx)
            self._cell_flat_img.append(flat_img)

            # Legacy fields
            self._cell_concat_idx.append(flat_idx)
            if need_uniq:
                uniq_imgs = torch.unique(flat_img)
            else:
                uniq_imgs = torch.empty(0, dtype=flat_img.dtype, device=dev)
            self._cell_img_ids.append(uniq_imgs)
            self._cell_img_starts.append(torch.empty(0, dtype=torch.long, device=dev))
            self._cell_img_lengths.append(torch.empty(0, dtype=torch.long, device=dev))
            self._cell_total_counts.append(N)

    # ---------- sampling helpers ----------
    @staticmethod
    def _split_support_query(N: int, S_target: int, Q_target: int):
        """Preserve S:Q ratio when underfilled."""
        if N >= (S_target + Q_target):
            return S_target, Q_target
        r = float(S_target) / float(Q_target)
        S = int(round(N * r / (1.0 + r)))
        S = max(0, min(S, N))
        Q = N - S
        return S, Q

    @staticmethod
    def _freq_sorted_unique(img_ix_tensor: torch.Tensor) -> List[int]:
        """Unique image ids sorted by frequency (desc)."""
        vals, cnt = torch.unique(img_ix_tensor, return_counts=True)
        order = torch.argsort(cnt, descending=True)
        return vals[order].tolist()

    def _pick_cell(self) -> Optional[int]:
        """Pick an eligible cell per policy."""
        if not self.eligible_cells:
            return None
        if self.cell_pick == "sequential":
            cid = self.eligible_cells[self._cursor % len(self.eligible_cells)]
            self._cursor += 1
            return int(cid)
        # uniform random among eligible cells
        idx = torch.randint(len(self.eligible_cells), (1,), generator=self.rng).item()
        return int(self.eligible_cells[idx])

    # ---------- image/ray sampling (simple & random) ----------
    def _per_image_cap(self, target: int, Ki: int) -> int:
        """Return a cap per image for the current split; prevents one image from dominating.
        If self.image_cap is given (0<cap<=1), use ceil(cap*target). Otherwise choose a mild
        cap = ceil(target / min(Ki, 8)) so at least ~8 images share when available.
        """
        if target <= 0:
            return 0
        if self.image_cap is not None and self.image_cap > 0:
            return max(1, int(math.ceil(self.image_cap * target)))
        return max(1, int(math.ceil(target / max(1, min(Ki, 8)))))

    def _random_pick_from_bucket(
        self,
        concat_idx: torch.Tensor,
        start: int,
        length: int,
        need: int,
        forbid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if need <= 0 or length <= 0:
            return concat_idx[:0]
        bucket = concat_idx[start : start + length]
        if forbid is not None and forbid.numel() > 0:
            mask = ~torch.isin(bucket, forbid)
            bucket = bucket[mask]
        if bucket.numel() == 0:
            return bucket
        k = min(int(need), int(bucket.numel()))
        perm = torch.randperm(bucket.numel(), generator=self.rng, device=bucket.device)[
            :k
        ]
        return bucket[perm]

    def _sample_split(
        self,
        cid: int,
        target: int,
        prefer_images: Optional[torch.Tensor] = None,
        forbid_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Random rays from the flat, shuffled pool of cell `cid`.
        - If `prefer_images` is given, restrict the pool to those image ids.
        - Always exclude `forbid_indices` (ray-level disjointness).
        - If `image_cap` is set, greedily enforce per-image max; else pure random.
        """
        if target <= 0:
            return self._cell_flat_idx[cid][:0]

        flat_idx = self._cell_flat_idx[cid]
        flat_img = self._cell_flat_img[cid]

        if flat_idx.numel() == 0:
            return flat_idx

        # Start from the whole pool
        mask = torch.ones(flat_idx.numel(), dtype=torch.bool, device=self.device)

        # Restrict to preferred images (if provided)
        if prefer_images is not None and prefer_images.numel() > 0:
            mask &= torch.isin(flat_img, prefer_images)

        # Exclude forbidden ray indices (support vs query disjointness)
        if forbid_indices is not None and forbid_indices.numel() > 0:
            mask &= ~torch.isin(flat_idx, forbid_indices)

        pool_idx = flat_idx[mask]
        pool_img = flat_img[mask]

        if pool_idx.numel() == 0:
            return flat_idx[:0]

        need = min(int(target), int(pool_idx.numel()))

        # Fresh random order every call (reproducible via self.rng)
        order = torch.randperm(pool_idx.numel(), generator=self.rng, device=self.device)

        # No per-image cap -> pure random
        if not (self.image_cap is not None and self.image_cap > 0):
            return pool_idx.index_select(0, order[:need])

        # With per-image cap -> greedy selection
        cap = max(1, int(math.ceil(float(self.image_cap) * need)))
        picked_positions: List[int] = []
        counts: Dict[int, int] = {}

        for pos in order.tolist():
            img_id = int(pool_img[pos].item())
            if counts.get(img_id, 0) >= cap:
                continue
            picked_positions.append(pos)
            counts[img_id] = counts.get(img_id, 0) + 1
            if len(picked_positions) >= need:
                break

        if not picked_positions:
            return pool_idx[:0]

        pos_tensor = torch.tensor(
            picked_positions, dtype=torch.long, device=self.device
        )
        return pool_idx.index_select(0, pos_tensor)

    # ---------- iterator ----------
    def __iter__(self):
        """Yield Task episodes endlessly (NeRF-style)."""
        # Make per-worker RNG deterministic but unique
        info = torch.utils.data.get_worker_info()
        if info is not None:
            # torch initial_seed already different per worker; combine with base seed
            self.rng.manual_seed(self.seed + info.id)

        if len(self.eligible_cells) == 0:
            return

        while True:
            cid = self._pick_cell()
            if cid is None:
                return

            N = self._cell_total_counts[cid]
            if N < self.min_rays_cell:
                continue  # rare due to eligibility

            # S/Q sizes (preserve ratio when underfilled)
            S, Q = self._split_support_query(N, self.S_target, self.Q_target)

            # ---- choose SUPPORT images (random subset)
            supp_imgs = self._choose_images_for_split(
                cid,
                min_imgs=self.min_images_support,
                max_imgs=self.max_images_support,
                forbid_imgs=None,
            )

            # ---- sample SUPPORT rays from chosen images
            sel_S = self._sample_split_from_images(
                cid=cid,
                target=S,
                images=supp_imgs,
                forbid_indices=None,
            )

            # ---- choose QUERY images, preferring image-disjoint from support
            query_imgs = self._choose_images_for_split(
                cid,
                min_imgs=self.min_images_query,
                max_imgs=self.max_images_query,
                forbid_imgs=supp_imgs if self.image_disjoint_splits else None,
            )

            # ---- sample QUERY rays from chosen query images (ray-disjoint from S)
            sel_Q = self._sample_split_from_images(
                cid=cid,
                target=Q,
                images=query_imgs,
                forbid_indices=torch.unique(sel_S),
            )

            # ---- fallback: if Q underfilled and image-disjoint pool was too small, allow borrowing
            if sel_Q.numel() < Q and self.image_disjoint_splits:
                need_more = Q - sel_Q.numel()
                # borrow from all images (including support), but keep ray-level disjointness
                borrow = self._sample_split_from_images(
                    cid=cid,
                    target=need_more,
                    images=torch.unique(self._cell_flat_img[cid]),
                    forbid_indices=torch.unique(torch.cat([sel_S, sel_Q], dim=0)),
                )
                if borrow.numel() > 0:
                    sel_Q = torch.cat([sel_Q, borrow], dim=0)

            # Hard disjointness assertions (debug)
            if self.debug:
                assert sel_S.numel() == sel_S.unique().numel()
                assert sel_Q.numel() == sel_Q.unique().numel()
                assert not self._has_overlap_1d(
                    sel_S, sel_Q
                ), "S/Q rays are not disjoint!"
                if self.image_disjoint_splits:
                    imgs_S = self.imgix.index_select(0, sel_S)
                    imgs_Q = self.imgix.index_select(0, sel_Q)
                    # Allow equality only if no feasible alternative; here we just report
                    if self._has_overlap_1d(imgs_S, imgs_Q):
                        warnings.warn("[debug] S/Q images overlap (fallback path).")
                # geometry sanity
                self._assert_cell_hits(self.rays, self.cell_bounds[cid], sel_S)
                self._assert_cell_hits(self.rays, self.cell_bounds[cid], sel_Q)

            # Pack outputs
            sup = {
                "rays": self.rays.index_select(0, sel_S),
                "rgbs": self.rgbs.index_select(0, sel_S),
                "img_indices": self.imgix.index_select(0, sel_S),
                "idx": sel_S,
            }
            sup["img_ids"] = sup["img_indices"]

            qry = {
                "rays": self.rays.index_select(0, sel_Q),
                "rgbs": self.rgbs.index_select(0, sel_Q),
                "img_indices": self.imgix.index_select(0, sel_Q),
                "idx": sel_Q,
            }
            qry["img_ids"] = qry["img_indices"]

            # Debug extras
            if self.debug and self.uv is not None:
                sup_uv_raw = self.uv.index_select(0, sel_S)
                qry_uv_raw = self.uv.index_select(0, sel_Q)
                sup["uv"] = torch.round(sup_uv_raw - 0.5).to(torch.int64)
                qry["uv"] = torch.round(qry_uv_raw - 0.5).to(torch.int64)

            if self.debug:
                sup_imgs_list = self._freq_sorted_unique(sup["img_indices"])
                qry_imgs_list = self._freq_sorted_unique(qry["img_indices"])
            else:
                sup_imgs_list = None
                qry_imgs_list = None

            # Metrics & warnings
            task_warnings: List[str] = []
            imgs_S = self.imgix.index_select(0, sel_S)
            imgs_Q = self.imgix.index_select(0, sel_Q)
            image_disjoint_ok = float(
                0.0 if self._has_overlap_1d(imgs_S, imgs_Q) else 1.0
            )
            if self.image_disjoint_splits and image_disjoint_ok == 0.0:
                task_warnings.append(
                    "[fallback] borrowed from support images (still ray-disjoint)"
                )

            metrics = {
                "S": float(sup["rays"].shape[0]),
                "Q": float(qry["rays"].shape[0]),
                "total_cell": float(N),
                "num_cells": float(self.cell_bounds.shape[0]),
                "routing_policy": (
                    1.0 if (self.routing_policy == "dda") else 0.0
                ),  # 1=dda, 0=alpha
                "alpha": float(self.assignment_checkpoint),
                "image_disjoint_ok": image_disjoint_ok,
            }

            yield Task(
                support=sup,
                query=qry,
                cell_id=self.cell_id,
                block_id=cid,
                bounds=self.cell_bounds[cid],
                support_imgs=sup_imgs_list,
                query_imgs=qry_imgs_list,
                warnings=task_warnings,
                metrics=metrics,
            )

    def __len__(self):
        return len(self.eligible_cells)

    # ---------- overlap checks ----------
    @staticmethod
    def _has_overlap_1d(a: torch.Tensor, b: torch.Tensor) -> bool:
        """True if any element overlaps between 1D tensors a and b (torch-version agnostic)."""
        if a.numel() == 0 or b.numel() == 0:
            return False
        if hasattr(torch, "isin"):
            return bool(torch.isin(a, b).any().item())
        u, cnt = torch.unique(torch.cat([a, b]), return_counts=True)
        return bool((cnt > 1).any().item())

    @staticmethod
    def _assert_cell_hits(
        rays: torch.Tensor, cell_aabb: torch.Tensor, idx: torch.Tensor
    ):
        """Assert ≥99% of selected rays have positive overlap with the cell."""
        if idx.numel() == 0:
            return
        o = rays.index_select(0, idx)[:, :3]
        d = rays.index_select(0, idx)[:, 3:6]
        near = rays.index_select(0, idx)[:, 6] if rays.shape[1] >= 8 else None
        far = rays.index_select(0, idx)[:, 7] if rays.shape[1] >= 8 else None
        hit, te, tx = TaskDataset._aabb_intersect(o, d, cell_aabb)
        zero = torch.zeros_like(te)
        t0 = torch.maximum(te, zero)
        t1 = tx.clone()
        if near is not None:
            t0 = torch.maximum(t0, near)
            t1 = torch.minimum(t1, far)
        len_t = torch.where(hit, (t1 - t0).clamp_min(0.0), torch.zeros_like(te))
        size = (cell_aabb[1] - cell_aabb[0]).abs()
        tol = 1e-6 * torch.clamp(size.max(), min=torch.tensor(1.0, device=size.device))
        ok = len_t > tol
        frac = float(ok.float().mean().item())
        assert frac > 0.99, f"Only {frac*100:.1f}% of selected rays overlap the cell!"
