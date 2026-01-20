from typing import List, Optional, OrderedDict, Literal, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.image_metadata import ImageMetadata
from models.metamodule import MetaModule
from models.encodings import SHEncoder, FrequencyEncoder
from models.inr.meta_vanilla import MetaNeRF
from models.inr.meta_ngp import MetaNGP


def build_expert(nerf_variant: str, **nerf_kwargs) -> nn.Module:
    """Factory for a single NeRF expert."""
    if nerf_variant == "instant":
        return MetaNGP(**nerf_kwargs)
    return MetaNeRF(**nerf_kwargs)


class MetaContainer(MetaModule):
    """
    Modular NeRF container with routing, background, and occupancy support.

    - Holds K NeRF experts (submodules) and routes queries to them.
    - Supports soft (inverse-distance) or hard (nearest-centroid) routing in DRB.
    - Optionally owns a background MLP and per-expert occupancy grids.
    """

    def __init__(
        self,
        num_submodules: int,
        centroids: torch.Tensor,
        aabb: torch.Tensor,
        nerf_variant: Literal["instant", "vanilla"] = "instant",
        boundary_margin: float = 1.0,
        cluster_2d: bool = True,
        joint_training: bool = False,
        use_bg_nerf: bool = True,
        bg_hidden: int = 32,
        bg_encoding: Literal["spherical", "fourier"] = "spherical",
        occ_conf: Optional[Dict] = None,
        **nerf_kwargs,
    ):
        super().__init__()
        assert num_submodules > 0
        assert centroids.ndim == 2 and centroids.size(0) == num_submodules
        assert boundary_margin >= 1.0

        occ_conf = occ_conf or {}

        self.register_buffer(
            "scene_aabb_vec",
            torch.cat([aabb[0], aabb[1]], dim=0).float(),
            persistent=True,
        )
        self.register_buffer("centroids", centroids.to(torch.float32), persistent=True)

        self.use_occ = bool(occ_conf.get("use_occ", False))
        self.boundary_margin = float(boundary_margin)
        self.cluster_2d = bool(cluster_2d)
        self.joint_training = bool(joint_training)
        self._coord_idx = (1, 2) if self.cluster_2d else (0, 1, 2)
        self.nerf_variant = nerf_variant
        self.dim_out = 4

        # Experts (each with its own SceneBox)
        expert_box_list = nerf_kwargs.pop("expert_box_list")
        expert_kwargs_base = {**nerf_kwargs, "occ_conf": occ_conf}
        self.submodules = nn.ModuleList()
        for box in expert_box_list:
            kw = {**expert_kwargs_base, "scene_box": box}
            self.submodules.append(build_expert(nerf_variant, **kw))

        # Background NeRF-style head
        self.use_bg_nerf = bool(use_bg_nerf)
        if self.use_bg_nerf:
            if bg_encoding == "spherical":
                self.bg_dir_enc = SHEncoder(levels=4, implementation="tcnn")
                in_ch_dir = self.bg_dir_enc.out_dim
            else:
                self.bg_dir_enc = FrequencyEncoder(
                    pe_dim=4, include_input=True, use_pi=False
                )
                in_ch_dir = self.bg_dir_enc.out_dim

            self.bg_hidden_dim = int(bg_hidden)
            self.bg_mlp = nn.Sequential(
                nn.Linear(in_ch_dir, self.bg_hidden_dim, bias=True),
                nn.ReLU(),
                nn.Linear(self.bg_hidden_dim, 3, bias=True),
                nn.Sigmoid(),
            )

    # *----------------------- routing -------------------------*

    def _routing(
        self, pts: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Route 3D points to experts.

        Args:
            pts: (N, 3) points in world DRB.

        Returns:
            (weights, hard_assign):
                - Soft: weights (N, K), None
                - Hard: None, hard_assign (N,)
        """
        assert pts.dim() == 2 and pts.shape[-1] == 3, "pts must be (N,3)"
        N = pts.shape[0]
        K = self.centroids.shape[0]
        assert K > 0, "No centroids provided."

        x_pos = pts.to(device=self.centroids.device, dtype=self.centroids.dtype)
        c_pos = self.centroids[:, :3].to(device=x_pos.device, dtype=x_pos.dtype)

        x_cluster = x_pos[:, self._coord_idx]  # (N, d)
        c_cluster = c_pos[:, self._coord_idx]  # (K, d)

        dist = torch.cdist(x_cluster.float(), c_cluster.float())  # (N, K)
        if self.boundary_margin > 1.0:
            dist = dist.clamp_min(1e-6)
            invd = 1.0 / dist
            mind = dist.min(dim=1, keepdim=True).values  # (N, 1)
            mask = dist <= (self.boundary_margin * mind)
            invd = invd * mask
            denom = invd.sum(dim=1, keepdim=True).clamp_min(1e-6)
            weights = (invd / denom).to(dtype=x_pos.dtype, device=x_pos.device)
            return weights, None

        hard_assign = dist.argmin(dim=1).to(device=x_pos.device)
        return None, hard_assign

    # *----------------------- network calls -------------------------*
    def color(
        self,
        xyz: torch.Tensor,  # (N,3)
        dirs: torch.Tensor,  # (N,3)
        params: Optional[OrderedDict] = None,
        active_module: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Routed RGB query.

        If active_module is set, queries only that expert; otherwise routes per xyz.
        Returns RGB in (N,3).
        """
        assert xyz.dim() == 2 and xyz.shape[-1] == 3, "xyz must be (N,3)"
        assert dirs.dim() == 2 and dirs.shape[-1] == 3, "dirs must be (N,3)"

        if dirs.device != xyz.device:
            dirs = dirs.to(xyz.device)
        dirs = F.normalize(dirs, dim=-1)

        N = xyz.shape[0]
        K = len(self.submodules)

        if params is not None:
            sub_params = [self.get_subdict(params, f"submodules.{k}") for k in range(K)]
        else:
            sub_params = [None] * K

        if active_module is not None:
            sub = self.submodules[active_module]
            dens_out = sub.density(
                xyz, params=sub_params[active_module], return_feats=True
            )
            return sub.color(
                dirs, dens_out["geo_feat"], params=sub_params[active_module]
            )

        with torch.no_grad():
            weights, hard_assign = self._routing(xyz)

        results = xyz.new_zeros(N, 3)

        if weights is not None:
            # Soft: rgb_mix = Σ_k w_k * rgb_k
            for k, sub in enumerate(self.submodules):
                wk = weights[:, k]
                sel = (wk > 0).nonzero(as_tuple=False).squeeze(1)
                if sel.numel() == 0:
                    if self.joint_training:
                        _ = sub.density(
                            xyz[:0], params=sub_params[k], return_feats=True
                        )
                    continue

                sub_xyz = xyz.index_select(0, sel)
                sub_dirs = dirs.index_select(0, sel)
                dens_out = sub.density(sub_xyz, params=sub_params[k], return_feats=True)
                rgb_loc = sub.color(
                    sub_dirs, dens_out["geo_feat"], params=sub_params[k]
                )
                results.index_add_(
                    0, sel, rgb_loc * wk.index_select(0, sel).unsqueeze(1)
                )
            return results

        # Hard: copy rgb from owner expert
        for k, sub in enumerate(self.submodules):
            sel = (hard_assign == k).nonzero(as_tuple=False).squeeze(1)
            if sel.numel() == 0:
                if self.joint_training:
                    _ = sub.density(xyz[:0], params=sub_params[k], return_feats=True)
                continue

            sub_xyz = xyz.index_select(0, sel)
            sub_dirs = dirs.index_select(0, sel)
            dens_out = sub.density(sub_xyz, params=sub_params[k], return_feats=True)
            rgb_loc = sub.color(sub_dirs, dens_out["geo_feat"], params=sub_params[k])
            results.index_copy_(0, sel, rgb_loc)
        return results

    def density(
        self,
        xyz: torch.Tensor,  # (N,3)
        params: Optional[OrderedDict] = None,
        active_module: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Routed density query.

        Returns σ in shape (N,) (squeezed if expert returns (N,1)).
        """
        assert xyz.dim() == 2 and xyz.shape[-1] == 3, "xyz must be (N,3)"
        N = xyz.shape[0]
        K = len(self.submodules)

        if params is not None:
            sub_params = [self.get_subdict(params, f"submodules.{k}") for k in range(K)]
        else:
            sub_params = [None] * K

        if active_module is not None:
            sub = self.submodules[active_module]
            sigma = sub.density(xyz, params=sub_params[active_module])
            return sigma.to(device=xyz.device, dtype=xyz.dtype).squeeze(-1)

        with torch.no_grad():
            weights, hard_assign = self._routing(xyz)

        sigmas = xyz.new_zeros(N)

        if weights is not None:
            for k, sub in enumerate(self.submodules):
                wk = weights[:, k]
                sel = (wk > 0).nonzero(as_tuple=False).squeeze(1)
                if sel.numel() == 0:
                    if self.joint_training:
                        _ = sub.density(xyz[:0], params=sub_params[k])
                    continue

                sub_xyz = xyz.index_select(0, sel)
                sigma_k = sub.density(sub_xyz, params=sub_params[k])
                sigma_k = sigma_k.to(device=xyz.device, dtype=xyz.dtype).squeeze(-1)
                sigmas.index_add_(0, sel, sigma_k * wk.index_select(0, sel))
            return sigmas

        for k, sub in enumerate(self.submodules):
            sel = (hard_assign == k).nonzero(as_tuple=False).squeeze(1)
            if sel.numel() == 0:
                if self.joint_training:
                    _ = sub.density(xyz[:0], params=sub_params[k])
                continue

            sub_xyz = xyz.index_select(0, sel)
            sigma_k = sub.density(sub_xyz, params=sub_params[k])
            sigma_k = sigma_k.to(device=xyz.device, dtype=xyz.dtype).squeeze(-1)
            sigmas.index_copy_(0, sel, sigma_k)
        return sigmas

    def forward(
        self,
        x: torch.Tensor,  # (N, D), D >= 6 (e.g. [o(3), d(3), near, far])
        params: Optional[OrderedDict] = None,
        active_module: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Routed forward through experts.

        - If active_module is not None: run only that expert on all inputs.
        - Otherwise: soft/hard route per-point via centroids in DRB.
        """
        assert x.dim() == 2 and x.shape[-1] >= 6, "x must be (N,D>=6)"
        N, _ = x.shape
        K = len(self.submodules)

        if params is not None:
            sub_params = [self.get_subdict(params, f"submodules.{k}") for k in range(K)]
        else:
            sub_params = [None] * K

        if active_module is not None:
            sub = self.submodules[active_module]
            return sub(x, params=sub_params[active_module])

        with torch.no_grad():
            weights, hard = self._routing(x[:, :3])

        results = None

        if weights is not None:
            # Soft routing: y_mix = Σ_k w_k * y_k
            for k, sub in enumerate(self.submodules):
                wk = weights[:, k]
                sel = (wk > 0).nonzero(as_tuple=False).squeeze(1)
                if sel.numel() == 0:
                    if self.joint_training:
                        _ = sub(x[:0], params=sub_params[k])
                    continue

                xk = x.index_select(0, sel)
                yk = sub(xk, params=sub_params[k])

                if results is None:
                    results = x.new_zeros(N, yk.shape[-1])

                results.index_add_(0, sel, yk * wk.index_select(0, sel).unsqueeze(1))
        else:
            # Hard routing: expert k owns its assigned region
            for k, sub in enumerate(self.submodules):
                sel = (hard == k).nonzero(as_tuple=False).squeeze(1)
                if sel.numel() == 0:
                    if self.joint_training:
                        _ = sub(x[:0], params=sub_params[k])
                    continue

                xk = x.index_select(0, sel)
                yk = sub(xk, params=sub_params[k])

                if results is None:
                    results = x.new_zeros(N, yk.shape[-1])

                results.index_copy_(0, sel, yk)

        if results is None:
            dim_out = getattr(self, "dim_out", x.shape[-1])
            results = x.new_zeros(N, dim_out)

        return results

    # *----------------------- background -----------------------*

    def background_color(self, d: torch.Tensor) -> torch.Tensor:
        """
        Background RGB given ray directions.

        Args:
            d: (N,3) or (B,N,3) directions.

        Returns:
            (N,3) or (B,N,3) RGB in [0,1].
        """
        if not self.use_bg_nerf:
            raise RuntimeError("background_color called but use_bg_nerf=False")

        if d.dim() == 2:
            dn = F.normalize(d, dim=-1)
            enc = self.bg_dir_enc(dn)
            first_linear = self.bg_mlp[0]
            enc = enc.to(
                dtype=first_linear.weight.dtype, device=first_linear.weight.device
            )
            return self.bg_mlp(enc)

        if d.dim() == 3:
            B, N, _ = d.shape
            dn = F.normalize(d, dim=-1).reshape(-1, 3)
            enc = self.bg_dir_enc(dn)
            first_linear = self.bg_mlp[0]
            enc = enc.to(
                dtype=first_linear.weight.dtype, device=first_linear.weight.device
            )
            rgb = self.bg_mlp(enc)
            return rgb.view(B, N, 3)

        raise ValueError(
            f"background_color expects (N,3) or (B,N,3), got {tuple(d.shape)}"
        )

    # *------------------------ expert occupancy handling ------------------------*

    def maybe_update_expert_occupancies(self, step: int, params=None) -> None:
        """Trigger per-expert occupancy grid update if supported."""
        for sub in self.submodules:
            sub.maybe_update_occ_grid(step, params)

    def freeze_expert_occupancies(self, flag: bool) -> None:
        """Freeze/unfreeze occupancy grids for all experts."""
        for sub in self.submodules:
            sub.occ_frozen = flag

    @torch.no_grad()
    def premark_invisible_expert_cells(
        self,
        metas: List[ImageMetadata],
        near_plane: float = 0.0,
        chunk: int = 32**3,
    ) -> List[int]:
        """
        One-time visibility pruning for all experts using camera metadata.

        Args:
            metas: list of camera metadata.
            near_plane: near plane used by ray sampling.
            chunk: chunk size for occ_grid.mark_invisible_cells.

        Returns:
            list[int]: number of invisible cells marked per expert.
        """
        if self.cells_premarked or not self.use_occ:
            return [0] * len(self.submodules)

        marked_counts: List[int] = []
        total_counts: List[int] = []

        for k, expert in enumerate(self.submodules):
            expert.premark_invisible_cells(
                metas,
                near_plane=float(near_plane),
                chunk=chunk,
            )

            n_marked = (
                int((expert.occ_grid.occs < 0).sum().item())
                if hasattr(expert.occ_grid, "occs")
                else 0
            )
            marked_counts.append(n_marked)
            total = (
                expert.occ_grid.occs.numel() if hasattr(expert.occ_grid, "occs") else 0
            )
            total_counts.append(total)
            print(
                f"[OCC] container: expert#{k} cams={len(metas)} "
                f"marked_invisible={n_marked} / {total} "
                f"({100.0 * n_marked / max(1, total):.2f}%)"
            )

        print("[OCC] container: premark complete for all experts.")
        return marked_counts

    @property
    def occ_ready(self) -> bool:
        """True if all experts report occ_ready=True."""
        return all(sub.occ_ready for sub in self.submodules)

    @property
    def cells_premarked(self) -> bool:
        """True if all experts report occ_premarked=True."""
        return all(sub.occ_premarked for sub in self.submodules)

    # *----------------------- param groups ------------------------*

    def get_param_groups(self) -> Dict[str, Dict]:
        """
        Aggregate parameter groups across experts and background.

        Returns a dict with optional keys:
          - "encoding":   expert encoders (hash grids / PE)
          - "sigma":      expert density / geometry MLPs
          - "color":      expert color MLPs
          - "background": bg_dir_enc + bg_mlp
        """
        encoding_params: List[torch.nn.Parameter] = []
        sigma_params: List[torch.nn.Parameter] = []
        color_params: List[torch.nn.Parameter] = []
        bg_params: List[torch.nn.Parameter] = []

        for sub in self.submodules:
            if hasattr(sub, "get_param_groups"):
                sub_groups = sub.get_param_groups()
                enc_group = sub_groups.get("encoding")
                if enc_group is not None and "params" in enc_group:
                    encoding_params.extend(list(enc_group["params"]))
                sig_group = sub_groups.get("sigma")
                if sig_group is not None and "params" in sig_group:
                    sigma_params.extend(list(sig_group["params"]))
                col_group = sub_groups.get("color")
                if col_group is not None and "params" in col_group:
                    color_params.extend(list(col_group["params"]))
            else:
                sigma_params.extend(list(sub.parameters()))

        if self.use_bg_nerf:
            if hasattr(self, "bg_dir_enc"):
                bg_params.extend(list(self.bg_dir_enc.parameters()))
            if hasattr(self, "bg_mlp"):
                bg_params.extend(list(self.bg_mlp.parameters()))

        groups: Dict[str, Dict] = {}
        if encoding_params:
            groups["encoding"] = {"params": encoding_params}
        if sigma_params:
            groups["sigma"] = {"params": sigma_params}
        if color_params:
            groups["color"] = {"params": color_params}
        if bg_params:
            groups["background"] = {"params": bg_params}
        return groups
