import math
from typing import List, Optional, Literal, Dict, Union, Tuple

import torch
from torch import Tensor
from nerfacc import OccGridEstimator

from data.image_metadata import ImageMetadata
from models.metamodule import MetaModule, MetaLinear, MetaLayerBlock, MetaSequential
from models.trunc_exp import trunc_exp
from models.encodings import FrequencyEncoder, HashGridEncoder, SHEncoder
from nerfs.scene_box import SceneBox


class MetaNGP(MetaModule):
    """
    Instant-NGP style meta-learnable NeRF with hash encoding and optional occupancy grid.
    """

    # *--------------- init ---------------*
    def __init__(
        self,
        *,
        occ_conf: Dict,
        scene_box: SceneBox,
        hidden: int = 64,
        sigma_depth: int = 2,
        color_hidden: int = 64,
        geo_feat_dim: int = 15,
        color_depth: int = 3,
        use_sigmoid_rgb: bool = True,
        hash_enc_conf=None,
        dir_encoding: Literal["spherical", "frequency"] = "spherical",
        **kwargs,
    ) -> None:
        super().__init__()
        self.register_buffer("aabb_extent", scene_box.extent)
        self.register_buffer(
            "enc_eps", torch.tensor(1e-6, dtype=torch.float32), persistent=False
        )

        hash_enc_conf = hash_enc_conf or {}
        occ_conf = occ_conf or {}
        self.use_occ = bool(occ_conf.get("use_occ", False))
        self.geo_feat_dim = int(geo_feat_dim)
        self.use_sigmoid_rgb = bool(use_sigmoid_rgb)
        self.scene_box = scene_box
        aabb = scene_box.aabb
        assert isinstance(aabb, torch.Tensor) and aabb.shape == (2, 3)

        # *--------------- xyz / dir encoders ---------------*
        self.xyz_encoder = HashGridEncoder(
            levels=hash_enc_conf.get("levels", 4),
            min_res=hash_enc_conf.get("min_res", 16),
            max_res=hash_enc_conf.get("max_res", 4096),
            log2_hashmap_size=hash_enc_conf.get("log2_hashmap_size", 19),
            features_per_level=hash_enc_conf.get("features_per_level", 2),
            interpolation=hash_enc_conf.get("interpolation", "Linear"),
        )
        in_ch_xyz = self.xyz_encoder.out_dim

        dir_encoding = dir_encoding.lower()
        if dir_encoding == "frequency":
            self.dir_encoder = FrequencyEncoder(
                in_dim=3, pe_dim=4, include_input=True, use_pi=False
            )
            in_ch_dir = self.dir_encoder.out_dim
        elif dir_encoding == "spherical":
            self.dir_encoder = SHEncoder(levels=4)
            in_ch_dir = self.dir_encoder.out_dim
        else:
            raise ValueError(f"Unsupported dir_encoding: {dir_encoding}")

        # *--------------- sigma trunk / heads ---------------*
        sigma_trunk = []
        last = in_ch_xyz
        for _ in range(max(int(sigma_depth), 0)):
            sigma_trunk.append(MetaLayerBlock(last, hidden, activation="relu"))
            last = hidden
        self.sigma_trunk = MetaSequential(*sigma_trunk)

        self.sigma_head = MetaLinear(last, 1)
        with torch.no_grad():
            self.sigma_head.bias.fill_(-1.0)

        self.geo_head = MetaLinear(last, self.geo_feat_dim)
        self.sigma_act = trunc_exp

        # *--------------- color MLP ---------------*
        color_mlp = []
        last = self.geo_feat_dim + in_ch_dir
        for _ in range(max(int(color_depth), 0)):
            color_mlp.append(MetaLayerBlock(last, color_hidden, activation="relu"))
            last = color_hidden
        color_mlp.append(MetaLinear(last, 3))
        self.color_mlp = MetaSequential(*color_mlp)

        self.rgb_act = (
            torch.nn.Sigmoid() if self.use_sigmoid_rgb else torch.nn.Identity()
        )

        # *--------------- occupancy init ---------------*
        if self.use_occ:
            self.render_step_size = (
                float(occ_conf["render_step_size"])
                if "render_step_size" in occ_conf
                and occ_conf["render_step_size"] is not None
                else float(scene_box.get_diagonal_length()) / 1000.0
            )

            self.occ_thre: float = float(occ_conf.get("occ_thre", 1e-2))
            self.alpha_thre: float = float(occ_conf.get("alpha_thre", 1e-2))
            self.cone_angle: float = float(occ_conf.get("cone_angle", 1.0 / 256.0))
            self.near_plane: float = float(occ_conf.get("near_plane", 0.05))
            self.far_plane: float = float(occ_conf.get("far_plane", 1e3))
            self.occ_update_interval: int = int(occ_conf.get("update_interval", 16))
            self.occ_warmup_steps: int = int(occ_conf.get("warmup_steps", 256))
            self.occ_cosine_anneal = bool(occ_conf.get("cosine_anneal", True))
            self.occ_alpha_thre_start: float = float(
                occ_conf.get("alpha_thre_start", 0.0)
            )
            self.occ_alpha_thre_end: float = float(
                occ_conf.get("alpha_thre_end", self.alpha_thre)
            )
            self.occ_ema_decay: float = float(occ_conf.get("ema_decay", 0.95))
            self.occ_resolution: int = int(occ_conf.get("resolution", 128))
            self.occ_levels: int = int(occ_conf.get("levels", 4))

            scene_aabb = torch.cat([self.scene_box.min, self.scene_box.max]).flatten()
            self.register_buffer("scene_aabb", scene_aabb)

            self.occ_grid = OccGridEstimator(
                roi_aabb=self.scene_aabb,
                resolution=self.occ_resolution,
                levels=self.occ_levels,
            )

            self._check_aabb()
            self.occ_frozen: bool = bool(occ_conf.get("occ_frozen", False))
            self.occ_ready: bool = bool(occ_conf.get("occ_ready", False))
            self.num_occ_updates = 0
            self.occ_premarked = False

    # *--------------- encoding helpers ---------------*
    def _check_aabb(self) -> None:
        """Validate AABB for occupancy use."""
        aabb = self.scene_aabb
        assert aabb.dtype == torch.float32 and aabb.isfinite().all()
        assert aabb.numel() == 6
        mn, mx = aabb[:3], aabb[3:]
        if not (mn < mx).all():
            raise ValueError(f"AABB invalid: min>=max ({mn} vs {mx})")
        assert aabb.device == self.occ_grid.aabbs.device

    def _world_to_unit(self, x: Tensor) -> Tensor:
        """Map world coords to [0,1]^3 with clamping for hash grid."""
        x01 = (x - self.scene_box.min) / self.aabb_extent
        return x01.clamp(self.enc_eps, 1.0 - self.enc_eps)

    def _enc_xyz(self, x_world: Tensor) -> Tensor:
        """Hash-encode xyz."""
        x01 = self._world_to_unit(x_world)
        return self.xyz_encoder(x01)

    def _enc_dir(self, d: Tensor) -> Tensor:
        """Encode unit directions with the chosen dir encoder."""
        d = d / d.norm(dim=-1, keepdim=True).clamp_min_(1e-9)
        return self.dir_encoder(d)

    # *----------------------- network calls -------------------------*
    def color(
        self,
        d: Tensor,
        geo_feat: Tensor,
        params: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        View-dependent color from direction and geometry features.

        Args:
            d: (...,3) viewing directions.
            geo_feat: (...,G) geometric features.

        Returns:
            (...,3) RGB in [0,1] if sigmoid is enabled, else linear.
        """
        d_enc = self._enc_dir(d)
        h_rgb = torch.cat([geo_feat, d_enc], dim=-1)
        h_rgb = self.color_mlp(h_rgb, params=self.get_subdict(params, "color_mlp"))
        return self.rgb_act(h_rgb)

    def density(
        self,
        x: Tensor,
        params: Optional[Dict[str, Tensor]] = None,
        return_feats: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        Density and optional geometry features at world coords.

        Args:
            x: (...,3) world coords.
            params: optional meta-params under 'sigma_trunk', 'sigma_head', 'geo_head'.
            return_feats: if True, also return geo features.

        Returns:
            sigma: (...,1) if return_feats=False
            or dict {'sigma': (...,1), 'geo_feat': (...,G)}.
        """
        h = self._enc_xyz(x)
        h = self.sigma_trunk(h, params=self.get_subdict(params, "sigma_trunk"))

        sigma_raw = self.sigma_head(
            h, params=self.get_subdict(params, "sigma_head")
        )  # (...,1)
        sigma = self.sigma_act(sigma_raw)

        if not return_feats:
            return sigma

        geo_feat = self.geo_head(
            h, params=self.get_subdict(params, "geo_head")
        )  # (...,G)
        return {"sigma": sigma, "geo_feat": geo_feat}

    def forward(self, x_d: Tensor, params=None) -> Tensor:
        """
        NeRF-style forward.

        Args:
            x_d: (...,6) concatenated [xyz(3), dir(3)].

        Returns:
            (...,4): [rgb(3), sigma(1)].
        """
        assert x_d.shape[-1] == 6, f"Expected (...,6) [xyz,dir], got {x_d.shape}"
        x, d = x_d[..., :3], x_d[..., 3:6]

        dens = self.density(x, params=params, return_feats=True)
        rgb = self.color(d, dens["geo_feat"], params=params)
        return torch.cat([rgb, dens["sigma"]], dim=-1)

    # *--------------- occupancy helpers ---------------*
    def _anneal_alpha_thre(self, step: int) -> None:
        """Ramp alpha threshold from start→end over warmup, then hold."""
        if step < self.occ_warmup_steps:
            t = step / max(1, self.occ_warmup_steps - 1)
            if self.occ_cosine_anneal:
                cos = 0.5 * (1 - math.cos(math.pi * t))
                self.alpha_thre = (
                    1 - cos
                ) * self.occ_alpha_thre_start + cos * self.occ_alpha_thre_end
            else:
                self.alpha_thre = (
                    1 - t
                ) * self.occ_alpha_thre_start + t * self.occ_alpha_thre_end
        else:
            self.alpha_thre = self.occ_alpha_thre_end

    @torch.no_grad()
    def _build_intrinsics_from_metadata(
        self,
        mds: List[ImageMetadata],
        device: torch.device,
    ) -> Tensor:
        """Stack intrinsics into (N,3,3) K matrices."""

        def _make_K(md: ImageMetadata) -> Tensor:
            Kraw = torch.as_tensor(md.intrinsics, dtype=torch.float32)
            if Kraw.numel() == 9:
                return Kraw.view(3, 3)
            if Kraw.numel() == 4:  # [fx, fy, cx, cy]
                fx, fy, cx, cy = Kraw.unbind()
                return torch.tensor(
                    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    dtype=torch.float32,
                )
            raise ValueError(f"Unsupported intrinsics shape: {tuple(Kraw.shape)}")

        K = torch.stack([_make_K(md) for md in mds], dim=0)
        return K.to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def _build_c2w_rdf_from_metadata(
        self,
        mds: List[ImageMetadata],
        device: torch.device,
    ) -> Tensor:
        """
        Convert c2w from RUB basis to RDF, keeping world frame unchanged.

        Returns:
            (N,3,4) or (N,4,4) c2w in RDF basis.
        """
        c2w = torch.stack(
            [torch.as_tensor(md.c2w, dtype=torch.float32) for md in mds], dim=0
        )

        C3 = torch.diag(
            torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32)
        )  # RUB->RDF

        if c2w.shape[1:] == (3, 4):
            R = c2w[:, :3, :3]
            t = c2w[:, :3, 3:]
            R_rdf = torch.einsum("nij,jk->nik", R, C3)
            c2w_rdf = torch.cat([R_rdf, t], dim=2)
        elif c2w.shape[1:] == (4, 4):
            c2w_rdf = c2w.clone()
            R = c2w_rdf[:, :3, :3]
            R_rdf = torch.einsum("nij,jk->nik", R, C3)
            c2w_rdf[:, :3, :3] = R_rdf
        else:
            raise ValueError(f"Unsupported c2w shape: {tuple(c2w.shape)}")

        return c2w_rdf.to(device=device, dtype=torch.float32)

    # *--------------- occupancy: premark / update / marching ---------------*
    @torch.no_grad()
    def premark_invisible_cells(
        self,
        mds: List[ImageMetadata],
        near_plane: float = 0.05,
        chunk: int = 32**3,
    ) -> None:
        """
        One-time visibility pruning for this expert's occupancy grid.

        Marks cells never visible from any camera as invisible (occ < 0).
        """
        if not self.use_occ or self.occ_premarked:
            return

        mds = [md for md in mds if md is not None]
        if len(mds) == 0:
            print("[OCC] premark skipped: empty metadata list.")
            self.occ_premarked = True
            return

        device = self.occ_grid.aabbs.device
        K = self._build_intrinsics_from_metadata(mds, device)
        c2w_rdf = self._build_c2w_rdf_from_metadata(mds, device)
        H, W = int(mds[0].H), int(mds[0].W)

        self.occ_grid.mark_invisible_cells(
            K=K,
            c2w=c2w_rdf,
            width=W,
            height=H,
            near_plane=float(near_plane),
            chunk=chunk,
        )
        self.occ_premarked = True

    @torch.no_grad()
    def maybe_update_occ_grid(
        self,
        step: int,
        params: Optional[Dict[str, Tensor]] = None,
    ) -> None:
        """
        Periodically update the occupancy grid during training.
        """
        if not (self.training and self.use_occ and not self.occ_frozen):
            return

        self.occ_ready = step >= self.occ_warmup_steps
        self._anneal_alpha_thre(step)

        def occ_eval_fn(x: Tensor) -> Tensor:
            return self.density(x, params=params).squeeze(-1) * self.render_step_size

        self.occ_grid.update_every_n_steps(
            step=step,
            occ_eval_fn=occ_eval_fn,
            occ_thre=self.occ_thre,
            ema_decay=self.occ_ema_decay,
            warmup_steps=self.occ_warmup_steps,
            n=self.occ_update_interval,
        )
        self.num_occ_updates += 1
        if step % self.occ_update_interval == 0:
            print(
                f"[OCC UPDATE {self.num_occ_updates}] "
                f"step={step:5d} warmup={step < self.occ_warmup_steps} "
                f"alpha_thre={self.alpha_thre:6.4f}"
            )

    @torch.no_grad()
    def occupancy_marching(
        self,
        rays: Tensor,
        *,
        params: Optional[Dict[str, Tensor]] = None,
        render_step_size: Optional[float] = None,
        alpha_thre: Optional[float] = None,
        cone_angle: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Volumetric ray marching for this expert using its occupancy grid.

        Args:
            rays: (N,8) [o(3), d(3), t_min, t_max] in the same world frame as the AABB.
            params: optional meta-params for sigma evaluation during training.

        Returns:
            (ray_indices, t_starts, t_ends).
        """
        if getattr(self, "occ_grid", None) is None:
            raise RuntimeError("MetaNGP: occ_grid missing")

        rays = rays.contiguous()
        o = rays[:, :3]
        d = rays[:, 3:6]
        t_min = rays[:, 6]
        t_max = rays[:, 7]

        sigma_fn = None
        if self.training:

            def sigma_fn(
                t_starts: Tensor,
                t_ends: Tensor,
                ray_indices: Tensor,
            ) -> Tensor:
                mids = 0.5 * (t_starts + t_ends)
                x = o[ray_indices] + d[ray_indices] * mids[:, None]
                return self.density(x, params=params).squeeze(-1)

        ray_indices, t_starts, t_ends = self.occ_grid.sampling(
            rays_o=o,
            rays_d=d,
            sigma_fn=sigma_fn,
            t_min=t_min,
            t_max=t_max,
            render_step_size=(
                self.render_step_size if render_step_size is None else render_step_size
            ),
            stratified=self.training,
            cone_angle=(self.cone_angle if cone_angle is None else cone_angle),
            alpha_thre=(self.alpha_thre if alpha_thre is None else alpha_thre),
        )
        return ray_indices, t_starts, t_ends

    # *--------------- optimizer groups ---------------*
    def get_param_groups(self) -> Dict[str, Dict]:
        """
        Parameter groups for optimizer configuration.

        Returns:
            {
                "encoding": {"params": [...]},
                "sigma":    {"params": [...]},
                "color":    {"params": [...]},
            }
        """
        encoding_params = list(self.xyz_encoder.parameters())
        sigma_params = (
            list(self.sigma_trunk.parameters())
            + list(self.sigma_head.parameters())
            + list(self.geo_head.parameters())
        )
        color_params = list(self.color_mlp.parameters())

        return {
            "encoding": {"params": encoding_params},
            "sigma": {"params": sigma_params},
            "color": {"params": color_params},
        }
