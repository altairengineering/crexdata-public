from typing import Optional, Dict, Literal, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor

from models.metamodule import MetaModule, MetaLinear, MetaLayerBlock, MetaSequential
from models.trunc_exp import trunc_exp
from ..encodings import FrequencyEncoder, SHEncoder, FrequencyEncoder as DirFreqEnc


class MetaNeRF(MetaModule):
    """
    Meta-learnable NeRF with Fourier xyz, NeRF-style skips, and view-dependent color.
    """

    # *--------------- init ---------------*
    def __init__(
        self,
        *,
        hidden: int = 256,
        sigma_depth: int = 8,
        skips: Tuple[int, ...] = (4,),
        geo_feat_dim: int = 15,
        color_hidden: int = 128,
        color_depth: int = 2,
        use_sigmoid_rgb: bool = True,
        # xyz Fourier PE
        pe_dim_xyz: int = 10,
        include_input_xyz: bool = True,
        # dir encoding
        encoding_dir: Literal["spherical", "frequency"] = "spherical",
        include_input_dir: bool = True,
    ) -> None:
        super().__init__()

        self.hidden = int(hidden)
        self.sigma_depth = int(sigma_depth)
        self.skips = tuple(int(s) for s in skips)
        self.geo_feat_dim = int(geo_feat_dim)
        self.color_hidden = int(color_hidden)
        self.color_depth = int(color_depth)
        self.use_sigmoid_rgb = bool(use_sigmoid_rgb)

        # *--------------- xyz encoder ---------------*
        self.xyz_encoder = FrequencyEncoder(
            in_dim=3,
            pe_dim=int(pe_dim_xyz),
            include_input=bool(include_input_xyz),
            use_pi=False,
        )
        in_ch_xyz = self.xyz_encoder.out_dim

        # *--------------- direction encoder ---------------*
        encoding_dir = encoding_dir.lower()
        if encoding_dir == "spherical":
            self.dir_encoder = SHEncoder(degree=4)
            in_ch_dir = self.dir_encoder.out_dim
        elif encoding_dir == "frequency":
            self.dir_encoder = DirFreqEnc(
                in_dim=3, pe_dim=4, include_input=include_input_dir, use_pi=False
            )
            in_ch_dir = self.dir_encoder.out_dim
        else:
            raise ValueError(f"Unsupported encoding_dir: {encoding_dir}")

        # *--------------- sigma trunk with NeRF skip ---------------*
        self.trunk = nn.ModuleList()
        for i in range(self.sigma_depth):
            in_dim = in_ch_xyz if i == 0 else self.hidden
            if i in self.skips and i != 0:
                in_dim = self.hidden + in_ch_xyz
            self.trunk.append(MetaLayerBlock(in_dim, self.hidden, activation="relu"))

        self.sigma_head = MetaLinear(self.hidden, 1)
        self.geo_head = MetaLinear(self.hidden, self.geo_feat_dim)
        self.sigma_act = trunc_exp

        # *--------------- color MLP ---------------*
        layers = []
        for l in range(self.color_depth):
            in_dim = self.geo_feat_dim + in_ch_dir if l == 0 else self.color_hidden
            is_last = l == self.color_depth - 1
            if is_last:
                layers.append(("color_out", MetaLinear(in_dim, 3)))
            else:
                layers.append(
                    (
                        f"layer{l}",
                        MetaLayerBlock(in_dim, self.color_hidden, activation="relu"),
                    )
                )
        self.color_mlp = MetaSequential(OrderedDict(layers))
        self.rgb_act = nn.Sigmoid() if self.use_sigmoid_rgb else nn.Identity()

        self._in_ch_xyz = in_ch_xyz  # cached for external uses if needed

    # *--------------- encoders ---------------*
    def _enc_xyz(self, x: Tensor) -> Tensor:
        """Encode xyz with Fourier features."""
        return self.xyz_encoder(x)

    def _enc_dir(self, d: Tensor) -> Tensor:
        """Encode view directions."""
        return self.dir_encoder(d)

    # *----------------------- network calls -------------------------*
    def color(
        self,
        d: Tensor,
        geo_feat: Tensor,
        params: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Color branch: [geo_feat, dir_enc] → color MLP → rgb.
        """
        d_enc = self._enc_dir(d)
        h_rgb = torch.cat([geo_feat, d_enc], dim=-1)
        h_rgb = self.color_mlp(h_rgb, params=self.get_subdict(params, "color_mlp"))
        return self.rgb_act(h_rgb)

    def density(
        self,
        x: Tensor,
        params: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        Density branch: xyz → trunk (with skips) → {'sigma': (...,1), 'geo_feat': (...,G)}.
        """
        enc = self._enc_xyz(x)
        h = enc
        for i, blk in enumerate(self.trunk):
            if i in self.skips and i != 0:
                h = torch.cat([h, enc], dim=-1)
            h = blk(h, params=self.get_subdict(params, f"trunk.{i}"))

        sigma_raw = self.sigma_head(h, params=self.get_subdict(params, "sigma_head"))
        geo_feat = self.geo_head(h, params=self.get_subdict(params, "geo_head"))
        sigma = self.sigma_act(sigma_raw)
        return {"sigma": sigma, "geo_feat": geo_feat}

    def forward(
        self,
        x: Tensor,  # (...,3)
        d: Tensor,  # (...,3)
        params: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        Full NeRF query: returns {'sigma': (...,1), 'rgb': (...,3)}.
        """
        g = self.density(x, params=params)
        rgb = self.color(d, g["geo_feat"], params=params)
        return {"sigma": g["sigma"], "rgb": rgb}
