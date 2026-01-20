import math

import torch
import torch.nn as nn

from models.metamodule import MetaModule, MetaSequential, MetaBatchLinear


class Sine(nn.Module):
    """Sine activation with frequency scaling (used in SIREN)."""

    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class PositionalEncoding(nn.Module):
    """
    Positional encoding module that encodes each input coordinate as
    a concatenation of sine and cosine signals of increasing frequencies.

    Converts x to: [..., sin(2^k * x), cos(2^k * x), ...]
    where k is linearly spaced between [0, max_freq].
    """

    def __init__(self, max_freq, num_freqs):
        super().__init__()
        freqs = 2 ** torch.linspace(0, max_freq, num_freqs)
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        x_proj = x.unsqueeze(-2) * self.freqs.unsqueeze(
            -1
        )  # shape: [..., num_freqs, in_features]
        x_proj = x_proj.reshape(*x.shape[:-1], -1)
        out = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        return out


class MetaSirenLayer(MetaModule):
    """
    A single meta-learnable SIREN layer.
    Uses frequency-aware initialization specific to sine activations.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        c=6.0,
        is_first=False,
        is_final=False,
        w0_type="uniform",
    ):
        super().__init__()
        self.linear = MetaBatchLinear(dim_in, dim_out)
        self.activation = nn.Identity() if is_final else Sine(w0)
        self.w0_type = w0_type
        self.init_(c=c, w0=w0, is_first=is_first)

    def init_(self, c, w0, is_first):
        dim_in = self.linear.weight.size(1)
        w_std = 1.0 / dim_in if is_first else (math.sqrt(c / dim_in) / w0)

        if self.w0_type == "uniform":
            nn.init.uniform_(self.linear.weight, -w_std, w_std)
            nn.init.uniform_(self.linear.bias, -w_std, w_std)
        elif self.w0_type == "sparse":
            nn.init.sparse_(self.linear.weight, sparsity=0.1)
            nn.init.uniform_(self.linear.bias, -w_std, w_std)
        elif self.w0_type == "orthogonal":
            nn.init.orthogonal_(self.linear.weight)
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

    def forward(self, x, params=None):
        return self.activation(self.linear(x, self.get_subdict(params, "linear")))


class MetaReLULayer(MetaModule):
    """
    A single meta-learnable layer with ReLU activation.
    Uses standard ReLU-style He initialization.
    """

    def __init__(self, dim_in, dim_out, w0=30.0, c=6.0, is_first=False, is_final=False):
        super().__init__()
        self.linear = MetaBatchLinear(dim_in, dim_out)
        self.activation = nn.Identity() if is_final else nn.ReLU()
        self.init_(c=c, w0=w0, is_first=is_first)

    def init_(self, c, w0, is_first):
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity="relu")
        self.linear.bias.data.fill_(0.0)

    def forward(self, x, params=None):
        return self.activation(self.linear(x, self.get_subdict(params, "linear")))


class MetaReLU(MetaModule):
    """
    Full meta-learnable network using ReLU layers with positional encoding.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers=5,
        w0=30.0,
        w0_initial=30.0,
        data_type="img",
        data_size=(128, 128, 3),
        w0_type="uniform",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.data_type = data_type
        self.w0 = w0

        layers = [PositionalEncoding(8, 20)]  # Add PE at input
        for ind in range(num_layers - 1):
            is_first = ind == 0
            layer_dim_in = 2 * dim_in * 20 if is_first else dim_hidden
            layers.append(
                MetaReLULayer(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=w0_initial if is_first else w0,
                    is_first=is_first,
                )
            )
        layers.append(
            MetaReLULayer(dim_in=dim_hidden, dim_out=dim_out, w0=w0, is_final=True)
        )

        self.layers = MetaSequential(*layers)

    def forward(self, x, params=None):
        return self.layers(x, params=self.get_subdict(params, "layers")) + 0.5


class MetaSiren(MetaModule):
    """
    Full meta-learnable SIREN network (for continuous signal representation).
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers=4,
        w0=30.0,
        w0_initial=30.0,
        data_type="img",
        data_size=(3, 178, 178),
        w0_type="uniform",
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.w0 = w0
        self.dim_out = dim_out

        layers = []
        for ind in range(num_layers - 1):
            is_first = ind == 0
            layer_dim_in = dim_in if is_first else dim_hidden
            layers.append(
                MetaSirenLayer(
                    dim_in=layer_dim_in,
                    dim_out=dim_hidden,
                    w0=w0_initial if is_first else w0,
                    is_first=is_first,
                    w0_type=w0_type,
                )
            )
        layers.append(
            MetaSirenLayer(
                dim_in=dim_hidden,
                dim_out=dim_out,
                w0=w0,
                is_final=True,
                w0_type=w0_type,
            )
        )

        self.layers = MetaSequential(*layers)

    def forward(self, x, params=None):
        return self.layers(x, params=self.get_subdict(params, "layers")) + 0.5


class ModularMetaSiren(MetaModule):
    """
    Modular meta-learnable SIREN network.
    Consists of K submodules (each a full MetaSiren network), with input-dependent routing.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers=4,
        w0=30.0,
        w0_initial=30.0,
        data_type="img",
        data_size=(3, 178, 178),
        w0_type="uniform",
        num_submodules=4,  # K: Number of submodules
        routing_order="colwise",  # Routing strategy
    ):
        super().__init__()
        self.num_submodules = num_submodules
        self.routing_order = routing_order
        self.data_type = data_type
        self.dim_out = dim_out
        self.submodules = nn.ModuleList(
            [
                MetaSiren(
                    dim_in,
                    dim_hidden,
                    dim_out,
                    num_layers,
                    w0=w0,
                    w0_initial=w0_initial,
                    data_type=data_type,
                    data_size=data_size,
                    w0_type=w0_type,
                )
                for _ in range(num_submodules)
            ]
        )

    def forward(self, x, params=None, region_ids=None):
        """
        Modular forward for batched coordinates.

        Args:
            x: (B, N, 2) input coords
            params: dict of meta-parameters (optional)
            region_ids: (N,) Optional tensor of region assignments for each point

        Returns:
            output: (B, N_active, D_out)
        """
        B, N, _ = x.shape
        if region_ids is None:
            region_ids = self._route_coords(x[0])  # compute from input coords

        out = x.new_zeros(B, N, self.dim_out)
        for region_id in torch.unique(region_ids).tolist():
            idx = (region_ids == region_id).nonzero(as_tuple=True)[0]  # (Nr,)
            if idx.numel() == 0:
                continue

            x_sub = x[:, idx, :]  # (B, Nr, 2)
            sub_params = (
                None
                if params is None
                else {
                    key[len(f"submodules.{region_id}.") :]: val
                    for key, val in params.items()
                    if key.startswith(f"submodules.{region_id}.")
                }
            )
            out[:, idx, :] = self.submodules[region_id](x_sub, params=sub_params)

        return out

    def _route_coords(self, coords):
        # coords: (N, 2) = (y, x) in [-1, 1]
        if self.routing_order in ("colwise", "rowwise"):
            bins = torch.linspace(
                -1.0, 1.0, self.num_submodules + 1, device=coords.device
            )
            if self.routing_order == "colwise":  # split by columns (x)
                region_ids = torch.bucketize(coords[:, 1].contiguous(), bins) - 1
            else:  # split by rows (y)
                region_ids = torch.bucketize(coords[:, 0].contiguous(), bins) - 1

        elif self.routing_order == "raster":
            n = int(self.num_submodules**0.5)
            assert n * n == self.num_submodules
            xbins = torch.linspace(-1.0, 1.0, n + 1, device=coords.device)
            ybins = torch.linspace(-1.0, 1.0, n + 1, device=coords.device)
            x_id = torch.bucketize(coords[:, 1].contiguous(), xbins) - 1
            y_id = torch.bucketize(coords[:, 0].contiguous(), ybins) - 1
            region_ids = y_id * n + x_id
        else:
            raise NotImplementedError

        return region_ids.clamp_(0, self.num_submodules - 1)


class MetaSirenPenultimate(MetaModule):
    """
    SIREN-based meta-network that exposes the penultimate feature representation.
    Useful for auxiliary tasks like regularization or mutual information.
    """

    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers=4,
        w0=30.0,
        w0_initial=30.0,
        data_type="img",
        data_size=(3, 178, 178),
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.w0 = w0

        self.layers = MetaSequential(
            *[
                MetaSirenLayer(
                    dim_in=(dim_in if ind == 0 else dim_hidden),
                    dim_out=dim_hidden,
                    w0=(w0_initial if ind == 0 else w0),
                    is_first=(ind == 0),
                )
                for ind in range(num_layers - 1)
            ]
        )

        self.last_layer = MetaSirenLayer(
            dim_in=dim_hidden, dim_out=dim_out, w0=w0, is_final=True
        )

    def forward(self, x, params=None, get_features=False):
        feature = self.layers(x, params=self.get_subdict(params, "layers"))
        out = (
            self.last_layer(feature, params=self.get_subdict(params, "last_layer"))
            + 0.5
        )

        if get_features:
            return out, feature
        else:
            return out
