from typing import Dict, Optional
from collections import OrderedDict
import re
import warnings
from einops import rearrange

import torch
import torch.nn as nn

from models.trunc_exp import trunc_exp


class MetaModule(nn.Module):
    """Base class for modules that support optional fast weights injection via a params dict."""

    def __init__(self):
        super(MetaModule, self).__init__()
        self._children_modules_parameters_cache = dict()

    def meta_named_parameters(self, prefix: str = "", recurse: bool = True):
        """Iterate over meta-learnable parameters in this module and nested MetaModules."""
        gen = self._named_members(
            lambda module: (
                module._parameters.items() if isinstance(module, MetaModule) else []
            ),
            prefix=prefix,
            recurse=recurse,
        )
        for elem in gen:
            yield elem

    def meta_parameters(self, recurse: bool = True):
        """Yield meta-learnable parameters only."""
        for _, param in self.meta_named_parameters(recurse=recurse):
            yield param

    def get_subdict(
        self, params: Optional[Dict[str, torch.Tensor]], key: Optional[str] = None
    ) -> Optional[OrderedDict]:
        """
        Return sub-dict of params relevant to child module `key`.

        Expects params with keys like "layer.weight", "layer.bias", etc.
        """
        if params is None:
            return None

        all_names = tuple(params.keys())

        if (key, all_names) not in self._children_modules_parameters_cache:
            if key is None:
                self._children_modules_parameters_cache[(key, all_names)] = all_names
            else:
                key_escape = re.escape(key)
                key_re = re.compile(rf"^{key_escape}\.(.+)")
                self._children_modules_parameters_cache[(key, all_names)] = [
                    key_re.sub(r"\1", k) for k in all_names if key_re.match(k)
                ]

        names = self._children_modules_parameters_cache[(key, all_names)]
        if not names:
            warnings.warn(
                f"Module `{self.__class__.__name__}` has no parameter for submodule `{key}` in `params`.\n"
                f"Using default parameters. Provided keys: [{', '.join(all_names)}]",
                stacklevel=2,
            )
            return None

        return OrderedDict([(name, params[f"{key}.{name}"]) for name in names])


class MetaSequential(nn.Sequential, MetaModule):
    """Sequential container that propagates optional meta-params to MetaModules."""

    def forward(self, input, params: Optional[Dict[str, torch.Tensor]] = None):
        for name, module in self._modules.items():
            if isinstance(module, MetaModule):
                input = module(input, params=self.get_subdict(params, name))
            elif isinstance(module, nn.Module):
                input = module(input)
            else:
                raise TypeError(
                    f"The module must be a `nn.Module` or `MetaModule`. Got: {type(module)}"
                )
        return input


class MetaBatchLinear(nn.Linear, MetaModule):
    """
    Batched meta-learnable Linear layer.

    Expects:
      inputs: (B, N, in_features)
      weight: (B, out_features, in_features)
      bias:   (B, out_features)
    """

    def forward(
        self, inputs: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None
    ):
        if params is None:
            params = OrderedDict(self.named_parameters())
            for name, param in params.items():
                params[name] = param[None, ...].repeat(
                    (inputs.size(0),) + (1,) * len(param.shape)
                )

        weight = params["weight"]
        bias = params.get("bias", None)

        if weight.dim() == 2:
            weight = weight.unsqueeze(0)
        if bias is not None:
            if bias.dim() == 1:
                bias = bias.unsqueeze(0)
            elif bias.dim() == 3 and bias.shape[1] == 1:
                bias = bias.squeeze(1)

        inputs = rearrange(inputs, "b n d -> b d n")
        output = torch.bmm(weight, inputs)
        output = rearrange(output, "b d n -> b n d")

        if bias is not None:
            output += bias.unsqueeze(1)

        return output


class MetaLinear(nn.Linear, MetaModule):
    """
    Meta-learnable Linear layer for single-task fast weights.

    inputs: (N, in_features)
    params (optional): {
        "weight": (out_features, in_features),
        "bias":   (out_features,)
    }
    """

    def forward(
        self, inputs: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        if params is None:
            weight = self.weight
            bias = self.bias
        else:
            weight = params.get("weight", self.weight)
            bias = params.get("bias", self.bias)

        if inputs.dtype != weight.dtype:
            inputs = inputs.to(weight.dtype)

        out = inputs.matmul(weight.t())
        if bias is not None:
            out = out + bias
        return out


class MetaLayerBlock(MetaModule):
    """Linear + activation block using MetaLinear or MetaBatchLinear."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: Optional[str] = None,
        batched: bool = False,
    ):
        super().__init__()
        self.linear = (
            MetaLinear(dim_in, dim_out)
            if not batched
            else MetaBatchLinear(dim_in, dim_out)
        )
        if activation is None:
            self.act = nn.Identity()
        elif activation.lower() == "relu":
            self.act = nn.ReLU()
        elif activation.lower() == "sigmoid":
            self.act = nn.Sigmoid()
        elif activation.lower() == "softplus":
            self.act = nn.Softplus()
        elif activation.lower() == "trunc_exp":
            self.act = trunc_exp
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(
        self, x: torch.Tensor, params: Optional[OrderedDict] = None
    ) -> torch.Tensor:
        x = self.linear(x, params=self.get_subdict(params, "linear"))
        return self.act(x)
