"""
The trunc exp implementation is an extension of
https://github.com/ashawkey/torch-ngp/blob/main/activation.py

Copyright (c) 2022 hawkey

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
"""

import torch
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd

_EXP_MAX = {
    torch.float16: 11.089866488,  # ~log(65504)
    torch.bfloat16: 88.722839111,  # ~log(3.4e38)
    torch.float32: 88.722839111,
    torch.float64: 709.782712893,
}


def _exp_clamp(x: torch.Tensor) -> torch.Tensor:
    m = _EXP_MAX.get(x.dtype, _EXP_MAX[torch.float32])
    return x.clamp(-m, m)


class _TruncExpFn(Function):
    @staticmethod
    @custom_fwd()
    def forward(ctx, x):
        xc = _exp_clamp(x)
        y = torch.exp(xc)
        ctx.save_for_backward(xc)
        return y

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_out):
        (xc,) = ctx.saved_tensors
        # dy/dx = exp(xc) — same clamp as forward
        return grad_out * torch.exp(xc)


def trunc_exp(x: torch.Tensor) -> torch.Tensor:
    return _TruncExpFn.apply(x)
