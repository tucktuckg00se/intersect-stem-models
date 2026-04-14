from functools import wraps
from collections import namedtuple

from packaging import version

import torch
from torch import einsum, nn
import torch.nn.functional as F


FlashAttentionConfig = namedtuple(
    "FlashAttentionConfig",
    ["enable_flash", "enable_math", "enable_mem_efficient"],
)


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return None
        called = True
        return fn(x)

    return inner


print_once = once(print)


class Attend(nn.Module):
    def __init__(self, dropout: float = 0.0, flash: bool = False, scale=None):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.flash = flash
        assert not (
            flash and version.parse(torch.__version__) < version.parse("2.0.0")
        ), "flash attention requires PyTorch 2.0+"

        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        if device_properties.major == 8 and device_properties.minor == 0:
            print_once("A100 GPU detected, enabling flash attention on CUDA")
            self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            print_once("Non-A100 GPU detected, using math or mem-efficient attention on CUDA")
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        is_cuda = q.is_cuda
        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        config = self.cuda_config if is_cuda else self.cpu_config
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
            )

    def forward(self, q, k, v):
        scale = default(self.scale, q.shape[-1] ** -0.5)
        if self.flash:
            return self.flash_attn(q, k, v)
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
        attn = self.attn_dropout(sim.softmax(dim=-1))
        return einsum("b h i j, b h j d -> b h i d", attn, v)

