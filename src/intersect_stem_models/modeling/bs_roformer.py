from __future__ import annotations

from functools import partial

import torch
from torch import Tensor, einsum, nn, tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from beartype import beartype
from beartype.typing import Callable
from einops import pack, rearrange, unpack
from rotary_embedding_torch import RotaryEmbedding

from .attend import Attend


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        rotary_embed=None,
        flash=True,
        learned_value_residual_mix=False,
    ):
        super().__init__()
        self.heads = heads
        dim_inner = heads * dim_head
        self.rotary_embed = rotary_embed
        self.attend = Attend(flash=flash, dropout=dropout)
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_value_residual_mix = (
            nn.Linear(dim, heads) if learned_value_residual_mix else None
        )
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x, value_residual=None):
        x = self.norm(x)
        q, k, v = rearrange(
            self.to_qkv(x), "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads
        )
        orig_v = v
        if exists(self.to_value_residual_mix):
            mix = rearrange(self.to_value_residual_mix(x), "b n h -> b h n 1").sigmoid()
            assert exists(value_residual)
            v = v.lerp(value_residual, mix)
        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)
        out = self.attend(q, k, v)
        gates = rearrange(self.to_gates(x), "b n h -> b h n 1").sigmoid()
        out = rearrange(out * gates, "b h n d -> b n (h d)")
        return self.to_out(out), orig_v


class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        flash_attn=True,
        add_value_residual=False,
        num_residual_streams=1,
        num_residual_fracs=1,
    ):
        super().__init__()
        del num_residual_streams, num_residual_fracs
        self.layers = ModuleList([])
        for _ in range(depth):
            self.layers.append(
                ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            dropout=attn_dropout,
                            rotary_embed=rotary_embed,
                            flash=flash_attn,
                            learned_value_residual_mix=add_value_residual,
                        ),
                        FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout),
                    ]
                )
            )
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x, value_residual=None):
        first_values = None
        for attn, ff in self.layers:
            attn_out, next_values = attn(x, value_residual=value_residual)
            x = x + attn_out
            first_values = default(first_values, next_values)
            x = x + ff(x)
        return self.norm(x), first_values


class BandSplit(Module):
    @beartype
    def __init__(self, dim, dim_inputs: tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList(
            [nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim)) for dim_in in dim_inputs]
        )

    def forward(self, x):
        chunks = x.split(self.dim_inputs, dim=-1)
        return torch.stack(
            [layer(chunk) for chunk, layer in zip(chunks, self.to_features)],
            dim=-2,
        )


def MLP(dim_in, dim_out, dim_hidden=None, depth=1, activation=nn.Tanh):
    dim_hidden = default(dim_hidden, dim_in)
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)
    layers = []
    for index, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = index == len(dims) - 2
        layers.append(nn.Linear(layer_dim_in, layer_dim_out))
        if not is_last:
            layers.append(activation())
    return nn.Sequential(*layers)


class MaskEstimator(Module):
    @beartype
    def __init__(self, dim, dim_inputs: tuple[int, ...], depth, mlp_expansion_factor=4):
        super().__init__()
        dim_hidden = dim * mlp_expansion_factor
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList(
            [
                nn.Sequential(
                    MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                    nn.GLU(dim=-1),
                )
                for dim_in in dim_inputs
            ]
        )

    def forward(self, x):
        bands = x.unbind(dim=-2)
        return torch.cat([mlp(band) for band, mlp in zip(bands, self.to_freqs)], dim=-1)


DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)


class BSRoformer(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        freqs_per_bands: tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
        freq_range: tuple[int, int] | None = None,
        dim_head=64,
        heads=8,
        attn_dropout=0.0,
        ff_dropout=0.0,
        flash_attn=True,
        num_residual_streams=4,
        num_residual_fracs=1,
        dim_freqs_in=1025,
        stft_n_fft=2048,
        stft_hop_length=512,
        stft_win_length=2048,
        stft_normalized=False,
        zero_dc=False,
        stft_window_fn: Callable | None = None,
        mask_estimator_depth=2,
        multi_stft_resolution_loss_weight=1.0,
        multi_stft_resolutions_window_sizes: tuple[int, ...] = (4096, 2048, 1024, 512, 256),
        multi_stft_hop_size=147,
        multi_stft_normalized=False,
        multi_stft_window_fn: Callable = torch.hann_window,
    ):
        super().__init__()
        del dim_freqs_in
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            num_residual_streams=num_residual_streams,
            num_residual_fracs=num_residual_fracs,
            norm_output=False,
        )
        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Transformer(
                            depth=time_transformer_depth,
                            rotary_embed=time_rotary_embed,
                            add_value_residual=False,
                            **transformer_kwargs,
                        ),
                        Transformer(
                            depth=freq_transformer_depth,
                            rotary_embed=freq_rotary_embed,
                            add_value_residual=False,
                            **transformer_kwargs,
                        ),
                    ]
                )
            )
        self.final_norm = RMSNorm(dim)
        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized,
        )
        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)
        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, return_complex=True).shape[1]
        freq_range = default(freq_range, (-1, -1))
        min_freq, max_freq = freq_range
        min_freq = 0 if min_freq == -1 else min_freq
        max_freq = freqs if max_freq == -1 else max_freq
        assert min_freq >= 0 and max_freq <= freqs and min_freq < max_freq
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.freq_slice = slice(min_freq, max_freq)
        self.freq_pad = (min_freq, freqs - max_freq)
        freqs = max_freq - min_freq
        assert len(freqs_per_bands) > 1
        assert sum(freqs_per_bands) == freqs
        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)
        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)
        self.mask_estimators = nn.ModuleList(
            [
                MaskEstimator(
                    dim=dim,
                    dim_inputs=freqs_per_bands_with_complex,
                    depth=mask_estimator_depth,
                )
                for _ in range(num_stems)
            ]
        )
        self.zero_dc = min_freq == 0 and zero_dc
        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn
        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized,
        )

    def waveform_to_spectrum(self, raw_audio: torch.Tensor) -> torch.Tensor:
        device = raw_audio.device
        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")
        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2)
        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, "* t")
        stft_window = self.stft_window_fn(device=device)
        stft_repr = torch.stft(
            raw_audio,
            **self.stft_kwargs,
            window=stft_window,
            return_complex=False,
        )
        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, "* f t c")
        stft_repr = stft_repr[:, :, self.freq_slice]
        return rearrange(stft_repr, "b s f t c -> b (f s) t c")

    def separator_core(self, stft_repr: torch.Tensor) -> torch.Tensor:
        x = self.band_split(rearrange(stft_repr, "b f t c -> b t (f c)"))
        time_v_residual = None
        freq_v_residual = None
        for time_transformer, freq_transformer in self.layers:
            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x, next_time_v_residual = time_transformer(x, value_residual=time_v_residual)
            time_v_residual = default(time_v_residual, next_time_v_residual)
            x, = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x, next_freq_v_residual = freq_transformer(x, value_residual=freq_v_residual)
            freq_v_residual = default(freq_v_residual, next_freq_v_residual)
            x, = unpack(x, ps, "* f d")
        x = self.final_norm(x)
        return torch.stack([fn(x) for fn in self.mask_estimators], dim=1)

    def apply_masks(self, stft_repr: torch.Tensor, mask_logits: torch.Tensor) -> torch.Tensor:
        mask = rearrange(mask_logits, "b n t (f c) -> b n f t c", c=2)
        stft_repr = rearrange(stft_repr, "b f t c -> b 1 f t c")
        stft_real = stft_repr[..., 0]
        stft_imag = stft_repr[..., 1]
        mask_real = mask[..., 0]
        mask_imag = mask[..., 1]
        out_real = stft_real * mask_real - stft_imag * mask_imag
        out_imag = stft_real * mask_imag + stft_imag * mask_real
        return torch.stack((out_real, out_imag), dim=-1)

    def spectrum_to_waveform(self, masked_stft_repr: torch.Tensor) -> torch.Tensor:
        device = masked_stft_repr.device
        num_stems = masked_stft_repr.shape[1]
        stft_repr = rearrange(masked_stft_repr, "b n (f s) t c -> (b n s) f t c", s=self.audio_channels)
        if any(self.freq_pad):
            pad_left, pad_right = self.freq_pad
            stft_repr = F.pad(stft_repr, (0, 0, 0, 0, pad_left, pad_right))
        stft_repr = torch.view_as_complex(stft_repr.contiguous())
        if self.zero_dc:
            stft_repr = stft_repr.index_fill(1, tensor(0, device=device), 0.0)
        stft_window = self.stft_window_fn(device=device)
        recon_audio = torch.istft(
            stft_repr,
            **self.stft_kwargs,
            window=stft_window,
            return_complex=False,
        )
        recon_audio = rearrange(
            recon_audio,
            "(b n s) t -> b n s t",
            s=self.audio_channels,
            n=num_stems,
        )
        if num_stems == 1:
            recon_audio = rearrange(recon_audio, "b 1 s t -> b s t")
        return recon_audio

    def forward(self, raw_audio, target=None, return_loss_breakdown=False):
        stft_repr = self.waveform_to_spectrum(raw_audio)
        mask_logits = self.separator_core(stft_repr)
        masked_stft_repr = self.apply_masks(stft_repr, mask_logits)
        recon_audio = self.spectrum_to_waveform(masked_stft_repr)
        if not exists(target):
            return recon_audio
        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems
        if target.ndim == 2:
            target = rearrange(target, "... t -> ... 1 t")
        target = target[..., : recon_audio.shape[-1]]
        loss = F.l1_loss(recon_audio, target)
        multi_stft_resolution_loss = 0.0
        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )
            recon_y = torch.stft(rearrange(recon_audio, "... s t -> (... s) t"), **res_stft_kwargs)
            target_y = torch.stft(rearrange(target, "... s t -> (... s) t"), **res_stft_kwargs)
            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_y, target_y)
        weighted_multi_resolution_loss = (
            multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight
        )
        total_loss = loss + weighted_multi_resolution_loss
        if not return_loss_breakdown:
            return total_loss
        return total_loss, (loss, multi_stft_resolution_loss)


class BSRoformerSpectralCore(Module):
    def __init__(self, model: BSRoformer):
        super().__init__()
        self.model = model

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        mask_logits = self.model.separator_core(spectrum)
        return self.model.apply_masks(spectrum, mask_logits)


class BSRoformerWaveformModel(Module):
    def __init__(self, model: BSRoformer):
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.model(audio)
