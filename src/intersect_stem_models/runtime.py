from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import AttrDict, load_yaml_config
from .modeling.bs_roformer import BSRoformer, BSRoformerSpectralCore, BSRoformerWaveformModel


def _tupleize_model_args(model_config: dict[str, Any]) -> dict[str, Any]:
    tuple_params = {"multi_stft_resolutions_window_sizes", "freqs_per_bands", "freq_range"}
    result = dict(model_config)
    for name in tuple_params:
        if name in result and isinstance(result[name], list):
            result[name] = tuple(result[name])
    return result


def get_model_from_config(model_type: str, config: AttrDict) -> torch.nn.Module:
    if model_type != "bs_roformer":
        raise ValueError(f"Unsupported model type: {model_type}")
    model_config = _tupleize_model_args(config.model.to_dict())
    valid_params = {
        "dim",
        "depth",
        "stereo",
        "num_stems",
        "time_transformer_depth",
        "freq_transformer_depth",
        "freqs_per_bands",
        "freq_range",
        "dim_head",
        "heads",
        "attn_dropout",
        "ff_dropout",
        "flash_attn",
        "num_residual_streams",
        "num_residual_fracs",
        "dim_freqs_in",
        "stft_n_fft",
        "stft_hop_length",
        "stft_win_length",
        "stft_normalized",
        "zero_dc",
        "stft_window_fn",
        "mask_estimator_depth",
        "multi_stft_resolution_loss_weight",
        "multi_stft_resolutions_window_sizes",
        "multi_stft_hop_size",
        "multi_stft_normalized",
        "multi_stft_window_fn",
    }
    return BSRoformer(**{k: v for k, v in model_config.items() if k in valid_params})


def load_model(
    config_path: str | Path,
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
    force_disable_flash_attn: bool = False,
) -> tuple[torch.nn.Module, AttrDict]:
    config = load_yaml_config(config_path)
    if force_disable_flash_attn and "flash_attn" in config.model.to_dict():
        config.model.to_dict()["flash_attn"] = False
    model = get_model_from_config("bs_roformer", config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        cleaned = {}
        for key, value in state.items():
            key = key.removeprefix("model.")
            key = key.removeprefix("module.")
            cleaned[key] = value
        state = cleaned
    model.load_state_dict(state, strict=True)
    model.eval()
    return model.to(device), config


def build_waveform_wrapper(model: BSRoformer) -> torch.nn.Module:
    wrapper = BSRoformerWaveformModel(model)
    wrapper.eval()
    return wrapper


def build_spectral_core_wrapper(model: BSRoformer) -> torch.nn.Module:
    wrapper = BSRoformerSpectralCore(model)
    wrapper.eval()
    return wrapper
