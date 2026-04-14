from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from .registry import DEFAULT_MODEL_ID, get_model_spec
from .runtime import build_spectral_core_wrapper, build_waveform_wrapper, load_model


@dataclass(frozen=True)
class ValidationResult:
    artifact_kind: str
    pytorch_shape: tuple[int, ...]
    onnx_shape: tuple[int, ...]
    max_abs_error: float
    mean_abs_error: float


def validate_export(
    repo_root: str | Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    artifact_kind: str = "spectral_core",
    onnx_path: str | Path | None = None,
    config_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    export_chunk_size: int = 131072,
) -> ValidationResult:
    root = Path(repo_root)
    spec = get_model_spec(model_id)
    if artifact_kind == "waveform_full":
        default_onnx = root / "artifacts" / "exports" / model_id / spec.export_filename
    elif artifact_kind == "spectral_core":
        default_onnx = root / "artifacts" / "exports" / model_id / "bs_roformer_sw_6stem_spectral_core.onnx"
    else:
        raise ValueError(f"Unsupported artifact_kind: {artifact_kind}")
    onnx_path = Path(onnx_path or default_onnx)
    source_root = root / "artifacts" / "sources" / model_id / "source"
    config_path = Path(config_path or source_root / "BS-Rofo-SW-Fixed.yaml")
    checkpoint_path = Path(checkpoint_path or source_root / "BS-Rofo-SW-Fixed.ckpt")

    model, _config = load_model(config_path, checkpoint_path, device="cpu", force_disable_flash_attn=True)
    chunk_size = int(export_chunk_size)
    rng = np.random.default_rng(0)
    audio = rng.normal(size=(1, spec.channels, chunk_size)).astype(np.float32)

    if artifact_kind == "waveform_full":
        pytorch_model = build_waveform_wrapper(model)
        input_name = "audio"
        output_name = "stems"
        pytorch_input = torch.from_numpy(audio)
        onnx_input = audio
    else:
        pytorch_model = build_spectral_core_wrapper(model)
        with torch.no_grad():
            spectrum = model.waveform_to_spectrum(torch.from_numpy(audio))
        input_name = "spectrum"
        output_name = "masked_stems"
        pytorch_input = spectrum
        onnx_input = spectrum.numpy()

    with torch.no_grad():
        pytorch_out = pytorch_model(pytorch_input).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run([output_name], {input_name: onnx_input})[0]

    diff = np.abs(pytorch_out - onnx_out)
    return ValidationResult(
        artifact_kind=artifact_kind,
        pytorch_shape=tuple(pytorch_out.shape),
        onnx_shape=tuple(onnx_out.shape),
        max_abs_error=float(diff.max()),
        mean_abs_error=float(diff.mean()),
    )
