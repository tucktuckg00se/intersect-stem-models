from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

import onnx
import torch

from .downloader import ensure_source_assets, sha256_file
from .registry import DEFAULT_MODEL_ID, ModelSpec, get_model_spec
from .runtime import (
    build_spectral_core_wrapper,
    build_waveform_wrapper,
    load_model,
)


@dataclass(frozen=True)
class ArtifactExportResult:
    artifact_kind: str
    onnx_path: str
    input_name: str
    output_name: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    onnx_sha256: str


@dataclass(frozen=True)
class ExportSummary:
    model_id: str
    config_path: str
    checkpoint_path: str
    config_sha256: str
    checkpoint_sha256: str
    waveform_artifact: ArtifactExportResult | None
    spectral_core_artifact: ArtifactExportResult | None
    waveform_failure_path: str | None
    training_chunk_size: int
    export_chunk_size: int


def _chunk_size_from_config(config) -> int:
    if hasattr(config, "audio") and hasattr(config.audio, "chunk_size"):
        return int(config.audio.chunk_size)
    raise ValueError("Config is missing audio.chunk_size")


def export_default_model(
    repo_root: str | Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    force_download: bool = False,
    opset: int = 18,
    export_chunk_size: int = 131072,
) -> ExportSummary:
    root = Path(repo_root)
    spec = get_model_spec(model_id)
    config_path, checkpoint_path = ensure_source_assets(
        spec,
        root / "artifacts" / "sources",
        force=force_download,
    )
    model, config = load_model(
        config_path,
        checkpoint_path,
        device="cpu",
        force_disable_flash_attn=True,
    )
    training_chunk_size = _chunk_size_from_config(config)
    chunk_size = int(export_chunk_size)
    if chunk_size <= 0:
        raise ValueError("export_chunk_size must be positive")

    export_root = root / "artifacts" / "exports" / model_id
    export_root.mkdir(parents=True, exist_ok=True)

    config_sha256 = sha256_file(config_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    waveform_artifact = _try_export_waveform_artifact(
        export_root=export_root,
        model=model,
        spec=spec,
        chunk_size=chunk_size,
        opset=opset,
        training_chunk_size=training_chunk_size,
    )
    spectral_core_artifact = _export_spectral_core_artifact(
        export_root=export_root,
        model=model,
        spec=spec,
        chunk_size=chunk_size,
        opset=opset,
    )

    summary = ExportSummary(
        model_id=model_id,
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        config_sha256=config_sha256,
        checkpoint_sha256=checkpoint_sha256,
        waveform_artifact=waveform_artifact,
        spectral_core_artifact=spectral_core_artifact,
        waveform_failure_path=str(export_root / "waveform-export-failure.json"),
        training_chunk_size=training_chunk_size,
        export_chunk_size=chunk_size,
    )
    _write_export_metadata(root, spec, summary)
    _update_manifest(root, summary)
    return summary


def _try_export_waveform_artifact(
    *,
    export_root: Path,
    model,
    spec: ModelSpec,
    chunk_size: int,
    opset: int,
    training_chunk_size: int,
) -> ArtifactExportResult | None:
    onnx_path = export_root / spec.export_filename
    dummy = torch.randn(1, spec.channels, chunk_size, dtype=torch.float32)
    waveform_model = build_waveform_wrapper(model)
    try:
        with torch.no_grad():
            _export_with_fallback(
                waveform_model,
                dummy,
                onnx_path,
                opset=opset,
                input_name="audio",
                output_name="stems",
            )
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        return ArtifactExportResult(
            artifact_kind="waveform_full",
            onnx_path=str(onnx_path),
            input_name="audio",
            output_name="stems",
            input_shape=(1, spec.channels, chunk_size),
            output_shape=(1, len(spec.stems), spec.channels, chunk_size),
            onnx_sha256=sha256_file(onnx_path),
        )
    except Exception as exc:
        _write_failure_report(
            export_root / "waveform-export-failure.json",
            spec=spec,
            training_chunk_size=training_chunk_size,
            export_chunk_size=chunk_size,
            error=exc,
            artifact_kind="waveform_full",
            next_step=(
                "Use the exported spectral-core artifact for integration while "
                "replacing the waveform STFT/ISTFT path with an ONNX-native implementation."
            ),
        )
        return None


def _export_spectral_core_artifact(
    *,
    export_root: Path,
    model,
    spec: ModelSpec,
    chunk_size: int,
    opset: int,
) -> ArtifactExportResult:
    spectral_model = build_spectral_core_wrapper(model)
    dummy_audio = torch.randn(1, spec.channels, chunk_size, dtype=torch.float32)
    with torch.no_grad():
        dummy_spectrum = model.waveform_to_spectrum(dummy_audio)
    onnx_path = export_root / "bs_roformer_sw_6stem_spectral_core.onnx"
    with torch.no_grad():
        _export_with_fallback(
            spectral_model,
            dummy_spectrum,
            onnx_path,
            opset=opset,
            input_name="spectrum",
            output_name="masked_stems",
        )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    return ArtifactExportResult(
        artifact_kind="spectral_core",
        onnx_path=str(onnx_path),
        input_name="spectrum",
        output_name="masked_stems",
        input_shape=tuple(dummy_spectrum.shape),
        output_shape=(1, len(spec.stems), *tuple(dummy_spectrum.shape[1:])),
        onnx_sha256=sha256_file(onnx_path),
    )


def _write_export_metadata(repo_root: Path, spec: ModelSpec, summary: ExportSummary) -> None:
    export_root = repo_root / "artifacts" / "exports" / summary.model_id
    metadata_path = export_root / "export-metadata.json"
    payload = {
        "model": asdict(spec),
        "summary": asdict(summary),
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _update_manifest(repo_root: Path, summary: ExportSummary) -> None:
    def rel(path_str: str | None) -> str | None:
        if path_str is None:
            return None
        try:
            return str(Path(path_str).resolve().relative_to(repo_root.resolve()))
        except Exception:
            return path_str

    manifest_path = repo_root / "models" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for model in manifest.get("models", []):
        if model.get("model_id") != summary.model_id:
            continue
        model["runtime"]["chunk_size_samples"] = summary.export_chunk_size
        model["artifacts"]["source_config_path"] = rel(summary.config_path)
        model["artifacts"]["source_checkpoint_path"] = rel(summary.checkpoint_path)
        model["artifacts"]["sha256"]["config"] = summary.config_sha256
        model["artifacts"]["sha256"]["checkpoint"] = summary.checkpoint_sha256

        waveform = summary.waveform_artifact
        spectral = summary.spectral_core_artifact
        model["artifacts"]["waveform_full"] = {
            "available": waveform is not None,
            "onnx_path": rel(waveform.onnx_path) if waveform else None,
            "input_name": waveform.input_name if waveform else "audio",
            "output_name": waveform.output_name if waveform else "stems",
            "input_shape": list(waveform.input_shape) if waveform else [1, 2, summary.export_chunk_size],
            "output_shape": list(waveform.output_shape) if waveform else [1, len(model["runtime"]["stems"]), 2, summary.export_chunk_size],
            "sha256": waveform.onnx_sha256 if waveform else None,
            "failure_report_path": rel(summary.waveform_failure_path) if waveform is None else None,
        }
        model["artifacts"]["spectral_core"] = {
            "available": spectral is not None,
            "onnx_path": rel(spectral.onnx_path) if spectral else None,
            "input_name": spectral.input_name if spectral else "spectrum",
            "output_name": spectral.output_name if spectral else "masked_stems",
            "input_shape": list(spectral.input_shape) if spectral else None,
            "output_shape": list(spectral.output_shape) if spectral else None,
            "sha256": spectral.onnx_sha256 if spectral else None,
        }
        model["artifacts"]["onnx_path"] = rel(waveform.onnx_path) if waveform else None
        model["artifacts"]["sha256"]["onnx"] = waveform.onnx_sha256 if waveform else None
        break
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_failure_report(
    path: Path,
    *,
    spec: ModelSpec,
    training_chunk_size: int,
    export_chunk_size: int,
    error: Exception,
    artifact_kind: str,
    next_step: str,
) -> None:
    payload = {
        "model_id": spec.model_id,
        "model_name": spec.name,
        "artifact_kind": artifact_kind,
        "training_chunk_size": training_chunk_size,
        "export_chunk_size": export_chunk_size,
        "error_type": type(error).__name__,
        "error": str(error),
        "next_step": next_step,
        "traceback": traceback.format_exc(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _export_with_fallback(
    model: torch.nn.Module,
    dummy: torch.Tensor,
    onnx_path: Path,
    *,
    opset: int,
    input_name: str,
    output_name: str,
) -> None:
    common_kwargs = dict(
        input_names=[input_name],
        output_names=[output_name],
        opset_version=opset,
        dynamic_axes=None,
        export_params=True,
        do_constant_folding=True,
    )
    try:
        torch.onnx.export(
            model,
            (dummy,),
            onnx_path,
            dynamo=True,
            **common_kwargs,
        )
        return
    except Exception as first_error:
        legacy_path = onnx_path.with_suffix(".legacy-attempt.onnx")
        try:
            torch.onnx.export(
                model,
                (dummy,),
                legacy_path,
                dynamo=False,
                **common_kwargs,
            )
            legacy_path.replace(onnx_path)
            return
        except Exception as second_error:
            raise RuntimeError(
                "ONNX export failed with both the dynamo and legacy exporters. "
                f"Dynamo error: {first_error}. Legacy error: {second_error}"
            ) from second_error
