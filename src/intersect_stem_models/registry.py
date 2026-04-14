from __future__ import annotations

from dataclasses import dataclass


DEFAULT_MODEL_ID = "bs-roformer-sw-6stem"


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    name: str
    architecture: str
    stems: tuple[str, ...]
    sample_rate: int
    channels: int
    source_config_url: str
    source_checkpoint_url: str
    source_repo: str
    export_model_type: str
    export_filename: str
    manifest_path: str


MODEL_SPECS: dict[str, ModelSpec] = {
    DEFAULT_MODEL_ID: ModelSpec(
        model_id=DEFAULT_MODEL_ID,
        name="BS RoFormer SW by jarredou",
        architecture="bs_roformer",
        stems=("vocals", "drums", "bass", "guitar", "piano", "other"),
        sample_rate=44100,
        channels=2,
        source_config_url="https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.yaml",
        source_checkpoint_url="https://huggingface.co/jarredou/BS-ROFO-SW-Fixed/resolve/main/BS-Rofo-SW-Fixed.ckpt",
        source_repo="https://github.com/openmirlab/bs-roformer-infer",
        export_model_type="bs_roformer",
        export_filename="bs_roformer_sw_6stem.onnx",
        manifest_path="models/manifest.json",
    ),
}


def get_model_spec(model_id: str = DEFAULT_MODEL_ID) -> ModelSpec:
    try:
        return MODEL_SPECS[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown model id: {model_id}") from exc

