from __future__ import annotations

import hashlib
from pathlib import Path

import requests

from .registry import ModelSpec


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, output_path: str | Path, *, chunk_size: int = 1024 * 1024) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(output, "wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    return output


def ensure_source_assets(spec: ModelSpec, assets_dir: str | Path, *, force: bool = False) -> tuple[Path, Path]:
    assets_root = Path(assets_dir)
    model_root = assets_root / spec.model_id
    config_path = model_root / "source" / "BS-Rofo-SW-Fixed.yaml"
    checkpoint_path = model_root / "source" / "BS-Rofo-SW-Fixed.ckpt"

    if force or not config_path.exists():
        download_file(spec.source_config_url, config_path)
    if force or not checkpoint_path.exists():
        download_file(spec.source_checkpoint_url, checkpoint_path)

    return config_path, checkpoint_path

