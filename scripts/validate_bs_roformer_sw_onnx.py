#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from intersect_stem_models.validate import validate_export


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PyTorch vs ONNX parity for BS RoFormer SW.")
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument(
        "--artifact-kind",
        choices=["spectral_core", "waveform_full"],
        default="spectral_core",
    )
    parser.add_argument("--export-chunk-size", type=int, default=131072)
    args = parser.parse_args()
    result = validate_export(
        args.repo_root,
        artifact_kind=args.artifact_kind,
        export_chunk_size=args.export_chunk_size,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
