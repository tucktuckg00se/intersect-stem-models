#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from intersect_stem_models.export import export_default_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Export BS RoFormer SW to ONNX.")
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1], type=Path)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--export-chunk-size",
        type=int,
        default=131072,
        help="Fixed waveform chunk size to bake into the ONNX graph.",
    )
    args = parser.parse_args()

    result = export_default_model(
        args.repo_root,
        force_download=args.force_download,
        opset=args.opset,
        export_chunk_size=args.export_chunk_size,
    )
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
